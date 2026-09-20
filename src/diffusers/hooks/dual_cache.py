# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Tuple, Union

import torch

from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = get_logger(__name__)  # pylint: disable=invalid-name

_DUAL_CACHE_LEADER_BLOCK_HOOK = "dual_cache_leader_block_hook"
_DUAL_CACHE_BLOCK_HOOK = "dual_cache_block_hook"

# Step types produced by ``DualCachePolicy``.
_COMPUTE = "compute"
_AGGRESSIVE = "aggressive"
_CONSERVATIVE = "conservative"


class DualCachePolicy:
    """
    Training-free scheduler that classifies every denoising step into one of the two caching strategies proposed in
    Dual Feature Caching (DuCa), or a fresh full computation.

    Each cache cycle spans ``cache_interval`` steps. The cycle opens with a full ``"compute"`` step that refreshes the
    cached residual, followed by ``aggressive_steps`` ``"aggressive"`` steps (the cached residual is reused verbatim
    while it is freshest) and finally the remaining ``"conservative"`` steps (the cached residual is reused with a
    damping correction as it ages). The two-phase reuse is DuCa's namesake "dual" schedule; keeping it deterministic and
    model-free makes the policy independently unit-testable.
    """

    def __init__(
        self,
        cache_interval: int = 3,
        aggressive_steps: int = 1,
        retention_ratio: float = 0.2,
        num_inference_steps: int = 28,
    ) -> None:
        if cache_interval < 1:
            raise ValueError(f"`cache_interval` must be >= 1, got {cache_interval}.")
        if aggressive_steps < 0:
            raise ValueError(f"`aggressive_steps` must be >= 0, got {aggressive_steps}.")
        # There are at most `cache_interval - 1` reuse slots after the leading compute step.
        self.aggressive_steps = min(aggressive_steps, max(cache_interval - 1, 0))
        self.cache_interval = cache_interval
        self.retention_ratio = retention_ratio
        self.num_inference_steps = num_inference_steps

    @property
    def retention_steps(self) -> int:
        return int(self.retention_ratio * self.num_inference_steps + 0.5)

    def classify(self, step_index: int) -> str:
        """Return the step type (``"compute"``/``"aggressive"``/``"conservative"``) for ``step_index``."""
        if step_index < self.retention_steps:
            return _COMPUTE
        position = (step_index - self.retention_steps) % self.cache_interval
        if position == 0:
            return _COMPUTE
        if position <= self.aggressive_steps:
            return _AGGRESSIVE
        return _CONSERVATIVE


@dataclass
class DualCacheConfig:
    r"""
    Configuration for [Dual Feature Caching (DuCa)](https://huggingface.co/papers/2412.18911).

    DuCa is a training-free feature-caching schedule for Diffusion Transformers. It alternates between an *aggressive*
    strategy that reuses cached block residuals verbatim for maximum speedup and a *conservative* strategy that damps
    the reused residual to arrest the quality drop caused by reusing stale features, refreshing the cache at fixed cycle
    boundaries.

    Args:
        cache_interval (`int`, defaults to `3`):
            Length of each cache cycle (`N` in the paper). A full recomputation happens on the first step of every
            cycle; the remaining `cache_interval - 1` steps reuse cached features.
        aggressive_steps (`int`, defaults to `1`):
            Number of steps immediately after a compute step that reuse the cached residual verbatim. The remaining
            steps of the cycle use the conservative strategy. Clamped to `cache_interval - 1`.
        conservative_scale (`float`, defaults to `0.95`):
            Multiplier applied to the cached residual on conservative steps. Values below `1.0` damp error accumulation
            from aging features. This scalar is a parameter-free proxy for DuCa's ToCa selective token recomputation,
            which the block-level hook architecture cannot host.
        retention_ratio (`float`, defaults to `0.2`):
            Fraction of initial steps during which caching is disabled for stability, mirroring the warmup convention
            used by [`MagCacheConfig`].
        num_inference_steps (`int`, defaults to `28`):
            Number of inference steps used by the pipeline, required to resolve `retention_ratio` into a step count.
    """

    cache_interval: int = 3
    aggressive_steps: int = 1
    conservative_scale: float = 0.95
    retention_ratio: float = 0.2
    num_inference_steps: int = 28

    def get_policy(self) -> DualCachePolicy:
        return DualCachePolicy(
            cache_interval=self.cache_interval,
            aggressive_steps=self.aggressive_steps,
            retention_ratio=self.retention_ratio,
            num_inference_steps=self.num_inference_steps,
        )


def _combine_residual(hidden_states: torch.Tensor, residual: torch.Tensor, scale: float) -> torch.Tensor:
    """Add a (optionally scaled) cached residual back onto ``hidden_states``, tolerating text+image concat layouts."""
    if residual.device != hidden_states.device:
        residual = residual.to(hidden_states.device)
    if scale != 1.0:
        residual = residual * scale

    if residual.shape == hidden_states.shape:
        return hidden_states + residual
    # Flux/SD3-style concatenation: the image tokens sit at the tail of the sequence dimension.
    if (
        hidden_states.ndim == 3
        and residual.ndim == 3
        and hidden_states.shape[0] == residual.shape[0]
        and hidden_states.shape[2] == residual.shape[2]
        and hidden_states.shape[1] > residual.shape[1]
    ):
        diff = hidden_states.shape[1] - residual.shape[1]
        hidden_states = hidden_states.clone()
        hidden_states[:, diff:, :] = hidden_states[:, diff:, :] + residual
        return hidden_states

    logger.warning(
        f"DualCache: cannot align residual {tuple(residual.shape)} with input {tuple(hidden_states.shape)}; "
        "returning input unchanged for this step."
    )
    return hidden_states


class DualCacheState(BaseState):
    def __init__(self) -> None:
        super().__init__()
        self.previous_residual: torch.Tensor = None
        self.head_block_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = None
        self.should_compute: bool = True
        self.step_index: int = 0

    def reset(self):
        self.previous_residual = None
        self.head_block_input = None
        self.should_compute = True
        self.step_index = 0


class DualCacheHeadHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager, config: DualCacheConfig):
        self.state_manager = state_manager
        self.config = config
        self.policy = config.get_policy()
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")

        arg_name = self._metadata.hidden_states_argument_name
        hidden_states = self._metadata._get_parameter_from_args_kwargs(arg_name, args, kwargs)

        state: DualCacheState = self.state_manager.get_state()
        state.head_block_input = hidden_states

        step_type = self.policy.classify(state.step_index)
        # A reuse step can only run once a residual has been cached.
        if step_type == _COMPUTE or state.previous_residual is None:
            state.should_compute = True
            return self.fn_ref.original_forward(*args, **kwargs)

        state.should_compute = False
        scale = 1.0 if step_type == _AGGRESSIVE else self.config.conservative_scale
        logger.debug(f"DualCache: reusing cache at step {state.step_index} ({step_type}, scale={scale})")

        output = _combine_residual(hidden_states, state.previous_residual, scale)

        if self._metadata.return_encoder_hidden_states_index is not None:
            original_encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                "encoder_hidden_states", args, kwargs
            )
            max_idx = max(
                self._metadata.return_hidden_states_index, self._metadata.return_encoder_hidden_states_index
            )
            ret_list = [None] * (max_idx + 1)
            ret_list[self._metadata.return_hidden_states_index] = output
            ret_list[self._metadata.return_encoder_hidden_states_index] = original_encoder_hidden_states
            return tuple(ret_list)
        return output

    def reset_state(self, module):
        self.state_manager.reset()
        return module


class DualCacheBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, config: DualCacheConfig, is_tail: bool = False):
        super().__init__()
        self.state_manager = state_manager
        self.config = config
        self.is_tail = is_tail
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")
        state: DualCacheState = self.state_manager.get_state()

        if not state.should_compute:
            arg_name = self._metadata.hidden_states_argument_name
            hidden_states = self._metadata._get_parameter_from_args_kwargs(arg_name, args, kwargs)
            if self.is_tail:
                self._advance_step(state)
            if self._metadata.return_encoder_hidden_states_index is not None:
                encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                    "encoder_hidden_states", args, kwargs
                )
                max_idx = max(
                    self._metadata.return_hidden_states_index, self._metadata.return_encoder_hidden_states_index
                )
                ret_list = [None] * (max_idx + 1)
                ret_list[self._metadata.return_hidden_states_index] = hidden_states
                ret_list[self._metadata.return_encoder_hidden_states_index] = encoder_hidden_states
                return tuple(ret_list)
            return hidden_states

        output = self.fn_ref.original_forward(*args, **kwargs)

        if self.is_tail:
            out_hidden = output[self._metadata.return_hidden_states_index] if isinstance(output, tuple) else output
            in_hidden = state.head_block_input
            if in_hidden is not None and out_hidden.shape == in_hidden.shape:
                state.previous_residual = out_hidden - in_hidden
            self._advance_step(state)

        return output

    def _advance_step(self, state: DualCacheState):
        state.step_index += 1
        if state.step_index >= self.config.num_inference_steps:
            state.step_index = 0
            state.previous_residual = None


def apply_dual_cache(module: torch.nn.Module, config: DualCacheConfig) -> None:
    """
    Applies [Dual Feature Caching (DuCa)](https://huggingface.co/papers/2412.18911) to a transformer module.

    A [`DualCacheHeadHook`] on the first transformer block decides, per step, whether to run a fresh forward pass or to
    reuse the cached residual (verbatim on aggressive steps, damped on conservative steps). A tail [`DualCacheBlockHook`]
    caches the full-stack residual after each fresh compute.

    Args:
        module (`torch.nn.Module`):
            The transformer module to apply DuCa to.
        config (`DualCacheConfig`):
            The configuration for Dual Feature Caching.
    """
    HookRegistry.check_if_exists_or_initialize(module)

    state_manager = StateManager(DualCacheState, (), {})
    blocks = []
    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for index, block in enumerate(submodule):
            blocks.append((f"{name}.{index}", block))

    if not blocks:
        logger.warning("DualCache: No transformer blocks found to apply hooks.")
        return

    if len(blocks) == 1:
        name, block = blocks[0]
        logger.info(f"DualCache: Applying head+tail hooks to single block '{name}'")
        _apply_dual_cache_block_hook(block, state_manager, config, is_tail=True)
        _apply_dual_cache_head_hook(block, state_manager, config)
        return

    head_block_name, head_block = blocks.pop(0)
    tail_block_name, tail_block = blocks.pop(-1)

    logger.info(f"DualCache: Applying head hook to '{head_block_name}'")
    _apply_dual_cache_head_hook(head_block, state_manager, config)
    for name, block in blocks:
        _apply_dual_cache_block_hook(block, state_manager, config)
    logger.info(f"DualCache: Applying tail hook to '{tail_block_name}'")
    _apply_dual_cache_block_hook(tail_block, state_manager, config, is_tail=True)


def _apply_dual_cache_head_hook(block: torch.nn.Module, state_manager: StateManager, config: DualCacheConfig) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    if registry.get_hook(_DUAL_CACHE_LEADER_BLOCK_HOOK) is not None:
        registry.remove_hook(_DUAL_CACHE_LEADER_BLOCK_HOOK)
    registry.register_hook(DualCacheHeadHook(state_manager, config), _DUAL_CACHE_LEADER_BLOCK_HOOK)


def _apply_dual_cache_block_hook(
    block: torch.nn.Module, state_manager: StateManager, config: DualCacheConfig, is_tail: bool = False
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    if registry.get_hook(_DUAL_CACHE_BLOCK_HOOK) is not None:
        registry.remove_hook(_DUAL_CACHE_BLOCK_HOOK)
    registry.register_hook(DualCacheBlockHook(state_manager, config, is_tail), _DUAL_CACHE_BLOCK_HOOK)
