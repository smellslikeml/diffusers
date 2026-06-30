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

import torch

from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = get_logger(__name__)  # pylint: disable=invalid-name

_CLOCKWORK_LEADER_BLOCK_HOOK = "clockwork_leader_block_hook"
_CLOCKWORK_BLOCK_HOOK = "clockwork_block_hook"


@dataclass
class ClockworkCacheConfig:
    r"""
    Configuration for [Clockwork
    Cache](https://huggingface.co/papers/2312.08128), an inference-time adaptation of the Clockwork Diffusion caching
    insight.

    Clockwork Diffusion observes that the *low-resolution / semantically important* feature maps inside a denoiser
    evolve slowly across diffusion steps, whereas the high-resolution feature maps are sensitive to small
    perturbations. It therefore reuses the slow-changing features across a window of steps and recomputes them only
    periodically, on a fixed "clock" schedule. This hook ports that cross-step reuse policy onto diffusers'
    transformer block stack: every `clock_interval` denoising steps the full block stack is recomputed, and on the
    steps in between the cached block-stack output is replayed instead. This is a *periodic* schedule, in contrast to
    [`apply_first_block_cache`], which decides reuse adaptively from the residual magnitude at every step.

    Note: the original paper additionally *distills* a student model that is robust to the frozen low-resolution
    features ("model-step distillation") to recover the quality lost by freezing them. That training step is out of
    scope for this inference-only hook and is left to whoever trains a compatible checkpoint — the other caches in
    this gallery ship the inference-time acceleration of their papers without the accompanying training procedures,
    and this one follows the same convention.

    Args:
        clock_interval (`int`, defaults to `3`):
            The number of denoising steps between two full recomputations of the transformer block stack. Every
            `clock_interval`-th step recomputes; the steps in between reuse the cached output. `1` disables caching
            (every step recomputes). Must be `>= 1`.
        warmup (`int`, defaults to `1`):
            The number of leading denoising steps that always recompute, to populate and stabilize the cache before
            any reuse happens. Must be `>= 0`.
    """

    clock_interval: int = 3
    warmup: int = 1

    def __post_init__(self):
        if self.clock_interval < 1:
            raise ValueError(f"`clock_interval` must be >= 1, got {self.clock_interval}.")
        if self.warmup < 0:
            raise ValueError(f"`warmup` must be >= 0, got {self.warmup}.")


class ClockworkCacheState(BaseState):
    def __init__(self) -> None:
        super().__init__()

        # Cached leader output and residual, used to reconstruct the block-stack output on reuse steps.
        self.head_block_output: torch.Tensor | tuple[torch.Tensor, ...] = None
        self.head_block_residual: torch.Tensor = None
        # Delta between the tail and leader outputs captured on the last compute step.
        self.tail_block_residuals: torch.Tensor | tuple[torch.Tensor, ...] = None
        # Per-step bookkeeping shared between the leader and the remaining blocks.
        self.should_compute: bool = True
        # Denoising-step counter, persisted across the whole `__call__` so the clock can reference it.
        self.step: int = -1
        self.last_compute_step: int = -1

    def reset(self):
        # Mirrors FirstBlockCache: only the per-step bookkeeping is reset; the leader/tail residuals and the step
        # counter persist across the generation (they are rebuilt from scratch by StateManager when the cache is
        # reset before a new generation).
        self.tail_block_residuals = None
        self.should_compute = True


class ClockworkLeaderBlockHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager, clock_interval: int, warmup: int):
        self.state_manager = state_manager
        self.clock_interval = clock_interval
        self.warmup = warmup
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        original_hidden_states = self._metadata._get_parameter_from_args_kwargs("hidden_states", args, kwargs)

        # The leader always recomputes itself; it is the reference against which the cached tail delta is applied.
        output = self.fn_ref.original_forward(*args, **kwargs)
        is_output_tuple = isinstance(output, tuple)

        if is_output_tuple:
            hidden_states_residual = output[self._metadata.return_hidden_states_index] - original_hidden_states
        else:
            hidden_states_residual = output - original_hidden_states

        shared_state: ClockworkCacheState = self.state_manager.get_state()
        shared_state.step += 1

        should_compute = self._should_compute(shared_state)
        shared_state.should_compute = should_compute

        if not should_compute:
            # Reuse step: replay the cached block-stack output by adding the stale tail delta to the current leader
            # output, then skip the remaining blocks (they pass this through unchanged).
            if is_output_tuple:
                hidden_states = (
                    shared_state.tail_block_residuals[0] + output[self._metadata.return_hidden_states_index]
                )
            else:
                hidden_states = shared_state.tail_block_residuals[0] + output

            encoder_hidden_states = None
            if self._metadata.return_encoder_hidden_states_index is not None:
                assert is_output_tuple
                encoder_hidden_states = (
                    shared_state.tail_block_residuals[1] + output[self._metadata.return_encoder_hidden_states_index]
                )

            if is_output_tuple:
                return_output = [None] * len(output)
                return_output[self._metadata.return_hidden_states_index] = hidden_states
                return_output[self._metadata.return_encoder_hidden_states_index] = encoder_hidden_states
                return_output = tuple(return_output)
            else:
                return_output = hidden_states
            output = return_output
        else:
            # Compute step: cache the leader output/residual so the tail can record the block-stack delta.
            shared_state.last_compute_step = shared_state.step
            if is_output_tuple:
                head_block_output = [None] * len(output)
                head_block_output[0] = output[self._metadata.return_hidden_states_index]
                head_block_output[1] = output[self._metadata.return_encoder_hidden_states_index]
            else:
                head_block_output = output
            shared_state.head_block_output = head_block_output
            shared_state.head_block_residual = hidden_states_residual

        return output

    def reset_state(self, module):
        self.state_manager.reset()
        return module

    @torch.compiler.disable
    def _should_compute(self, shared_state: ClockworkCacheState) -> bool:
        # No cache populated yet -> the first step must always compute.
        if shared_state.head_block_residual is None:
            return True
        # Warmup window always recomputes to stabilize the cache.
        if shared_state.step < self.warmup:
            return True
        # Clock: recompute every `clock_interval` steps since the last compute.
        if shared_state.step - shared_state.last_compute_step >= self.clock_interval:
            return True
        return False


class ClockworkBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, is_tail: bool = False):
        super().__init__()
        self.state_manager = state_manager
        self.is_tail = is_tail
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        original_hidden_states = self._metadata._get_parameter_from_args_kwargs("hidden_states", args, kwargs)
        original_encoder_hidden_states = None
        if self._metadata.return_encoder_hidden_states_index is not None:
            original_encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                "encoder_hidden_states", args, kwargs
            )

        shared_state = self.state_manager.get_state()

        if shared_state.should_compute:
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.is_tail:
                # Record the delta between the tail and leader outputs for reuse on later steps.
                if isinstance(output, tuple):
                    hidden_states_residual = (
                        output[self._metadata.return_hidden_states_index] - shared_state.head_block_output[0]
                    )
                    encoder_hidden_states_residual = (
                        output[self._metadata.return_encoder_hidden_states_index] - shared_state.head_block_output[1]
                    )
                else:
                    hidden_states_residual = output - shared_state.head_block_output
                    encoder_hidden_states_residual = None
                shared_state.tail_block_residuals = (hidden_states_residual, encoder_hidden_states_residual)
            return output

        # Reuse step: pass the input through unchanged; the cached delta is applied by the leader block.
        if original_encoder_hidden_states is None:
            return_output = original_hidden_states
        else:
            return_output = [None, None]
            return_output[self._metadata.return_hidden_states_index] = original_hidden_states
            return_output[self._metadata.return_encoder_hidden_states_index] = original_encoder_hidden_states
            return_output = tuple(return_output)
        return return_output


def apply_clockwork_cache(module: torch.nn.Module, config: ClockworkCacheConfig) -> None:
    """
    Applies [Clockwork Cache](https://huggingface.co/papers/2312.08128) to a given module.

    Clockwork Cache reuses the slowly-evolving transformer block stack output across a window of denoising steps and
    recomputes it on a fixed periodic schedule (the "clock"). It is built on the same `StateManager` mechanism as
    [`apply_first_block_cache`], but selects reuse steps by a periodic schedule rather than by per-step residual
    magnitude.

    Args:
        module (`torch.nn.Module`):
            The pytorch module to apply Clockwork Cache to. Typically this should be a transformer architecture
            supported in Diffusers, such as `CogVideoXTransformer3DModel`, but external implementations may also
            work.
        config (`ClockworkCacheConfig`):
            The configuration to use for applying the Clockwork Cache method.

    Example:
        ```python
        >>> import torch
        >>> from diffusers import CogView4Pipeline
        >>> from diffusers.hooks import ClockworkCacheConfig, apply_clockwork_cache

        >>> pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> apply_clockwork_cache(pipe.transformer, ClockworkCacheConfig(clock_interval=3, warmup=1))

        >>> prompt = "A photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt, generator=torch.Generator().manual_seed(42)).images[0]
        >>> image.save("output.png")
        ```
    """

    state_manager = StateManager(ClockworkCacheState, (), {})
    remaining_blocks = []

    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for index, block in enumerate(submodule):
            remaining_blocks.append((f"{name}.{index}", block))

    if len(remaining_blocks) < 2:
        raise ValueError(
            "Clockwork Cache requires at least two transformer blocks to cache (a leader block and a tail block). "
            f"Found {len(remaining_blocks)} block(s) under known transformer-block identifiers "
            f"({_ALL_TRANSFORMER_BLOCK_IDENTIFIERS})."
        )

    head_block_name, head_block = remaining_blocks.pop(0)
    tail_block_name, tail_block = remaining_blocks.pop(-1)

    logger.debug(f"Applying ClockworkLeaderBlockHook to '{head_block_name}'")
    _apply_clockwork_leader_block_hook(head_block, state_manager, config)

    for name, block in remaining_blocks:
        logger.debug(f"Applying ClockworkBlockHook to '{name}'")
        _apply_clockwork_block_hook(block, state_manager)

    logger.debug(f"Applying ClockworkBlockHook to tail block '{tail_block_name}'")
    _apply_clockwork_block_hook(tail_block, state_manager, is_tail=True)


def _apply_clockwork_leader_block_hook(
    block: torch.nn.Module, state_manager: StateManager, config: ClockworkCacheConfig
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    hook = ClockworkLeaderBlockHook(state_manager, config.clock_interval, config.warmup)
    registry.register_hook(hook, _CLOCKWORK_LEADER_BLOCK_HOOK)


def _apply_clockwork_block_hook(block: torch.nn.Module, state_manager: StateManager, is_tail: bool = False) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    hook = ClockworkBlockHook(state_manager, is_tail)
    registry.register_hook(hook, _CLOCKWORK_BLOCK_HOOK)
