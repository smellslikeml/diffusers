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

r"""
Dual Feature Caching (DuCa) for Diffusion Transformers.

Adapted from "Accelerating Diffusion Transformers with Dual Feature Caching"
(https://huggingface.co/papers/2412.18911). DuCa runs a cyclic caching schedule of period ``cache_interval``: each cycle
starts with a *fresh* step that fully computes and caches the transformer residual, followed by alternating *aggressive*
steps (skip every block and reuse the full cached residual) and *conservative* steps (recompute a subset of blocks to
correct the drift accumulated by aggressive skipping, reusing the cached residual for the remaining blocks). The key
insight ported here is DuCa's dual schedule: aggressive caching is cheap but drifts, and interleaved conservative
caching corrects that drift, giving a better speed/quality trade-off than a single fixed caching strategy.

Adaptation note: the paper's conservative step selects *tokens* to recompute via a value-norm criterion (the
flash-attention-friendly "V-caching" estimator). That token-wise estimator is replaced here with a parameter-free,
block-wise proxy: the leading ``conservative_fraction`` of transformer blocks (which carry the largest feature change)
are recomputed while the deeper blocks reuse their cached contribution. This keeps DuCa's three-state dual schedule at
full fidelity while dropping the auxiliary token-selection machinery, which does not map onto the block-granular hook
interface used by the other cache hooks in this repo.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = get_logger(__name__)  # pylint: disable=invalid-name

_DUAL_CACHE_BLOCK_HOOK = "dual_cache_block_hook"

_MODE_FRESH = "fresh"
_MODE_AGGRESSIVE = "aggressive"
_MODE_CONSERVATIVE = "conservative"


@dataclass
class DualCacheConfig:
    r"""
    Configuration for [DuCa (Dual Feature Caching)](https://huggingface.co/papers/2412.18911).

    Args:
        cache_interval (`int`, defaults to `3`):
            The period `N` of the caching cycle. Each cycle is one *fresh* (full-compute) step followed by `N - 1` cache
            steps that alternate *aggressive* and *conservative* caching. `N = 3` reproduces the paper's default
            fresh -> aggressive -> conservative pattern.
        conservative_fraction (`float`, defaults to `0.5`):
            The fraction of leading transformer blocks recomputed on a conservative step (the block-wise proxy for the
            paper's token-wise V-caching). The remaining deeper blocks reuse their cached residual contribution. Must be
            in the open interval `(0, 1)`.
        warmup_steps (`int`, defaults to `0`):
            The number of initial steps that are always fully computed before caching kicks in. Larger values trade
            speed for stability on the early, high-variance steps.
    """

    cache_interval: int = 3
    conservative_fraction: float = 0.5
    warmup_steps: int = 0

    def __post_init__(self):
        if self.cache_interval < 1:
            raise ValueError(f"`cache_interval` must be a positive integer, got {self.cache_interval}.")
        if not 0.0 < self.conservative_fraction < 1.0:
            raise ValueError(f"`conservative_fraction` must be in the open interval (0, 1), got {self.conservative_fraction}.")
        if self.warmup_steps < 0:
            raise ValueError(f"`warmup_steps` must be non-negative, got {self.warmup_steps}.")


def _residual(output: torch.Tensor, base: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Compute ``output - base`` when the shapes match, otherwise return ``None`` so callers fall back to recompute."""
    if base is not None and output.shape == base.shape:
        return output - base
    return None


def _add_residual(output: torch.Tensor, residual: Optional[torch.Tensor]) -> torch.Tensor:
    """Add a cached residual back onto a passed-through hidden state, tolerating text+image concatenation shapes."""
    if residual is None:
        return output
    if residual.device != output.device:
        residual = residual.to(output.device)
    if residual.shape == output.shape:
        return output + residual
    if (
        output.ndim == 3
        and residual.ndim == 3
        and output.shape[0] == residual.shape[0]
        and output.shape[2] == residual.shape[2]
        and output.shape[1] > residual.shape[1]
    ):
        # Standard Flux/SD3 concatenation where the image tokens are at the tail.
        diff = output.shape[1] - residual.shape[1]
        output = output.clone()
        output[:, diff:, :] = output[:, diff:, :] + residual
        return output
    logger.warning(
        "DualCache: residual shape %s is incompatible with output shape %s; skipping residual reuse this step.",
        tuple(residual.shape),
        tuple(output.shape),
    )
    return output


class DualCacheState(BaseState):
    def __init__(self) -> None:
        super().__init__()
        # Absolute step counter, used to place each step in the caching cycle.
        self.step_index: int = 0
        # Caching mode selected by the head block for the current forward pass.
        self.mode: str = _MODE_FRESH
        # Input to the first (head) block; base for the full residual.
        self.head_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = None
        # Input to the split block during a fresh step; base for the deep residual.
        self.fresh_split_input: torch.Tensor = None
        # Cached residual across all blocks (reused by aggressive steps).
        self.full_residual: torch.Tensor = None
        # Cached residual across the deep (skipped) blocks only (reused by conservative steps).
        self.deep_residual: torch.Tensor = None

    def reset(self):
        self.step_index = 0
        self.mode = _MODE_FRESH
        self.head_input = None
        self.fresh_split_input = None
        self.full_residual = None
        self.deep_residual = None


class DualCacheBlockHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        state_manager: StateManager,
        config: DualCacheConfig,
        block_index: int,
        num_blocks: int,
        split_index: int,
    ) -> None:
        super().__init__()
        self.state_manager = state_manager
        self.config = config
        self.block_index = block_index
        self.num_blocks = num_blocks
        self.split_index = split_index
        self.is_head = block_index == 0
        self.is_tail = block_index == num_blocks - 1
        self._metadata = None

    def initialize_hook(self, module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def _hidden(self, args, kwargs) -> torch.Tensor:
        arg_name = self._metadata.hidden_states_argument_name
        return self._metadata._get_parameter_from_args_kwargs(arg_name, args, kwargs)

    def _pack(self, hidden, args, kwargs):
        """Return ``hidden`` in the same layout the wrapped block would (bare tensor or (hidden, encoder) tuple)."""
        if self._metadata.return_encoder_hidden_states_index is None:
            return hidden
        encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs("encoder_hidden_states", args, kwargs)
        max_idx = max(self._metadata.return_hidden_states_index, self._metadata.return_encoder_hidden_states_index)
        ret_list = [None] * (max_idx + 1)
        ret_list[self._metadata.return_hidden_states_index] = hidden
        ret_list[self._metadata.return_encoder_hidden_states_index] = encoder_hidden_states
        return tuple(ret_list)

    def _decide_mode(self, step: int) -> str:
        cfg = self.config
        if self.num_blocks < 2 or step < cfg.warmup_steps:
            return _MODE_FRESH
        position = step % cfg.cache_interval
        if position == 0:
            return _MODE_FRESH
        # After each fresh step, alternate aggressive (odd) and conservative (even) cache steps.
        return _MODE_AGGRESSIVE if position % 2 == 1 else _MODE_CONSERVATIVE

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")
        state: DualCacheState = self.state_manager.get_state()

        if self.is_head:
            state.head_input = self._hidden(args, kwargs)
            state.mode = self._decide_mode(state.step_index)

        if state.mode == _MODE_AGGRESSIVE:
            return self._forward_aggressive(state, args, kwargs)
        if state.mode == _MODE_CONSERVATIVE:
            return self._forward_conservative(state, args, kwargs)
        return self._forward_fresh(state, args, kwargs)

    def _forward_fresh(self, state: DualCacheState, args, kwargs):
        if self.block_index == self.split_index:
            state.fresh_split_input = self._hidden(args, kwargs)
        output = self.fn_ref.original_forward(*args, **kwargs)
        if self.is_tail:
            out_hidden = output[self._metadata.return_hidden_states_index] if isinstance(output, tuple) else output
            state.full_residual = _residual(out_hidden, state.head_input)
            state.deep_residual = _residual(out_hidden, state.fresh_split_input)
            state.step_index += 1
        return output

    def _forward_aggressive(self, state: DualCacheState, args, kwargs):
        # Cold cache (should only happen if the schedule is disturbed): fall back to a full compute.
        if state.full_residual is None:
            return self._forward_fresh(state, args, kwargs)
        hidden = self._hidden(args, kwargs)
        if self.is_tail:
            output = _add_residual(hidden, state.full_residual)
            state.step_index += 1
            return self._pack(output, args, kwargs)
        # Non-tail blocks pass their input straight through; the residual is re-applied once, at the tail.
        return self._pack(hidden, args, kwargs)

    def _forward_conservative(self, state: DualCacheState, args, kwargs):
        if state.deep_residual is None:
            return self._forward_fresh(state, args, kwargs)
        if self.block_index < self.split_index:
            # Recompute the leading (high-signal) blocks to correct the drift from aggressive steps.
            return self.fn_ref.original_forward(*args, **kwargs)
        # Deep blocks are skipped; their cached contribution is re-applied once, at the tail.
        hidden = self._hidden(args, kwargs)
        if self.is_tail:
            output = _add_residual(hidden, state.deep_residual)
            state.step_index += 1
            return self._pack(output, args, kwargs)
        return self._pack(hidden, args, kwargs)

    def reset_state(self, module):
        self.state_manager.reset()
        return module


def apply_dual_cache(module: torch.nn.Module, config: DualCacheConfig) -> None:
    """
    Applies Dual Feature Caching (DuCa) to a given module (typically a Diffusion Transformer).

    Args:
        module (`torch.nn.Module`):
            The module to apply DuCa to.
        config (`DualCacheConfig`):
            The configuration for DuCa.
    """
    # Initialize registry on the root module so the pipeline can set the caching context.
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

    num_blocks = len(blocks)
    # Recompute the leading `conservative_fraction` of blocks on conservative steps; clamp so at least one block is
    # recomputed and at least one deep block is cached (a non-trivial dual split needs num_blocks >= 2).
    split_index = min(max(1, round(config.conservative_fraction * num_blocks)), max(1, num_blocks - 1))

    for index, (name, block) in enumerate(blocks):
        registry = HookRegistry.check_if_exists_or_initialize(block)
        # Automatically remove an existing hook to allow re-application (e.g. switching configs).
        if registry.get_hook(_DUAL_CACHE_BLOCK_HOOK) is not None:
            registry.remove_hook(_DUAL_CACHE_BLOCK_HOOK)
        hook = DualCacheBlockHook(state_manager, config, index, num_blocks, split_index)
        registry.register_hook(hook, _DUAL_CACHE_BLOCK_HOOK)

    logger.info(
        f"DualCache: applied hooks to {num_blocks} blocks "
        f"(conservative split at block {split_index}, cache interval {config.cache_interval})."
    )
