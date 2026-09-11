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

import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..utils import logging
from .hooks import HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)
_CHEBYSHEV_CACHE_HOOK = "chebyshev_cache"
_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = (
    "^blocks.*attn",
    "^transformer_blocks.*attn",
    "^single_transformer_blocks.*attn",
)
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("^temporal_transformer_blocks.*attn",)
_TRANSFORMER_BLOCK_IDENTIFIERS = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS + _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS


@dataclass
class ChebyshevCacheConfig:
    """
    Configuration for Chebyshev-extrapolation cache. Adapted from "ChebBooster: A Training-Free Approach for Efficient
    Diffusion Transformer Inference via Chebyshev-Inspired Extrapolation" (https://huggingface.co/papers/2608.23429).

    Like TaylorSeer, this hook reuses expensive module outputs across denoising steps by extrapolating from previously
    computed values. Unlike TaylorSeer's Taylor-series (divided-difference) extrapolation — which is prone to Runge
    oscillations over long cache intervals — this hook evaluates a barycentric interpolant with Chebyshev-Lobatto
    weights, which stays numerically stable as the extrapolation order grows.

    Attributes:
        cache_interval (`int`, defaults to `4`):
            The interval between full computation steps. After a full computation, the extrapolated outputs are reused
            for this many subsequent denoising steps before refreshing with a new full forward pass.

        disable_cache_before_step (`int`, defaults to `3`):
            The denoising step index before which caching is disabled. The initial steps run full computations to seed
            the history buffer used by the barycentric extrapolant.

        disable_cache_after_step (`int`, *optional*, defaults to `None`):
            The denoising step index after which caching is disabled. For steps `>=` this value all modules run full
            computations, restoring accuracy in the final refinement steps.

        max_order (`int`, defaults to `3`):
            The polynomial degree of the barycentric extrapolant. The hook keeps the last `max_order + 1` computed
            outputs per module as interpolation nodes. Higher orders capture more curvature; the Chebyshev weighting
            keeps them stable where a Taylor expansion of the same order would oscillate.

        factors_dtype (`torch.dtype`, defaults to `torch.float32`):
            Data type used for storing the history buffer and computing the barycentric extrapolation. `float32`
            preserves the conditioning of the barycentric formula.

        skip_predict_identifiers (`list[str]`, *optional*, defaults to `None`):
            Regex patterns (using `re.fullmatch`) for module names to place in "skip" mode, where the module returns a
            zero tensor (matching the recorded shape) during prediction steps to skip computation cheaply.

        cache_identifiers (`list[str]`, *optional*, defaults to `None`):
            Regex patterns (using `re.fullmatch`) for module names to place in Chebyshev-extrapolation caching mode. If
            neither this nor `skip_predict_identifiers` is provided, all attention-like modules are hooked by default.

    Notes:
        - Patterns are matched using `re.fullmatch` on the module name.
        - The barycentric weights depend only on the number of history nodes, so they are precomputed once (the paper's
          "offline weight precomputation" stage) and reused on every prediction step (the "online application" stage).
    """

    cache_interval: int = 4
    disable_cache_before_step: int = 3
    disable_cache_after_step: int | None = None
    max_order: int = 3
    factors_dtype: torch.dtype | None = torch.float32
    skip_predict_identifiers: list[str] | None = None
    cache_identifiers: list[str] | None = None

    def __repr__(self) -> str:
        return (
            "ChebyshevCacheConfig("
            f"cache_interval={self.cache_interval}, "
            f"disable_cache_before_step={self.disable_cache_before_step}, "
            f"disable_cache_after_step={self.disable_cache_after_step}, "
            f"max_order={self.max_order}, "
            f"factors_dtype={self.factors_dtype}, "
            f"skip_predict_identifiers={self.skip_predict_identifiers}, "
            f"cache_identifiers={self.cache_identifiers})"
        )


def chebyshev_barycentric_weights(num_nodes: int) -> list[float]:
    """
    Barycentric weights for the second (Chebyshev-Lobatto) barycentric form.

    These weights depend only on the node count, not the node positions, so they can be precomputed offline. Applied to
    the (roughly equispaced) history of compute steps they yield Berrut's pole-free rational interpolant, which — unlike
    a Taylor expansion of the same order — does not exhibit Runge oscillations when extrapolating.
    """
    if num_nodes <= 1:
        return [1.0]
    weights = []
    for j in range(num_nodes):
        half_endpoint = 0.5 if (j == 0 or j == num_nodes - 1) else 1.0
        weights.append(((-1.0) ** j) * half_endpoint)
    return weights


class ChebyshevCacheState:
    def __init__(
        self,
        factors_dtype: torch.dtype | None = torch.float32,
        max_order: int = 3,
        is_inactive: bool = False,
    ):
        self.factors_dtype = factors_dtype
        self.num_history = max(1, max_order + 1)
        self.is_inactive = is_inactive

        self.module_dtypes: tuple[torch.dtype, ...] = ()
        self.device: torch.device | None = None
        self.current_step: int = -1
        # Ring buffers of interpolation nodes: the step indices and the outputs computed at those steps.
        self.history_steps: list[int] = []
        self.history_outputs: list[tuple[torch.Tensor, ...]] = []
        self.inactive_shapes: tuple[tuple[int, ...], ...] | None = None

    def reset(self) -> None:
        self.current_step = -1
        self.device = None
        self.history_steps = []
        self.history_outputs = []
        self.inactive_shapes = None

    def update(self, outputs: tuple[torch.Tensor, ...]) -> None:
        self.module_dtypes = tuple(output.dtype for output in outputs)
        self.device = outputs[0].device

        if self.is_inactive:
            self.inactive_shapes = tuple(output.shape for output in outputs)
            return

        self.history_steps.append(self.current_step)
        self.history_outputs.append(tuple(output.to(self.factors_dtype) for output in outputs))
        # Keep only the most recent `num_history` nodes as the extrapolation window.
        if len(self.history_steps) > self.num_history:
            self.history_steps = self.history_steps[-self.num_history :]
            self.history_outputs = self.history_outputs[-self.num_history :]

    @torch.compiler.disable
    def predict(self) -> list[torch.Tensor]:
        if self.is_inactive:
            if self.inactive_shapes is None:
                raise ValueError("Inactive shapes not set during prediction.")
            return [
                torch.zeros(shape, dtype=self.module_dtypes[i], device=self.device)
                for i, shape in enumerate(self.inactive_shapes)
            ]

        if not self.history_outputs:
            raise ValueError("History buffer empty during prediction.")

        num_outputs = len(self.history_outputs[-1])
        nodes = self.history_steps
        num_nodes = len(nodes)

        # Single node -> constant extrapolation; no barycentric evaluation needed.
        if num_nodes == 1:
            return [self.history_outputs[0][i].to(self.module_dtypes[i]) for i in range(num_outputs)]

        weights = chebyshev_barycentric_weights(num_nodes)
        # Second barycentric form: p(x) = sum_j (w_j / (x - x_j)) f_j / sum_j (w_j / (x - x_j)).
        # `current_step` is a prediction step, so it never coincides with a compute node -> no division by zero.
        coeffs = [weights[j] / (self.current_step - nodes[j]) for j in range(num_nodes)]
        denom = sum(coeffs)

        outputs = []
        for i in range(num_outputs):
            acc = torch.zeros_like(self.history_outputs[-1][i])
            for j in range(num_nodes):
                acc = acc + self.history_outputs[j][i] * coeffs[j]
            outputs.append((acc / denom).to(self.module_dtypes[i]))
        return outputs


class ChebyshevCacheHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        cache_interval: int,
        disable_cache_before_step: int,
        state_manager: StateManager,
        disable_cache_after_step: int | None = None,
    ):
        super().__init__()
        self.cache_interval = cache_interval
        self.disable_cache_before_step = disable_cache_before_step
        self.disable_cache_after_step = disable_cache_after_step
        self.state_manager = state_manager

    def initialize_hook(self, module: torch.nn.Module):
        return module

    def reset_state(self, module: torch.nn.Module) -> None:
        self.state_manager.reset()

    @torch.compiler.disable
    def _measure_should_compute(self):
        state: ChebyshevCacheState = self.state_manager.get_state()
        state.current_step += 1
        current_step = state.current_step
        is_warmup_phase = current_step < self.disable_cache_before_step
        is_compute_interval = (current_step - self.disable_cache_before_step - 1) % self.cache_interval == 0
        is_cooldown_phase = self.disable_cache_after_step is not None and current_step >= self.disable_cache_after_step
        should_compute = is_warmup_phase or is_compute_interval or is_cooldown_phase
        return should_compute, state

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        should_compute, state = self._measure_should_compute()
        if should_compute:
            outputs = self.fn_ref.original_forward(*args, **kwargs)
            wrapped_outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
            state.update(wrapped_outputs)
            return outputs

        outputs_list = state.predict()
        return outputs_list[0] if len(outputs_list) == 1 else tuple(outputs_list)


def _resolve_patterns(config: ChebyshevCacheConfig) -> tuple[list[str], list[str]]:
    inactive_patterns = config.skip_predict_identifiers or []
    active_patterns = config.cache_identifiers or []
    return inactive_patterns, active_patterns


def apply_chebyshev_cache(module: torch.nn.Module, config: ChebyshevCacheConfig):
    """
    Applies the Chebyshev-extrapolation cache to a model subtree (typically the transformer / UNet).

    Selected modules are hooked to extrapolate their outputs across denoising steps using a numerically stable
    barycentric Chebyshev interpolant, reducing redundant computation in the diffusion loop.

    Args:
        module (torch.nn.Module): The model subtree to apply the hooks to.
        config (ChebyshevCacheConfig): Configuration for the cache.

    Example:
    ```python
    >>> import torch
    >>> from diffusers import PixArtSigmaPipeline, ChebyshevCacheConfig

    >>> pipe = PixArtSigmaPipeline.from_pretrained(
    ...     "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
    ... )
    >>> pipe.to("cuda")

    >>> config = ChebyshevCacheConfig(cache_interval=4, max_order=3, disable_cache_before_step=3)
    >>> pipe.transformer.enable_cache(config)
    ```
    """
    inactive_patterns, active_patterns = _resolve_patterns(config)
    active_patterns = active_patterns or list(_TRANSFORMER_BLOCK_IDENTIFIERS)

    for name, submodule in module.named_modules():
        matches_inactive = any(re.fullmatch(pattern, name) for pattern in inactive_patterns)
        matches_active = any(re.fullmatch(pattern, name) for pattern in active_patterns)
        if not (matches_inactive or matches_active):
            continue
        _apply_chebyshev_cache_hook(module=submodule, config=config, is_inactive=matches_inactive)


def _apply_chebyshev_cache_hook(module: nn.Module, config: ChebyshevCacheConfig, is_inactive: bool):
    state_manager = StateManager(
        ChebyshevCacheState,
        init_kwargs={
            "factors_dtype": config.factors_dtype,
            "max_order": config.max_order,
            "is_inactive": is_inactive,
        },
    )

    registry = HookRegistry.check_if_exists_or_initialize(module)

    hook = ChebyshevCacheHook(
        cache_interval=config.cache_interval,
        disable_cache_before_step=config.disable_cache_before_step,
        disable_cache_after_step=config.disable_cache_after_step,
        state_manager=state_manager,
    )

    registry.register_hook(hook, _CHEBYSHEV_CACHE_HOOK)
