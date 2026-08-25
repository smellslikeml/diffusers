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
from collections import deque
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
_BLOCK_IDENTIFIERS = ("^[^.]*block[^.]*\\.[^.]+$",)
_PROJ_OUT_IDENTIFIERS = ("^proj_out$",)


@dataclass
class ChebyshevCacheConfig:
    """
    Configuration for ChebBooster (Chebyshev) cache. See: https://arxiv.org/abs/2608.23429

    ChebBooster keeps the same cache machinery as [`TaylorSeerCacheConfig`] but replaces the Taylor-series factors
    with a numerically-stable barycentric Chebyshev interpolation over a rolling window of full-compute activations
    ("A Training-Free Approach for Efficient Diffusion Transformer Inference", Lu, Deng, He, Luo & Li). The
    interpolation math is ported from the Apache-2.0 `ChebBooster-FLUX` subtree of
    https://github.com/Kiramei/ChebBooster (`src/flux/cheb_utils/__init__.py`).

    Attributes:
        cache_interval (`int`, defaults to `5`):
            The interval between full computation steps. After a full computation, the cached (predicted) outputs are
            reused for this many subsequent denoising steps before refreshing with a new full forward pass.

        disable_cache_before_step (`int`, defaults to `3`):
            The denoising step index before which caching is disabled, meaning full computation is performed for the
            initial steps (0 to disable_cache_before_step - 1) to gather the history window for the Chebyshev
            interpolation (the paper's `first_enhance`). During these steps, the feature history is updated, but
            caching/predictions are not applied. Caching begins at this step.

        disable_cache_after_step (`int`, *optional*, defaults to `None`):
            The denoising step index after which caching is disabled (the paper's `end_enhance`). If set, for steps
            >= this value, all modules run full computations without predictions, ensuring accuracy in later stages
            if needed.

        cheb_order (`int`, defaults to `6`):
            The maximum number of full-compute activations kept in the rolling history window (the paper's `n`).
            The interpolation uses at most this many nodes; fewer are used while the window is still filling up
            during warmup.

        cheb_factors_dtype (`torch.dtype`, defaults to `torch.float32`):
            Data type used for computing the barycentric interpolation weights and accumulating the weighted sum.
            Unlike the reference implementation, which hardcodes bfloat16 preset tables, weights are computed
            in-memory per prediction; float32 is the recommended default for numerical stability.

        skip_predict_identifiers (`list[str]`, *optional*, defaults to `None`):
            Regex patterns (using `re.fullmatch`) for module names to place as "skip" in "cache" mode. In this mode,
            the module computes fully during initial or refresh steps but returns a zero tensor (matching recorded
            shape) during prediction steps to skip computation cheaply.

        cache_identifiers (`list[str]`, *optional*, defaults to `None`):
            Regex patterns (using `re.fullmatch`) for module names to place in Chebyshev caching mode, where outputs
            are interpolated and cached for reuse.

        use_lite_mode (`bool`, *optional*, defaults to `False`):
            Enables a lightweight variant that minimizes memory usage by applying predefined patterns for skipping
            and caching (e.g., skipping blocks and caching projections). This overrides any custom
            `skip_predict_identifiers` or `cache_identifiers`.

    Notes:
        - Patterns are matched using `re.fullmatch` on the module name.
        - If `skip_predict_identifiers` or `cache_identifiers` are provided, only matching modules are hooked.
        - If neither is provided, all attention-like modules are hooked by default.
    """

    cache_interval: int = 5
    disable_cache_before_step: int = 3
    disable_cache_after_step: int | None = None
    cheb_order: int = 6
    cheb_factors_dtype: torch.dtype | None = torch.float32
    skip_predict_identifiers: list[str] | None = None
    cache_identifiers: list[str] | None = None
    use_lite_mode: bool = False

    def __repr__(self) -> str:
        return (
            "ChebyshevCacheConfig("
            f"cache_interval={self.cache_interval}, "
            f"disable_cache_before_step={self.disable_cache_before_step}, "
            f"disable_cache_after_step={self.disable_cache_after_step}, "
            f"cheb_order={self.cheb_order}, "
            f"cheb_factors_dtype={self.cheb_factors_dtype}, "
            f"skip_predict_identifiers={self.skip_predict_identifiers}, "
            f"cache_identifiers={self.cache_identifiers}, "
            f"use_lite_mode={self.use_lite_mode})"
        )


class ChebyshevCacheState:
    def __init__(
        self,
        cheb_factors_dtype: torch.dtype | None = torch.float32,
        cheb_order: int = 6,
        is_inactive: bool = False,
    ):
        self.cheb_factors_dtype = cheb_factors_dtype
        self.cheb_order = cheb_order
        self.is_inactive = is_inactive

        self.module_dtypes: tuple[torch.dtype, ...] = ()
        # Rolling window of (step, feature) pairs per module output, capped at `cheb_order` nodes.
        # Mirrors `cheb_derivative_approximation` in the reference (`cheb_utils/__init__.py`, ~L135-158),
        # which appends to `history` and pops the oldest entry beyond `max_history`.
        self.feature_history: dict[int, deque] = {}
        self.inactive_shapes: tuple[tuple[int, ...], ...] | None = None
        self.device: torch.device | None = None
        self.current_step: int = -1

    def reset(self) -> None:
        self.current_step = -1
        self.feature_history = {}
        self.inactive_shapes = None
        self.device = None

    def update(
        self,
        outputs: tuple[torch.Tensor, ...],
    ) -> None:
        self.module_dtypes = tuple(output.dtype for output in outputs)
        self.device = outputs[0].device

        if self.is_inactive:
            self.inactive_shapes = tuple(output.shape for output in outputs)
        else:
            for i, features in enumerate(outputs):
                history = self.feature_history.get(i)
                if history is None:
                    history = deque(maxlen=self.cheb_order)
                    self.feature_history[i] = history
                if history and self.current_step == history[-1][0]:
                    raise ValueError("Delta step cannot be zero for Chebyshev cache update.")
                history.append((self.current_step, features.to(self.cheb_factors_dtype)))

    @staticmethod
    def _barycentric_weights(
        history_steps: list[int],
        target_step: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Barycentric interpolation weights for `target_step` over the nodes `history_steps`.

        Ported from `precompute_chebyshev_weights` + `chebyshev_barycentric_weights` in the Apache-2.0
        `ChebBooster-FLUX` reference (`src/flux/cheb_utils/__init__.py`, L13-35 and L161-220), computing per
        prediction in-memory instead of reading a disk `__preset__/*.pt` table:

        1. Map the nodes and the target affinely to [-1, 1] using the window extent (reference L173-191; the map
           leaves the interpolant unchanged and only improves conditioning).
        2. Use the Chebyshev barycentric weights w_j = (-1)^j * delta_j with delta_j = 1/2 at the window endpoints
           (reference L25-32).
        3. weights_j = (w_j / (x_target - x_j)) / sum_k (w_k / (x_target - x_k)) (reference L192-205), with the
           exact-node one-hot guard when the target coincides with a node (reference L196-216).
        """
        n = len(history_steps)
        if n == 1:
            # Matches the reference fallback of reusing the latest feature when no interpolation
            # window is available (`cheb_formula`, ~L57-59).
            return torch.ones(1, dtype=torch.float64, device=device)

        steps = torch.tensor(history_steps, dtype=torch.float64, device=device)
        s_min, s_max = steps[0], steps[-1]
        x_nodes = 2.0 * (steps - s_min) / (s_max - s_min) - 1.0
        x_target = 2.0 * (float(target_step) - s_min) / (s_max - s_min) - 1.0

        bary_w = torch.ones(n, dtype=torch.float64, device=device)
        bary_w[1::2] *= -1.0
        bary_w[0] *= 0.5
        bary_w[-1] *= 0.5

        x_diff = x_target - x_nodes
        exact_node = torch.abs(x_diff) < 1e-8
        if exact_node.any():
            weights = torch.zeros(n, dtype=torch.float64, device=device)
            weights[exact_node] = 1.0
            return weights

        w_over_diff = bary_w / x_diff
        return w_over_diff / w_over_diff.sum()

    @torch.compiler.disable
    def predict(self) -> list[torch.Tensor]:
        if not self.is_inactive and not self.feature_history:
            raise ValueError("Cannot predict without prior initialization/update.")

        outputs = []
        if self.is_inactive:
            if self.inactive_shapes is None:
                raise ValueError("Inactive shapes not set during prediction.")
            for i in range(len(self.module_dtypes)):
                outputs.append(
                    torch.zeros(
                        self.inactive_shapes[i],
                        dtype=self.module_dtypes[i],
                        device=self.device,
                    )
                )
        else:
            for i in range(len(self.feature_history)):
                output_dtype = self.module_dtypes[i]
                history = self.feature_history[i]
                history_steps = [step for step, _ in history]
                # Note on indexing: the reference stores steps in reverse scheduler order and remaps with
                # `num_steps - step - 1` (`cheb_formula.v2`, ~L91-109). This hook counts denoising steps forward
                # from 0 (same convention as `TaylorSeerState`), so no remap is applied here.
                weights = self._barycentric_weights(history_steps, self.current_step, self.device)
                output = torch.zeros_like(history[0][1], dtype=output_dtype)
                # Weighted-sum prediction, mirroring `cheb_formula.v2` (reference ~L97-110).
                for weight, (_, feature) in zip(weights, history):
                    output = output + feature.to(output_dtype) * weight.to(output_dtype)
                outputs.append(output)
        return outputs


class ChebyshevCacheHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        cache_interval: int,
        disable_cache_before_step: int,
        cheb_factors_dtype: torch.dtype,
        state_manager: StateManager,
        disable_cache_after_step: int | None = None,
    ):
        super().__init__()
        self.cache_interval = cache_interval
        self.disable_cache_before_step = disable_cache_before_step
        self.disable_cache_after_step = disable_cache_after_step
        self.cheb_factors_dtype = cheb_factors_dtype
        self.state_manager = state_manager

    def initialize_hook(self, module: torch.nn.Module):
        return module

    def reset_state(self, module: torch.nn.Module) -> None:
        """
        Reset state between sampling runs.
        """
        self.state_manager.reset()

    @torch.compiler.disable
    def _measure_should_compute(self) -> bool:
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
    """
    Resolve effective inactive and active pattern lists from config + templates.
    """

    inactive_patterns = config.skip_predict_identifiers if config.skip_predict_identifiers is not None else None
    active_patterns = config.cache_identifiers if config.cache_identifiers is not None else None

    return inactive_patterns or [], active_patterns or []


def apply_chebyshev_cache(module: torch.nn.Module, config: ChebyshevCacheConfig):
    """
    Applies the ChebBooster (Chebyshev) cache to a given pipeline (typically the transformer / UNet).

    This function hooks selected modules in the model to enable caching or skipping based on the provided
    configuration, reducing redundant computations in diffusion denoising loops.

    Args:
        module (torch.nn.Module): The model subtree to apply the hooks to.
        config (ChebyshevCacheConfig): Configuration for the cache.

    Example:
    ```python
    >>> import torch
    >>> from diffusers import FluxPipeline, ChebyshevCacheConfig

    >>> pipe = FluxPipeline.from_pretrained(
    ...     "black-forest-labs/FLUX.1-dev",
    ...     torch_dtype=torch.bfloat16,
    ... )
    >>> pipe.to("cuda")

    >>> config = ChebyshevCacheConfig(
    ...     cache_interval=5,
    ...     cheb_order=6,
    ...     disable_cache_before_step=3,
    ...     cheb_factors_dtype=torch.float32,
    ... )
    >>> pipe.transformer.enable_cache(config)
    ```
    """
    inactive_patterns, active_patterns = _resolve_patterns(config)

    active_patterns = active_patterns or _TRANSFORMER_BLOCK_IDENTIFIERS

    if config.use_lite_mode:
        logger.info("Using Chebyshev Lite variant for cache.")
        active_patterns = _PROJ_OUT_IDENTIFIERS
        inactive_patterns = _BLOCK_IDENTIFIERS
        if config.skip_predict_identifiers or config.cache_identifiers:
            logger.warning("Lite mode overrides user patterns.")

    for name, submodule in module.named_modules():
        matches_inactive = any(re.fullmatch(pattern, name) for pattern in inactive_patterns)
        matches_active = any(re.fullmatch(pattern, name) for pattern in active_patterns)
        if not (matches_inactive or matches_active):
            continue
        _apply_chebyshev_cache_hook(
            module=submodule,
            config=config,
            is_inactive=matches_inactive,
        )


def _apply_chebyshev_cache_hook(
    module: nn.Module,
    config: ChebyshevCacheConfig,
    is_inactive: bool,
):
    """
    Registers the Chebyshev cache hook on the specified nn.Module.

    Args:
        name: Name of the module.
        module: The nn.Module to be hooked.
        config: Cache configuration.
        is_inactive: Whether this module should operate in "inactive" mode.
    """
    state_manager = StateManager(
        ChebyshevCacheState,
        init_kwargs={
            "cheb_factors_dtype": config.cheb_factors_dtype,
            "cheb_order": config.cheb_order,
            "is_inactive": is_inactive,
        },
    )

    registry = HookRegistry.check_if_exists_or_initialize(module)

    hook = ChebyshevCacheHook(
        cache_interval=config.cache_interval,
        disable_cache_before_step=config.disable_cache_before_step,
        cheb_factors_dtype=config.cheb_factors_dtype,
        disable_cache_after_step=config.disable_cache_after_step,
        state_manager=state_manager,
    )

    registry.register_hook(hook, _CHEBYSHEV_CACHE_HOOK)
