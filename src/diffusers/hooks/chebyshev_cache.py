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
    Configuration for Chebyshev-inspired extrapolation cache. See: https://arxiv.org/abs/2608.23429

    This is a numerically stable sibling of [`TaylorSeerCacheConfig`]. Rather than expanding a Taylor series around a
    single anchor step -- which is prone to Runge oscillations over long cache intervals -- the cached feature at a
    skipped step is recovered by evaluating a polynomial interpolant through the last few *computed* steps using the
    Barycentric formulation of Lagrange interpolation, which is the numerically stable evaluation form advocated by
    ChebBooster.

    Attributes:
        cache_interval (`int`, defaults to `5`):
            The interval between full computation steps. After a full computation, the cached (predicted) outputs are
            reused for this many subsequent denoising steps before refreshing with a new full forward pass.

        disable_cache_before_step (`int`, defaults to `3`):
            The denoising step index before which caching is disabled. Full computation is performed for the initial
            steps (0 to disable_cache_before_step - 1) so that enough interpolation nodes are gathered before
            extrapolation begins.

        disable_cache_after_step (`int`, *optional*, defaults to `None`):
            The denoising step index after which caching is disabled. If set, for steps >= this value all modules run
            full computations, ensuring accuracy in later stages if needed.

        num_nodes (`int`, defaults to `3`):
            The number of most-recent computed steps used as interpolation nodes (i.e. one more than the polynomial
            degree). `num_nodes=1` collapses to constant reuse; `num_nodes=2` is linear extrapolation; higher values
            follow a higher-degree Barycentric interpolant. More nodes improve accuracy but increase memory.

        cache_factors_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
            Data type used for storing cached feature snapshots. Lower precision reduces memory but may affect
            stability; higher precision improves accuracy at the cost of more memory.

        skip_predict_identifiers (`list[str]`, *optional*, defaults to `None`):
            Regex patterns (using `re.fullmatch`) for module names to place as "skip" modules. These compute fully
            during refresh steps but return a zero tensor (matching recorded shape) during prediction steps.

        cache_identifiers (`list[str]`, *optional*, defaults to `None`):
            Regex patterns (using `re.fullmatch`) for module names whose outputs are extrapolated and cached for reuse.

        use_lite_mode (`bool`, *optional*, defaults to `False`):
            Enables a lightweight variant that skips block outputs and only caches projections, minimizing memory. This
            overrides any custom `skip_predict_identifiers` or `cache_identifiers`.

    Notes:
        - Patterns are matched using `re.fullmatch` on the module name.
        - If neither `skip_predict_identifiers` nor `cache_identifiers` is provided, all attention-like modules are
          hooked by default.
    """

    cache_interval: int = 5
    disable_cache_before_step: int = 3
    disable_cache_after_step: int | None = None
    num_nodes: int = 3
    cache_factors_dtype: torch.dtype | None = torch.bfloat16
    skip_predict_identifiers: list[str] | None = None
    cache_identifiers: list[str] | None = None
    use_lite_mode: bool = False

    def __repr__(self) -> str:
        return (
            "ChebyshevCacheConfig("
            f"cache_interval={self.cache_interval}, "
            f"disable_cache_before_step={self.disable_cache_before_step}, "
            f"disable_cache_after_step={self.disable_cache_after_step}, "
            f"num_nodes={self.num_nodes}, "
            f"cache_factors_dtype={self.cache_factors_dtype}, "
            f"skip_predict_identifiers={self.skip_predict_identifiers}, "
            f"cache_identifiers={self.cache_identifiers}, "
            f"use_lite_mode={self.use_lite_mode})"
        )


def _barycentric_weights(nodes: list[float]) -> list[float]:
    """
    Compute Barycentric interpolation weights ``w_j = 1 / prod_{k != j} (x_j - x_k)`` for the given interpolation
    nodes. This is the numerically stable weight formulation used by ChebBooster; for the small windows used here the
    cost is negligible and the weights are recomputed online from the actual computed-step positions.
    """
    n = len(nodes)
    weights = [1.0] * n
    for j in range(n):
        for k in range(n):
            if k != j:
                weights[j] /= nodes[j] - nodes[k]
    return weights


class ChebyshevExtrapolationState:
    def __init__(
        self,
        cache_factors_dtype: torch.dtype | None = torch.bfloat16,
        num_nodes: int = 3,
        is_inactive: bool = False,
    ):
        self.cache_factors_dtype = cache_factors_dtype
        self.num_nodes = max(1, num_nodes)
        self.is_inactive = is_inactive

        self.module_dtypes: tuple[torch.dtype, ...] = ()
        self.device: torch.device | None = None
        # Rolling window of (step_index, tuple_of_feature_snapshots) at computed steps.
        self.nodes: deque = deque(maxlen=self.num_nodes)
        self.inactive_shapes: tuple[tuple[int, ...], ...] | None = None
        self.current_step: int = -1

    def reset(self) -> None:
        self.current_step = -1
        self.nodes = deque(maxlen=self.num_nodes)
        self.inactive_shapes = None
        self.device = None

    def update(self, outputs: tuple[torch.Tensor, ...]) -> None:
        self.module_dtypes = tuple(output.dtype for output in outputs)
        self.device = outputs[0].device

        if self.is_inactive:
            self.inactive_shapes = tuple(output.shape for output in outputs)
        else:
            snapshot = tuple(output.to(self.cache_factors_dtype) for output in outputs)
            self.nodes.append((self.current_step, snapshot))

    @torch.compiler.disable
    def predict(self) -> list[torch.Tensor]:
        if self.is_inactive:
            if self.inactive_shapes is None:
                raise ValueError("Inactive shapes not set during prediction.")
            return [
                torch.zeros(self.inactive_shapes[i], dtype=self.module_dtypes[i], device=self.device)
                for i in range(len(self.module_dtypes))
            ]

        if not self.nodes:
            raise ValueError("Cannot predict without a prior computed step.")

        node_steps = [float(step) for step, _ in self.nodes]
        num_outputs = len(self.nodes[-1][1])

        # A single node collapses to constant reuse of the most recent computation.
        if len(self.nodes) == 1:
            return [self.nodes[-1][1][i].to(self.module_dtypes[i]) for i in range(num_outputs)]

        x = float(self.current_step)
        # Exact hit on a node (defensive; predictions target skipped steps): return that snapshot directly.
        for step, snapshot in self.nodes:
            if float(step) == x:
                return [snapshot[i].to(self.module_dtypes[i]) for i in range(num_outputs)]

        weights = _barycentric_weights(node_steps)
        terms = [w / (x - xj) for w, xj in zip(weights, node_steps)]
        denom = sum(terms)

        outputs = []
        for i in range(num_outputs):
            output_dtype = self.module_dtypes[i]
            acc = torch.zeros_like(self.nodes[-1][1][i], dtype=torch.float32)
            for term, (_, snapshot) in zip(terms, self.nodes):
                acc = acc + snapshot[i].to(torch.float32) * term
            outputs.append((acc / denom).to(output_dtype))
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
        """
        Reset state between sampling runs.
        """
        self.state_manager.reset()

    @torch.compiler.disable
    def _measure_should_compute(self):
        state: ChebyshevExtrapolationState = self.state_manager.get_state()
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
    Resolve effective inactive and active pattern lists from config.
    """
    inactive_patterns = config.skip_predict_identifiers if config.skip_predict_identifiers is not None else None
    active_patterns = config.cache_identifiers if config.cache_identifiers is not None else None
    return inactive_patterns or [], active_patterns or []


def apply_chebyshev_cache(module: torch.nn.Module, config: ChebyshevCacheConfig):
    """
    Applies the Chebyshev-inspired extrapolation cache to a given model (typically the transformer / UNet).

    This hooks selected modules so that, on skipped denoising steps, their outputs are recovered from a Barycentric
    Lagrange interpolant through the last few computed steps instead of being recomputed, reducing redundant
    computation in DiT denoising loops.

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

    >>> config = ChebyshevCacheConfig(cache_interval=5, num_nodes=3, disable_cache_before_step=3)
    >>> pipe.transformer.enable_cache(config)
    ```
    """
    inactive_patterns, active_patterns = _resolve_patterns(config)

    active_patterns = active_patterns or _TRANSFORMER_BLOCK_IDENTIFIERS

    if config.use_lite_mode:
        logger.info("Using Chebyshev cache lite variant.")
        active_patterns = _PROJ_OUT_IDENTIFIERS
        inactive_patterns = _BLOCK_IDENTIFIERS
        if config.skip_predict_identifiers or config.cache_identifiers:
            logger.warning("Lite mode overrides user patterns.")

    for name, submodule in module.named_modules():
        matches_inactive = any(re.fullmatch(pattern, name) for pattern in inactive_patterns)
        matches_active = any(re.fullmatch(pattern, name) for pattern in active_patterns)
        if not (matches_inactive or matches_active):
            continue
        _apply_chebyshev_cache_hook(module=submodule, config=config, is_inactive=matches_inactive)


def _apply_chebyshev_cache_hook(module: nn.Module, config: ChebyshevCacheConfig, is_inactive: bool):
    """
    Registers the Chebyshev cache hook on the specified nn.Module.
    """
    state_manager = StateManager(
        ChebyshevExtrapolationState,
        init_kwargs={
            "cache_factors_dtype": config.cache_factors_dtype,
            "num_nodes": config.num_nodes,
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
