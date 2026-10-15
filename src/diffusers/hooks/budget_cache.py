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

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = get_logger(__name__)  # pylint: disable=invalid-name

_BUDGET_CACHE_LEADER_BLOCK_HOOK = "budget_cache_leader_block_hook"
_BUDGET_CACHE_BLOCK_HOOK = "budget_cache_block_hook"


def _uniform_schedule(num_steps: int, compute_budget: int) -> List[bool]:
    """Spread ``compute_budget`` compute steps as evenly as possible across ``num_steps`` (step 0 always computes)."""
    if compute_budget >= num_steps:
        return [True] * num_steps

    indices = set()
    if compute_budget == 1:
        indices.add(0)
    else:
        for i in range(compute_budget):
            indices.add(int(round(i * (num_steps - 1) / (compute_budget - 1))))
    # Rounding collisions can leave fewer than ``compute_budget`` indices; fill deterministically.
    cursor = 0
    while len(indices) < compute_budget:
        if cursor not in indices:
            indices.add(cursor)
        cursor += 1

    schedule = [False] * num_steps
    for idx in indices:
        schedule[idx] = True
    schedule[0] = True
    return schedule


def _schedule_cost(compute_schedule: List[bool], step_errors: List[float]) -> float:
    """
    Surrogate trajectory error of a cache policy.

    Each skipped step reuses a cached residual that grows staler the longer compute is deferred, so its contribution is
    weighted by the number of steps since the last computed step. Minimizing this cost spreads compute toward steps the
    error profile marks as expensive to skip and discourages long skip runs.
    """
    cost = 0.0
    gap = 0
    for i, compute in enumerate(compute_schedule):
        if compute:
            gap = 0
        else:
            gap += 1
            cost += float(step_errors[i]) * gap
    return cost


def _propose_swap(schedule: List[bool], rng: random.Random) -> Optional[List[bool]]:
    """Swap one computed step (never step 0) with one skipped step, preserving the compute budget."""
    compute_idx = [i for i, c in enumerate(schedule) if c and i != 0]
    skip_idx = [i for i, c in enumerate(schedule) if not c]
    if not compute_idx or not skip_idx:
        return None
    off = rng.choice(compute_idx)
    on = rng.choice(skip_idx)
    neighbor = list(schedule)
    neighbor[off] = False
    neighbor[on] = True
    return neighbor


def _hill_climb(schedule: List[bool], step_errors: List[float]) -> Tuple[List[bool], float]:
    """Deterministic local search: repeatedly apply the single budget-preserving swap that most reduces the cost."""
    best = list(schedule)
    best_cost = _schedule_cost(best, step_errors)
    improved = True
    while improved:
        improved = False
        compute_idx = [i for i, c in enumerate(best) if c and i != 0]
        skip_idx = [i for i, c in enumerate(best) if not c]
        for off in compute_idx:
            for on in skip_idx:
                candidate = list(best)
                candidate[off] = False
                candidate[on] = True
                cost = _schedule_cost(candidate, step_errors)
                if cost < best_cost - 1e-12:
                    best, best_cost = candidate, cost
                    improved = True
                    break
            if improved:
                break
    return best, best_cost


def search_cache_schedule(
    step_errors: Union[torch.Tensor, List[float]],
    compute_budget: int,
    *,
    search_iterations: int = 2000,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.995,
    seed: int = 0,
) -> List[bool]:
    """
    Offline search for a budget-constrained step-level cache policy (the core of BudCache).

    Rather than letting a per-step error threshold dictate runtime cost, the compute budget is fixed in advance and we
    search for the binary compute/skip schedule that best preserves the final output. Combines Simulated Annealing
    (global exploration via budget-preserving swaps under a cooling temperature) with deterministic Hill Climbing
    (local refinement) over the surrogate cost in [`_schedule_cost`]. The search is deterministic given ``seed`` and
    runs offline, so inference incurs no thresholding or search overhead.

    Args:
        step_errors (`torch.Tensor` or `List[float]`):
            Per-step skip cost, e.g. derived from MagCache-style residual magnitude ratios. Large values mark steps
            that are expensive to skip.
        compute_budget (`int`):
            Number of steps (out of ``len(step_errors)``) that must be computed. Step 0 always computes.

    Returns:
        `List[bool]`: schedule of length ``len(step_errors)`` with exactly ``compute_budget`` entries set to `True`.
    """
    if torch.is_tensor(step_errors):
        step_errors = step_errors.detach().float().cpu().tolist()
    else:
        step_errors = [float(x) for x in step_errors]

    num_steps = len(step_errors)
    if not 1 <= compute_budget <= num_steps:
        raise ValueError(f"`compute_budget` must be in [1, {num_steps}], got {compute_budget}.")
    if compute_budget == num_steps:
        return [True] * num_steps

    current = _uniform_schedule(num_steps, compute_budget)
    current_cost = _schedule_cost(current, step_errors)
    best, best_cost = list(current), current_cost

    rng = random.Random(seed)
    temperature = max(initial_temperature, 1e-8)
    for _ in range(search_iterations):
        neighbor = _propose_swap(current, rng)
        if neighbor is None:
            break
        neighbor_cost = _schedule_cost(neighbor, step_errors)
        delta = neighbor_cost - current_cost
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current, current_cost = neighbor, neighbor_cost
            if current_cost < best_cost:
                best, best_cost = list(current), current_cost
        temperature *= cooling_rate

    best, _ = _hill_climb(best, step_errors)
    return best


@dataclass
class BudgetCacheConfig:
    r"""
    Configuration for [BudCache](https://github.com/Westlake-AGI-Lab/BudCache).

    BudCache inverts threshold-based step-level caching: instead of per-step error thresholds dictating runtime cost,
    the compute budget is fixed and an offline search ([`search_cache_schedule`]) finds the cache policy that best
    preserves the final output. The resulting schedule is replayed at inference with no online search overhead.

    Args:
        num_inference_steps (`int`, defaults to `28`):
            The number of denoising steps the pipeline runs.
        compute_budget (`int`, *optional*):
            Number of steps that must be computed (the rest reuse cached residuals). Required unless
            `compute_schedule` is given directly.
        compute_schedule (`List[bool]`, *optional*):
            A precomputed per-step schedule (`True` == compute). If provided it is used as-is; otherwise it is derived
            from `compute_budget`.
        step_errors (`torch.Tensor` or `List[float]`, *optional*):
            Per-step skip cost profile (e.g. MagCache-style residual ratios) used to drive the offline search. If
            omitted, a uniformly spaced budget-respecting schedule is used as a heuristic baseline.
        search_iterations (`int`, defaults to `2000`):
            Number of Simulated Annealing iterations.
        initial_temperature (`float`, defaults to `1.0`):
            Starting SA temperature.
        cooling_rate (`float`, defaults to `0.995`):
            Per-iteration multiplicative SA cooling factor.
        seed (`int`, defaults to `0`):
            Seed making the offline search deterministic.
    """

    num_inference_steps: int = 28
    compute_budget: Optional[int] = None
    compute_schedule: Optional[List[bool]] = None
    step_errors: Optional[Union[torch.Tensor, List[float]]] = None
    search_iterations: int = 2000
    initial_temperature: float = 1.0
    cooling_rate: float = 0.995
    seed: int = 0

    def __post_init__(self):
        if self.compute_schedule is not None:
            self.compute_schedule = [bool(x) for x in self.compute_schedule]
            if len(self.compute_schedule) != self.num_inference_steps:
                raise ValueError(
                    f"`compute_schedule` length ({len(self.compute_schedule)}) must equal "
                    f"`num_inference_steps` ({self.num_inference_steps})."
                )
            if not self.compute_schedule[0]:
                raise ValueError("The first step must always be computed (`compute_schedule[0]` must be `True`).")
            return

        if self.compute_budget is None:
            raise ValueError("Either `compute_budget` or `compute_schedule` must be provided.")
        if not 1 <= self.compute_budget <= self.num_inference_steps:
            raise ValueError(
                f"`compute_budget` must be in [1, {self.num_inference_steps}], got {self.compute_budget}."
            )

        if self.step_errors is not None:
            errors = self.step_errors
            if torch.is_tensor(errors):
                errors = errors.detach().float().cpu().tolist()
            else:
                errors = [float(x) for x in errors]
            if len(errors) != self.num_inference_steps:
                raise ValueError(
                    f"`step_errors` length ({len(errors)}) must equal `num_inference_steps` "
                    f"({self.num_inference_steps})."
                )
            self.compute_schedule = search_cache_schedule(
                errors,
                self.compute_budget,
                search_iterations=self.search_iterations,
                initial_temperature=self.initial_temperature,
                cooling_rate=self.cooling_rate,
                seed=self.seed,
            )
        else:
            self.compute_schedule = _uniform_schedule(self.num_inference_steps, self.compute_budget)


class BudgetCacheState(BaseState):
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


class BudgetCacheHeadHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager, config: BudgetCacheConfig):
        self.state_manager = state_manager
        self.config = config
        self._metadata = None

    def initialize_hook(self, module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def _should_compute(self, step_index: int) -> bool:
        schedule = self.config.compute_schedule
        if step_index >= len(schedule):
            return True
        # The first step has no cached residual yet, so it must always be computed.
        return bool(schedule[step_index]) or step_index == 0

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")

        arg_name = self._metadata.hidden_states_argument_name
        hidden_states = self._metadata._get_parameter_from_args_kwargs(arg_name, args, kwargs)

        state: BudgetCacheState = self.state_manager.get_state()
        state.head_block_input = hidden_states

        should_compute = self._should_compute(state.step_index)
        if state.previous_residual is None:
            # No residual cached yet (e.g. forced budget on step 0) — cannot skip.
            should_compute = True
        state.should_compute = should_compute

        if should_compute:
            return self.fn_ref.original_forward(*args, **kwargs)

        logger.debug(f"BudgetCache: reusing cached residual at step {state.step_index}")
        output = hidden_states
        res = state.previous_residual
        if res.device != output.device:
            res = res.to(output.device)

        if res.shape == output.shape:
            output = output + res
        elif (
            output.ndim == 3
            and res.ndim == 3
            and output.shape[0] == res.shape[0]
            and output.shape[2] == res.shape[2]
        ):
            diff = output.shape[1] - res.shape[1]
            if diff > 0:
                output = output.clone()
                output[:, diff:, :] = output[:, diff:, :] + res
            else:
                logger.warning(
                    f"BudgetCache: residual shape {res.shape} incompatible with input {output.shape}; skipping reuse."
                )
        else:
            logger.warning(
                f"BudgetCache: residual shape {res.shape} incompatible with input {output.shape}; skipping reuse."
            )

        if self._metadata.return_encoder_hidden_states_index is not None:
            original_encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                "encoder_hidden_states", args, kwargs
            )
            max_idx = max(self._metadata.return_hidden_states_index, self._metadata.return_encoder_hidden_states_index)
            ret_list = [None] * (max_idx + 1)
            ret_list[self._metadata.return_hidden_states_index] = output
            ret_list[self._metadata.return_encoder_hidden_states_index] = original_encoder_hidden_states
            return tuple(ret_list)
        return output

    def reset_state(self, module):
        self.state_manager.reset()
        return module


class BudgetCacheBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, is_tail: bool = False, config: BudgetCacheConfig = None):
        super().__init__()
        self.state_manager = state_manager
        self.is_tail = is_tail
        self.config = config
        self._metadata = None

    def initialize_hook(self, module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def _advance_step(self, state: BudgetCacheState):
        state.step_index += 1
        if state.step_index >= self.config.num_inference_steps:
            state.step_index = 0
            state.previous_residual = None

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")
        state: BudgetCacheState = self.state_manager.get_state()

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
            if in_hidden is not None:
                if out_hidden.shape == in_hidden.shape:
                    state.previous_residual = out_hidden - in_hidden
                elif out_hidden.ndim == 3 and in_hidden.ndim == 3 and out_hidden.shape[2] == in_hidden.shape[2]:
                    state.previous_residual = out_hidden - in_hidden[:, in_hidden.shape[1] - out_hidden.shape[1] :, :]
                else:
                    state.previous_residual = out_hidden
            self._advance_step(state)

        return output


def apply_budget_cache(module: torch.nn.Module, config: BudgetCacheConfig) -> None:
    """
    Apply BudCache budget-constrained step-level caching to a transformer module.

    Reference: https://github.com/Westlake-AGI-Lab/BudCache

    Args:
        module (`torch.nn.Module`):
            The transformer to cache (its `compute_schedule` is replayed across denoising steps).
        config (`BudgetCacheConfig`):
            The BudCache configuration, carrying the precomputed compute/skip schedule.
    """
    HookRegistry.check_if_exists_or_initialize(module)
    state_manager = StateManager(BudgetCacheState, (), {})

    remaining_blocks = []
    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for index, block in enumerate(submodule):
            remaining_blocks.append((f"{name}.{index}", block))

    if not remaining_blocks:
        logger.warning("BudgetCache: No transformer blocks found to apply hooks.")
        return

    if len(remaining_blocks) == 1:
        _, block = remaining_blocks[0]
        _apply_budget_cache_block_hook(block, state_manager, config, is_tail=True)
        _apply_budget_cache_head_hook(block, state_manager, config)
        return

    head_block_name, head_block = remaining_blocks.pop(0)
    tail_block_name, tail_block = remaining_blocks.pop(-1)

    logger.info(f"BudgetCache: Applying Head Hook to {head_block_name}")
    _apply_budget_cache_head_hook(head_block, state_manager, config)
    for _, block in remaining_blocks:
        _apply_budget_cache_block_hook(block, state_manager, config)
    logger.info(f"BudgetCache: Applying Tail Hook to {tail_block_name}")
    _apply_budget_cache_block_hook(tail_block, state_manager, config, is_tail=True)


def _apply_budget_cache_head_hook(block: torch.nn.Module, state_manager: StateManager, config: BudgetCacheConfig):
    registry = HookRegistry.check_if_exists_or_initialize(block)
    if registry.get_hook(_BUDGET_CACHE_LEADER_BLOCK_HOOK) is not None:
        registry.remove_hook(_BUDGET_CACHE_LEADER_BLOCK_HOOK)
    registry.register_hook(BudgetCacheHeadHook(state_manager, config), _BUDGET_CACHE_LEADER_BLOCK_HOOK)


def _apply_budget_cache_block_hook(
    block: torch.nn.Module, state_manager: StateManager, config: BudgetCacheConfig, is_tail: bool = False
):
    registry = HookRegistry.check_if_exists_or_initialize(block)
    if registry.get_hook(_BUDGET_CACHE_BLOCK_HOOK) is not None:
        registry.remove_hook(_BUDGET_CACHE_BLOCK_HOOK)
    registry.register_hook(BudgetCacheBlockHook(state_manager, is_tail, config), _BUDGET_CACHE_BLOCK_HOOK)
