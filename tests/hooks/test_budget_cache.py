# Copyright 2025 HuggingFace Inc.
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

import pytest
import torch

from diffusers import BudgetCacheConfig, apply_budget_cache
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.budget_cache import _schedule_cost, search_cache_schedule


class DummyBlock(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        # Output is double input -> with two blocks, residual = 4*x - x = 3*x.
        return hidden_states * 2.0


class DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock(), DummyBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states


@pytest.fixture(autouse=True)
def register_dummy_blocks():
    TransformerBlockRegistry.register(
        DummyBlock,
        TransformerBlockMetadata(return_hidden_states_index=None, return_encoder_hidden_states_index=None),
    )


def _set_context(model, context_name):
    for module in model.modules():
        if hasattr(module, "_diffusers_hook"):
            module._diffusers_hook._set_context(context_name)


def test_search_respects_budget_and_first_step():
    step_errors = [5.0, 0.1, 0.1, 4.0, 0.1, 0.1, 3.0, 0.1]
    schedule = search_cache_schedule(step_errors, compute_budget=4, seed=0)

    assert len(schedule) == len(step_errors)
    assert sum(schedule) == 4
    assert schedule[0] is True  # first step is always computed


def test_search_beats_uniform_heuristic():
    # Errors concentrated on a few steps; a good policy should compute those steps.
    step_errors = [3.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 4.0, 0.0, 0.0]
    budget = 4

    from diffusers.hooks.budget_cache import _uniform_schedule

    uniform = _uniform_schedule(len(step_errors), budget)
    searched = search_cache_schedule(step_errors, budget, seed=0)

    assert _schedule_cost(searched, step_errors) <= _schedule_cost(uniform, step_errors)


def test_config_schedule_validation():
    with pytest.raises(ValueError):
        BudgetCacheConfig(num_inference_steps=4)  # neither budget nor schedule
    with pytest.raises(ValueError):
        BudgetCacheConfig(num_inference_steps=4, compute_budget=99)  # budget out of range
    with pytest.raises(ValueError):
        BudgetCacheConfig(num_inference_steps=4, compute_schedule=[False, True, True, True])  # step 0 not computed


def test_budget_cache_replays_schedule():
    """The applied hook computes on scheduled steps and reuses the cached residual otherwise."""
    model = DummyTransformer()
    # Step 0 computes (mandatory), step 1 skips (reuse residual).
    config = BudgetCacheConfig(num_inference_steps=2, compute_schedule=[True, False])
    assert config.compute_schedule == [True, False]

    apply_budget_cache(model, config)
    _set_context(model, "test_context")

    # Step 0: input 10 -> output 40 (2 blocks * 2x). Residual = 30.
    out0 = model(torch.tensor([[[10.0]]]))
    assert torch.allclose(out0, torch.tensor([[[40.0]]]))

    # Step 1: scheduled skip -> output = input(11) + residual(30) = 41.
    out1 = model(torch.tensor([[[11.0]]]))
    assert torch.allclose(out1, torch.tensor([[[41.0]]])), f"Expected cached reuse (41.0), got {out1.item()}"


def test_budget_cache_full_budget_computes_every_step():
    model = DummyTransformer()
    config = BudgetCacheConfig(num_inference_steps=2, compute_budget=2)
    assert config.compute_schedule == [True, True]

    apply_budget_cache(model, config)
    _set_context(model, "test_context")

    model(torch.tensor([[[10.0]]]))  # step 0
    # Step 1 must compute: 11 * 4 = 44 (not the cached-reuse value 41).
    out1 = model(torch.tensor([[[11.0]]]))
    assert torch.allclose(out1, torch.tensor([[[44.0]]])), f"Expected full compute (44.0), got {out1.item()}"
