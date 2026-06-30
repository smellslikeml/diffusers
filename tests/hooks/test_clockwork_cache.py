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

from diffusers import ClockworkCacheConfig, apply_clockwork_cache
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.models import ModelMixin


class CountingBlock(torch.nn.Module):
    """A transformer block that doubles its input and records how often its forward runs."""

    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        self.forward_calls += 1
        return hidden_states * 2.0


class CountingTransformer(ModelMixin):
    def __init__(self, num_blocks: int = 3):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([CountingBlock() for _ in range(num_blocks)])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states


@pytest.fixture(autouse=True)
def register_counting_blocks():
    TransformerBlockRegistry.register(
        CountingBlock,
        TransformerBlockMetadata(return_hidden_states_index=None, return_encoder_hidden_states_index=None),
    )


def _set_context(model, context_name):
    """Helper to set context on all hooks in the model."""
    for module in model.modules():
        if hasattr(module, "_diffusers_hook"):
            module._diffusers_hook._set_context(context_name)


def test_clockwork_cache_validation():
    """Invalid config is rejected."""
    with pytest.raises(ValueError):
        ClockworkCacheConfig(clock_interval=0)
    with pytest.raises(ValueError):
        ClockworkCacheConfig(warmup=-1)


def test_clockwork_cache_clock_schedule():
    """
    Every `clock_interval` steps the block stack recomputes; the steps in between reuse the cached output, so the
    non-leader blocks are skipped. With clock_interval=3 and warmup=1 over 4 steps: steps 0 and 3 compute, steps 1
    and 2 reuse.
    """
    model = CountingTransformer(num_blocks=3)
    apply_clockwork_cache(model, ClockworkCacheConfig(clock_interval=3, warmup=1))

    _set_context(model, "default")
    x = torch.randn(1, 4)

    for _ in range(4):
        model(x)

    leader, middle, tail = model.transformer_blocks
    # The leader recomputes itself every step (it is the reference for the cached delta).
    assert leader.forward_calls == 4
    # Middle and tail only run on the compute steps (0 and 3).
    assert middle.forward_calls == 2
    assert tail.forward_calls == 2


def test_clockwork_cache_replays_correctly_for_static_input():
    """
    For a deterministic block, the cached replay on a reuse step must match a full recomputation of the same input.
    """
    cached_model = CountingTransformer(num_blocks=3)
    plain_model = CountingTransformer(num_blocks=3)
    apply_clockwork_cache(cached_model, ClockworkCacheConfig(clock_interval=3, warmup=1))

    _set_context(cached_model, "default")
    x = torch.randn(1, 4)

    expected = plain_model(x)

    # Step 0 is a compute step (warmup) and must reproduce the full forward.
    assert torch.allclose(cached_model(x), expected)
    # Step 1 is a reuse step and must replay the cached output.
    assert torch.allclose(cached_model(x), expected)


def test_clockwork_cache_requires_multiple_blocks():
    """A single-block transformer cannot be cached (needs a leader and a tail)."""
    model = CountingTransformer(num_blocks=1)
    with pytest.raises(ValueError, match="at least two transformer blocks"):
        apply_clockwork_cache(model, ClockworkCacheConfig())
