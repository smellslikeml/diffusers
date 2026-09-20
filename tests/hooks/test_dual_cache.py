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

from diffusers import DualCacheConfig, apply_dual_cache
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.dual_cache import DualCachePolicy
from diffusers.models import ModelMixin
from diffusers.models.cache_utils import CacheMixin


class DummyBlock(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        return hidden_states * 2.0


class DummyTransformer(ModelMixin, CacheMixin):
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


def test_policy_dual_schedule():
    """The policy must open each cycle with a compute step, then aggressive, then conservative steps."""
    policy = DualCachePolicy(cache_interval=3, aggressive_steps=1, retention_ratio=0.0, num_inference_steps=9)
    types = [policy.classify(i) for i in range(9)]
    assert types == [
        "compute",
        "aggressive",
        "conservative",
        "compute",
        "aggressive",
        "conservative",
        "compute",
        "aggressive",
        "conservative",
    ]


def test_policy_retention_forces_compute():
    """Steps inside the warmup window are always full computes regardless of the cycle."""
    policy = DualCachePolicy(cache_interval=2, aggressive_steps=1, retention_ratio=0.5, num_inference_steps=8)
    assert policy.retention_steps == 4
    assert [policy.classify(i) for i in range(4)] == ["compute"] * 4
    # After warmup the dual cycle resumes.
    assert policy.classify(4) == "compute"
    assert policy.classify(5) == "aggressive"


def test_policy_clamps_aggressive_steps():
    """`aggressive_steps` can never exceed the number of reuse slots in a cycle."""
    policy = DualCachePolicy(cache_interval=2, aggressive_steps=5, retention_ratio=0.0, num_inference_steps=4)
    assert policy.aggressive_steps == 1
    assert [policy.classify(i) for i in range(4)] == ["compute", "aggressive", "compute", "aggressive"]


def test_aggressive_step_reuses_residual_verbatim():
    """A fresh compute caches the residual; the next (aggressive) step reuses it as-is."""
    model = DummyTransformer()
    config = DualCacheConfig(
        cache_interval=2, aggressive_steps=1, retention_ratio=0.0, num_inference_steps=2
    )
    apply_dual_cache(model, config)
    _set_context(model, "test_context")

    # Step 0 (compute): input 10 -> 4x -> 40, cached residual = 30.
    assert torch.allclose(model(torch.tensor([[[10.0]]])), torch.tensor([[[40.0]]]))
    # Step 1 (aggressive): reuse -> 11 + 30 = 41 (not the computed 44).
    assert torch.allclose(model(torch.tensor([[[11.0]]])), torch.tensor([[[41.0]]]))


def test_conservative_step_damps_residual():
    """Conservative steps reuse the cached residual scaled by `conservative_scale`."""
    model = DummyTransformer()
    config = DualCacheConfig(
        cache_interval=3,
        aggressive_steps=1,
        conservative_scale=0.5,
        retention_ratio=0.0,
        num_inference_steps=3,
    )
    apply_dual_cache(model, config)
    _set_context(model, "test_context")

    model(torch.tensor([[[10.0]]]))  # compute -> residual 30
    model(torch.tensor([[[11.0]]]))  # aggressive -> 41
    # Step 2 (conservative): 12 + 30 * 0.5 = 27.
    assert torch.allclose(model(torch.tensor([[[12.0]]])), torch.tensor([[[27.0]]]))


def test_enable_cache_dispatches_dual_cache():
    """The CacheMixin.enable_cache dispatch must route DualCacheConfig through apply_dual_cache."""
    model = DummyTransformer()
    assert not model.is_cache_enabled

    model.enable_cache(DualCacheConfig(cache_interval=2, retention_ratio=0.0, num_inference_steps=2))
    assert model.is_cache_enabled
    assert isinstance(model._cache_config, DualCacheConfig)
    # The head hook must be registered on the first transformer block by the dispatch.
    assert model.transformer_blocks[0]._diffusers_hook.get_hook("dual_cache_leader_block_hook") is not None

    model.disable_cache()
    assert not model.is_cache_enabled
    assert model.transformer_blocks[0]._diffusers_hook.get_hook("dual_cache_leader_block_hook") is None
