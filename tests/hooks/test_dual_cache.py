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
from diffusers.models import ModelMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.utils import logging


logger = logging.get_logger(__name__)


class DummyBlock(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        # Each block doubles its input, so the residual across all blocks is deterministic.
        return hidden_states * 2.0


class DummyTransformer(ModelMixin, CacheMixin):
    def __init__(self, num_blocks=4):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock() for _ in range(num_blocks)])

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


def test_dual_cache_config_validation():
    with pytest.raises(ValueError):
        DualCacheConfig(cache_interval=0)
    with pytest.raises(ValueError):
        DualCacheConfig(conservative_fraction=0.0)
    with pytest.raises(ValueError):
        DualCacheConfig(conservative_fraction=1.0)


def test_dual_cache_schedule():
    """Exercise a full fresh -> aggressive -> conservative cycle through the DuCa hooks."""
    # 4 blocks, each doubles: a full forward computes input * 16.
    model = DummyTransformer(num_blocks=4)
    # conservative_fraction=0.5 -> recompute the first 2 blocks on conservative steps.
    config = DualCacheConfig(cache_interval=3, conservative_fraction=0.5, warmup_steps=0)
    apply_dual_cache(model, config)
    _set_context(model, "test_context")

    # Step 0 (FRESH): full compute. Input 1 -> 16.
    # full_residual = 16 - 1 = 15 (across all 4 blocks).
    # deep_residual = 16 - 4 = 12 (blocks 2..3, since input to block 2 during fresh is 1*2*2 = 4).
    out0 = model(torch.tensor([[[1.0]]]))
    assert torch.allclose(out0, torch.tensor([[[16.0]]]))

    # Step 1 (AGGRESSIVE): skip everything, reuse full_residual. Input 2 -> 2 + 15 = 17.
    out1 = model(torch.tensor([[[2.0]]]))
    assert torch.allclose(out1, torch.tensor([[[17.0]]])), f"aggressive step got {out1.item()}"

    # Step 2 (CONSERVATIVE): recompute first 2 blocks then reuse deep_residual.
    # Input 3 -> blocks 0,1 compute -> 3*2*2 = 12; + deep_residual(12) = 24.
    out2 = model(torch.tensor([[[3.0]]]))
    assert torch.allclose(out2, torch.tensor([[[24.0]]])), f"conservative step got {out2.item()}"

    # Step 3 (FRESH again, cycle restart): full compute. Input 5 -> 80.
    out3 = model(torch.tensor([[[5.0]]]))
    assert torch.allclose(out3, torch.tensor([[[80.0]]])), f"fresh restart got {out3.item()}"


def test_dual_cache_warmup_forces_compute():
    """During warmup, every step is fully computed regardless of the cycle position."""
    model = DummyTransformer(num_blocks=4)
    config = DualCacheConfig(cache_interval=3, conservative_fraction=0.5, warmup_steps=2)
    apply_dual_cache(model, config)
    _set_context(model, "test_context")

    model(torch.tensor([[[1.0]]]))  # step 0 (warmup -> fresh)
    # step 1 would be aggressive without warmup; warmup forces full compute: 2 -> 32.
    out1 = model(torch.tensor([[[2.0]]]))
    assert torch.allclose(out1, torch.tensor([[[32.0]]])), f"warmup step got {out1.item()}"


def test_dual_cache_enable_disable_via_cache_mixin():
    """The DuCa config routes through CacheMixin.enable_cache/disable_cache wiring."""
    model = DummyTransformer(num_blocks=4)
    config = DualCacheConfig(cache_interval=3)

    model.enable_cache(config)
    assert model.is_cache_enabled
    assert any(
        hasattr(m, "_diffusers_hook") and m._diffusers_hook.get_hook("dual_cache_block_hook") is not None
        for m in model.modules()
    ), "expected dual_cache_block_hook to be registered on the transformer blocks"

    model.disable_cache()
    assert not model.is_cache_enabled
    assert all(
        not hasattr(m, "_diffusers_hook") or m._diffusers_hook.get_hook("dual_cache_block_hook") is None
        for m in model.modules()
    ), "expected dual_cache_block_hook to be removed after disable_cache"
