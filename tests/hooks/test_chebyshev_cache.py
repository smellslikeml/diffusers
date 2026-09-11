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

import torch

from diffusers import ChebyshevCacheConfig, apply_chebyshev_cache
from diffusers.hooks.chebyshev_cache import chebyshev_barycentric_weights
from diffusers.models import ModelMixin
from diffusers.models.cache_utils import CacheMixin


class DummyBlock(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        # Output is double the input, so a cached (extrapolated) step is
        # distinguishable from a freshly computed one.
        return hidden_states * 2.0


class DummyTransformer(ModelMixin, CacheMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return hidden_states


def _set_context(model, context_name):
    """Helper to set the state-manager context on all hooks in the model."""
    for module in model.modules():
        if hasattr(module, "_diffusers_hook"):
            module._diffusers_hook._set_context(context_name)


def test_chebyshev_barycentric_weights():
    # Single node -> constant extrapolation.
    assert chebyshev_barycentric_weights(1) == [1.0]
    # For two nodes the Chebyshev-Lobatto weights coincide (up to a global
    # scale that cancels in the barycentric quotient) with the exact linear
    # barycentric weights, so extrapolation is exact for affine data.
    assert chebyshev_barycentric_weights(2) == [0.5, -0.5]
    # Alternating signs with halved endpoints for higher orders.
    assert chebyshev_barycentric_weights(4) == [0.5, -1.0, 1.0, -0.5]


def test_chebyshev_cache_linear_extrapolation():
    """Two seeded nodes -> the cached step must be the exact linear extrapolant, not a recompute."""
    model = DummyTransformer()
    config = ChebyshevCacheConfig(
        cache_interval=100,
        disable_cache_before_step=2,
        max_order=1,
        factors_dtype=torch.float32,
        cache_identifiers=[r"transformer_blocks\.0"],
    )
    apply_chebyshev_cache(model, config)
    _set_context(model, "test_context")

    # Step 0 (warmup, compute): 2 * 1.0 -> 2.0, seeds node (step=0, value=2.0).
    out0 = model(torch.tensor([[[1.0]]]))
    assert torch.allclose(out0, torch.tensor([[[2.0]]])), f"Step 0 should compute, got {out0.item()}"

    # Step 1 (warmup, compute): 2 * 2.0 -> 4.0, seeds node (step=1, value=4.0).
    out1 = model(torch.tensor([[[2.0]]]))
    assert torch.allclose(out1, torch.tensor([[[4.0]]])), f"Step 1 should compute, got {out1.item()}"

    # Step 2 (cache): linear extrapolation of (0, 2.0) and (1, 4.0) at x=2 -> 6.0.
    # A recompute would instead give 2 * 100.0 = 200.0, so this asserts the skip path.
    out2 = model(torch.tensor([[[100.0]]]))
    assert torch.allclose(out2, torch.tensor([[[6.0]]])), f"Step 2 should extrapolate to 6.0, got {out2.item()}"


def test_chebyshev_cache_constant_reproduction():
    """Berrut's rational interpolant reproduces constants exactly, at any order."""
    model = DummyTransformer()
    config = ChebyshevCacheConfig(
        cache_interval=100,
        disable_cache_before_step=3,
        max_order=2,
        factors_dtype=torch.float32,
        cache_identifiers=[r"transformer_blocks\.0"],
    )
    apply_chebyshev_cache(model, config)
    _set_context(model, "test_context")

    const_in = torch.tensor([[[5.0]]])
    for _ in range(3):  # warmup computes -> history is constant 10.0
        model(const_in)

    # Cached step with a different input: must return the constant 10.0, not 2 * 999.
    out = model(torch.tensor([[[999.0]]]))
    assert torch.allclose(out, torch.tensor([[[10.0]]])), f"Constant extrapolation failed, got {out.item()}"


def test_chebyshev_cache_enable_disable_dispatch():
    """enable_cache / disable_cache must route ChebyshevCacheConfig through the CacheMixin dispatcher."""
    model = DummyTransformer()
    config = ChebyshevCacheConfig(cache_identifiers=[r"transformer_blocks\.0"])

    model.enable_cache(config)
    assert model.is_cache_enabled
    assert any(hasattr(m, "_diffusers_hook") for m in model.modules())

    model.disable_cache()
    assert not model.is_cache_enabled
