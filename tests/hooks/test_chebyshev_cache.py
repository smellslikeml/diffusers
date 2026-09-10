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

from diffusers import ChebyshevCacheConfig
from diffusers.models import ModelMixin
from diffusers.models.cache_utils import CacheMixin


class IdentityAttention(torch.nn.Module):
    """Leaf module that returns its input unchanged, so the cached snapshot equals the fed value."""

    def forward(self, hidden_states):
        return hidden_states


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = IdentityAttention()

    def forward(self, hidden_states):
        return self.attn(hidden_states)


class DummyTransformer(ModelMixin, CacheMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([Block()])

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def _set_context(model, context_name):
    for module in model.modules():
        if hasattr(module, "_diffusers_hook"):
            module._diffusers_hook._set_context(context_name)


def test_enable_cache_dispatches_to_chebyshev():
    """`CacheMixin.enable_cache` should route a ChebyshevCacheConfig to the new apply function and hook the module."""
    model = DummyTransformer()
    model.enable_cache(ChebyshevCacheConfig(cache_interval=5, disable_cache_before_step=2, num_nodes=2))

    assert model.is_cache_enabled
    hook = model.transformer_blocks[0].attn._diffusers_hook
    assert hook.get_hook("chebyshev_cache") is not None

    model.disable_cache()
    assert not model.is_cache_enabled
    assert not hasattr(model.transformer_blocks[0].attn, "_diffusers_hook") or (
        model.transformer_blocks[0].attn._diffusers_hook.get_hook("chebyshev_cache") is None
    )


def test_barycentric_linear_extrapolation():
    """
    With two interpolation nodes the Barycentric interpolant reduces to linear extrapolation: for computed values
    f0 at step 0 and f1 at step 1, a skipped step 2 must yield 2*f1 - f0, independent of the fed input.
    """
    model = DummyTransformer()
    model.enable_cache(
        ChebyshevCacheConfig(
            cache_interval=5,
            disable_cache_before_step=2,
            num_nodes=2,
            cache_factors_dtype=torch.float32,
        )
    )
    _set_context(model, "test_context")

    # Step 0 (warmup / compute): output echoes input.
    out0 = model(torch.tensor([[[1.0]]]))
    assert torch.allclose(out0, torch.tensor([[[1.0]]]))

    # Step 1 (warmup / compute): output echoes input.
    out1 = model(torch.tensor([[[2.0]]]))
    assert torch.allclose(out1, torch.tensor([[[2.0]]]))

    # Step 2 (skip / extrapolate): 2 * f1 - f0 = 3.0, and the fed 99.0 is ignored.
    out2 = model(torch.tensor([[[99.0]]]))
    assert torch.allclose(out2, torch.tensor([[[3.0]]])), f"Expected linear extrapolation 3.0, got {out2.item()}"


def test_single_node_falls_back_to_constant_reuse():
    """num_nodes=1 keeps only the most recent computation, so a skipped step reuses it verbatim."""
    model = DummyTransformer()
    model.enable_cache(
        ChebyshevCacheConfig(
            cache_interval=5,
            disable_cache_before_step=1,
            num_nodes=1,
            cache_factors_dtype=torch.float32,
        )
    )
    _set_context(model, "test_context")

    out0 = model(torch.tensor([[[7.0]]]))  # Step 0: compute
    assert torch.allclose(out0, torch.tensor([[[7.0]]]))

    out1 = model(torch.tensor([[[99.0]]]))  # Step 1: skip -> reuse last computed (7.0)
    assert torch.allclose(out1, torch.tensor([[[7.0]]])), f"Expected constant reuse 7.0, got {out1.item()}"
