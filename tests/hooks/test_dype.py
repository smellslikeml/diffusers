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

from diffusers.hooks import HookRegistry, apply_dype
from diffusers.hooks.dype import (
    DyPEHook,
    _DyPEPosEmbed,
    _dype_rotary_pos_embed,
    find_correction_factor,
    find_correction_range,
    find_newbase_ntk,
    linear_ramp_mask,
)
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.transformers.transformer_flux import FluxPosEmbed


# Flux trains at 1024x1024 with patch size 16, i.e. 64 patches per spatial axis.
BASE_PATCHES = 64
AXES_DIM = [16, 56, 56]
THETA = 10000


def build_flux_style_ids(num_txt_tokens: int, patch_grid: int) -> torch.Tensor:
    # Mirrors the ids fed to `FluxTransformer2DModel.pos_embed`: `txt_ids` are all-zero and `img_ids` carry the (h, w)
    # patch grid positions on the two spatial axes.
    txt_ids = torch.zeros(num_txt_tokens, 3)
    h, w = torch.meshgrid(torch.arange(patch_grid), torch.arange(patch_grid), indexing="ij")
    img_ids = torch.cat([torch.zeros(patch_grid * patch_grid, 1), torch.stack((h, w), dim=-1).reshape(-1, 2)], dim=1)
    return torch.cat((txt_ids, img_ids), dim=0).float()


class TestDypeScheduleHelpers:
    def test_find_correction_factor(self):
        # (dim * ln(max_pe / (n_rot * 2pi))) / (2 * ln(base))
        assert find_correction_factor(1.25, 56, THETA, BASE_PATCHES) == pytest.approx(6.37763064832401)
        assert find_correction_factor(0.75, 56, THETA, BASE_PATCHES) == pytest.approx(7.9305718956385)
        assert find_correction_factor(16, 56, THETA, BASE_PATCHES) == pytest.approx(-1.3728391392110684)

    def test_find_correction_range(self):
        # Floor/ceil of the correction factors, clamped to [0, dim - 1]. At kappa=1 (timestep 1.0), the beta ramp
        # covers rotations in (1.25, 0.75) and the gamma ramp in (16, 2).
        assert find_correction_range(1.25, 0.75, 56, THETA, BASE_PATCHES) == (6.0, 8.0)
        assert find_correction_range(16, 2, 56, THETA, BASE_PATCHES) == (0.0, 5.0)

    def test_linear_ramp_mask(self):
        torch.testing.assert_close(
            linear_ramp_mask(2, 6, 8), torch.tensor([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
        )
        # A degenerate (min == max) range is guarded against singularity.
        torch.testing.assert_close(linear_ramp_mask(3, 3, 4), torch.zeros(4))

    def test_find_newbase_ntk(self):
        assert find_newbase_ntk(56, THETA, 4.0) == pytest.approx(42107.40810555822)
        # `scale` may also be a 0-dim tensor, as in the yarn path.
        assert find_newbase_ntk(56, THETA, torch.tensor(4.0, dtype=torch.float64)).item() == pytest.approx(
            42107.40810555822
        )


class TestDypeRotaryPosEmbed:
    def test_noop_at_trained_resolution(self):
        # With max_pe_len <= ori_max_pe_len the yarn branch must not engage, and the plain branch must be bitwise
        # identical to the stock rotary embedding used by FluxPosEmbed.
        pos = torch.arange(BASE_PATCHES)
        common_kwargs = {
            "theta": THETA,
            "use_real": True,
            "repeat_interleave_real": True,
            "freqs_dtype": torch.float64,
        }
        for max_pe_len in (BASE_PATCHES, BASE_PATCHES // 2):
            cos, sin = _dype_rotary_pos_embed(
                56, pos, yarn=True, max_pe_len=max_pe_len, ori_max_pe_len=BASE_PATCHES, dype=True, **common_kwargs
            )
            expected_cos, expected_sin = get_1d_rotary_pos_embed(56, pos, **common_kwargs)
            assert torch.equal(cos, expected_cos)
            assert torch.equal(sin, expected_sin)

    def test_yarn_engages_above_trained_resolution(self):
        pos = torch.arange(128)
        cos, sin = _dype_rotary_pos_embed(
            56,
            pos,
            yarn=True,
            max_pe_len=128,
            ori_max_pe_len=BASE_PATCHES,
            dype=True,
            current_timestep=1.0,
            theta=THETA,
            use_real=True,
            repeat_interleave_real=True,
            freqs_dtype=torch.float64,
        )
        expected_cos, expected_sin = get_1d_rotary_pos_embed(
            56, pos, theta=THETA, use_real=True, repeat_interleave_real=True, freqs_dtype=torch.float64
        )

        assert cos.shape == expected_cos.shape == (128, 56)
        assert not torch.equal(cos, expected_cos)
        assert not torch.equal(sin, expected_sin)

        # The yarn attention temperature (mscale = 0.1 * ln(scale) + 1.0, scale = 128 / 64 = 2) scales cos/sin, so
        # position 0 evaluates to exactly mscale (cos(0) = 1) and 0 (sin(0) = 0).
        assert cos[0, 0].item() == pytest.approx(1.0693147180559945)
        assert sin[0, 0].item() == pytest.approx(0.0, abs=1e-7)

    def test_yarn_follows_the_timestep_schedule(self):
        pos = torch.arange(128)
        common_kwargs = {
            "yarn": True,
            "max_pe_len": 128,
            "ori_max_pe_len": BASE_PATCHES,
            "theta": THETA,
            "use_real": True,
            "repeat_interleave_real": True,
            "freqs_dtype": torch.float64,
        }
        cos_noise, sin_noise = _dype_rotary_pos_embed(56, pos, dype=True, current_timestep=1.0, **common_kwargs)
        cos_mid, sin_mid = _dype_rotary_pos_embed(56, pos, dype=True, current_timestep=0.5, **common_kwargs)
        cos_static, sin_static = _dype_rotary_pos_embed(56, pos, dype=False, **common_kwargs)

        # kappa = t^2 shifts the beta/gamma correction ranges, so the embedding must change across timesteps.
        assert not torch.equal(cos_noise, cos_mid)
        assert not torch.equal(sin_noise, sin_mid)
        # At t = 1.0, kappa = 1 and the DyPE schedule degenerates to static yarn.
        assert torch.equal(cos_noise, cos_static)
        assert torch.equal(sin_noise, sin_static)


class TestDypePosEmbed:
    def test_matches_stock_flux_pos_embed_at_trained_resolution(self):
        pos_embed = _DyPEPosEmbed(THETA, AXES_DIM)
        stock_pos_embed = FluxPosEmbed(THETA, AXES_DIM)

        # Byte-identical output at 1024x1024 (64 patches) and below.
        for patch_grid in (8, 32, BASE_PATCHES):
            ids = build_flux_style_ids(num_txt_tokens=8, patch_grid=patch_grid)
            cos, sin = pos_embed(ids)
            expected_cos, expected_sin = stock_pos_embed(ids)
            assert torch.equal(cos, expected_cos)
            assert torch.equal(sin, expected_sin)

    def test_engages_above_trained_resolution(self):
        pos_embed = _DyPEPosEmbed(THETA, AXES_DIM)
        stock_pos_embed = FluxPosEmbed(THETA, AXES_DIM)
        pos_embed.set_timestep(0.5)

        ids = build_flux_style_ids(num_txt_tokens=8, patch_grid=128)
        cos, sin = pos_embed(ids)
        expected_cos, expected_sin = stock_pos_embed(ids)

        assert cos.shape == expected_cos.shape
        assert not torch.equal(cos, expected_cos)
        assert not torch.equal(sin, expected_sin)

        # The first (text) axis is plain RoPE, so its slice must stay untouched.
        assert torch.equal(cos[:, : AXES_DIM[0]], expected_cos[:, : AXES_DIM[0]])
        assert torch.equal(sin[:, : AXES_DIM[0]], expected_sin[:, : AXES_DIM[0]])

    def test_base_method_disables_dype(self):
        pos_embed = _DyPEPosEmbed(THETA, AXES_DIM, method="base", dype=True)
        assert pos_embed.dype is False


class DummyFluxLikeTransformer(torch.nn.Module):
    # Minimal stand-in for `FluxTransformer2DModel` with the same forward signature prefix, so that the hook's
    # positional fallback for `timestep` can be exercised.
    def __init__(self):
        super().__init__()
        self.pos_embed = FluxPosEmbed(THETA, AXES_DIM)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        pooled_projections=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
    ):
        ids = torch.cat((txt_ids, img_ids), dim=0)
        cos, sin = self.pos_embed(ids)
        return cos, sin, timestep


class TestDypeHook:
    def test_swaps_and_restores_pos_embed(self):
        model = DummyFluxLikeTransformer()
        original_pos_embed = model.pos_embed
        ids = build_flux_style_ids(num_txt_tokens=8, patch_grid=8)
        hidden_states = torch.randn(8, 4)

        apply_dype(model)

        registry = HookRegistry.check_if_exists_or_initialize(model)
        assert isinstance(registry.get_hook("dype_hook"), DyPEHook)
        assert isinstance(model.pos_embed, _DyPEPosEmbed)
        assert model.pos_embed.theta == original_pos_embed.theta
        assert model.pos_embed.axes_dim == original_pos_embed.axes_dim
        assert model.pos_embed.method == "yarn"
        assert model.pos_embed.dype is True

        registry.remove_hook("dype_hook")

        assert model.pos_embed is original_pos_embed
        cos, sin, _ = model(hidden_states, img_ids=ids[8:], txt_ids=ids[:8], timestep=torch.tensor(1.0))
        assert cos.shape == (8 + 64, sum(AXES_DIM))

    def test_timestep_is_read_from_kwargs(self):
        model = DummyFluxLikeTransformer()
        apply_dype(model)
        ids = build_flux_style_ids(num_txt_tokens=8, patch_grid=8)
        hidden_states = torch.randn(8, 4)

        # Stock Flux passes the timestep already normalized to [0, 1], where 1 is pure noise.
        model(hidden_states, img_ids=ids[8:], txt_ids=ids[:8], timestep=torch.tensor([0.75]))
        assert model.pos_embed.current_timestep == 0.75

        model(hidden_states, img_ids=ids[8:], txt_ids=ids[:8], timestep=0.25)
        assert model.pos_embed.current_timestep == 0.25

    def test_timestep_is_read_from_args(self):
        model = DummyFluxLikeTransformer()
        apply_dype(model)
        ids = build_flux_style_ids(num_txt_tokens=8, patch_grid=8)
        hidden_states = torch.randn(8, 4)

        model(hidden_states, None, None, torch.tensor(0.5), ids[8:], ids[:8])
        assert model.pos_embed.current_timestep == 0.5

    def test_timestep_fed_via_native_forward_pre_hook(self):
        # Regression: accelerate's `enable_model_cpu_offload` re-wraps the transformer's `forward`, which bypasses
        # `ModelHook.pre_forward`. DyPE therefore feeds the timestep with a native forward pre-hook, which
        # `nn.Module._call_impl` runs before `forward` regardless of how `forward` is subsequently wrapped.
        model = DummyFluxLikeTransformer()
        assert len(model._forward_pre_hooks) == 0

        apply_dype(model)
        assert len(model._forward_pre_hooks) == 1  # native pre-hook installed by initialize_hook

        # The timestep must still reach the embedding when `forward` is replaced by an external wrapper.
        import functools

        inner_forward = model.forward
        model.forward = functools.update_wrapper(lambda *a, **k: inner_forward(*a, **k), inner_forward)
        ids = build_flux_style_ids(num_txt_tokens=8, patch_grid=8)
        model(torch.randn(8, 4), img_ids=ids[8:], txt_ids=ids[:8], timestep=torch.tensor([0.4]))
        assert model.pos_embed.current_timestep == pytest.approx(0.4)

        registry = HookRegistry.check_if_exists_or_initialize(model)
        registry.remove_hook("dype_hook")
        assert len(model._forward_pre_hooks) == 0  # torn down on removal

    def test_apply_dype_validation(self):
        with pytest.raises(ValueError, match='must be "yarn"'):
            apply_dype(DummyFluxLikeTransformer(), method="ntk")

        class NoPosEmbedModel(torch.nn.Module):
            def forward(self, hidden_states):
                return hidden_states

        with pytest.raises(ValueError, match="`pos_embed`"):
            apply_dype(NoPosEmbedModel())
