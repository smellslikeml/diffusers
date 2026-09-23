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

# Import the new guider through the existing `diffusers.guiders` package (the call-site wiring), and pull the existing
# ClassifierFreeGuidance from the same package to assert the S-CFG guider is a faithful drop-in extension of it.
from diffusers.guiders import ClassifierFreeGuidance, SemanticAwareGuidance


def _cfg_pred(guider, pred_cond, pred_uncond):
    return guider.forward(pred_cond, pred_uncond).pred


def test_registered_alongside_classifier_free_guidance():
    # Both guiders resolve from the existing package and share the CFG contract.
    guider = SemanticAwareGuidance(guidance_scale=7.5)
    assert guider._input_predictions == ["pred_cond", "pred_uncond"]
    assert guider.num_conditions == 2


def test_reduces_to_scalar_cfg_when_guidance_is_spatially_uniform():
    # When the guidance term has a constant magnitude everywhere, S-CFG's scale map collapses to the scalar
    # guidance_scale, so its output must match the existing ClassifierFreeGuidance exactly.
    torch.manual_seed(0)
    pred_uncond = torch.randn(2, 4, 8, 8)
    diff = torch.full((2, 4, 8, 8), 0.5)
    pred_cond = pred_uncond + diff

    cfg = ClassifierFreeGuidance(guidance_scale=7.5)
    scfg = SemanticAwareGuidance(guidance_scale=7.5)

    torch.testing.assert_close(_cfg_pred(scfg, pred_cond, pred_uncond), _cfg_pred(cfg, pred_cond, pred_uncond))


def test_equalizes_semantic_strength_across_regions():
    # A strong region (large guidance magnitude) should receive a smaller effective scale than a weak region, so that
    # the per-region guidance strengths are pulled toward each other -- the core S-CFG behavior.
    pred_uncond = torch.zeros(1, 4, 8, 8)
    diff = torch.zeros(1, 4, 8, 8)
    diff[..., :4] = 2.0  # strong left half
    diff[..., 4:] = 0.5  # weak right half
    pred_cond = pred_uncond + diff

    guidance_scale = 6.0
    scfg = SemanticAwareGuidance(guidance_scale=guidance_scale, window_size=1, rescale_clamp=4.0)
    pred = _cfg_pred(scfg, pred_cond, pred_uncond)

    # Recover the applied scale per region: pred = pred_uncond + scale * diff -> scale = (pred - pred_uncond) / diff.
    strong_scale = (pred[..., :4] / diff[..., :4]).mean().item()
    weak_scale = (pred[..., 4:] / diff[..., 4:]).mean().item()

    assert strong_scale < guidance_scale < weak_scale
    # Effective guidance strengths (scale * magnitude) are closer than the raw magnitudes were.
    assert strong_scale * 2.0 - weak_scale * 0.5 < 2.0 - 0.5


def test_falls_back_to_scalar_cfg_for_non_spatial_predictions():
    # Sequence-shaped (3D) predictions have no spatial grid, so S-CFG must behave exactly like scalar CFG.
    pred_uncond = torch.randn(2, 16, 32)
    pred_cond = pred_uncond + torch.randn(2, 16, 32)

    cfg = ClassifierFreeGuidance(guidance_scale=5.0)
    scfg = SemanticAwareGuidance(guidance_scale=5.0)

    torch.testing.assert_close(_cfg_pred(scfg, pred_cond, pred_uncond), _cfg_pred(cfg, pred_cond, pred_uncond))
