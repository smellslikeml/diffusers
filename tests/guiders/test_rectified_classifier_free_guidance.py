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

# Import through the public API (exercises the guider registration wiring) and pull the
# existing CFG guider (a non-new module) to check ReCFG's reduction-to-CFG property.
from diffusers import ClassifierFreeGuidance, RectifiedClassifierFreeGuidance


def _norm(x):
    return torch.linalg.vector_norm(x.flatten(1), dim=1)


def test_reduces_to_cfg_when_norms_match():
    # When the conditional and unconditional predictions have equal per-sample norm, the
    # rectification coefficient rho == 1, so ReCFG must match standard CFG exactly.
    torch.manual_seed(0)
    pred_cond = torch.randn(2, 3, 8, 8)
    pred_uncond = torch.randn(2, 3, 8, 8)
    # Rescale uncond so each sample matches the norm of the corresponding cond sample.
    pred_uncond = pred_uncond * (_norm(pred_cond) / _norm(pred_uncond)).view(-1, 1, 1, 1)

    cfg = ClassifierFreeGuidance(guidance_scale=5.0)
    recfg = RectifiedClassifierFreeGuidance(guidance_scale=5.0)

    cfg_out = cfg(cfg.prepare_inputs({"noise_pred": (pred_cond, pred_uncond)}))
    recfg_out = recfg(recfg.prepare_inputs({"noise_pred": (pred_cond, pred_uncond)}))

    assert torch.allclose(recfg_out.pred, cfg_out.pred, atol=1e-5)


def test_rectification_matches_closed_form():
    # ReCFG rescales the unconditional term by rho = ||pred_cond|| / ||pred_uncond|| and then
    # applies ordinary CFG. Assert the output equals that closed form and differs from plain CFG.
    torch.manual_seed(1)
    pred_cond = torch.randn(2, 4)
    pred_uncond = 0.25 * torch.randn(2, 4)  # deliberately smaller norm -> rho != 1

    guidance_scale = 6.0
    recfg = RectifiedClassifierFreeGuidance(guidance_scale=guidance_scale)
    out = recfg(recfg.prepare_inputs({"noise_pred": (pred_cond, pred_uncond)}))

    rho = (_norm(pred_cond) / (_norm(pred_uncond) + 1e-8)).view(-1, 1)
    uncond_rect = pred_uncond * rho
    expected = uncond_rect + guidance_scale * (pred_cond - uncond_rect)

    assert torch.allclose(out.pred, expected, atol=1e-5)

    cfg = ClassifierFreeGuidance(guidance_scale=guidance_scale)
    cfg_out = cfg(cfg.prepare_inputs({"noise_pred": (pred_cond, pred_uncond)}))
    assert not torch.allclose(out.pred, cfg_out.pred, atol=1e-3)


def test_rectification_scale_zero_recovers_cfg():
    # rectification_scale=0.0 forces rho -> 1, recovering standard CFG even when norms differ.
    torch.manual_seed(2)
    pred_cond = torch.randn(1, 3, 4, 4)
    pred_uncond = 2.0 * torch.randn(1, 3, 4, 4)

    cfg = ClassifierFreeGuidance(guidance_scale=4.0)
    recfg = RectifiedClassifierFreeGuidance(guidance_scale=4.0, rectification_scale=0.0)

    cfg_out = cfg(cfg.prepare_inputs({"noise_pred": (pred_cond, pred_uncond)}))
    recfg_out = recfg(recfg.prepare_inputs({"noise_pred": (pred_cond, pred_uncond)}))

    assert torch.allclose(recfg_out.pred, cfg_out.pred, atol=1e-5)
