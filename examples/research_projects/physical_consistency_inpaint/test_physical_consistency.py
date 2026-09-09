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
import os
import sys

import torch

# Existing (non-new) module: the diffusers VAE the losses integrate with.
from diffusers import AutoencoderKL


sys.path.insert(0, os.path.dirname(__file__))

from physical_consistency import (  # noqa: E402
    PhysicalConsistencyLoss,
    cielab_chromaticity_loss,
    reconstruction_loss_with_physical_consistency,
    rgb_to_lab,
    sobel_gradient_loss,
)


def _tiny_vae() -> AutoencoderKL:
    return AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        latent_channels=4,
    )


def test_rgb_to_lab_matches_reference_red():
    # Pure sRGB red converts to CIELAB ~ (53.24, 80.09, 67.20).
    red = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)
    lab = rgb_to_lab(red).view(3)
    assert torch.allclose(lab, torch.tensor([53.2408, 80.0925, 67.2032]), atol=1e-2)


def test_identical_images_have_zero_loss():
    image = torch.rand(2, 3, 32, 32)
    assert cielab_chromaticity_loss(image, image).item() < 1e-4
    assert sobel_gradient_loss(image, image).item() < 1e-4


def test_hue_shift_raises_chromaticity_more_than_luminance_shift():
    torch.manual_seed(0)
    target = torch.rand(1, 3, 32, 32)
    hue_shifted = target.clone()
    hue_shifted[:, 0] = (hue_shifted[:, 0] + 0.4).clamp(0.0, 1.0)  # push the red channel
    darkened = (target * 0.7).clamp(0.0, 1.0)  # lightness-only change
    # Chromaticity term reacts to the hue shift and mostly ignores darkening.
    assert cielab_chromaticity_loss(hue_shifted, target) > cielab_chromaticity_loss(darkened, target)


def test_edge_blur_raises_sobel_loss():
    torch.manual_seed(0)
    target = torch.rand(1, 3, 48, 48)
    # Blur destroys edge structure -> Sobel term should grow.
    kernel = torch.ones(3, 1, 5, 5) / 25.0
    blurred = torch.nn.functional.conv2d(
        torch.nn.functional.pad(target, (2, 2, 2, 2), mode="replicate"), kernel, groups=3
    )
    assert sobel_gradient_loss(blurred, target) > sobel_gradient_loss(target, target)


def test_mask_restricts_loss_to_region():
    torch.manual_seed(0)
    target = torch.rand(1, 3, 32, 32)
    prediction = target.clone()
    prediction[:, :, :16, :] = torch.rand(1, 3, 16, 32)  # corrupt the top half only
    top_mask = torch.zeros(1, 1, 32, 32)
    top_mask[:, :, :16, :] = 1.0
    bottom_mask = torch.zeros(1, 1, 32, 32)
    bottom_mask[:, :, 16:, :] = 1.0
    # The bottom is untouched, so a bottom mask sees ~no error; the top does.
    assert cielab_chromaticity_loss(prediction, target, top_mask) > 1e-3
    assert cielab_chromaticity_loss(prediction, target, bottom_mask) < 1e-4


def test_reconstruction_helper_integrates_with_diffusers_vae():
    torch.manual_seed(0)
    vae = _tiny_vae().eval()
    target_latents = torch.randn(1, 4, 16, 16)
    predicted_latents = (target_latents + 0.1 * torch.randn_like(target_latents)).requires_grad_(True)
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1.0

    losses = reconstruction_loss_with_physical_consistency(
        vae,
        predicted_latents,
        target_latents,
        mask=mask,
        physical_loss=PhysicalConsistencyLoss(lambda_cielab=1.0, lambda_sobel=0.5),
    )

    assert set(losses) == {"cielab", "sobel", "total"}
    for value in losses.values():
        assert torch.isfinite(value)
    # The wiring is differentiable end-to-end through the VAE decode.
    losses["total"].backward()
    assert predicted_latents.grad is not None
    assert torch.isfinite(predicted_latents.grad).all()
