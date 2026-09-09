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
"""Minimal, self-contained demo of the physical-consistency losses.

It composes a small diffusers ``AutoencoderKL`` with
``reconstruction_loss_with_physical_consistency`` to show exactly where the
losses plug into an inpainting training step — no checkpoints or datasets are
downloaded, so it doubles as a smoke test:

    python example_physical_consistency.py

In a real SD1.5-inpaint / ControlNet-inpaint loop you would swap the tiny VAE
below for the pipeline's ``vae``, feed the model's predicted ``x0`` latents as
``predicted_latents``, the ground-truth latents as ``target_latents``, and add
``losses["total"]`` to the usual latent MSE before ``backward()``.
"""

import torch
from physical_consistency import PhysicalConsistencyLoss, reconstruction_loss_with_physical_consistency

from diffusers import AutoencoderKL


def build_tiny_vae() -> AutoencoderKL:
    """A CPU-sized VAE mirroring the config used across the diffusers test suite."""
    return AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        latent_channels=4,
    )


def main() -> None:
    torch.manual_seed(0)
    vae = build_tiny_vae().eval()

    # Stand in for a training batch: ground-truth latents plus a noisier
    # prediction and a mask marking the inpainted region.
    target_latents = torch.randn(1, 4, 16, 16)
    predicted_latents = target_latents + 0.1 * torch.randn_like(target_latents)
    predicted_latents.requires_grad_(True)
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1.0

    physical_loss = PhysicalConsistencyLoss(lambda_cielab=1.0, lambda_sobel=0.5)
    losses = reconstruction_loss_with_physical_consistency(
        vae, predicted_latents, target_latents, mask=mask, physical_loss=physical_loss
    )

    losses["total"].backward()
    print(f"cielab: {losses['cielab'].item():.4f}")
    print(f"sobel:  {losses['sobel'].item():.4f}")
    print(f"total:  {losses['total'].item():.4f}")
    print(f"grad flows back to latents: {predicted_latents.grad is not None}")


if __name__ == "__main__":
    main()
