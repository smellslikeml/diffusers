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
"""Physical-consistency losses for diffusion inpainting.

Adapted from "Structured-Prior-Guided Diffusion Inpainting with Physical
Consistency for Traffic Sign Augmentation" (https://arxiv.org/abs/2609.02348).

The paper observes that general-purpose inpainting models shift colours and
deform edge structure when applied to physically-composed objects (traffic
signs, plates, digits). It constrains the decoded prediction with two
parameter-free, differentiable pixel-space terms:

* a **CIELAB chromaticity L1** term that penalises colour drift, measured in a
  perceptually-uniform space so equal numeric errors are roughly equal
  perceived errors, and
* a **Sobel gradient** term that penalises edge/structure drift.

Only these two loss terms are ported here (the paper's full method also injects
semantic/appearance/geometric priors through a JSON prompt, an IP-Adapter
template and a ControlNet template — those are orthogonal conditioning
pathways, out of scope for this module). The terms operate on decoded RGB
images and plug into any existing diffusers reconstruction/inpainting training
loop after the VAE decode, optionally restricted to the inpainted region via a
mask.
"""

import torch
import torch.nn.functional as F


# sRGB <-> linear RGB <-> CIE XYZ (D65) <-> CIELAB constants.
_SRGB_THRESHOLD = 0.04045
_LAB_DELTA = 6.0 / 29.0
_D65_WHITE = (0.95047, 1.0, 1.08883)
_EPS = 1e-8


def _srgb_to_linear(channel: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB gamma expansion for a single channel in [0, 1]."""
    channel = channel.clamp(0.0, 1.0)
    low = channel / 12.92
    high = ((channel + 0.055) / 1.055).clamp(min=_EPS) ** 2.4
    return torch.where(channel <= _SRGB_THRESHOLD, low, high)


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """Convert an sRGB image in [0, 1] to CIELAB.

    Args:
        image: Tensor of shape ``(N, 3, H, W)`` with values in ``[0, 1]``.

    Returns:
        Tensor of shape ``(N, 3, H, W)`` holding the ``L*``, ``a*`` and ``b*``
        channels. The conversion is fully differentiable, so it can sit inside
        a training graph.
    """
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3, H, W) RGB tensor, got shape {tuple(image.shape)}.")

    linear = _srgb_to_linear(image)
    r, g, b = linear[:, 0], linear[:, 1], linear[:, 2]

    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    x = x / _D65_WHITE[0]
    y = y / _D65_WHITE[1]
    z = z / _D65_WHITE[2]

    def _f(t: torch.Tensor) -> torch.Tensor:
        cube_root = t.clamp(min=_EPS) ** (1.0 / 3.0)
        linear_part = t / (3.0 * _LAB_DELTA * _LAB_DELTA) + 4.0 / 29.0
        return torch.where(t > _LAB_DELTA**3, cube_root, linear_part)

    fx, fy, fz = _f(x), _f(y), _f(z)
    lightness = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return torch.stack([lightness, a, b_star], dim=1)


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """L1 distance, optionally averaged over a region mask.

    ``mask`` is broadcast over the channel dimension; a ``None`` mask reduces
    over the whole image.
    """
    diff = (prediction - target).abs()
    if mask is None:
        return diff.mean()
    channels = prediction.shape[1]
    denominator = mask.sum() * channels + _EPS
    return (diff * mask).sum() / denominator


def cielab_chromaticity_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """CIELAB chromaticity (``a*``, ``b*``) L1 loss between two sRGB images.

    Only the two chromaticity channels are compared, so the term penalises hue
    and saturation drift while staying indifferent to lightness changes the
    diffusion prior may legitimately introduce.

    Args:
        prediction: Predicted sRGB image ``(N, 3, H, W)`` in ``[0, 1]``.
        target: Reference sRGB image ``(N, 3, H, W)`` in ``[0, 1]``.
        mask: Optional ``(N, 1, H, W)`` region mask (e.g. the inpainted area).

    Returns:
        Scalar loss tensor.
    """
    pred_ab = rgb_to_lab(prediction)[:, 1:]
    target_ab = rgb_to_lab(target)[:, 1:]
    return _masked_l1(pred_ab, target_ab, mask)


def _sobel_gradients(image: torch.Tensor) -> torch.Tensor:
    """Return stacked horizontal/vertical Sobel responses per channel."""
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    )
    kernel_y = kernel_x.t()
    channels = image.shape[1]
    weight = torch.stack([kernel_x, kernel_y]).unsqueeze(1)  # (2, 1, 3, 3)
    weight = weight.repeat(channels, 1, 1, 1)  # (2 * C, 1, 3, 3)
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, weight, groups=channels)


def sobel_gradient_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """Sobel edge-structure L1 loss between two sRGB images.

    Gradient magnitude is compared per channel, which keeps digit strokes and
    geometric outlines aligned between the prediction and the reference.

    Args:
        prediction: Predicted sRGB image ``(N, 3, H, W)`` in ``[0, 1]``.
        target: Reference sRGB image ``(N, 3, H, W)`` in ``[0, 1]``.
        mask: Optional ``(N, 1, H, W)`` region mask (e.g. the inpainted area).

    Returns:
        Scalar loss tensor.
    """
    pred_grad = _sobel_gradients(prediction)
    target_grad = _sobel_gradients(target)
    pred_mag = torch.sqrt(pred_grad[:, 0::2] ** 2 + pred_grad[:, 1::2] ** 2 + _EPS)
    target_mag = torch.sqrt(target_grad[:, 0::2] ** 2 + target_grad[:, 1::2] ** 2 + _EPS)
    # A single-channel mask broadcasts over the per-channel magnitudes; _masked_l1
    # already normalises by the channel count.
    return _masked_l1(pred_mag, target_mag, mask)


class PhysicalConsistencyLoss(torch.nn.Module):
    """Weighted sum of the CIELAB chromaticity and Sobel gradient terms.

    Args:
        lambda_cielab: Weight of the CIELAB chromaticity term.
        lambda_sobel: Weight of the Sobel gradient term.
    """

    def __init__(self, lambda_cielab: float = 1.0, lambda_sobel: float = 1.0):
        super().__init__()
        self.lambda_cielab = lambda_cielab
        self.lambda_sobel = lambda_sobel

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
    ) -> dict:
        """Compute the individual and combined physical-consistency terms.

        Returns a dict with ``cielab``, ``sobel`` and ``total`` scalar tensors
        so a training loop can log each term while backpropagating ``total``.
        """
        cielab = cielab_chromaticity_loss(prediction, target, mask)
        sobel = sobel_gradient_loss(prediction, target, mask)
        total = self.lambda_cielab * cielab + self.lambda_sobel * sobel
        return {"cielab": cielab, "sobel": sobel, "total": total}


def _decode_to_images(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode diffusion latents into sRGB images in ``[0, 1]`` via the VAE."""
    images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return (images / 2.0 + 0.5).clamp(0.0, 1.0)


def reconstruction_loss_with_physical_consistency(
    vae,
    predicted_latents: torch.Tensor,
    target_latents: torch.Tensor,
    mask: torch.Tensor = None,
    physical_loss: PhysicalConsistencyLoss = None,
) -> dict:
    """Decode a prediction and reference and score physical consistency.

    This is the call site that stitches the losses above onto an existing
    diffusers inpainting loop: pass the model's predicted ``x0`` latents and
    the ground-truth latents together with any diffusers ``AutoencoderKL``
    (SD1.5-inpaint, ControlNet-inpaint, ... all share this VAE interface). The
    latents are decoded to pixel space and both terms are computed there, since
    colour and edge structure are physical quantities that only exist after
    decoding.

    Args:
        vae: A diffusers ``AutoencoderKL`` (or any VAE exposing ``decode`` and a
            ``config.scaling_factor``).
        predicted_latents: Predicted ``(N, C, h, w)`` latents.
        target_latents: Reference ``(N, C, h, w)`` latents.
        mask: Optional latent-resolution ``(N, 1, h, w)`` mask; it is upsampled
            to the decoded image resolution and used to restrict both terms to
            the inpainted region.
        physical_loss: Optional pre-configured :class:`PhysicalConsistencyLoss`;
            a default (equal weights) instance is created when omitted.

    Returns:
        The dict returned by :class:`PhysicalConsistencyLoss`.
    """
    if physical_loss is None:
        physical_loss = PhysicalConsistencyLoss()

    predicted_images = _decode_to_images(vae, predicted_latents)
    target_images = _decode_to_images(vae, target_latents)

    if mask is not None:
        mask = F.interpolate(mask, size=predicted_images.shape[-2:], mode="nearest")

    return physical_loss(predicted_images, target_images, mask)
