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
"""Source-anchored masked-flow blending for training-free localized editing.

Adapted from "SAM-Flow: Source-Anchored Masked Flow for Training-Free Image
Editing" (https://arxiv.org/abs/2606.06228). The paper observes that mask-based
flow-matching edits leak into the background because the binary blend
``(1 - mask) * source + mask * edit`` produces hard boundaries and treats every
denoising step identically. SAM-Flow instead applies differential velocity
updates only inside the editable region while anchoring the rest of the latent
to the source-image trajectory, using a *time-varying* projection with dynamic
soft masks, transition regions, and temporal mask accumulation for spatial
stability and natural boundaries.

The functions here implement that projection as a drop-in replacement for the
binary blend in the flow-matching inpaint/edit loop. They operate purely on
tensors and are backbone-agnostic (FLUX packed latents, SD3 spatial latents),
so no fine-tuning or extra checkpoints are required.
"""

import torch


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    """Hermite ``3x^2 - 2x^3`` smoothstep, monotonic on ``[0, 1]``."""
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def dynamic_soft_mask(mask: torch.Tensor, transition_width: float = 0.0) -> torch.Tensor:
    """Widen the boundary of ``mask`` into a graded transition region.

    The hard ``{0, 1}`` boundary of an inpaint mask makes edits and the anchored
    background meet abruptly. Mapping the mask through a smoothstep centered on
    ``0.5`` turns partially-covered values into a soft band whose half-width is
    controlled by ``transition_width`` (a fraction of the mask range). With
    ``transition_width <= 0`` the mask is returned unchanged, which keeps the
    original binary behavior bit-for-bit.

    Args:
        mask (`torch.Tensor`): Edit mask, values in ``[0, 1]``, any shape.
        transition_width (`float`): Half-width of the soft transition band in
            ``[0, 1]``. ``0`` disables softening.

    Returns:
        `torch.Tensor`: Soft mask with the same shape as ``mask``.
    """
    if transition_width <= 0.0:
        return mask
    half = float(min(max(transition_width, 0.0), 1.0)) + 1e-6
    # Map [0.5 - half, 0.5 + half] -> [0, 1] and smoothstep the band.
    normalized = (mask - (0.5 - half)) / (2.0 * half)
    return _smoothstep(normalized).to(mask.dtype)


def accumulate_mask(current: torch.Tensor, accumulated: torch.Tensor | None) -> torch.Tensor:
    """Temporal mask accumulation: monotonically grow the editable region.

    SAM-Flow accumulates the per-step masks so that a location, once marked
    editable, stays editable for the remainder of the trajectory. This prevents
    the editable region from flickering between steps, which otherwise injects
    high-frequency artifacts. Implemented as a running element-wise maximum.

    Args:
        current (`torch.Tensor`): This step's (soft) mask.
        accumulated (`torch.Tensor` or `None`): The running mask, or ``None`` on
            the first step.

    Returns:
        `torch.Tensor`: The updated running mask.
    """
    if accumulated is None:
        return current
    return torch.maximum(accumulated, current)


def anchor_weight(step_index: int, num_steps: int, schedule: str = "constant") -> float:
    """Time-varying edit strength for the source-anchored projection.

    Returns a scalar in ``[0, 1]`` multiplied into the edit mask so that the
    balance between *editing* and *anchoring to the source trajectory* can vary
    over the diffusion time. Late steps shape fine structure and boundaries, so
    schedules that taper the edit strength toward the end (``"linear_decay"``,
    ``"cosine"``) increase background preservation and boundary naturalness,
    matching the paper's time-varying projection. ``"constant"`` reproduces the
    original uniform blend.

    Args:
        step_index (`int`): Current step, ``0``-based.
        num_steps (`int`): Total number of denoising steps.
        schedule (`str`): One of ``"constant"``, ``"linear_decay"``, ``"cosine"``.

    Returns:
        `float`: Edit-strength multiplier in ``[0, 1]``.
    """
    if schedule == "constant" or num_steps <= 1:
        return 1.0
    progress = min(max(step_index / (num_steps - 1), 0.0), 1.0)
    if schedule == "linear_decay":
        return 1.0 - progress
    if schedule == "cosine":
        # Smoothly taper from 1 -> 0 following the first quarter of a cosine.
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    raise ValueError(f"Unknown anchor schedule '{schedule}'. Expected one of 'constant', 'linear_decay', 'cosine'.")


def build_source_anchored_mask(
    mask: torch.Tensor,
    step_index: int,
    num_steps: int,
    *,
    transition_width: float = 0.0,
    temporal_accumulation: bool = False,
    accumulated_mask: torch.Tensor | None = None,
    anchor_schedule: str = "constant",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compose the time-varying soft blend mask for one denoising step.

    Combines the three SAM-Flow ingredients — dynamic soft masks
    ([`dynamic_soft_mask`]), temporal accumulation ([`accumulate_mask`]) and a
    time-varying anchor weight ([`anchor_weight`]) — into a single blend mask to
    hand to [`source_anchored_blend`].

    With the default arguments (``transition_width=0``, ``temporal_accumulation
    =False``, ``anchor_schedule="constant"``) the returned blend mask equals
    ``mask`` exactly, so callers can opt in without changing baseline behavior.

    Args:
        mask (`torch.Tensor`): Edit mask, ``1`` where the edit may apply.
        step_index (`int`): Current denoising step, ``0``-based.
        num_steps (`int`): Total number of denoising steps.
        transition_width (`float`): Soft-boundary half-width, see
            [`dynamic_soft_mask`].
        temporal_accumulation (`bool`): Whether to grow the mask over time.
        accumulated_mask (`torch.Tensor` or `None`): Running mask state to thread
            across steps when ``temporal_accumulation`` is enabled.
        anchor_schedule (`str`): Edit-strength schedule, see [`anchor_weight`].

    Returns:
        `tuple[torch.Tensor, torch.Tensor | None]`: ``(blend_mask, new_state)``
        where ``new_state`` should be passed back as ``accumulated_mask`` on the
        next step.
    """
    soft = dynamic_soft_mask(mask, transition_width)

    new_state = accumulated_mask
    if temporal_accumulation:
        new_state = accumulate_mask(soft, accumulated_mask)
        soft = new_state

    weight = anchor_weight(step_index, num_steps, anchor_schedule)
    blend_mask = soft * weight
    return blend_mask, new_state


def source_anchored_blend(
    edit_latents: torch.Tensor,
    source_latents: torch.Tensor,
    blend_mask: torch.Tensor,
) -> torch.Tensor:
    """Project the step's edit back onto the source trajectory outside the mask.

    Equivalent to ``(1 - blend_mask) * source_latents + blend_mask * edit_latents``:
    the masked region receives the differential (edited) velocity update while
    everything else is anchored to the source-image latent trajectory. This is
    the masked-flow update of SAM-Flow.

    Args:
        edit_latents (`torch.Tensor`): Latents after the scheduler step.
        source_latents (`torch.Tensor`): Source-image latents re-noised to the
            same timestep (the anchor trajectory).
        blend_mask (`torch.Tensor`): Soft blend mask from
            [`build_source_anchored_mask`].

    Returns:
        `torch.Tensor`: Blended latents.
    """
    return (1 - blend_mask) * source_latents + blend_mask * edit_latents
