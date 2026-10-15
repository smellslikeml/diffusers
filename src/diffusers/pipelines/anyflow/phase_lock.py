# Copyright 2026 The HuggingFace Team. All rights reserved.
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
#
# Training-free motion-phase locking for image-to-video diffusion, adapted from
# "Physics in 2-Steps: Locking Motion Priors Before Visual Refinement Erases Them"
# (PhaseLock, https://arxiv.org/abs/2606.06361). The paper observes that the *phase*
# of the latent spectrum carries the physically-consistent motion prior and erodes
# (~18%) over a long denoising trajectory, while the *magnitude* (visual fidelity)
# stays stable. PhaseLock captures the phase from an early, few-step latent and
# re-imposes it on later high-fidelity latents, preserving motion without retraining.

from typing import Optional, Tuple

import torch


def _spectral_dims(ndim: int) -> Tuple[int, ...]:
    """Spectral axes that carry motion: temporal + spatial, never batch or channel.

    AnyFlow runs the denoising loop in the ``(B, T, C, H, W)`` layout, so the motion
    axes are ``(T, H, W)`` -> ``(-4, -2, -1)``. Plain ``(B, C, H, W)`` image latents
    fall back to the spatial axes.
    """
    if ndim >= 5:
        return (-4, -2, -1)
    if ndim == 4:
        return (-2, -1)
    return tuple(range(1, ndim))


def extract_motion_phase(latents: torch.Tensor) -> torch.Tensor:
    """Return the phase angle of the latent spectrum over the motion axes.

    This is the "motion prior" PhaseLock locks onto: the argument of the complex FFT,
    which encodes structure/motion independently of amplitude (visual fidelity).
    """
    dims = _spectral_dims(latents.ndim)
    spectrum = torch.fft.fftn(latents.float(), dim=dims)
    return torch.angle(spectrum)


def apply_phase_lock(
    latents: torch.Tensor,
    prior_phase: torch.Tensor,
    lock_strength: float,
    freq_cutoff_ratio: Optional[float] = None,
) -> torch.Tensor:
    """Re-impose ``prior_phase`` onto ``latents`` while keeping the current magnitude.

    Implements PhaseLock's Latent Delta Guidance: the current latent's magnitude
    spectrum (visual fidelity) is preserved exactly, and only the phase (motion) is
    nudged along the shortest arc toward the early-step prior by ``lock_strength``.

    Args:
        latents (`torch.Tensor`): Current high-fidelity latent to refine in place.
        prior_phase (`torch.Tensor`): Phase captured from the few-step motion prior.
        lock_strength (`float`): Fraction of the phase delta to apply, in ``[0, 1]``.
            ``0.0`` is an exact no-op; ``1.0`` snaps the phase onto the prior.
        freq_cutoff_ratio (`float`, *optional*): When set in ``(0, 1]``, only the
            lowest fraction of spectral coefficients (where physical motion lives) is
            locked, leaving high-frequency appearance detail untouched.

    Returns:
        `torch.Tensor`: The phase-locked latent, cast back to the input dtype.
    """
    if lock_strength <= 0.0:
        return latents

    dims = _spectral_dims(latents.ndim)
    spectrum = torch.fft.fftn(latents.float(), dim=dims)
    magnitude = torch.abs(spectrum)
    current_phase = torch.angle(spectrum)

    # Shortest-arc delta toward the prior phase, robust to 2*pi wrap-around.
    delta = torch.angle(torch.exp(1j * (prior_phase - current_phase)))

    weight = lock_strength
    if freq_cutoff_ratio is not None:
        weight = lock_strength * _low_pass_mask(spectrum.shape, dims, freq_cutoff_ratio, latents.device)

    locked_phase = current_phase + weight * delta
    locked = magnitude * torch.exp(1j * locked_phase)
    refined = torch.fft.ifftn(locked, dim=dims).real
    return refined.to(latents.dtype)


def _low_pass_mask(
    shape: torch.Size,
    dims: Tuple[int, ...],
    cutoff_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Build a broadcastable low-pass mask that is 1 on low frequencies, 0 elsewhere."""
    radius_sq = torch.zeros(1, device=device)
    view_shape = [1] * len(shape)
    for d in dims:
        n = shape[d]
        freq = torch.fft.fftfreq(n, device=device)  # in [-0.5, 0.5)
        axis_shape = list(view_shape)
        axis_shape[d] = n
        radius_sq = radius_sq + (freq.view(axis_shape) * 2.0) ** 2  # normalize to [-1, 1]
    radius = torch.sqrt(radius_sq)
    return (radius <= cutoff_ratio).to(radius.dtype)


class PhaseLockGuidance:
    """Stateful, training-free PhaseLock guidance for an I2V denoising loop.

    Captures the motion-prior phase from an early ("few-step") latent, then re-imposes
    it on every subsequent high-fidelity latent. Designed to be invoked once per
    denoising step from a pipeline's per-step latent hook — exactly the I/O exposed by
    `~AnyFlowPipeline`'s ``callback_on_step_end`` contract.

    Args:
        prior_step (`int`, defaults to `1`): Zero-based step index at which to capture
            the motion prior. The paper's "2-step" prior corresponds to capturing after
            two scheduler updates.
        lock_strength (`float`, defaults to `0.5`): Fraction of the phase delta applied
            per step. ``0.0`` disables the effect.
        freq_cutoff_ratio (`float`, *optional*): Lock only the lowest fraction of
            spectral coefficients (motion), preserving high-frequency appearance.
        cutoff_step_ratio (`float`, defaults to `1.0`): Stop applying the lock after
            this fraction of the trajectory, letting the final steps refine freely.
    """

    def __init__(
        self,
        prior_step: int = 1,
        lock_strength: float = 0.5,
        freq_cutoff_ratio: Optional[float] = None,
        cutoff_step_ratio: float = 1.0,
    ):
        if not 0.0 <= lock_strength <= 1.0:
            raise ValueError(f"`lock_strength` must be in [0, 1], got {lock_strength}.")
        if freq_cutoff_ratio is not None and not 0.0 < freq_cutoff_ratio <= 1.0:
            raise ValueError(f"`freq_cutoff_ratio` must be in (0, 1], got {freq_cutoff_ratio}.")
        self.prior_step = prior_step
        self.lock_strength = lock_strength
        self.freq_cutoff_ratio = freq_cutoff_ratio
        self.cutoff_step_ratio = cutoff_step_ratio
        self._prior_phase: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Drop the captured prior so the guidance can be reused across generations."""
        self._prior_phase = None

    @property
    def has_prior(self) -> bool:
        return self._prior_phase is not None

    def __call__(self, latents: torch.Tensor, step_index: int, num_inference_steps: int) -> torch.Tensor:
        """Capture the prior or lock the latent's phase for ``step_index``.

        Returns ``latents`` unchanged until the prior has been captured, and during any
        steps past ``cutoff_step_ratio``; otherwise returns the phase-locked latent.
        """
        if step_index < self.prior_step:
            return latents

        if self._prior_phase is None:
            # The early, physically-consistent latent: snapshot its phase as the prior.
            self._prior_phase = extract_motion_phase(latents)
            return latents

        if step_index >= self.cutoff_step_ratio * num_inference_steps:
            return latents

        return apply_phase_lock(
            latents,
            self._prior_phase,
            self.lock_strength,
            freq_cutoff_ratio=self.freq_cutoff_ratio,
        )
