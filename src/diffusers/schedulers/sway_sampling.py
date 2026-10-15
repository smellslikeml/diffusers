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

"""Inference-time *sway sampling* for flow-matching schedulers.

Implements the flow-step sampling strategy from F5-TTS (Chen et al., 2024), which
redistributes the uniformly spaced flow steps used at inference time via a
monotone map over the [0, 1] progress coordinate. Concentrating steps toward one
end of the denoising trajectory gives the ODE solver finer resolution where it
matters most, improving the quality/efficiency trade-off **without retraining** --
the paper notes the strategy "can be easily applied to existing flow matching
based models without retraining."

The mapping (F5-TTS, Sec. 3.2) is::

    f_sway(u; s) = u + s * (cos(pi/2 * u) - 1 + u) ,  u in [0, 1]

``f_sway(0) = 0`` and ``f_sway(1) = 1`` for every ``s``, and the map is monotone on
``s in [-1, 2 / (pi - 2)]``:

* ``s < 0``  -> steps are pulled toward the *start* of the trajectory (high noise
  level), giving the solver finer resolution on the early integration steps.
* ``s > 0``  -> steps are pulled toward the *end* of the trajectory.
* ``s = 0``  -> recovers the original (uniform) schedule.

F5-TTS uses ``s = -1`` by default; that is the default here too.

Reference: "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow
Matching", https://arxiv.org/abs/2410.06885
"""

import math

import numpy as np
import torch


# Monotonicity interval for the sway coefficient, as derived in the F5-TTS paper.
SWAY_COEF_MIN = -1.0
SWAY_COEF_MAX = 2.0 / (math.pi - 2.0)
# F5-TTS default (paper Sec. 5).
DEFAULT_SWAY_COEF = -1.0


def sway_sampling(u, sway_coef=DEFAULT_SWAY_COEF):
    """Map flow-step positions through F5-TTS *sway sampling*.

    Args:
        u (`np.ndarray` or `torch.Tensor`):
            Flow-step positions in `[0, 1]` -- typically a uniform grid spanning
            the inference steps. The endpoints `0` and `1` are fixed points.
        sway_coef (`float`, defaults to `-1.0`):
            Sway coefficient `s`. Must lie in `[-1, 2 / (pi - 2)]` so the map
            stays monotone. Negative values bias steps toward the start of the
            trajectory, positive values toward the end, and `0` leaves the grid
            unchanged.

    Returns:
        An array of the same type and shape as `u` holding the sway-mapped
        positions in `[0, 1]`.

    Raises:
        ValueError: if `sway_coef` falls outside its monotone interval.
    """
    if not (SWAY_COEF_MIN - 1e-9 <= sway_coef <= SWAY_COEF_MAX + 1e-9):
        raise ValueError(
            f"`sway_coef` must be in [{SWAY_COEF_MIN:.4f}, {SWAY_COEF_MAX:.4f}] to keep sway sampling monotone, "
            f"got {sway_coef}."
        )
    # f_sway(u; s) = u + s * (cos(pi/2 * u) - 1 + u). Dispatch on the backend so
    # the function works for both numpy arrays and torch tensors.
    if isinstance(u, torch.Tensor):
        half_pi = math.pi / 2.0
        return u + sway_coef * (torch.cos(half_pi * u) - 1.0 + u)
    return u + sway_coef * (np.cos(np.pi / 2.0 * u) - 1.0 + u)


def apply_sway_sampling(scheduler, sway_coef=DEFAULT_SWAY_COEF):
    """Re-space a flow-matching scheduler's schedule via sway sampling.

    Operates on any scheduler that exposes ``sigmas`` (a 1-D tensor of length
    ``num_inference_steps + 1`` decreasing from the starting noise level down to
    ``0``) and derives ``timesteps`` as ``sigmas * num_train_timesteps`` -- e.g.
    [`~diffusers.FlowMatchEulerDiscreteScheduler`]. The already-configured
    schedule (shift, Karras, ...) is left intact: sway sampling only re-spaces
    *where along the trajectory* each step lands, exactly as F5-TTS applies it at
    inference time.

    Call this immediately after ``scheduler.set_timesteps(...)`` and before the
    sampling loop. The change is in-place and requires no retraining.

    Args:
        scheduler:
            A configured flow-matching scheduler with `sigmas`/`timesteps` set.
        sway_coef (`float`, defaults to `-1.0`):
            Sway coefficient; see [`sway_sampling`].
    """
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        raise ValueError("Scheduler has no sigma schedule; call `set_timesteps` before applying sway sampling.")
    num_nodes = int(sigmas.shape[0])
    # With fewer than three nodes there is no interior point to re-space.
    if num_nodes < 3:
        return

    device, dtype = sigmas.device, sigmas.dtype
    sigmas_np = sigmas.detach().to("cpu", dtype=torch.float32).numpy()

    # Uniform progress coordinate over the schedule nodes (0 = start, 1 = clean).
    u = np.linspace(0.0, 1.0, num_nodes, dtype=np.float32)
    # Sway-map the progress coordinate and resample the original noise levels at
    # the mapped positions; `np.interp` keeps the endpoints fixed by construction.
    t = np.asarray(sway_sampling(u, sway_coef), dtype=np.float32)
    new_sigmas = np.interp(t, u, sigmas_np).astype(np.float32)

    new_sigmas_t = torch.from_numpy(new_sigmas).to(device=device, dtype=dtype)
    num_train_timesteps = scheduler.config.num_train_timesteps
    new_timesteps = new_sigmas_t[:-1] * num_train_timesteps

    scheduler.sigmas = new_sigmas_t
    scheduler.timesteps = new_timesteps.to(dtype=torch.float32)
    scheduler._step_index = None
    scheduler._begin_index = None
