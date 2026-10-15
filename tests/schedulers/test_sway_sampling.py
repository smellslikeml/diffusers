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

import numpy as np
import pytest
import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.sway_sampling import (
    DEFAULT_SWAY_COEF,
    SWAY_COEF_MAX,
    SWAY_COEF_MIN,
    apply_sway_sampling,
    sway_sampling,
)


def test_sway_sampling_formula():
    u = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    # s = 0 must be the identity mapping.
    np.testing.assert_allclose(sway_sampling(u, 0.0), u, atol=1e-7)
    # Endpoints are fixed for every admissible coefficient.
    endpoints = np.array([0.0, 1.0], dtype=np.float32)
    for s in (SWAY_COEF_MIN, -0.5, 0.5, SWAY_COEF_MAX):
        out = sway_sampling(endpoints, s)
        assert out[0] == 0.0
        assert out[-1] == 1.0
    # The map is monotone increasing on the admissible interval.
    fine = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    mapped = sway_sampling(fine, DEFAULT_SWAY_COEF)
    assert np.all(np.diff(mapped) > 0)


def test_sway_sampling_rejects_non_monotone_coefficients():
    u = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    with pytest.raises(ValueError):
        sway_sampling(u, SWAY_COEF_MIN - 0.5)
    with pytest.raises(ValueError):
        sway_sampling(u, SWAY_COEF_MAX + 0.5)


def test_sway_sampling_works_on_torch_tensors():
    u = torch.linspace(0.0, 1.0, 9)
    out = sway_sampling(u, DEFAULT_SWAY_COEF)
    assert isinstance(out, torch.Tensor)
    assert out.shape == u.shape
    np_out = sway_sampling(u.numpy(), DEFAULT_SWAY_COEF)
    np.testing.assert_allclose(out.numpy(), np_out, atol=1e-6)


def _make_scheduler(num_inference_steps=8, shift=1.0):
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    scheduler.set_timesteps(num_inference_steps)
    return scheduler


def test_apply_sway_sampling_preserves_schedule_invariants():
    scheduler = _make_scheduler(num_inference_steps=8)
    orig_sigmas = scheduler.sigmas.clone()
    orig_timesteps = scheduler.timesteps.clone()

    apply_sway_sampling(scheduler, sway_coef=DEFAULT_SWAY_COEF)

    # Lengths are unchanged: N timesteps, N+1 sigmas (terminal 0 included).
    assert scheduler.sigmas.shape[0] == orig_sigmas.shape[0]
    assert scheduler.timesteps.shape[0] == orig_timesteps.shape[0]
    # The trajectory endpoints (start noise level, clean 0) are preserved.
    assert torch.allclose(scheduler.sigmas[0], orig_sigmas[0])
    assert torch.allclose(scheduler.sigmas[-1], orig_sigmas[-1])
    # Sigmas stay strictly decreasing along the interior.
    assert bool(torch.all(scheduler.sigmas[:-1] > scheduler.sigmas[1:]))
    # Timesteps remain consistent with the resampled sigmas.
    assert torch.allclose(scheduler.timesteps, scheduler.sigmas[:-1] * scheduler.config.num_train_timesteps)


def test_apply_sway_sampling_zero_coefficient_is_identity():
    scheduler = _make_scheduler(num_inference_steps=8)
    before = scheduler.sigmas.clone()
    apply_sway_sampling(scheduler, sway_coef=0.0)
    # s = 0 maps every flow step to itself, so the schedule is recovered exactly.
    assert torch.allclose(scheduler.sigmas, before, atol=1e-6)


def test_apply_sway_sampling_default_biases_steps_toward_start():
    # With s = -1, flow steps are pulled toward the start of the trajectory
    # (high noise level), so the midpoint step lands at a higher sigma than it
    # does on the uniform schedule.
    scheduler = _make_scheduler(num_inference_steps=8)
    uniform_midpoint_sigma = scheduler.sigmas[4].clone()
    apply_sway_sampling(scheduler, sway_coef=DEFAULT_SWAY_COEF)
    assert bool(scheduler.sigmas[4] > uniform_midpoint_sigma)
