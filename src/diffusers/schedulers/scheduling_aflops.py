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

# DISCLAIMER: check https://arxiv.org/abs/2509.00036 for more info. This is a clean-room implementation
# from the A-FloPS paper (CC-BY-4.0), following the structure of
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_unipc_multistep.py

import math
from typing import Literal

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import deprecate
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    max_beta: float = 0.999,
    alpha_transform_type: Literal["cosine", "exp", "laplace"] = "cosine",
) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.

    Args:
        num_diffusion_timesteps (`int`):
            The number of betas to produce.
        max_beta (`float`, defaults to `0.999`):
            The maximum beta to use; use values lower than 1 to avoid numerical instability.
        alpha_transform_type (`str`, defaults to `"cosine"`):
            The type of noise schedule for `alpha_bar`. Choose from `cosine`, `exp`, or `laplace`.

    Returns:
        `torch.Tensor`:
            The betas used by the scheduler to step the model outputs.
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "laplace":

        def alpha_bar_fn(t):
            lmb = -0.5 * math.copysign(1, 0.5 - t) * math.log(1 - 2 * math.fabs(0.5 - t) + 1e-6)
            snr = math.exp(lmb)
            return math.sqrt(snr / (1 + snr))

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class AFloPSScheduler(SchedulerMixin, ConfigMixin):
    """
    `AFloPSScheduler` is a training-free, plug-and-play sampler for fast diffusion sampling introduced in
    [A-FloPS: Adaptive Flow Path Sampler](https://arxiv.org/abs/2509.00036) (AAAI 2026).

    A VP noise schedule (`x_t = alpha_t * x_0 + sigma_t * epsilon`) is reparameterized into a flow-matching
    coordinate `z = x * scale_t`, whose flow time `floweq_t = alpha_t * scale_t` interpolates linearly between
    pure noise (`floweq = 0`) and clean data (`floweq = 1`). The inference timesteps are chosen uniformly in
    flow time instead of VP time. Each update decomposes the flow velocity adaptively into a per-sample linear
    drift and a residual, and integrates both with a second-order step built from the two most recently cached
    (sample, velocity) pairs. A first-order flow-Euler step is used to warm up, to finish, and whenever `mode=0`.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the
    generic methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to `1000`):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to `0.0001`):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to `0.02`):
            The final `beta` value.
        beta_schedule (`"linear"`, `"scaled_linear"`, or `"squaredcos_cap_v2"`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        solver_order (`int`, defaults to `2`):
            The order of the solver, which can be `1` (first-order flow-Euler) or `2` (the A-FloPS adaptive
            second-order update).
        mode (`int`, defaults to `1`):
            Whether to use the adaptive second-order update (`mode=1`) or the first-order flow-Euler update
            (`mode=0`) whenever two steps of history are available.
        c_max (`float`, defaults to `1.0`):
            The linear drift coefficient `c` is clamped to `[-c_max, c_max]` to keep the second-order update
            stable.
        prediction_type (`"epsilon"`, `"sample"`, or `"v_prediction"`, defaults to `"epsilon"`):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion
            process), `sample` (directly predicts the clean sample), or `v_prediction` (see section 2.4 of
            [Imagen Video](https://huggingface.co/papers/2210.02303)).
        final_sigmas_type (`"zero"` or `"sigma_min"`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the smallest sigma in the training schedule. If `zero`, the final sigma is set to
            0, so sampling stops on the clean-data end of the flow (`floweq = 1`).
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: np.ndarray | list[float] | None = None,
        solver_order: int = 2,
        mode: int = 1,
        c_max: float = 1.0,
        prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
        final_sigmas_type: Literal["zero", "sigma_min"] = "zero",
    ) -> None:
        if solver_order not in [1, 2]:
            raise NotImplementedError(f"solver_order {solver_order} is not implemented for {self.__class__.__name__}")
        if mode not in [0, 1]:
            raise NotImplementedError(f"mode {mode} is not implemented for {self.__class__.__name__}")

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Currently we only support VP-type noise schedule: x_t = alpha_t * x_0 + sigma_t * epsilon
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.samples = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        # training-schedule sigmas plus their flow-matching coordinates, kept on cpu to avoid too much
        # CPU/GPU communication; `set_timesteps` rederives them for the inference schedule
        self.sigmas = self.sigma_t
        self.scales = 1.0 / (self.alpha_t + self.sigma_t)
        self.floweqs = self.alpha_t * self.scales

    @property
    def step_index(self) -> int:
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self) -> int:
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`, defaults to `0`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        The timesteps are the training timesteps whose flow times are closest to a uniform grid
        `linspace(0, 1, num_inference_steps, endpoint=False)`, so the sampling path is uniform in flow time
        rather than in VP time.

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        # flow time of every training timestep; `floweq` decreases from ~1 (t=0, clean data) to ~0
        # (t=num_train_timesteps-1, pure noise), so matching the ascending grid yields a descending
        # timestep schedule as expected by `step`
        floweq_all = (self.alpha_t / (self.alpha_t + self.sigma_t)).numpy()
        flow_grid = np.linspace(0, 1, num_inference_steps, endpoint=False)
        step_indices = np.abs(floweq_all[:, None] - flow_grid[None, :]).argmin(axis=0)

        sigmas = self.sigma_t.numpy()[step_indices]
        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = float(self.sigma_t[0])
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0.0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(step_indices.astype(np.int64)).to(device=device, dtype=torch.int64)

        # flow-matching coordinate of the inference schedule; the appended final sigma is the update target
        # of the last step, where "zero" means the clean-data end of the flow (alpha = 1, scale = 1)
        alpha_ts = (1 - self.sigmas**2) ** 0.5
        self.scales = 1.0 / (alpha_ts + self.sigmas)
        self.floweqs = alpha_ts * self.scales

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.samples = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.scales = self.scales.to("cpu")
        self.floweqs = self.floweqs.to("cpu")

    def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert sigma values to the VP (alpha_t, sigma_t) pair used by A-FloPS, where a sample is parameterized as
        `x_t = alpha_t * x_0 + sigma_t * epsilon` (so `alpha_t**2 + sigma_t**2 = 1`).

        Args:
            sigma (`torch.Tensor`):
                The sigma value(s) to convert.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`:
                A tuple containing the (alpha_t, sigma_t) values.
        """
        alpha_t = (1 - sigma**2) ** 0.5
        sigma_t = sigma

        return alpha_t, sigma_t

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Convert the model output to the flow velocity `dz/d(floweq) = x_0 - epsilon` that the A-FloPS update
        integrates.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)

        if self.config.prediction_type == "epsilon":
            x0_pred = (sample - sigma_t * model_output) / alpha_t
            epsilon = model_output
        elif self.config.prediction_type == "sample":
            x0_pred = model_output
            epsilon = (sample - alpha_t * model_output) / sigma_t
        elif self.config.prediction_type == "v_prediction":
            x0_pred = alpha_t * sample - sigma_t * model_output
            epsilon = sigma_t * sample + alpha_t * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction` for the AFloPSScheduler."
            )

        return x0_pred - epsilon

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.index_for_timestep
    def index_for_timestep(
        self,
        timestep: int | torch.Tensor,
        schedule_timesteps: torch.Tensor | None = None,
    ) -> int:
        """
        Find the index for a given timestep in the schedule.

        Args:
            timestep (`int` or `torch.Tensor`):
                The timestep for which to find the index.
            schedule_timesteps (`torch.Tensor`, *optional*):
                The timestep schedule to search in. If `None`, uses `self.timesteps`.

        Returns:
            `int`:
                The index of the timestep in the schedule.
        """
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        index_candidates = (schedule_timesteps == timestep).nonzero()

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()

        return step_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index
    def _init_step_index(self, timestep: int | torch.Tensor) -> None:
        """
        Initialize the step_index counter for the scheduler.

        Args:
            timestep (`int` or `torch.Tensor`):
                The current timestep for which to initialize the step index.
        """

        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the adaptive flow path update of A-FloPS.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int` or `torch.Tensor`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        flow = self.convert_model_output(model_output, sample=sample)

        # roll the caches: [-1] holds the current step, [-2] the previous one. Unlike other multistep
        # schedulers, A-FloPS also caches the samples themselves in flow coordinates z = x * scale
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.samples[i] = self.samples[i + 1]
        self.model_outputs[-1] = flow
        self.samples[-1] = sample * self.scales[self.step_index]

        # warm up the multistep caches with first order, and drop to lower order on the final steps
        this_order = min(self.config.solver_order, self.lower_order_nums + 1, len(self.timesteps) - self.step_index)
        assert this_order > 0
        floweq_s0 = self.floweqs[self.step_index]
        delta = self.floweqs[self.step_index + 1] - floweq_s0

        if (
            this_order == 1
            or self.config.mode == 0
            # the second-order update needs two distinct flow times, so repeated timesteps fall back to
            # the first-order update (only reachable once a previous step has been taken)
            or floweq_s0 == self.floweqs[self.step_index - 1]
        ):
            # first-order flow-Euler update
            prev_sample = (self.samples[-1] + self.model_outputs[-1] * delta) / self.scales[self.step_index + 1]
        else:
            z_s0, z_s1 = self.samples[-1], self.samples[-2]
            flows0, flows1 = self.model_outputs[-1], self.model_outputs[-2]

            # per-sample linear drift coefficient c = <f, s> / <s, s> between the two cached steps, reduced
            # over all non-batch dimensions and clamped to [-c_max, c_max]; no CFG batch-shape is assumed
            f = flows0 - flows1
            s = z_s0 - z_s1
            reduce_dims = tuple(range(1, f.ndim))
            c = torch.sum(f * s, dim=reduce_dims, keepdim=True) / torch.sum(s * s, dim=reduce_dims, keepdim=True)
            c = torch.clamp(c, -self.config.c_max, self.config.c_max)

            # residual velocities m = flow - c * sample at the two cached steps
            m0 = flows0 - c * z_s0
            m1 = flows1 - c * z_s1
            d1 = (m0 - m1) / (floweq_s0 - self.floweqs[self.step_index - 1])

            prev_sample = (
                z_s0 * (1 + c * delta + c**2 * delta**2) + m0 * (delta - 0.5 * c * delta**2) + d1 * delta**2 / 2
            ) / self.scales[self.step_index + 1]

        prev_sample = prev_sample.to(sample.dtype)

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples according to the noise schedule at the specified timesteps.

        Args:
            original_samples (`torch.Tensor`):
                The original samples without noise.
            noise (`torch.Tensor`):
                The noise to add to the samples.
            timesteps (`torch.IntTensor`):
                The timesteps at which to add noise to the samples.

        Returns:
            `torch.Tensor`:
                The noisy samples.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # begin_index is None when the scheduler is used for training or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def __len__(self) -> int:
        return self.config.num_train_timesteps
