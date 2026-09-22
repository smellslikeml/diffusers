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

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..configuration_utils import register_to_config
from .guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg


if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState


class RectifiedClassifierFreeGuidance(BaseGuidance):
    """
    Rectified Classifier-Free Guidance (ReCFG): https://huggingface.co/papers/2410.18737

    Standard CFG combines the conditional and unconditional predictions with two coefficients that sum to one
    (`gamma` and `1 - gamma`). The ReCFG paper shows that this sum-to-one configuration cannot be expressed as a
    proper reverse diffusion process, and relaxes the constraint by introducing two independent coefficients
    `gamma_1` (on the conditional term) and `gamma_0` (on the unconditional term) that need *not* sum to one:

    ```
    pred = gamma_1 * pred_cond + gamma_0 * pred_uncond
    ```

    Requiring the guided prediction to have zero expectation (so that denoising aligns with diffusion theory) yields
    the closed form `gamma_0 = (1 - gamma_1) * rho`, where `rho = E[pred_cond] / E[pred_uncond]` is the ratio of the
    expected conditional and unconditional predictions (paper Eq. 42). Substituting this back is equivalent to running
    ordinary CFG with the unconditional prediction rescaled by `rho`, so the technique reduces exactly to
    `ClassifierFreeGuidance` when `rho == 1`.

    **Estimating `rho`.** The paper precomputes `rho` per timestep offline by traversing a calibration set and storing
    a lookup table of `E[pred_cond] / E[pred_uncond]`. The guider abstraction only observes the current step's batch of
    predictions, so this implementation instead estimates `rho` online, per sample, as the ratio of the conditional and
    unconditional prediction norms of the current batch. This is a parameter-free proxy for the paper's expectation
    ratio (in the spirit of the online rescaling used by
    [`~guiders.classifier_free_zero_star_guidance.ClassifierFreeZeroStarGuidance`]) and requires no calibration data.

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Corresponds to `gamma_1` above. Higher values result in
            stronger conditioning on the text prompt, while lower values allow for more freedom in generation.
        rectification_scale (`float`, defaults to `1.0`):
            A multiplier on the estimated rectification coefficient `rho`, controlling how strongly the unconditional
            term is rectified. `0.0` recovers standard CFG (`rho` forced to 1); `1.0` applies the full estimated
            rectification.
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time. See
            [`~guiders.classifier_free_guidance.ClassifierFreeGuidance`] for more details.
        start (`float`, defaults to `0.0`):
            The fraction of the total number of denoising steps after which guidance starts.
        stop (`float`, defaults to `1.0`):
            The fraction of the total number of denoising steps after which guidance stops.
        enabled (`bool`, defaults to `True`):
            Whether guidance is enabled. Set to `False` to disable guidance entirely (uses only conditional
            predictions).
    """

    _input_predictions = ["pred_cond", "pred_uncond"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 7.5,
        rectification_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)

        self.guidance_scale = guidance_scale
        self.rectification_scale = rectification_scale
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

    def prepare_inputs(self, data: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> list["BlockState"]:
        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, self._input_predictions):
            data_batch = self._prepare_batch(data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def prepare_inputs_from_block_state(
        self, data: "BlockState", input_fields: dict[str, str | tuple[str, str]]
    ) -> list["BlockState"]:
        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, self._input_predictions):
            data_batch = self._prepare_batch_from_block_state(input_fields, data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def forward(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor | None = None) -> GuiderOutput:
        pred = None

        if not self._is_cfg_enabled():
            pred = pred_cond
        else:
            # ReCFG: rectify the unconditional term by rho so the two coefficients need not sum to one.
            rho = rectified_guidance_scale(pred_cond, pred_uncond)
            rho = 1.0 + self.rectification_scale * (rho - 1.0)
            pred_uncond_rect = pred_uncond * rho
            shift = pred_cond - pred_uncond_rect
            pred = pred_cond if self.use_original_formulation else pred_uncond_rect
            pred = pred + self.guidance_scale * shift

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)

    @property
    def is_conditional(self) -> bool:
        return self._count_prepared == 1

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
            num_conditions += 1
        return num_conditions

    def _is_cfg_enabled(self) -> bool:
        if not self._enabled:
            return False

        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self._start * self._num_inference_steps)
            skip_stop_step = int(self._stop * self._num_inference_steps)
            is_within_range = skip_start_step <= self._step < skip_stop_step

        is_close = False
        if self.use_original_formulation:
            is_close = math.isclose(self.guidance_scale, 0.0)
        else:
            is_close = math.isclose(self.guidance_scale, 1.0)

        return is_within_range and not is_close


def rectified_guidance_scale(cond: torch.Tensor, uncond: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Estimates the ReCFG rectification coefficient `rho = E[pred_cond] / E[pred_uncond]` (paper Eq. 42).

    The paper's expectation ratio is approximated online, per sample, by the ratio of the conditional and
    unconditional prediction norms of the current batch. Returns a tensor shaped to broadcast against `cond`, so that
    `uncond * rho` rescales each sample independently. The value is `1.0` when the two predictions share the same
    magnitude, in which case ReCFG reduces exactly to standard CFG.
    """
    cond_dtype = cond.dtype
    cond_flat = cond.float().flatten(1)
    uncond_flat = uncond.float().flatten(1)
    cond_norm = torch.linalg.vector_norm(cond_flat, dim=1, keepdim=True)
    uncond_norm = torch.linalg.vector_norm(uncond_flat, dim=1, keepdim=True) + eps
    rho = cond_norm / uncond_norm
    rho = rho.view(-1, *(1,) * (cond.ndim - 1))
    return rho.to(dtype=cond_dtype)
