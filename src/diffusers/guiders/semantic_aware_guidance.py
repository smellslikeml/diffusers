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

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..configuration_utils import register_to_config
from .guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg


if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState


class SemanticAwareGuidance(BaseGuidance):
    """
    Semantic-aware Classifier-Free Guidance (S-CFG): https://huggingface.co/papers/2404.05384

    A global CFG scale applies the same guidance strength everywhere, so regions whose text guidance already produces a
    large update get over-guided while weak regions stay under-guided -- the "spatial inconsistency" the paper targets.
    S-CFG replaces the scalar with a per-region scale map that equalizes the guidance strength across the latent, then
    applies `pred = pred_uncond + scale_map * (pred_cond - pred_uncond)` (the standard guider contract).

    **Adaptation note.** The paper derives its semantic regions from cross-/self-attention maps collected via per-model
    attention hooks. That segmentation is an auxiliary signal that is not available at the guider contract (which only
    receives `pred_cond` / `pred_uncond`) and is architecture-specific. This implementation keeps the paper's core
    mechanism -- a spatially varying CFG scale map that equalizes per-region guidance strength -- at full fidelity, but
    substitutes the attention-based segmentation with a parameter-free proxy: the local guidance magnitude computed
    directly from `pred_cond - pred_uncond` and pooled over a `window_size` neighborhood to reach region-level (rather
    than pixel-level) granularity. Each position is rescaled toward the sample's mean guidance strength, so weak regions
    are boosted and strong regions are damped, exactly as in S-CFG. When the predictions are not spatial (`ndim != 4`,
    e.g. sequence-shaped transformer outputs) the guider falls back to standard scalar CFG.

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The reference CFG scale. The per-region scale map is centered on this value and rescaled around it to
            equalize semantic strengths. Higher values give stronger prompt conditioning.
        window_size (`int`, defaults to `3`):
            Odd side length of the square neighborhood used to pool the local guidance magnitude into a region-level
            estimate. `1` disables pooling (pixel-level); larger values approximate coarser semantic regions.
        rescale_clamp (`float`, defaults to `2.0`):
            Maximum multiplicative deviation of the per-region scale from `guidance_scale`. The scale map is clamped to
            `[guidance_scale / rescale_clamp, guidance_scale * rescale_clamp]` to keep low-magnitude regions from
            producing runaway guidance.
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time. See
            [~guiders.classifier_free_guidance.ClassifierFreeGuidance] for more details.
        start (`float`, defaults to `0.0`):
            The fraction of the total number of denoising steps after which guidance starts.
        stop (`float`, defaults to `1.0`):
            The fraction of the total number of denoising steps after which guidance stops.
    """

    _input_predictions = ["pred_cond", "pred_uncond"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 7.5,
        window_size: int = 3,
        rescale_clamp: float = 2.0,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)

        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f"Expected `window_size` to be a positive odd integer, but got {window_size}.")
        if rescale_clamp < 1.0:
            raise ValueError(f"Expected `rescale_clamp` to be >= 1.0, but got {rescale_clamp}.")

        self.guidance_scale = guidance_scale
        self.window_size = window_size
        self.rescale_clamp = rescale_clamp
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
            pred = semantic_aware_guidance(
                pred_cond,
                pred_uncond,
                self.guidance_scale,
                self.window_size,
                self.rescale_clamp,
                self.use_original_formulation,
            )

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


def semantic_scale_map(
    diff: torch.Tensor,
    guidance_scale: float,
    window_size: int = 3,
    rescale_clamp: float = 2.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Builds the per-region CFG scale map used by S-CFG from the guidance term `diff = pred_cond - pred_uncond`.

    The local guidance strength is the channel-norm of `diff` at each spatial position, pooled over a `window_size`
    neighborhood (a parameter-free proxy for the paper's attention-derived semantic regions). Each position is rescaled
    toward the per-sample mean strength so weak regions are boosted and strong regions are damped, then clamped to keep
    the guidance well-conditioned. Expects a 4D `(B, C, H, W)` tensor.
    """
    magnitude = diff.pow(2).sum(dim=1, keepdim=True).clamp_min(0.0).sqrt()
    if window_size > 1:
        pad = window_size // 2
        magnitude = F.avg_pool2d(magnitude, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)
    reference = magnitude.mean(dim=(2, 3), keepdim=True)
    scale_map = guidance_scale * reference / magnitude.clamp_min(eps)
    scale_map = scale_map.clamp(guidance_scale / rescale_clamp, guidance_scale * rescale_clamp)
    return scale_map


def semantic_aware_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    window_size: int = 3,
    rescale_clamp: float = 2.0,
    use_original_formulation: bool = False,
) -> torch.Tensor:
    diff = pred_cond - pred_uncond
    pred = pred_cond if use_original_formulation else pred_uncond

    if diff.ndim != 4:
        # Non-spatial predictions (e.g. sequence-shaped transformer outputs): fall back to scalar CFG.
        return pred + guidance_scale * diff

    scale_map = semantic_scale_map(diff, guidance_scale, window_size, rescale_clamp)
    return pred + scale_map * diff
