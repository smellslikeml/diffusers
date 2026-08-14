# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor


try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained(
        ...     "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, custom_pipeline="pipeline_flux_hrdit"
        ... ).to("cuda")
        >>> image = pipe(
        ...     "a photo of a mountain lake at dawn", height=4096, width=4096, num_inference_steps=28
        ... ).images[0]
        >>> image.save("hrdit_4096.png")
        ```
"""


# ---------------------------------------------------------------------------------------
# SPA: Spatial Position Alignment
# ---------------------------------------------------------------------------------------


def build_bundle_id_variants(
    height: int,
    width: int,
    bundle_size: int = 64,
    group_num: int = 4,
    device=None,
    dtype=None,
) -> List[torch.Tensor]:
    """
    Spatial Position Alignment (SPA) position ids for a packed latent grid of `height` x `width` tokens.

    Off-the-shelf Flux models are trained at 1024x1024, i.e. a 64x64 packed latent grid, so the rotary position
    ids never exceed 64 during training. Generating at higher resolutions produces out-of-range positions, which
    manifests as the "spatial disorder" artifacts described in the HRDiT paper.

    SPA remaps every token's absolute grid coordinate into the trained RoPE window by wrapping it into sliding
    `bundle_size` bundles. A single bundle partition would introduce seams at the bundle boundaries, so `group_num`
    partitions with shifted origins ("bundle variants") are produced; the pipeline averages the transformer output
    over the variants, which removes the seams.

    Adapted from HRDiT (https://arxiv.org/abs/2608.07003), `hrdit/spa.py::build_bundle_id_variants`. At or below
    the trained resolution a single variant equal to the stock `FluxPipeline` ids is returned.

    Args:
        height (`int`): Packed latent grid height (pixels / 16).
        width (`int`): Packed latent grid width (pixels / 16).
        bundle_size (`int`, defaults to 64): Side length of one bundle, i.e. the trained packed grid size.
        group_num (`int`, defaults to 4): Number of sliding bundle variants.
        device, dtype: Placed on / cast to the latent dtype, matching `_prepare_latent_image_ids`.

    Returns:
        `List[torch.Tensor]` of shape `(height * width, 3)` each, one per bundle variant.
    """
    if height <= bundle_size and width <= bundle_size:
        return [FluxPipeline._prepare_latent_image_ids(1, height, width, device, dtype)]

    variants = []
    ys = torch.arange(height, device=device)
    xs = torch.arange(width, device=device)
    for variant in range(group_num):
        # Slide the bundle partition origin; wrap coordinates into the trained RoPE window.
        shift = (variant * bundle_size) // group_num
        variant_ys = ((ys - shift) % height % bundle_size).to(dtype)
        variant_xs = ((xs - shift) % width % bundle_size).to(dtype)
        ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        ids[..., 1] = variant_ys[:, None]
        ids[..., 2] = variant_xs[None, :]
        variants.append(ids.reshape(height * width, 3))
    return variants


def upsample_packed_latents(latents: torch.Tensor, old_grid: tuple, new_grid: tuple) -> torch.Tensor:
    """
    Bilinearly upsample packed Flux latents (B, old_h * old_w, C * 4) to a `new_grid` (new_h, new_w) packed layout.

    Used between progressive generation stages: the previous stage's denoised latent is the structural prior for
    the next, higher-resolution stage.
    """
    batch_size, _, channels = latents.shape
    num_channels = channels // 4
    old_grid_h, old_grid_w = int(old_grid[0]), int(old_grid[1])
    new_grid_h, new_grid_w = int(new_grid[0]), int(new_grid[1])

    unpacked = latents.view(batch_size, old_grid_h, old_grid_w, num_channels, 2, 2)
    unpacked = unpacked.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, num_channels, old_grid_h * 2, old_grid_w * 2)
    upsampled = F.interpolate(
        unpacked.float(), size=(new_grid_h * 2, new_grid_w * 2), mode="bilinear", align_corners=False
    ).to(latents.dtype)
    return upsampled.view(batch_size, num_channels, new_grid_h, 2, new_grid_w, 2).permute(0, 2, 4, 1, 3, 5).reshape(
        batch_size, new_grid_h * new_grid_w, channels
    )


# ---------------------------------------------------------------------------------------
# HAP: Head-adaptive Attention Pruning
# ---------------------------------------------------------------------------------------


def build_head_scope_plan(num_heads: int, window: int = 64, full_period: int = 4) -> torch.Tensor:
    """
    Per-head attention scope plan for HAP as an int64 tensor of length `num_heads`.

    An entry of `-1` gives the head full (global) scope; a positive entry is the head's window radius in packed
    latent-grid cells. Heads keep text keys inside their scope regardless.

    The paper reads a per-head scope plan from `configs/scope_plan_flux.json`; the checkpoint-specific plan is not
    redistributable here, so this deterministic round-robin plan (every `full_period`-th head global, the rest
    windowed) substitutes for it.
    """
    plan = torch.full((num_heads,), window, dtype=torch.long)
    plan[::full_period] = -1
    return plan


def build_mask_mod(pos_h: torch.Tensor, pos_w: torch.Tensor, windows: torch.Tensor, num_txt: int) -> Callable:
    """
    Build a `flex_attention` mask_mod implementing the per-head scopes of HAP.

    `pos_h` / `pos_w` map an image token index to its packed grid coordinates; `windows` is the output of
    [`build_head_scope_plan`]. Text keys and text queries always stay in scope.
    """

    def mask_mod(b, h, q_idx, kv_idx):
        window = windows[h]
        text_query = q_idx < num_txt
        text_key = kv_idx < num_txt
        img_kv = (kv_idx - num_txt).clamp(min=0)
        img_q = (q_idx - num_txt).clamp(min=0)
        row_dist = (pos_h[img_q] - pos_h[img_kv]).abs()
        col_dist = (pos_w[img_q] - pos_w[img_kv]).abs()
        return (window < 0) | text_query | text_key | ((row_dist <= window) & (col_dist <= window))

    return mask_mod


class _HeadScopeState:
    """
    Carries the current HAP scope plan and packed grid between the pipeline and the attention processors.

    The transformer blocks share one processor instance and do not see the grid directly, so the pipeline arms
    this module-level state around the denoising loop and the processors read from it.
    """

    def __init__(self):
        self.windows = None
        self.grid_height = 0
        self.grid_width = 0
        self._block_masks: Dict[tuple, Any] = {}

    @property
    def enabled(self):
        return self.windows is not None

    def arm(self, windows: torch.Tensor):
        self.windows = windows
        self._block_masks = {}

    def set_grid(self, grid_height: int, grid_width: int):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self._block_masks = {}

    def disarm(self):
        self.windows = None
        self._block_masks = {}

    def get_block_mask(self, seq_len: int, num_txt: int, device):
        key = (seq_len, num_txt, device)
        if key in self._block_masks:
            return self._block_masks[key]

        num_img = seq_len - num_txt
        pos_h = torch.div(torch.arange(num_img, device=device), self.grid_width, rounding_mode="floor")
        pos_w = torch.arange(num_img, device=device) % self.grid_width
        mask_mod = build_mask_mod(pos_h, pos_w, self.windows.to(device), num_txt)
        block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
        self._block_masks[key] = block_mask
        return block_mask


_HAP_STATE = _HeadScopeState()


class HRDiTFluxAttnProcessor(FluxAttnProcessor):
    """
    Flux attention processor with head-adaptive attention pruning (HAP).

    Only the joint (double) blocks — the calls that pass `encoder_hidden_states` — take the pruned path; the
    single blocks fall back to the stock processor, and so does everything when FlexAttention is unavailable.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        if not (_HAP_STATE.enabled and FLEX_ATTENTION_AVAILABLE) or encoder_hidden_states is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        return self._scoped_attention(attn, hidden_states, encoder_hidden_states, image_rotary_emb)

    def _scoped_attention(self, attn, hidden_states, encoder_hidden_states, image_rotary_emb):
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
        encoder_query = attn.norm_added_q(encoder_query)
        encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        num_txt = encoder_hidden_states.shape[1]
        block_mask = _HAP_STATE.get_block_mask(key.shape[1], num_txt, query.device)
        hidden_states = flex_attention(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            block_mask=block_mask,
        ).transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        text_hidden_states, image_hidden_states = hidden_states.split_with_sizes(
            [num_txt, hidden_states.shape[1] - num_txt], dim=1
        )
        image_hidden_states = attn.to_out[0](image_hidden_states.contiguous())
        image_hidden_states = attn.to_out[1](image_hidden_states)
        text_hidden_states = attn.to_add_out(text_hidden_states.contiguous())
        return image_hidden_states, text_hidden_states


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------


class HRDiTFluxPipeline(FluxPipeline):
    r"""
    Training-free high-resolution (up to 4096x4096) text-to-image with off-the-shelf Flux models.

    Adapted from HRDiT, "Training-Free High-Resolution Image Generation with Off-the-Shelf Diffusion Transformer
    Models" (https://arxiv.org/abs/2608.07003); reference implementation at https://github.com/zylwithxy/HRDiT.

    Three training-free pieces on top of the stock `FluxPipeline` denoise loop:

    - **SPA (Spatial Position Alignment)** — `build_bundle_id_variants` wraps high-resolution rotary position ids
      into the trained 64x64 window across `group_num` sliding bundle variants; the transformer output is averaged
      over the variants. The paper averages attention inside each attention layer; this pipeline averages at the
      transformer output, which keeps the stock `FluxTransformer2DModel` untouched and needs no custom processor.
    - **HAP (Head-adaptive Attention Pruning)** — `HRDiTFluxAttnProcessor` prunes attention outside each head's
      scope via FlexAttention (torch >= 2.7). The paper's checkpoint-specific scope plan file is replaced by the
      deterministic `build_head_scope_plan`; if FlexAttention is unavailable the processor falls back to full
      attention and SPA alone remains active.
    - **Progressive generation** — denoising climbs a resolution ladder (1024 -> 2048 -> 4096 by default), with the
      previous stage's latent bilinearly upsampled and re-noised at the next stage's starting sigma.

    Args:
        prompt (`str` or `List[str]`): The prompt to render.
        height / width (`int`): Final output resolution. Defaults to 1024.
        resolutions (`List[int]`, optional): Progressive resolution ladder (square side lengths). Defaults to
            doubling from 1024 up to the target resolution.
        group_num (`int`, defaults to 4): Number of SPA bundle variants averaged per step.
        bundle_size (`int`, defaults to 64): Trained packed latent grid side (1024px for Flux).
        use_hap (`bool`, defaults to True): Enable head-adaptive attention pruning when FlexAttention is available.
        hap_window (`int`, defaults to 64): Window radius (packed grid cells) of the windowed heads.
        stage_strength (`float`, defaults to 0.6): Fraction of the schedule each upsampled stage re-noises through.

    Example: see `EXAMPLE_DOC_STRING`.
    """

    def _resolution_ladder(self, height: int, width: int, resolutions: Optional[List[int]]) -> List[int]:
        if resolutions is None:
            target = max(height, width)
            ladder = []
            side = min(1024, target)
            while side < target:
                ladder.append(side)
                side = min(side * 2, target)
            ladder.append(target)
            return ladder

        ladder = [int(res) for res in resolutions]
        if not ladder or any(ladder[i] >= ladder[i + 1] for i in range(len(ladder) - 1)):
            raise ValueError(f"`resolutions` must be a non-empty, strictly increasing list, got {resolutions}.")
        return ladder

    @staticmethod
    def _stage_dimensions(height: int, width: int, target: int, side: int, quant: int) -> tuple:
        if side >= target:
            return height, width
        stage_height = max(quant, int(round(height * side / target)) // quant * quant)
        stage_width = max(quant, int(round(width * side / target)) // quant * quant)
        return stage_height, stage_width

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        resolutions: Optional[List[int]] = None,
        group_num: int = 4,
        bundle_size: int = 64,
        use_hap: bool = True,
        hap_window: int = 64,
        stage_strength: float = 0.6,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
    ):
        r"""Generate a high-resolution image, training-free, with HRDiT (SPA + optional HAP).

        Accepts the standard [`FluxPipeline`] arguments plus SPA controls (`resolutions`,
        `group_num`, `bundle_size`) and HAP / progressive-ladder controls (`use_hap`,
        `hap_window`, `stage_strength`). `height` and `width` set the final resolution; the
        pipeline renders progressively up to it.

        Examples:
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        quant = self.vae_scale_factor * 2
        if int(height) % quant != 0 or int(width) % quant != 0:
            raise ValueError(f"`height` and `width` must be multiples of {quant}, got {height} and {width}.")
        if not 0.0 < stage_strength <= 1.0:
            raise ValueError(f"`stage_strength` must be in (0, 1], got {stage_strength}.")

        ladder = self._resolution_ladder(height, width, resolutions)
        target = max(height, width)

        device = self._execution_device
        dtype = prompt_embeds.dtype if prompt_embeds is not None else self.transformer.dtype

        # 1. Encode prompt (guidance-distilled models like FLUX.1-dev need no true CFG pass).
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
        batch_size = prompt_embeds.shape[0]

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(batch_size)
        else:
            guidance = None

        self._joint_attention_kwargs = {}

        # 2. Optionally arm HAP.
        original_attn_processors = None
        if use_hap:
            if FLEX_ATTENTION_AVAILABLE:
                original_attn_processors = dict(self.transformer.attn_processors)
                self.transformer.set_attn_processor(HRDiTFluxAttnProcessor())
                num_heads = getattr(self.transformer.config, "num_attention_heads", 24)
                _HAP_STATE.arm(build_head_scope_plan(num_heads, window=hap_window))
            else:
                logger.warning(
                    "use_hap=True but FlexAttention is unavailable (needs torch >= 2.7); "
                    "falling back to full attention. SPA stays active."
                )

        try:
            # 3. Progressive denoising over the resolution ladder.
            all_sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            num_channels_latents = self.transformer.config.in_channels // 4
            latents = latents.to(device=device, dtype=dtype) if latents is not None else None
            old_grid = None
            self._num_timesteps = 0

            for stage, side in enumerate(ladder):
                stage_height, stage_width = self._stage_dimensions(height, width, target, side, quant)
                latent_height = 2 * (stage_height // quant)
                latent_width = 2 * (stage_width // quant)
                grid_height, grid_width = latent_height // 2, latent_width // 2

                if stage == 0:
                    if latents is not None and latents.shape[-2:] != (latent_height, latent_width):
                        raise ValueError(
                            f"Provided `latents` of spatial shape {latents.shape[-2:]} do not match the first "
                            f"progressive stage ({latent_height}, {latent_width})."
                        )
                    if latents is None:
                        latents = randn_tensor(
                            (batch_size, num_channels_latents, latent_height, latent_width),
                            generator=generator,
                            device=device,
                            dtype=dtype,
                        )
                    latents = self._pack_latents(
                        latents, batch_size, num_channels_latents, latent_height, latent_width
                    )
                    stage_sigmas = all_sigmas
                else:
                    latents = upsample_packed_latents(latents, old_grid, (grid_height, grid_width))
                    # Later stages re-noise through the tail of the schedule (`stage_strength` of it).
                    num_stage_steps = max(1, int(round(num_inference_steps * stage_strength)))
                    stage_sigmas = all_sigmas[-num_stage_steps:]
                old_grid = (grid_height, grid_width)

                image_id_variants = build_bundle_id_variants(
                    grid_height, grid_width, bundle_size=bundle_size, group_num=group_num, device=device, dtype=dtype
                )

                if _HAP_STATE.enabled:
                    _HAP_STATE.set_grid(grid_height, grid_width)

                mu = calculate_shift(
                    grid_height * grid_width,
                    self.scheduler.config.get("base_image_seq_len", 256),
                    self.scheduler.config.get("max_image_seq_len", 4096),
                    self.scheduler.config.get("base_shift", 0.5),
                    self.scheduler.config.get("max_shift", 1.15),
                )
                timesteps, _ = retrieve_timesteps(
                    self.scheduler, len(stage_sigmas), device, sigmas=stage_sigmas, mu=mu
                )
                self.scheduler.set_begin_index(0)
                self._num_timesteps += len(timesteps)

                if stage > 0:
                    # Flow-match interpolation at the stage's (shift-adjusted) starting sigma.
                    start_sigma = float(self.scheduler.sigmas[0])
                    noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=dtype)
                    latents = (1.0 - start_sigma) * latents + start_sigma * noise

                self.set_progress_bar_config(desc=f"HRDiT {stage_width}x{stage_height}")
                with self.progress_bar(total=len(timesteps)) as progress_bar:
                    for t in timesteps:
                        self._current_timestep = t
                        timestep = t.expand(latents.shape[0]).to(latents.dtype)
                        noise_pred = None
                        for image_ids in image_id_variants:
                            with self.transformer.cache_context("cond"):
                                variant_pred = self.transformer(
                                    hidden_states=latents,
                                    timestep=timestep / 1000,
                                    guidance=guidance,
                                    pooled_projections=pooled_prompt_embeds,
                                    encoder_hidden_states=prompt_embeds,
                                    txt_ids=text_ids,
                                    img_ids=image_ids,
                                    joint_attention_kwargs=self.joint_attention_kwargs,
                                    return_dict=False,
                                )[0]
                            noise_pred = variant_pred if noise_pred is None else noise_pred + variant_pred
                        noise_pred = noise_pred / len(image_id_variants)
                        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                        progress_bar.update()

            self._current_timestep = None

            if output_type == "latent":
                image = latents
            else:
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)

            self.maybe_free_model_hooks()
        finally:
            if original_attn_processors is not None:
                self.transformer.set_attn_processor(original_attn_processors)
                _HAP_STATE.disarm()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
