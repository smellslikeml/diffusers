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

import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# FLUX is trained at 1024x1024 -> a 64x64 packed grid (4096 image tokens) plus 512 text tokens.
_TRAIN_SEQ_LEN = 64 ** 2 + 512

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


def _phi(x: torch.Tensor, n1: int, size: int) -> torch.Tensor:
    """Bundle mapping: 0 for x < n1, else ceil((x + 1 - n1) / size). Monotonic non-decreasing."""
    return torch.where(x < n1, torch.zeros_like(x), (x + 1 - n1 + size - 1) // size)


def build_bundle_id_variants(img_ids: torch.Tensor, group_num: int) -> List[torch.Tensor]:
    """
    Spatial Position Alignment (SPA) bundle-index variants of the packed-latent position ids ``img_ids``.

    Off-the-shelf FLUX is trained on a 64x64 packed grid, so its rotary position ids never exceed ~64. Generating
    at higher resolution pushes ids out of the trained range, which is the "spatial disorder" the HRDiT paper
    describes. SPA maps each token's grid coordinate into a small number of *bundles* via a monotonic (non-wrapping)
    coarsening ``_phi`` -- many neighbouring tokens then share a position id inside the trained range. Because the
    mapping is monotonic it introduces no periodic tiling; the residual bundle-boundary seams are averaged out by
    sliding the boundary origin across ``group_num`` variants (see [`HRDiTFluxAttnProcessor`], which averages the
    per-variant attention outputs -- element-wise identical to averaging the attention maps, at O(T*D) memory).

    Adapted from HRDiT (https://arxiv.org/abs/2608.07003), ``hrdit/spa.py::build_bundle_id_variants``.

    Args:
        img_ids (`torch.Tensor`): Packed-latent position ids of shape `(T, 3)`; column 1 is the row index and
            column 2 the column index (as produced by `FluxPipeline._prepare_latent_image_ids`).
        group_num (`int`): Controls the bundle size `ceil(max_index / (group_num - 1))`; larger values give finer
            bundles (more distinct positions, kept inside the trained window). Must be >= 2.

    Returns:
        `List[torch.Tensor]` of shape `(T, 3)` each, one per sliding bundle-boundary variant.
    """
    if group_num < 2:
        raise ValueError(f"`group_num` must be >= 2 for SPA, got {group_num}.")

    rows = img_ids[:, 1].long()
    cols = img_ids[:, 2].long()
    s_row = max(1, math.ceil(rows.max().item() / (group_num - 1)))
    s_col = max(1, math.ceil(cols.max().item() / (group_num - 1)))

    def variant(n1_row: int, n1_col: int) -> torch.Tensor:
        ids = img_ids.clone()
        ids[:, 1] = _phi(rows, n1_row, s_row).to(img_ids.dtype)
        ids[:, 2] = _phi(cols, n1_col, s_col).to(img_ids.dtype)
        return ids

    variants = [variant(s_row, s_col)]
    variants += [variant(n, s_col) for n in range(1, s_row)]
    variants += [variant(s_row, m) for m in range(1, s_col)]
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
# SPA attention processor (averaging happens *inside* attention)
# ---------------------------------------------------------------------------------------


class _SPAState:
    """
    Module-level carrier for the current SPA rotary-embedding variants.

    The transformer blocks share one processor instance and are not aware of SPA, so the pipeline precomputes the
    per-variant rotary embeddings once per stage and arms them here; every processor reads from this state.
    """

    def __init__(self):
        self.rope_variants = None  # List[(cos, sin)] covering the full [text; image] sequence
        self.proportional = True

    @property
    def enabled(self):
        return self.rope_variants is not None

    def arm(self, rope_variants):
        self.rope_variants = rope_variants

    def disarm(self):
        self.rope_variants = None


_SPA_STATE = _SPAState()


class HRDiTFluxAttnProcessor(FluxAttnProcessor):
    """
    Flux attention processor implementing HRDiT's Spatial Position Alignment (SPA).

    When SPA is armed (via [`_SPAState`]) the processor ignores the transformer's own rotary embedding and instead
    runs attention once per bundle-index variant -- applying that variant's RoPE to the query/key -- then averages
    the attention *outputs*. Since `mean_n(softmax(A_n)) @ V == mean_n(softmax(A_n) @ V)`, averaging the outputs is
    element-wise identical to the paper's average-over-attention-maps, at O(T*D) memory instead of O(V*T^2). A
    proportional attention scale ``sqrt(log_train(seq_len) / head_dim)`` compensates for the longer high-resolution
    sequence. When SPA is disarmed the processor is exactly the stock `FluxAttnProcessor`.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        if not _SPA_STATE.enabled:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        head_dim = query.shape[-1]
        seq_len = query.shape[1]
        if _SPA_STATE.proportional and seq_len > 1:
            scale = math.sqrt(math.log(seq_len, _TRAIN_SEQ_LEN) / head_dim)
        else:
            scale = head_dim ** -0.5

        value_t = value.transpose(1, 2).contiguous()  # [B, H, S, D]
        acc = None
        for rope_variant in _SPA_STATE.rope_variants:
            query_v = apply_rotary_emb(query, rope_variant, sequence_dim=1).transpose(1, 2).contiguous()
            key_v = apply_rotary_emb(key, rope_variant, sequence_dim=1).transpose(1, 2).contiguous()
            out = F.scaled_dot_product_attention(query_v, key_v, value_t, dropout_p=0.0, is_causal=False, scale=scale)
            acc = out if acc is None else acc + out
        hidden_states = (acc / len(_SPA_STATE.rope_variants)).transpose(1, 2)  # [B, S, H, D]
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            num_txt = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [num_txt, hidden_states.shape[1] - num_txt], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        return hidden_states


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------


class HRDiTFluxPipeline(FluxPipeline):
    r"""
    Training-free high-resolution (up to 4096x4096) text-to-image with off-the-shelf Flux models.

    Adapted from HRDiT, "Training-Free High-Resolution Image Generation with Off-the-Shelf Diffusion Transformer
    Models" (https://arxiv.org/abs/2608.07003); reference implementation at https://github.com/zylwithxy/HRDiT.

    Two training-free pieces on top of the stock `FluxPipeline` denoise loop:

    - **SPA (Spatial Position Alignment)** -- `build_bundle_id_variants` maps high-resolution rotary position ids
      into the trained ~64x64 window via a monotonic bundle coarsening (no wrapping, so no periodic tiling), across
      several sliding bundle-boundary variants. `HRDiTFluxAttnProcessor` runs attention once per variant and averages
      the outputs, with a proportional attention scale for the longer sequence. This is the core training-free fix
      for high-resolution "spatial disorder".
    - **Progressive generation** -- denoising climbs a resolution ladder (1024 -> 2048 -> 4096 by default); each
      stage bilinearly upsamples the previous stage's latent and re-noises it through the tail of the schedule. SPA
      is active only on the upscale stages (the base stage is in-distribution and uses stock RoPE).

    Not ported from the reference (documented follow-ups): the checkpoint-specific HAP head-scope pruning
    (`configs/scope_plan_flux.json`), NTK-aware RoPE scaling, and per-step SPA scheduling. SPA here runs on every
    step of every upscale stage and on every attention block.

    Args:
        prompt (`str` or `List[str]`): The prompt to render.
        height / width (`int`): Final output resolution. Defaults to 1024.
        resolutions (`List[int]`, optional): Progressive resolution ladder (square side lengths). Defaults to
            doubling from 1024 up to the target resolution.
        group_num (`int`, defaults to 80): SPA bundle granularity; bundle size is `ceil(max_index / (group_num - 1))`.
            Larger keeps more distinct positions inside the trained window. `group_num - 1` also bounds the number of
            averaged variants.
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

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        resolutions: Optional[List[int]] = None,
        group_num: int = 80,
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
        r"""Generate a high-resolution image, training-free, with HRDiT (SPA + progressive generation).

        Accepts the standard [`FluxPipeline`] arguments plus `resolutions` / `group_num` (SPA) and `stage_strength`
        (progressive re-noising). `height` and `width` set the final resolution; the pipeline renders progressively
        up to it.

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

        # 2. Install the SPA processor (armed per-stage below; disarmed => stock attention).
        original_attn_processors = dict(self.transformer.attn_processors)
        self.transformer.set_attn_processor(HRDiTFluxAttnProcessor())

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

                image_ids = self._prepare_latent_image_ids(batch_size, grid_height, grid_width, device, dtype)

                # SPA on the upscale stages only: coarsen the out-of-range ids into the trained window and
                # precompute the per-variant rotary embeddings over the full [text; image] sequence.
                if grid_height > 64 or grid_width > 64:
                    variants = build_bundle_id_variants(image_ids, group_num)
                    rope_variants = [
                        self.transformer.pos_embed(torch.cat([text_ids, variant_ids], dim=0))
                        for variant_ids in variants
                    ]
                    _SPA_STATE.arm(rope_variants)
                else:
                    _SPA_STATE.disarm()

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
                        with self.transformer.cache_context("cond"):
                            noise_pred = self.transformer(
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
            _SPA_STATE.disarm()
            self.transformer.set_attn_processor(original_attn_processors)

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
