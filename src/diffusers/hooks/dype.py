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

import numpy as np
import torch

from ..utils import get_logger
from ..utils.torch_utils import maybe_adjust_dtype_for_device
from .hooks import HookRegistry, ModelHook


logger = get_logger(__name__)  # pylint: disable=invalid-name


_DYPE_HOOK = "dype_hook"


# Adapted from https://github.com/guyyariv/DyPE (MIT). DyPE: "Dynamic Position Extrapolation for Ultra High
# Resolution Diffusion" (https://arxiv.org/abs/2510.20766).


def find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    # Inverse dim formula to find the dimension index of a given number of rotations
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    """
    Find the correction range for NTK-by-parts interpolation.
    """
    low = np.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = np.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def find_newbase_ntk(dim, base, scale):
    """
    Calculate the new base for NTK-aware scaling.
    """
    return base * (scale ** (dim / (dim - 2)))


def _dype_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  # torch.float32, torch.float64 (flux)
    yarn=False,
    max_pe_len=None,
    ori_max_pe_len=64,
    dype=False,
    current_timestep=1.0,
):
    r"""
    Precompute the frequency tensor for complex exponentials (cis) with RoPE. Supports YaRN interpolation, optionally
    modulated by the DyPE timestep schedule.

    Args:
        dim (`int`):
            Dimension of the frequency tensor.
        pos (`torch.Tensor`):
            Position indices for the frequency tensor. [S] or scalar.
        theta (`float`, *optional*, defaults to `10000.0`):
            Scaling factor for frequency computation.
        use_real (`bool`, *optional*, defaults to `False`):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to `1.0`):
            Scaling factor for linear interpolation.
        ntk_factor (`float`, *optional*, defaults to `1.0`):
            Scaling factor for NTK-Aware RoPE.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If True and use_real, real and imaginary parts are interleaved with themselves to reach `dim`. Otherwise,
            they are concatenated.
        freqs_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Data type of the frequency tensor. `torch.float64` is used by models such as Flux.
        yarn (`bool`, *optional*, defaults to `False`):
            If True, use YaRN interpolation combining NTK, linear, and base methods.
        max_pe_len (`int` or `torch.Tensor`, *optional*):
            Maximum position encoding length (current patches per axis for vision models).
        ori_max_pe_len (`int`, *optional*, defaults to `64`):
            Original maximum position encoding length (base patches per axis, 1024 // 16 = 64 for Flux).
        dype (`bool`, *optional*, defaults to `False`):
            If True, enable DyPE (Dynamic Position Extrapolation) with timestep-aware scaling of the correction
            ranges (`kappa = current_timestep**2`).
        current_timestep (`float`, *optional*, defaults to `1.0`):
            Current timestep for DyPE, normalized to [0, 1] where 1 is pure noise.

    Returns:
        `torch.Tensor`: Precomputed frequency tensor for complex exponentials. [S, D/2]. If `use_real=True`, returns a
        tuple of `(cos, sin)` tensors.
    """
    assert dim % 2 == 0

    device = pos.device

    if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len, dtype=freqs_dtype, device=device)

        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

        beta_0 = 1.25
        beta_1 = 0.75
        gamma_0 = 16
        gamma_1 = 2

        exponents = torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim
        freqs_base = 1.0 / (theta**exponents)
        # Position interpolation (PI) frequencies
        freqs_linear = 1.0 / (scale * theta**exponents)

        new_base = find_newbase_ntk(dim, theta, scale)
        if new_base.dim() > 0:
            new_base = new_base.view(-1, 1)
        freqs_ntk = 1.0 / torch.pow(new_base, exponents)
        if freqs_ntk.dim() > 1:
            freqs_ntk = freqs_ntk.squeeze()

        if dype:
            kappa = current_timestep**2.0  # kappa(t) = t^lambda_t, with lambda_t = 2
            beta_0 = beta_0 * kappa
            beta_1 = beta_1 * kappa

        low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)

        freqs_mask = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype)
        freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask

        if dype:
            gamma_0 = gamma_0 * kappa
            gamma_1 = gamma_1 * kappa

        low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)

        freqs_mask = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype)
        freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask
    else:
        theta_ntk = theta * ntk_factor
        exponents = torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim
        freqs = 1.0 / (theta_ntk**exponents) / linear_factor

    freqs = torch.outer(pos, freqs)

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()

    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]

        if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
            # YaRN attention temperature. `torch.ones_like` is used instead of a plain `torch.tensor(1.0)` so the
            # constant is materialized on the same device/dtype as `scale`.
            mscale = torch.where(scale <= 1.0, torch.ones_like(scale), 0.1 * torch.log(scale) + 1.0)
            freqs_cos = freqs_cos * mscale
            freqs_sin = freqs_sin * mscale

        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class _DyPEPosEmbed(torch.nn.Module):
    r"""
    Drop-in replacement for the positional embedding of `FluxTransformer2DModel` that applies the DyPE schedule to the
    spatial axes. The first axis (text positions) always uses plain RoPE, and the scheduled path only engages on a
    spatial axis when the number of patches on that axis (`max_pos + 1`) exceeds the number of patches at the trained
    resolution (`base_resolution // patch_size = 1024 // 16 = 64`). As a result, generation at or below the trained
    resolution is a no-op compared to the stock positional embedding.
    """

    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        method: str = "yarn",
        dype: bool = True,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.base_resolution = 1024
        self.patch_size = 16
        self.base_patches = self.base_resolution // self.patch_size
        self.method = method
        self.dype = dype if method != "base" else False
        self.current_timestep = 1.0

    def set_timestep(self, timestep: float):
        """Set current timestep for DyPE. Timestep normalized to [0, 1] where 1 is pure noise."""
        self.current_timestep = timestep

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, ids.device)
        for i in range(n_axes):
            common_kwargs = {
                "dim": self.axes_dim[i],
                "pos": pos[:, i],
                "theta": self.theta,
                "repeat_interleave_real": True,
                "use_real": True,
                "freqs_dtype": freqs_dtype,
            }

            if i > 0:
                max_pos = pos[:, i].max().item()
                current_patches = max_pos + 1

                if self.method == "yarn" and current_patches > self.base_patches:
                    max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = _dype_rotary_pos_embed(
                        **common_kwargs,
                        yarn=True,
                        max_pe_len=max_pe_len,
                        ori_max_pe_len=self.base_patches,
                        dype=self.dype,
                        current_timestep=self.current_timestep,
                    )

                else:
                    cos, sin = _dype_rotary_pos_embed(**common_kwargs)
            else:
                cos, sin = _dype_rotary_pos_embed(**common_kwargs)

            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class DyPEHook(ModelHook):
    r"""
    A hook that swaps the positional embedding of a Flux-like transformer for a `_DyPEPosEmbed` and feeds it the
    current (normalized) timestep at every forward pass, enabling training-free ultra-high-resolution generation.
    """

    def __init__(self, method: str = "yarn", dype: bool = True) -> None:
        super().__init__()

        self.method = method
        self.dype = dype
        self._original_pos_embed = None

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        pos_embed = getattr(module, "pos_embed", None)
        if pos_embed is None:
            raise ValueError(
                "DyPE requires the module to have a `pos_embed` attribute with `theta` and `axes_dim` attributes, as "
                "found on `FluxTransformer2DModel`. Please apply the hook to a compatible transformer."
            )

        self._original_pos_embed = pos_embed
        module.pos_embed = _DyPEPosEmbed(
            theta=pos_embed.theta,
            axes_dim=pos_embed.axes_dim,
            method=self.method,
            dype=self.dype,
        )
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        timestep = kwargs.get("timestep", None)
        if timestep is None and len(args) > 3:
            # Stock `FluxTransformer2DModel.forward` receives (hidden_states, encoder_hidden_states,
            # pooled_projections, timestep, ...) positionally when not passed as a kwarg.
            timestep = args[3]

        if timestep is not None:
            if torch.is_tensor(timestep):
                timestep = timestep.flatten()[0]
            module.pos_embed.set_timestep(float(timestep))

        return args, kwargs

    def deinitalize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if self._original_pos_embed is not None:
            module.pos_embed = self._original_pos_embed
            self._original_pos_embed = None
        return module


def apply_dype(module: torch.nn.Module, method: str = "yarn", dype: bool = True) -> None:
    r"""
    Applies [DyPE](https://huggingface.co/papers/2510.20766) to a given transformer to enable training-free
    ultra-high-resolution generation.

    Args:
        module (`torch.nn.Module`):
            The transformer to apply DyPE to. This should be a RoPE-based DiT with a `pos_embed` attribute exposing
            `theta` and `axes_dim`, such as the stock `FluxTransformer2DModel`. At or below the trained resolution
            (1024x1024 for Flux), the hook is a no-op.
        method (`str`, defaults to `"yarn"`):
            The position extrapolation method to use. Only `"yarn"` (YaRN / NTK-by-parts, DyPE's default) is currently
            supported.
        dype (`bool`, defaults to `True`):
            Whether to modulate the position extrapolation schedule by the diffusion timestep (`kappa = t^2`). If
            `False`, the extrapolation schedule is static across timesteps.

    Example:
    ```python
    >>> import torch
    >>> from diffusers import FluxPipeline, apply_dype

    >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> apply_dype(pipe.transformer)
    >>> image = pipe("a photo of a cat", height=4096, width=4096, guidance_scale=4.5).images[0]
    ```
    """

    if method != "yarn":
        raise ValueError(f'`method` must be "yarn", but got {method!r}. Other methods are not supported yet.')

    logger.debug(f"Enabling DyPE (method={method}, dype={dype}) on {module.__class__.__name__}")

    hook = DyPEHook(method=method, dype=dype)
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(hook, _DYPE_HOOK)
