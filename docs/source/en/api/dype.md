<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Resolution extrapolation

Training-free methods that let RoPE-based diffusion transformers such as [`FluxTransformer2DModel`] generate above their trained resolution (for example 4096x4096 from a model trained at 1024x1024), with no fine-tuning and no extra sampling cost.

[DyPE](https://huggingface.co/papers/2510.20766) (Dynamic Position Extrapolation) swaps the transformer's rotary positional embedding for a timestep-aware YaRN / NTK-by-parts schedule that only engages above the trained resolution. [SEGA](https://huggingface.co/papers/2605.22668) (Spectral-Energy Guided Attention) additionally applies a per-frequency, content-aware attention temperature derived from the latent's spectrum, which suppresses the high-frequency speckle a scalar temperature can leave in flat regions at very high resolutions. Both are enabled through [`apply_dype`].

```python
import torch
from diffusers import FluxPipeline, apply_dype

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# method="yarn" (default) is plain DyPE; method="spectral" adds SEGA spectral attention.
apply_dype(pipe.transformer, method="spectral")

# Above the trained resolution, also flatten the flow-matching shift schedule so the sampler
# does not stall near pure noise (the default shift `mu` grows with the image sequence length).
pipe.scheduler.register_to_config(base_shift=1.15, max_shift=1.15)

image = pipe(
    "a sunlit alpine meadow, snow-capped peaks, clear blue sky",
    height=4096,
    width=4096,
    guidance_scale=4.5,
    num_inference_steps=28,
).images[0]
```

> [!TIP]
> `apply_dype` is a no-op at or below the trained resolution (1024x1024 for Flux), so the hook can stay applied for standard-resolution generation.

## apply_dype

[[autodoc]] apply_dype
