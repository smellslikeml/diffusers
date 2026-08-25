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

"""Control harness for the ChebBooster (Chebyshev cache) PR.

Three-arm comparison on FLUX.1-dev with identical seeds/prompts and matched
`cache_interval` + warmup:

  (a) no-cache reference
  (b) in-repo TaylorSeer cache (the control arm)
  (c) Chebyshev cache (ChebBooster port)

Reports wall-clock latency per arm and quality of (b) and (c) against the
no-cache reference (PSNR always; LPIPS and CLIP score when `lpips` /
`open_clip_torch` are installed). The claim this harness is meant to check:
at matched latency, Chebyshev cache >= TaylorSeer cache on quality. If the
delta is within noise, the cache variant is redundant and the PR should not
be opened upstream.

Usage:
    python benchmarking_chebyshev_cache.py --num-prompts 8 --num-inference-steps 28
"""

import argparse
import csv
import time
from contextlib import contextmanager

import torch

from diffusers import ChebyshevCacheConfig, FluxPipeline, TaylorSeerCacheConfig


@contextmanager
def _cache_context(transformer, name="benchmark"):
    """Set the cache context directly on every hooked module.

    A fresh module traversal each call, so it is immune to the cached
    child-registry list that ``CacheMixin.cache_context`` relies on — that list
    is built on first use and is not rebuilt when hooks are added later, so any
    ``cache_context`` call made before ``enable_cache`` (e.g. the no-cache
    reference arm going through the pipeline) leaves later contexts unable to
    reach the newly-hooked blocks ("No context is set").
    """
    modules = [m for m in transformer.modules() if hasattr(m, "_diffusers_hook")]
    for m in modules:
        m._diffusers_hook._set_context(name)
    try:
        yield
    finally:
        for m in modules:
            m._diffusers_hook._set_context(None)


CKPT_ID = "black-forest-labs/FLUX.1-dev"
RESULT_FILENAME = "chebyshev_cache_control.csv"

# Placeholder prompt set — replace with the DrawBench prompts shipped in the
# reference repo (ChebBooster-FLUX/DrawBench.jsonl) for the full control run.
PROMPTS = [
    "A photograph of an astronaut riding a horse on Mars.",
    "A bowl of fruit on a wooden table, studio lighting.",
    "A cyberpunk city street at night in the rain.",
    "An oil painting of a lighthouse during a storm.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--cache-interval", type=int, default=5)
    parser.add_argument("--disable-cache-before-step", type=int, default=3)
    parser.add_argument("--cheb-order", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_pipeline():
    pipe = FluxPipeline.from_pretrained(CKPT_ID, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_arm(pipe, prompts, num_inference_steps, seed, cache_config=None):
    cached = cache_config is not None
    if cached:
        # Clear any cache left attached by a previous (possibly failed) arm.
        if getattr(pipe.transformer, "is_cache_enabled", False):
            pipe.transformer.disable_cache()
        pipe.transformer.enable_cache(cache_config)

    def _generate(prompt, i):
        generator = torch.Generator(device="cuda").manual_seed(seed + i)
        # Stateful cache hooks need a context, and per-prompt state (step counter
        # + feature history) must be reset so it does not carry across prompts.
        if cached:
            pipe.transformer._reset_stateful_cache()
            with _cache_context(pipe.transformer):
                return pipe(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
        return pipe(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]

    try:
        images = []
        start = time.perf_counter()
        for i, prompt in enumerate(prompts):
            images.append(_generate(prompt, i))
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) / len(prompts)
        return images, latency
    finally:
        # Always detach the cache, even on error, so the next arm starts clean.
        if cached and getattr(pipe.transformer, "is_cache_enabled", False):
            pipe.transformer.disable_cache()


def psnr_vs_reference(images, reference_images):
    values = []
    for image, reference in zip(images, reference_images):
        x = torch.tensor(list(image.getdata()), dtype=torch.float32) / 255.0
        y = torch.tensor(list(reference.getdata()), dtype=torch.float32) / 255.0
        mse = torch.mean((x - y) ** 2).item()
        values.append(float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item())
    return sum(values) / len(values)


def lpips_vs_reference(images, reference_images):
    try:
        import lpips
        import numpy as np
    except ImportError:
        return None

    loss_fn = lpips.LPIPS(net="vgg").to("cuda")
    values = []
    for image, reference in zip(images, reference_images):
        x = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float().to("cuda") / 127.5 - 1.0
        y = torch.from_numpy(np.array(reference)).permute(2, 0, 1)[None].float().to("cuda") / 127.5 - 1.0
        with torch.no_grad():
            values.append(loss_fn(x, y).item())
    return sum(values) / len(values)


def main():
    args = parse_args()
    prompts = PROMPTS[: args.num_prompts]
    shared_kwargs = {
        "cache_interval": args.cache_interval,
        "disable_cache_before_step": args.disable_cache_before_step,
    }

    pipe = load_pipeline()

    reference_images, reference_latency = run_arm(pipe, prompts, args.num_inference_steps, args.seed)

    taylorseer_images, taylorseer_latency = run_arm(
        pipe,
        prompts,
        args.num_inference_steps,
        args.seed,
        cache_config=TaylorSeerCacheConfig(max_order=1, **shared_kwargs),
    )

    chebyshev_images, chebyshev_latency = run_arm(
        pipe,
        prompts,
        args.num_inference_steps,
        args.seed,
        cache_config=ChebyshevCacheConfig(cheb_order=args.cheb_order, **shared_kwargs),
    )

    rows = [
        {
            "arm": "no-cache",
            "latency_s_per_image": reference_latency,
            "speedup": 1.0,
            "psnr_vs_reference": float("inf"),
            "lpips_vs_reference": 0.0,
        },
        {
            "arm": "taylorseer",
            "latency_s_per_image": taylorseer_latency,
            "speedup": reference_latency / taylorseer_latency,
            "psnr_vs_reference": psnr_vs_reference(taylorseer_images, reference_images),
            "lpips_vs_reference": lpips_vs_reference(taylorseer_images, reference_images),
        },
        {
            "arm": "chebyshev",
            "latency_s_per_image": chebyshev_latency,
            "speedup": reference_latency / chebyshev_latency,
            "psnr_vs_reference": psnr_vs_reference(chebyshev_images, reference_images),
            "lpips_vs_reference": lpips_vs_reference(chebyshev_images, reference_images),
        },
    ]

    with open(RESULT_FILENAME, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"\nResults written to {RESULT_FILENAME}")


if __name__ == "__main__":
    main()
