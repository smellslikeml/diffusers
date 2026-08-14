"""Benchmark HRDiT progressive high-resolution generation against naive single-pass generation.

HRDiT (https://arxiv.org/abs/2608.07003) targets two failure modes of off-the-shelf DiT models at high
resolution: spatial disorder (fixed by SPA, which this benchmark runs with) and long generation time
(addressed by HAP pruning and by the progressive 1024 -> 2048 -> 4096 ladder). This script times the
end-to-end pipeline call for the naive single-pass baseline and for the HRDiT stages.

Run on a GPU machine with the FLUX.1-dev checkpoint available:

    python benchmarks/benchmarking_flux_hrdit.py --height 2048 --width 2048
"""

import argparse
from pathlib import Path

import torch

from benchmarking_utils import benchmark_fn, flush

from diffusers import FluxPipeline
from diffusers.utils.testing_utils import torch_device


CKPT_ID = "black-forest-labs/FLUX.1-dev"
CUSTOM_PIPELINE_PATH = str(Path(__file__).resolve().parents[1] / "examples" / "community" / "pipeline_flux_hrdit.py")
RESULT_FILENAME = "flux_hrdit.csv"


def load_pipeline():
    return FluxPipeline.from_pretrained(
        CKPT_ID,
        torch_dtype=torch.bfloat16,
        custom_pipeline=CUSTOM_PIPELINE_PATH,
    ).to(torch_device)


def run_benchmarks(height, width, num_inference_steps, group_num, use_hap):
    pipe = load_pipeline()

    settings = {
        # Single-pass generation straight at the target resolution: SPA/HAP disabled, one stage.
        "naive": dict(resolutions=[max(height, width)], group_num=1, use_hap=False),
        # HRDiT: progressive ladder from 1024 up, SPA bundle averaging, optional HAP pruning.
        "hrdit": dict(resolutions=None, group_num=group_num, use_hap=use_hap),
    }
    results = []
    for name, kwargs in settings.items():
        flush()
        latency = benchmark_fn(
            pipe,
            "a photo of a mountain lake at dawn",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            **kwargs,
        )
        max_memory = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
        results.append((name, latency, max_memory))
        print(f"{name:>6}: {latency:.3f}s, peak memory {max_memory:.2f} GiB")

    with open(RESULT_FILENAME, "w") as f:
        f.write("setting,latency_s,peak_memory_gib\n")
        for name, latency, max_memory in results:
            f.write(f"{name},{latency},{max_memory}\n")
    print(f"Results saved to {RESULT_FILENAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--group_num", type=int, default=4, help="number of SPA bundle variants")
    parser.add_argument("--no_hap", action="store_true", help="disable head-adaptive attention pruning")
    args = parser.parse_args()

    run_benchmarks(args.height, args.width, args.num_inference_steps, args.group_num, not args.no_hap)
