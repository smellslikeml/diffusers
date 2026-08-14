# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline


REPO_ROOT = Path(__file__).parents[2]
PIPELINE_PATH = REPO_ROOT / "examples" / "community" / "pipeline_flux_hrdit.py"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def _load_module(name, path, extra_sys_path=None):
    if extra_sys_path is not None and str(extra_sys_path) not in sys.path:
        sys.path.insert(0, str(extra_sys_path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hrdit = _load_module("pipeline_flux_hrdit", PIPELINE_PATH)


class BuildBundleIdVariantsTests(unittest.TestCase):
    def test_single_variant_at_trained_resolution(self):
        # Below the trained 64x64 packed grid, SPA must reduce to the stock Flux position ids.
        variants = hrdit.build_bundle_id_variants(32, 64, bundle_size=64, group_num=4)

        self.assertEqual(len(variants), 1)
        expected = FluxPipeline._prepare_latent_image_ids(1, 32, 64, None, None)
        self.assertTrue(torch.equal(variants[0], expected))

    def test_variants_wrap_into_trained_window(self):
        variants = hrdit.build_bundle_id_variants(128, 96, bundle_size=64, group_num=4)

        self.assertEqual(len(variants), 4)
        for variant in variants:
            self.assertEqual(variant.shape, (128 * 96, 3))
            self.assertLessEqual(variant[:, 1].max().item(), 63)
            self.assertLessEqual(variant[:, 2].max().item(), 63)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(torch.equal(variants[i], variants[j]))

    def test_first_variant_is_unshifted_partition(self):
        variants = hrdit.build_bundle_id_variants(128, 64, bundle_size=64, group_num=4)

        ys = (torch.arange(128) % 64).repeat_interleave(64)
        self.assertTrue(torch.equal(variants[0][:, 1], ys.to(variants[0].dtype)))


class UpsamplePackedLatentsTests(unittest.TestCase):
    def test_upsample_shape(self):
        latents = torch.randn(2, 16 * 16, 64)
        upsampled = hrdit.upsample_packed_latents(latents, (16, 16), (32, 32))

        self.assertEqual(upsampled.shape, (2, 32 * 32, 64))

    def test_upsample_is_identity_at_same_grid(self):
        latents = torch.randn(1, 8 * 8, 64)
        upsampled = hrdit.upsample_packed_latents(latents, (8, 8), (8, 8))

        self.assertTrue(torch.allclose(upsampled, latents, atol=1e-5))


class HeadScopeTests(unittest.TestCase):
    def test_scope_plan_round_robin(self):
        plan = hrdit.build_head_scope_plan(24, window=64, full_period=4)

        self.assertEqual(plan.shape, (24,))
        self.assertEqual((plan == -1).sum().item(), 6)
        self.assertEqual((plan == 64).sum().item(), 18)

    def test_mask_mod_respects_scopes(self):
        grid_height, grid_width, num_txt = 8, 8, 16
        num_img = grid_height * grid_width
        pos_h = torch.div(torch.arange(num_img), grid_width, rounding_mode="floor")
        pos_w = torch.arange(num_img) % grid_width
        windows = hrdit.build_head_scope_plan(8, window=2, full_period=4)
        mask_mod = hrdit.build_mask_mod(pos_h, pos_w, windows, num_txt)

        windowed_head = 1
        full_head = 0
        image_query = torch.tensor(num_txt)  # grid position (0, 0)
        near_key = torch.tensor(num_txt + 1)  # grid position (0, 1)
        far_key = torch.tensor(num_txt + 3 * grid_width)  # grid position (3, 0)
        text_key = torch.tensor(3)

        self.assertTrue(bool(mask_mod(0, windowed_head, image_query, text_key)))
        self.assertTrue(bool(mask_mod(0, windowed_head, image_query, near_key)))
        self.assertFalse(bool(mask_mod(0, windowed_head, image_query, far_key)))
        self.assertTrue(bool(mask_mod(0, full_head, image_query, far_key)))


class PipelineIntegrationTests(unittest.TestCase):
    def test_pipeline_subclasses_flux_pipeline(self):
        self.assertTrue(issubclass(hrdit.HRDiTFluxPipeline, FluxPipeline))

    def test_attention_processor_subclasses_stock_flux_processor(self):
        self.assertTrue(issubclass(hrdit.HRDiTFluxAttnProcessor, FluxAttnProcessor))

    def test_benchmark_wiring(self):
        benchmark = _load_module(
            "benchmarking_flux_hrdit", BENCHMARKS_DIR / "benchmarking_flux_hrdit.py", extra_sys_path=BENCHMARKS_DIR
        )

        self.assertEqual(benchmark.RESULT_FILENAME, "flux_hrdit.csv")
        self.assertEqual(benchmark.CKPT_ID, "black-forest-labs/FLUX.1-dev")
        self.assertTrue(callable(benchmark.run_benchmarks))
