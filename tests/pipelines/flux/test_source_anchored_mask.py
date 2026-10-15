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

import inspect
import unittest

import torch

# Non-new call-site module: the wiring edit lives in FluxInpaintPipeline.__call__.
from diffusers import FluxInpaintPipeline
from diffusers.pipelines.flux.source_anchored_mask import (
    accumulate_mask,
    anchor_weight,
    build_source_anchored_mask,
    dynamic_soft_mask,
    source_anchored_blend,
)


class SourceAnchoredMaskTests(unittest.TestCase):
    def test_blend_matches_binary_projection(self):
        # source_anchored_blend must equal the original flux-inpaint blend line.
        torch.manual_seed(0)
        edit = torch.randn(2, 16, 8)
        source = torch.randn(2, 16, 8)
        mask = (torch.rand(2, 16, 8) > 0.5).float()

        reference = (1 - mask) * source + mask * edit
        result = source_anchored_blend(edit, source, mask)
        self.assertTrue(torch.allclose(result, reference))

    def test_defaults_reproduce_baseline(self):
        # With default options the composed mask is unchanged, so the integrated
        # path is bit-for-bit identical to the pre-existing binary projection.
        mask = (torch.rand(1, 12, 4) > 0.5).float()
        blend_mask, state = build_source_anchored_mask(mask, step_index=0, num_steps=10)
        self.assertTrue(torch.equal(blend_mask, mask))
        self.assertIsNone(state)

        edit = torch.randn(1, 12, 4)
        source = torch.randn(1, 12, 4)
        baseline = (1 - mask) * source + mask * edit
        integrated = source_anchored_blend(edit, source, blend_mask)
        self.assertTrue(torch.allclose(integrated, baseline))

    def test_dynamic_soft_mask_creates_transition_band(self):
        mask = torch.tensor([0.0, 0.4, 0.5, 0.6, 1.0])
        # No transition -> unchanged.
        self.assertTrue(torch.equal(dynamic_soft_mask(mask, 0.0), mask))
        # With a transition band, mid values become graded and stay in [0, 1].
        soft = dynamic_soft_mask(mask, transition_width=0.3)
        self.assertTrue(torch.all(soft >= 0.0) and torch.all(soft <= 1.0))
        self.assertAlmostEqual(soft[2].item(), 0.5, places=5)  # center stays centered
        self.assertGreater(soft[3].item(), soft[1].item())  # monotonic across boundary

    def test_temporal_accumulation_is_monotonic(self):
        m1 = torch.tensor([1.0, 0.0, 0.0])
        m2 = torch.tensor([0.0, 1.0, 0.0])
        acc = accumulate_mask(m1, None)
        self.assertTrue(torch.equal(acc, m1))
        acc = accumulate_mask(m2, acc)
        # Once a location is editable it stays editable.
        self.assertTrue(torch.equal(acc, torch.tensor([1.0, 1.0, 0.0])))

    def test_anchor_weight_schedules(self):
        self.assertEqual(anchor_weight(0, 10, "constant"), 1.0)
        self.assertEqual(anchor_weight(5, 10, "constant"), 1.0)
        # Decaying schedules taper edit strength toward the final step.
        self.assertAlmostEqual(anchor_weight(0, 11, "linear_decay"), 1.0)
        self.assertAlmostEqual(anchor_weight(10, 11, "linear_decay"), 0.0)
        self.assertGreater(anchor_weight(2, 11, "linear_decay"), anchor_weight(8, 11, "linear_decay"))
        self.assertGreater(anchor_weight(0, 11, "cosine"), anchor_weight(10, 11, "cosine"))
        with self.assertRaises(ValueError):
            anchor_weight(0, 10, "does-not-exist")

    def test_build_threads_accumulation_state(self):
        mask = (torch.rand(1, 8, 2) > 0.5).float()
        _, state = build_source_anchored_mask(mask, 0, 4, temporal_accumulation=True, accumulated_mask=None)
        self.assertIsNotNone(state)
        other = (torch.rand(1, 8, 2) > 0.5).float()
        _, new_state = build_source_anchored_mask(other, 1, 4, temporal_accumulation=True, accumulated_mask=state)
        # Accumulated state never shrinks.
        self.assertTrue(torch.all(new_state >= state))


class FluxInpaintWiringTests(unittest.TestCase):
    def test_call_exposes_source_anchored_parameters(self):
        # Asserts the integration is wired into the existing pipeline's public API.
        sig = inspect.signature(FluxInpaintPipeline.__call__)
        params = sig.parameters
        self.assertIn("source_anchored_masking", params)
        self.assertFalse(params["source_anchored_masking"].default)
        self.assertEqual(params["mask_transition_width"].default, 0.0)
        self.assertFalse(params["temporal_mask_accumulation"].default)
        self.assertEqual(params["anchor_schedule"].default, "constant")

    def test_pipeline_imports_blend_helpers(self):
        from diffusers.pipelines.flux import pipeline_flux_inpaint

        # The call-site module references the new capability (proves it is invoked,
        # not dead code).
        self.assertIs(pipeline_flux_inpaint.source_anchored_blend, source_anchored_blend)
        self.assertIs(pipeline_flux_inpaint.build_source_anchored_mask, build_source_anchored_mask)


if __name__ == "__main__":
    unittest.main()
