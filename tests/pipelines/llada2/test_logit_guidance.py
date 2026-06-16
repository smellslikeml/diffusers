# Copyright 2025 HuggingFace Inc.
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

"""Tests for training-free reward guidance (GILC) wired into the LLaDA2 pipeline.

The integration tests drive the real `LLaDA2Pipeline.__call__` (a non-new module)
so the wiring edit at the refinement-loop call site is actually exercised.
"""

import inspect
import unittest

import torch

from diffusers import BlockRefinementScheduler, LLaDA2Pipeline
from diffusers.pipelines.llada2 import RewardLogitGuidance


class _DummyModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _DummyCausalLM(torch.nn.Module):
    """Position-dependent logits so unguided top-k commits are deterministic."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.register_buffer("_device_anchor", torch.empty(0))

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return self._device_anchor.device

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), device=input_ids.device, dtype=torch.float32)
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.float32).view(1, seq_len, 1)
        token_ids = (torch.arange(seq_len, device=input_ids.device) % (self.vocab_size - 2)).view(1, seq_len, 1)
        logits.scatter_(2, token_ids.expand(batch_size, -1, -1), 1.0 + positions.expand(batch_size, -1, -1) * 0.1)
        return _DummyModelOutput(logits=logits)


def _make_pipeline():
    return LLaDA2Pipeline(model=_DummyCausalLM(vocab_size=32), scheduler=BlockRefinementScheduler())


_GEN_KWARGS = {
    "use_chat_template": False,
    "gen_length": 24,
    "block_length": 8,
    "num_inference_steps": 8,
    "temperature": 0.0,
    "threshold": 2.0,  # force top-k commits
    "minimal_topk": 1,
    "editing_threshold": 0.0,  # disable editing for a deterministic comparison
    "eos_early_stop": False,
    "mask_token_id": 31,
    "eos_token_id": None,
    "output_type": "seq",
}


class RewardLogitGuidanceUnitTest(unittest.TestCase):
    def test_shape_and_dtype_preserved(self):
        logits = torch.randn(2, 5, 7)
        guidance = RewardLogitGuidance(torch.zeros(7), guidance_scale=1.0)
        out = guidance(logits)
        self.assertEqual(out.shape, logits.shape)
        self.assertEqual(out.dtype, logits.dtype)

    def test_zero_scale_is_noop(self):
        logits = torch.randn(1, 3, 6)
        guidance = RewardLogitGuidance(torch.arange(6, dtype=torch.float32), guidance_scale=0.0)
        self.assertTrue(torch.equal(guidance(logits), logits))

    def test_gradient_correction_raises_favored_logit(self):
        # Uniform logits -> the correction direction is exactly the centered reward.
        logits = torch.zeros(1, 1, 4)
        reward = torch.tensor([0.0, 0.0, 10.0, 0.0])
        out = RewardLogitGuidance(reward, guidance_scale=1.0, mode="gradient")(logits)
        delta = (out - logits).reshape(-1)
        self.assertEqual(int(delta.argmax()), 2)
        self.assertGreater(delta[2].item(), 0.0)

    def test_tilt_matches_logprob_shift(self):
        # In tilt mode the distribution equals softmax(logits + scale * reward).
        logits = torch.randn(1, 2, 5)
        reward = torch.randn(5)
        out = RewardLogitGuidance(reward, guidance_scale=0.8, mode="tilt")(logits)
        got = torch.softmax(out, dim=-1)
        expected = torch.softmax(logits + 0.8 * reward, dim=-1)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_active_only_skips_committed_positions(self):
        logits = torch.zeros(1, 2, 4)
        tokens = torch.tensor([[31, 5]])  # position 0 masked, position 1 committed
        guidance = RewardLogitGuidance(torch.tensor([0.0, 9.0, 0.0, 0.0]), guidance_scale=1.0)
        out = guidance(logits, tokens=tokens, mask_token_id=31)
        self.assertTrue((out[0, 0] != logits[0, 0]).any())  # masked position corrected
        self.assertTrue(torch.equal(out[0, 1], logits[0, 1]))  # committed position untouched

    def test_from_token_rewards(self):
        guidance = RewardLogitGuidance.from_token_rewards({3: 5.0, 1: -2.0}, vocab_size=6)
        reward = guidance.reward
        self.assertEqual(reward.shape, (6,))
        self.assertEqual(reward[3].item(), 5.0)
        self.assertEqual(reward[1].item(), -2.0)
        self.assertEqual(reward[0].item(), 0.0)


class RewardLogitGuidancePipelineTest(unittest.TestCase):
    def test_call_signature_exposes_logit_guidance(self):
        # The wiring edit must surface the new parameter on the public pipeline.
        params = inspect.signature(LLaDA2Pipeline.__call__).parameters
        self.assertIn("logit_guidance", params)

    def test_guidance_steers_committed_tokens(self):
        target_token = 10
        input_ids = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.long)

        unguided = _make_pipeline().to("cpu")(input_ids=input_ids, **_GEN_KWARGS).sequences

        # `tilt` mode is the direct log-prob shift, which reliably overrides the model's prior.
        guidance = RewardLogitGuidance.from_token_rewards(
            {target_token: 100.0}, vocab_size=32, guidance_scale=1.0, mode="tilt"
        )
        guided = _make_pipeline().to("cpu")(input_ids=input_ids, logit_guidance=guidance, **_GEN_KWARGS).sequences

        guided_hits = int((guided == target_token).sum())
        unguided_hits = int((unguided == target_token).sum())

        # Guidance should dominate the committed tokens and clearly beat the unguided baseline.
        self.assertGreater(guided_hits, unguided_hits)
        self.assertGreaterEqual(guided_hits, int(0.9 * guided.numel()))

    def test_callable_guidance_is_invoked(self):
        # A plain callable (logits -> logits) is also accepted at the call site.
        calls = {"n": 0}

        def reward_fn(block_logits):
            calls["n"] += 1
            bias = torch.zeros(block_logits.shape[-1])
            bias[3] = 50.0
            return block_logits + bias

        input_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        out = _make_pipeline().to("cpu")(input_ids=input_ids, logit_guidance=reward_fn, **_GEN_KWARGS).sequences

        self.assertGreater(calls["n"], 0)
        self.assertEqual(out.shape, (1, 24))


if __name__ == "__main__":
    unittest.main()
