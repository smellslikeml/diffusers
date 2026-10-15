# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import (
    AnyFlowPipeline,
    AnyFlowTransformer3DModel,
    AutoencoderKLWan,
    FlowMapEulerDiscreteScheduler,
)
from diffusers.pipelines.anyflow import PhaseLockGuidance
from diffusers.pipelines.anyflow.phase_lock import apply_phase_lock, extract_motion_phase

from ...testing_utils import enable_full_determinism


enable_full_determinism()


def _dummy_components():
    torch.manual_seed(0)
    vae = AutoencoderKLWan(
        base_dim=3,
        z_dim=16,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
    )

    torch.manual_seed(0)
    scheduler = FlowMapEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
    text_encoder = T5EncoderModel(config)
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

    torch.manual_seed(0)
    transformer = AnyFlowTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=16,
        out_channels=16,
        text_dim=32,
        freq_dim=256,
        ffn_dim=32,
        num_layers=2,
        cross_attn_norm=True,
        rope_max_seq_len=32,
        gate_value=0.25,
        deltatime_type="r",
    )
    return {
        "transformer": transformer,
        "vae": vae,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    }


def _dummy_inputs(device, seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        "prompt": "dance monkey",
        "negative_prompt": "negative",
        "generator": generator,
        "num_inference_steps": 2,
        "guidance_scale": 6.0,
        "height": 16,
        "width": 16,
        "num_frames": 9,
        "max_sequence_length": 16,
        "output_type": "latent",
    }


class PhaseLockGuidanceUnitTests(unittest.TestCase):
    """Spectral guarantees of PhaseLock, exercised in isolation."""

    def test_zero_strength_is_identity(self):
        latents = torch.randn(1, 5, 16, 8, 8)
        prior_phase = extract_motion_phase(torch.randn_like(latents))
        out = apply_phase_lock(latents, prior_phase, lock_strength=0.0)
        self.assertTrue(torch.equal(out, latents))

    def test_full_strength_preserves_magnitude_and_snaps_phase(self):
        # At full strength the constructed spectrum stays conjugate-symmetric, so the
        # inverse FFT is exactly real: magnitude (fidelity) is preserved bit-for-bit
        # while the phase (motion) snaps onto the prior.
        latents = torch.randn(1, 5, 16, 8, 8)
        prior_phase = extract_motion_phase(torch.randn_like(latents))
        locked = apply_phase_lock(latents, prior_phase, lock_strength=1.0)

        mag_before = torch.abs(torch.fft.fftn(latents.float(), dim=(-4, -2, -1)))
        mag_after = torch.abs(torch.fft.fftn(locked.float(), dim=(-4, -2, -1)))
        self.assertTrue(torch.allclose(mag_before, mag_after, atol=1e-4))
        self.assertTrue(torch.allclose(extract_motion_phase(locked), prior_phase, atol=1e-4))

    def test_partial_lock_keeps_magnitude_stable_and_moves_phase(self):
        # PhaseLock's central claim: magnitude (visual fidelity) stays relatively stable
        # while the phase (motion) is steered toward the early-step prior.
        latents = torch.randn(1, 5, 16, 8, 8)
        prior_phase = extract_motion_phase(torch.randn_like(latents))
        locked = apply_phase_lock(latents, prior_phase, lock_strength=0.7)

        mag_before = torch.abs(torch.fft.fftn(latents.float(), dim=(-4, -2, -1)))
        mag_after = torch.abs(torch.fft.fftn(locked.float(), dim=(-4, -2, -1)))
        rel_mag_err = torch.norm(mag_after - mag_before) / torch.norm(mag_before)
        self.assertLess(rel_mag_err.item(), 0.1)

        # The locked phase must sit closer to the prior than the original did.
        def dist_to_prior(x):
            return torch.angle(torch.exp(1j * (prior_phase - extract_motion_phase(x)))).abs().mean()

        self.assertLess(dist_to_prior(locked).item(), dist_to_prior(latents).item())

    def test_guidance_captures_then_applies(self):
        guidance = PhaseLockGuidance(prior_step=0, lock_strength=0.5)
        latents = torch.randn(1, 5, 16, 8, 8)

        self.assertFalse(guidance.has_prior)
        captured = guidance(latents, step_index=0, num_inference_steps=4)
        self.assertTrue(guidance.has_prior)
        self.assertTrue(torch.equal(captured, latents))  # capture step is a no-op

        applied = guidance(torch.randn_like(latents), step_index=1, num_inference_steps=4)
        self.assertFalse(torch.allclose(applied, latents))


class AnyFlowPhaseLockIntegrationTests(unittest.TestCase):
    """PhaseLock wired through the real AnyFlow denoising loop call site."""

    def test_call_signature_exposes_phase_lock(self):
        sig = inspect.signature(AnyFlowPipeline.__call__)
        self.assertIn("phase_lock", sig.parameters)

    def test_pipeline_runs_and_phase_lock_alters_output(self):
        device = "cpu"
        pipe = AnyFlowPipeline(**_dummy_components()).to(device)
        pipe.set_progress_bar_config(disable=True)

        baseline = pipe(**_dummy_inputs(device), phase_lock=None).frames

        guided = pipe(
            **_dummy_inputs(device),
            phase_lock=PhaseLockGuidance(prior_step=0, lock_strength=0.8),
        ).frames

        self.assertEqual(guided.shape, baseline.shape)
        # The guidance fires after the scheduler step at the wired call site, so the
        # locked trajectory diverges from the untouched baseline by far more than the
        # pipeline's own CPU run-to-run noise (~1e-3).
        self.assertGreater((guided - baseline).abs().mean().item(), 1e-2)

    def test_zero_strength_is_a_noop_at_call_site(self):
        device = "cpu"
        pipe = AnyFlowPipeline(**_dummy_components()).to(device)
        pipe.set_progress_bar_config(disable=True)

        baseline = pipe(**_dummy_inputs(device), phase_lock=None).frames
        # Establish the pipeline's intrinsic run-to-run noise floor on this hardware.
        noise = (pipe(**_dummy_inputs(device), phase_lock=None).frames - baseline).abs().max().item()

        noop = pipe(
            **_dummy_inputs(device),
            phase_lock=PhaseLockGuidance(prior_step=0, lock_strength=0.0),
        ).frames
        # A zero-strength lock returns the latent untouched, so the output may only
        # differ from the baseline within that intrinsic noise.
        self.assertLessEqual((noop - baseline).abs().max().item(), max(noise * 2.0, 1e-3))


if __name__ == "__main__":
    unittest.main()
