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

import unittest

import torch

from diffusers import ChebyshevCacheConfig, apply_chebyshev_cache
from diffusers.hooks.chebyshev_cache import _CHEBYSHEV_CACHE_HOOK, ChebyshevCacheState
from diffusers.models import ModelMixin
from diffusers.models.cache_utils import CacheMixin


class CountingBlock(torch.nn.Module):
    """Doubles its input and counts how many times it actually ran."""

    def __init__(self):
        super().__init__()
        self.compute_count = 0

    def forward(self, hidden_states, **kwargs):
        self.compute_count += 1
        return hidden_states * 2.0


class DummyTransformer(ModelMixin, CacheMixin):
    # CacheMixin provides enable_cache / disable_cache / cache_context — real
    # model classes (e.g. FluxTransformer2DModel) inherit both mixins, and the
    # schedule tests below drive caching through that public API.
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([CountingBlock()])

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class ChebyshevCacheScheduleTests(unittest.TestCase):
    """
    Warmup/refresh/cooldown boundary tests for the Chebyshev cache schedule.

    TaylorSeer needed a follow-up fix (huggingface/diffusers#13806) for an off-by-one exactly at the
    warmup -> prediction boundary, so these tests pin down the full expected schedule step by step.
    """

    def _get_model_and_config(self, **config_overrides):
        model = DummyTransformer()
        config_kwargs = {
            "cache_interval": 3,
            "disable_cache_before_step": 2,
            "cheb_order": 4,
            "cache_identifiers": ["^transformer_blocks.*"],
        }
        config_kwargs.update(config_overrides)
        config = ChebyshevCacheConfig(**config_kwargs)
        apply_chebyshev_cache(model, config)
        return model, config

    def _run_steps(self, model, num_steps):
        block = model.transformer_blocks[0]
        compute_steps = []
        with model.cache_context("schedule_test"):
            for step in range(num_steps):
                before = block.compute_count
                output = model(torch.randn(1, 4))
                assert not torch.isnan(output).any(), f"NaN output at step {step}."
                if block.compute_count > before:
                    compute_steps.append(step)
        return compute_steps

    def test_warmup_refresh_schedule(self):
        model, _ = self._get_model_and_config()
        compute_steps = self._run_steps(model, num_steps=10)

        # cache_interval=3, disable_cache_before_step=2:
        # - steps 0, 1: warmup, always full compute
        # - step 2 (== disable_cache_before_step): FIRST prediction step, using the warmup history;
        #   this is the boundary that was off-by-one in the TaylorSeer follow-up
        # - steps 3, 6, 9: periodic refresh, (step - disable_cache_before_step - 1) % cache_interval == 0
        # - all other steps: predicted
        self.assertEqual(compute_steps, [0, 1, 3, 6, 9])

    def test_prediction_at_warmup_boundary_uses_partial_window(self):
        # At the first prediction step the history window only holds the warmup steps
        # (fewer nodes than cheb_order); prediction must still work.
        model, _ = self._get_model_and_config(cheb_order=6, disable_cache_before_step=2)
        block = model.transformer_blocks[0]

        with model.cache_context("boundary_test"):
            model(torch.ones(1, 4))  # step 0: warmup compute
            model(2.0 * torch.ones(1, 4))  # step 1: warmup compute
            self.assertEqual(block.compute_count, 2)

            output = model(3.0 * torch.ones(1, 4))  # step 2: boundary prediction with n=2 nodes
            self.assertEqual(block.compute_count, 2, "Step at disable_cache_before_step should be predicted.")

        # Linear extrapolation from (step 0 -> 2) to (step 1 -> 4) evaluated at step 2 gives 6.
        self.assertTrue(torch.allclose(output, 6.0 * torch.ones(1, 4), atol=1e-5))

    def test_cooldown_schedule(self):
        model, _ = self._get_model_and_config(disable_cache_after_step=5)
        compute_steps = self._run_steps(model, num_steps=8)

        # steps 0, 1: warmup; step 2: predict; step 3: refresh; step 4: predict;
        # steps 5, 6, 7: cooldown (>= disable_cache_after_step), always full compute
        self.assertEqual(compute_steps, [0, 1, 3, 5, 6, 7])

    def test_hooks_registered_and_removed(self):
        model = DummyTransformer()
        config = ChebyshevCacheConfig(cache_identifiers=["^transformer_blocks.*"])
        model.enable_cache(config)

        block = model.transformer_blocks[0]
        self.assertIsNotNone(block._diffusers_hook.get_hook(_CHEBYSHEV_CACHE_HOOK))

        model.disable_cache()
        self.assertIsNone(block._diffusers_hook.get_hook(_CHEBYSHEV_CACHE_HOOK))


class ChebyshevCacheStateTests(unittest.TestCase):
    """Unit tests for the barycentric Chebyshev math in ChebyshevCacheState."""

    def _make_state(self, cheb_order=6):
        return ChebyshevCacheState(cheb_factors_dtype=torch.float32, cheb_order=cheb_order)

    def _update(self, state, step, value):
        state.current_step = step
        state.update((value * torch.ones(2, 2),))

    def test_single_node_history_returns_cached_feature(self):
        state = self._make_state()
        self._update(state, step=0, value=3.0)
        state.current_step = 2
        (output,) = state.predict()
        self.assertTrue(torch.allclose(output, 3.0 * torch.ones(2, 2)))

    def test_exact_node_guard(self):
        # Predicting at a step that coincides with a history node must return that node's
        # feature exactly (the reference's one-hot guard, cheb_utils L196-216).
        state = self._make_state()
        for step, value in [(0, 1.0), (2, 5.0), (4, 9.0)]:
            self._update(state, step=step, value=value)
        state.current_step = 2
        (output,) = state.predict()
        self.assertTrue(torch.allclose(output, 5.0 * torch.ones(2, 2), atol=1e-5))

    def test_constant_features_reproduced(self):
        # The barycentric weights form a partition of unity, so constant features are
        # reproduced exactly at any target step.
        state = self._make_state()
        for step in [0, 1, 2, 3]:
            self._update(state, step=step, value=7.0)
        state.current_step = 5
        (output,) = state.predict()
        self.assertTrue(torch.allclose(output, 7.0 * torch.ones(2, 2), atol=1e-5))

    def test_linear_features_reproduced_with_two_nodes(self):
        # With two nodes the Chebyshev barycentric weights coincide with the true barycentric
        # weights, so the linear interpolant/extrapolant is exact.
        state = self._make_state()
        self._update(state, step=0, value=1.0)  # f(s) = 2s + 1
        self._update(state, step=1, value=3.0)
        state.current_step = 4
        (output,) = state.predict()
        self.assertTrue(torch.allclose(output, 9.0 * torch.ones(2, 2), atol=1e-5))

    def test_weights_match_reference_implementation(self):
        # Direct comparison against the weight-table computation of the Apache-2.0 reference
        # (ChebBooster-FLUX, src/flux/cheb_utils/__init__.py: chebyshev_barycentric_weights L13-35,
        # precompute_chebyshev_weights L161-220), re-implemented inline here for the test.
        history_steps = [1, 3, 4, 7]
        target_step = 5

        s_min, s_max = history_steps[0], history_steps[-1]
        n = len(history_steps)
        bary_w = torch.ones(n, dtype=torch.float64)
        bary_w[1::2] *= -1.0
        bary_w[0] *= 0.5
        bary_w[-1] *= 0.5
        x_map = 2.0 * (torch.tensor(history_steps, dtype=torch.float64) - s_min) / (s_max - s_min) - 1.0
        x_target = 2.0 * (float(target_step) - s_min) / (s_max - s_min) - 1.0
        x_diff = x_target - x_map
        w_over_diff = bary_w / x_diff
        expected = w_over_diff / w_over_diff.sum()

        actual = ChebyshevCacheState._barycentric_weights(history_steps, target_step, torch.device("cpu"))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12))
        self.assertAlmostEqual(actual.sum().item(), 1.0, places=12)

    def test_history_window_capped_at_cheb_order(self):
        state = self._make_state(cheb_order=3)
        for step in range(6):
            self._update(state, step=step, value=float(step))
        history = state.feature_history[0]
        self.assertEqual(len(history), 3)
        self.assertEqual([step for step, _ in history], [3, 4, 5])

    def test_update_at_same_step_raises(self):
        state = self._make_state()
        self._update(state, step=0, value=1.0)
        with self.assertRaises(ValueError):
            self._update(state, step=0, value=2.0)

    def test_predict_before_update_raises(self):
        state = self._make_state()
        state.current_step = 0
        with self.assertRaises(ValueError):
            state.predict()

    def test_reset_clears_state(self):
        state = self._make_state()
        self._update(state, step=0, value=1.0)
        state.reset()
        self.assertEqual(state.current_step, -1)
        self.assertEqual(state.feature_history, {})

    def test_prediction_accumulates_in_factor_dtype_not_module_dtype(self):
        # Features arrive in bf16 (the FLUX.1-dev setting) while
        # cheb_factors_dtype=float32. The weighted sum must accumulate in
        # float32 and cast to the module dtype only at the end — accumulating
        # in bf16 amplifies the alternating-sign cancellation in the
        # barycentric weights and defeats the stability rationale.
        torch.manual_seed(0)
        state = ChebyshevCacheState(cheb_factors_dtype=torch.float32, cheb_order=6)
        steps = [0, 3, 6, 9]
        feats = {s: (torch.randn(1024) * 40.0).to(torch.bfloat16) for s in steps}
        for s in steps:
            state.current_step = s
            state.update((feats[s],))
        state.current_step = 7
        (out,) = state.predict()
        self.assertEqual(out.dtype, torch.bfloat16, "output must be cast back to the module dtype")

        weights = ChebyshevCacheState._barycentric_weights(steps, 7, torch.device("cpu"))
        gt = sum(feats[s].double() * w for s, w in zip(steps, weights))
        # Differential check: the implementation must be closer to the high-
        # precision truth than a bf16-accumulated sum would be.
        acc_bf16 = torch.zeros(1024, dtype=torch.bfloat16)
        for s, w in zip(steps, weights):
            acc_bf16 = acc_bf16 + feats[s].to(torch.bfloat16) * w.to(torch.bfloat16)
        err_impl = (out.double() - gt).abs().mean().item()
        err_bf16 = (acc_bf16.double() - gt).abs().mean().item()
        self.assertLess(err_impl, err_bf16, f"impl err {err_impl} not better than bf16-accum {err_bf16}")

    def test_predict_before_any_compute_falls_back_to_full_forward(self):
        # disable_cache_before_step=0 must NOT crash: with no history recorded,
        # has_recorded() is False so the hook forces a full compute at step 0.
        model = DummyTransformer()
        config = ChebyshevCacheConfig(
            cache_interval=3,
            disable_cache_before_step=0,
            cache_identifiers=["^transformer_blocks.*"],
        )
        apply_chebyshev_cache(model, config)
        block = model.transformer_blocks[0]
        with model.cache_context("zero_warmup"):
            out = model(torch.ones(1, 4))  # step 0 — must compute, not predict
        self.assertEqual(block.compute_count, 1)
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
