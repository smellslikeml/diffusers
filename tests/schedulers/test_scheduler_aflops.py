import tempfile

import torch

from diffusers import AFloPSScheduler

from .test_schedulers import SchedulerCommonTest


class AFloPSSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (AFloPSScheduler,)
    forward_default_kwargs = (("num_inference_steps", 25),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "solver_order": 2,
            "mode": 1,
            "c_max": 1.0,
            "prediction_type": "epsilon",
            "final_sigmas_type": "zero",
        }

        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.10]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)
            # copy over dummy past residuals
            scheduler.model_outputs = dummy_past_residuals[: scheduler.config.solver_order]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                new_scheduler.set_timesteps(num_inference_steps)
                # copy over dummy past residuals
                new_scheduler.model_outputs = dummy_past_residuals[: new_scheduler.config.solver_order]

            output, new_output = sample, sample
            for t in range(time_step, time_step + scheduler.config.solver_order + 1):
                t = scheduler.timesteps[t]
                output = scheduler.step(residual, t, output, **kwargs).prev_sample
                new_output = new_scheduler.step(residual, t, new_output, **kwargs).prev_sample

                assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.10]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            # copy over dummy past residuals (must be after setting timesteps)
            scheduler.model_outputs = dummy_past_residuals[: scheduler.config.solver_order]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                # copy over dummy past residuals
                new_scheduler.set_timesteps(num_inference_steps)
                new_scheduler.model_outputs = dummy_past_residuals[: new_scheduler.config.solver_order]

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        return sample

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            time_step_0 = scheduler.timesteps[5]
            time_step_1 = scheduler.timesteps[6]

            output_0 = scheduler.step(residual, time_step_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, time_step_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [25, 50, 100, 999, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_timesteps_are_uniform_in_flow_time(self):
        scheduler = self.scheduler_classes[0](**self.get_scheduler_config())
        scheduler.set_timesteps(10)

        # the schedule runs from the noisy end (flow time 0) to the clean end and every timestep is the
        # training timestep whose flow time is closest to the uniform grid linspace(0, 1, 10, endpoint=False)
        self.assertTrue(torch.all(scheduler.timesteps[:-1] >= scheduler.timesteps[1:]))

        floweq_all = scheduler.alpha_t / (scheduler.alpha_t + scheduler.sigma_t)
        flow_grid = torch.linspace(0, 1, 11)[:-1]
        nearest = (floweq_all[:, None] - flow_grid[None, :]).abs().argmin(dim=0)
        self.assertTrue(torch.all(nearest == scheduler.timesteps))

        # the final sigma appended by `final_sigmas_type="zero"` is the clean-data end of the flow
        self.assertEqual(scheduler.floweqs[-1].item(), 1.0)

    def test_inference_steps(self):
        for num_inference_steps in [1, 2, 3, 5, 10, 25, 50, 100, 999, 1000]:
            self.check_over_forward(num_inference_steps=num_inference_steps, time_step=0)

    def test_solver_order_and_mode(self):
        for order in [1, 2]:
            for mode in [0, 1]:
                self.check_over_configs(solver_order=order, mode=mode)
                sample = self.full_loop(solver_order=order, mode=mode)
                assert not torch.isnan(sample).any(), "Samples have nan numbers"

        # mode=0 always uses the first-order flow-Euler update, so it matches solver_order=1
        self.assertTrue(torch.allclose(self.full_loop(solver_order=2, mode=0), self.full_loop(solver_order=1, mode=1)))

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "sample", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_c_max(self):
        for c_max in [0.0, 0.5, 1.0]:
            sample = self.full_loop(c_max=c_max)
            assert not torch.isnan(sample).any(), "Samples have nan numbers"

        # clamping the linear drift to 0 changes the second-order update
        self.assertFalse(torch.allclose(self.full_loop(c_max=0.0), self.full_loop(c_max=1.0)))

    def test_final_sigmas_type(self):
        for final_sigmas_type in ["zero", "sigma_min"]:
            self.check_over_configs(final_sigmas_type=final_sigmas_type)
            sample = self.full_loop(final_sigmas_type=final_sigmas_type)
            assert not torch.isnan(sample).any(), "Samples have nan numbers"

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 0.2726) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 0.1021) < 1e-3

    def test_full_loop_without_cfg_pairing(self):
        # the linear drift coefficient is estimated per sample, so batches that are not CFG-doubled
        # (here: an odd batch size) sample normally
        scheduler_class = self.scheduler_classes[0]
        scheduler = scheduler_class(**self.get_scheduler_config())
        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter[:3]

        scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        assert not torch.isnan(sample).any(), "Samples have nan numbers"
        assert not torch.isinf(sample).any(), "Samples have inf numbers"

    def test_fp16_support(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter.half()
        scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        assert sample.dtype == torch.float16
