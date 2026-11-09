<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AFloPSScheduler

`AFloPSScheduler` is a training-free, plug-and-play sampler for fast diffusion sampling introduced in
[A-FloPS: Adaptive Flow Path Sampler](https://arxiv.org/abs/2509.00036) (AAAI 2026).

A VP noise schedule (`x_t = alpha_t * x_0 + sigma_t * epsilon`) is reparameterized into a flow-matching
coordinate whose flow time linearly interpolates between pure noise and clean data. The inference timesteps are
chosen uniformly in flow time instead of VP time, and each update decomposes the flow velocity adaptively into a
per-sample linear drift and a residual that are integrated with a second-order step built from the two most recently
cached (sample, velocity) pairs. A first-order flow-Euler step is used to warm up, to finish, and with `mode=0`.

## AFloPSScheduler
[[autodoc]] AFloPSScheduler

## SchedulerOutput
[[autodoc]] schedulers.scheduling_utils.SchedulerOutput
