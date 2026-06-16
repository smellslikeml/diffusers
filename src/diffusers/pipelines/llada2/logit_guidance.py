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

"""Training-free, plug-and-play reward guidance for discrete-diffusion logits.

This implements a focused slice of *Gradient-Informed Logit Correction* (GILC),
"Plug-and-Play Guidance for Discrete Diffusion Models via Gradient-Informed Logit
Correction" (https://arxiv.org/abs/2606.06303). GILC steers a discrete-diffusion
sampler toward a reward without any retraining by correcting the clean-prediction
logits in place: logits in -> reward-guided logits out, same shape.

The correction is *Jacobian-free*: the reward function is evaluated on the
candidate-token axis directly (so it may be non-differentiable), and the guidance
signal is the analytic gradient of the expected reward with respect to the logits
under the model's own clean-prediction distribution. No gradient is taken through
the denoising network, which is what makes the step stable in the high-dimensional
discrete space.

For clean-prediction logits ``l`` with ``p = softmax(l)`` and a per-candidate
reward ``r`` (a vector over the vocabulary, broadcast over batch/position):

- ``mode="gradient"`` (default) takes one gradient-ascent step on the expected
  reward ``E_p[r] = sum_v p_v r_v``. Its gradient w.r.t. logit ``l_v`` is
  ``p_v (r_v - E_p[r])``, so the corrected logit is
  ``l_v + guidance_scale * p_v * (r_v - E_p[r])``.
- ``mode="tilt"`` applies the exponential reward-tilt ``p'(v) ∝ p(v) exp(g r_v)``,
  i.e. ``l_v + guidance_scale * (r_v - E_p[r])`` (centering only re-scales, since
  softmax is shift-invariant).

Both forms are training-free and accept differentiable or non-differentiable
rewards. Out of scope here (and not needed for the in-pipeline value) is GILC's
multi-step variational proxy that re-runs the denoiser to score the *fully*
denoised sequence, plus the paper's domain reward models (DNA / protein /
molecule). The in-place logit correction is the reusable core.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch


class RewardLogitGuidance:
    """Reward-guided logit correction for the LLaDA2 block-refinement loop.

    An instance is callable as ``guidance(logits) -> corrected_logits`` and is
    meant to be applied to the clean-prediction logits of a refinement step, just
    before they are handed to the scheduler. The output keeps the input shape and
    dtype, so it is a drop-in correction that changes no other contract.

    Args:
        reward (`torch.Tensor` or `Callable`):
            Per-candidate reward. Either a tensor broadcastable to the logits
            (e.g. shape `[vocab]`, `[1, 1, vocab]`, or the full `[batch, seq,
            vocab]`), or a callable mapping the logits to such a tensor. Larger
            reward favors the corresponding token. May be non-differentiable.
        guidance_scale (`float`, defaults to `1.0`):
            Strength of the correction. `0.0` is a no-op (unguided generation).
        mode (`str`, defaults to `"gradient"`):
            `"gradient"` for the GILC gradient-of-expected-reward step, or
            `"tilt"` for the exponential reward-tilt.
        active_only (`bool`, defaults to `True`):
            When the still-masked token positions are known (passed as `tokens` /
            `mask_token_id`), restrict the correction to those positions so
            already-committed context is left untouched.
    """

    def __init__(
        self,
        reward: "torch.Tensor | Callable[[torch.Tensor], torch.Tensor]",
        guidance_scale: float = 1.0,
        mode: str = "gradient",
        active_only: bool = True,
    ):
        if mode not in {"gradient", "tilt"}:
            raise ValueError(f"`mode` must be 'gradient' or 'tilt', got {mode!r}.")
        if not callable(reward) and not torch.is_tensor(reward):
            raise TypeError("`reward` must be a torch.Tensor or a callable returning one.")
        self.reward = reward
        self.guidance_scale = float(guidance_scale)
        self.mode = mode
        self.active_only = active_only

    @classmethod
    def from_token_rewards(
        cls,
        token_rewards: "Mapping[int, float] | torch.Tensor",
        vocab_size: int,
        default: float = 0.0,
        **kwargs,
    ) -> "RewardLogitGuidance":
        """Build guidance from a sparse map of `token_id -> reward`.

        Convenient for length / syntax control, e.g. boosting an end-of-sequence
        token to favor shorter outputs, or penalizing a set of disallowed tokens.
        """
        if torch.is_tensor(token_rewards):
            reward = token_rewards.to(dtype=torch.float32).reshape(-1)
            if reward.numel() != vocab_size:
                raise ValueError(
                    f"`token_rewards` has {reward.numel()} entries but `vocab_size` is {vocab_size}."
                )
        else:
            reward = torch.full((vocab_size,), float(default), dtype=torch.float32)
            for token_id, value in token_rewards.items():
                if not 0 <= int(token_id) < vocab_size:
                    raise ValueError(f"token id {token_id} out of range for vocab_size={vocab_size}.")
                reward[int(token_id)] = float(value)
        return cls(reward, **kwargs)

    def _reward_tensor(self, logits: torch.Tensor) -> torch.Tensor:
        reward = self.reward(logits) if callable(self.reward) else self.reward
        if not torch.is_tensor(reward):
            raise TypeError("Reward callable must return a torch.Tensor.")
        reward = reward.to(device=logits.device, dtype=torch.float32)
        # Broadcast-check against the logits shape; let torch raise on a real mismatch.
        return reward.expand_as(logits) if reward.shape != logits.shape else reward

    def __call__(
        self,
        logits: torch.Tensor,
        *,
        tokens: torch.Tensor | None = None,
        mask_token_id: int | None = None,
    ) -> torch.Tensor:
        """Return reward-corrected logits with the same shape and dtype as `logits`."""
        if self.guidance_scale == 0.0:
            return logits
        if logits.ndim != 3:
            raise ValueError(f"`logits` must be `[batch, seq, vocab]`, got shape {tuple(logits.shape)}.")

        reward = self._reward_tensor(logits)
        probs = torch.softmax(logits.float(), dim=-1)
        expected = (probs * reward).sum(dim=-1, keepdim=True)
        centered = reward - expected

        if self.mode == "gradient":
            correction = self.guidance_scale * probs * centered
        else:  # "tilt"
            correction = self.guidance_scale * centered

        if self.active_only and tokens is not None and mask_token_id is not None:
            active = (tokens == mask_token_id).unsqueeze(-1)
            correction = torch.where(active, correction, torch.zeros_like(correction))

        return logits + correction.to(logits.dtype)
