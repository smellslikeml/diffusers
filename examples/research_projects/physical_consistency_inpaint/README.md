# Physical-consistency losses for diffusion inpainting

Adapted from **"Structured-Prior-Guided Diffusion Inpainting with Physical
Consistency for Traffic Sign Augmentation"**
(https://arxiv.org/abs/2609.02348).

General-purpose inpainting models shift colours and deform edge structure when
applied to physically-composed objects (traffic signs, license plates, digits).
The paper adds two parameter-free, differentiable pixel-space losses on top of a
standard Stable Diffusion 1.5 inpainting backbone to hold those quantities
fixed:

* **CIELAB chromaticity L1** — penalises hue/saturation drift in a
  perceptually-uniform colour space (the `a*`, `b*` channels), while staying
  indifferent to legitimate lightness changes.
* **Sobel gradient L1** — penalises edge/structure drift so digit strokes and
  geometric outlines stay aligned.

## What this project ports

Only the two **physical-consistency loss terms** are ported here, at full
fidelity, as a drop-in `diffusers`-native module. They operate on the decoded
RGB prediction and plug into any existing inpainting reconstruction loop after
the VAE decode, optionally restricted to the inpainted region via a mask.

The paper's three **structured-prior conditioning pathways** — a JSON-formatted
text prompt (semantic), an IP-Adapter front-view template (appearance) and a
ControlNet affine template (geometric) — are **out of scope** for this module:
they are orthogonal conditioning inputs that `diffusers` already exposes through
its existing IP-Adapter and ControlNet interfaces, and can be layered on
separately. The in-house AMAP training set and downstream detection benchmark
are likewise not reproduced.

## Usage

```python
from diffusers import AutoencoderKL

from physical_consistency import (
    PhysicalConsistencyLoss,
    reconstruction_loss_with_physical_consistency,
)

physical_loss = PhysicalConsistencyLoss(lambda_cielab=1.0, lambda_sobel=0.5)

# Inside your inpainting training step, alongside the usual latent MSE:
losses = reconstruction_loss_with_physical_consistency(
    pipeline.vae,
    predicted_x0_latents,   # model prediction, decoded internally
    target_latents,         # ground-truth latents
    mask=inpaint_mask,      # latent-resolution mask of the edited region
    physical_loss=physical_loss,
)
total = latent_mse + losses["total"]
total.backward()
```

Run the self-contained demo (no checkpoints/datasets required):

```sh
pip install -r requirements.txt
python example_physical_consistency.py
```

## Tests

```sh
pytest test_physical_consistency.py
```
