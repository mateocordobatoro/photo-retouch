---
title: Photo Retouch
emoji: 🖌️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: "Remove objects from photos: our model + diffusion"
models:
  - stable-diffusion-v1-5/stable-diffusion-inpainting
---

# Photo Retouch

Upload a photo, paint over a person or object, and press **Remove**.

## How it works

The fill is produced by two models in sequence:

1. **Our model (stage 1).** A coarse U-Net and a gated-convolution refiner,
   trained from scratch for the course *Advanced Mathematics for Deep
   Learning* on landscape and street photos at 128 × 128. It fills a square
   crop around the painted area.
2. **Pretrained refinement (stage 2).** [Stable Diffusion 1.5
   Inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting),
   used as is, not retrained. It starts from stage 1's fill with part of it
   replaced by noise and completes it at 512 × 512. **Refinement strength**
   sets how much of stage 1's fill it may change; at 0 the result is our model
   alone.

Outside the painted area and a blend margin of a few pixels, the photo is not
changed. The **Stages** gallery shows the crop at each step.

## Limitations

- Stage 1 works at 128 × 128, so on its own its fill is soft and can show a
  fine grid pattern. Stage 2 adds the detail.
- The fill's resolution comes from stage 2 at 512 × 512 around the painted
  area. Large removals in high-resolution photos look softer than their
  surroundings. Photos larger than 2048 px are scaled down first.
- The diffusion model can invent content. Lower strength keeps more of stage
  1's fill.

## Credits and licenses

- Stage 1 models and this app: the Photo Retouch course project, trained on
  the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
  dataset.
- Stage 2: Stable Diffusion 1.5 Inpainting by Runway, distributed under the
  CreativeML OpenRAIL-M license, which restricts some uses; see the
  [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting).
  Use of this demo is subject to those restrictions.
