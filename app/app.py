"""Photo Retouch: paint over part of a photo and remove it.

Stage 1 is this project's inpainting model (trained from scratch at
128 × 128). Stage 2 is a pretrained diffusion model that completes stage 1's
fill at a higher resolution. Run locally from the project root:

    python app/app.py

The same file runs on a Hugging Face ZeroGPU Space, where the package is
copied into ``src/`` next to it.
"""
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# In the Space, the package is copied into src/ next to this file; in the
# project, it is the editable install.
SPACE_SRC = Path(__file__).resolve().parent / "src"
if SPACE_SRC.is_dir():
    sys.path.insert(0, str(SPACE_SRC))

try:
    import spaces  # Hugging Face ZeroGPU: a GPU is attached only inside @gpu calls.

    gpu = spaces.GPU(duration=60)
    DIFFUSION_DEVICE = "cuda"
except ImportError:
    def gpu(function):
        return function

    DIFFUSION_DEVICE = None  # CUDA, then Apple Metal, then CPU

from photo_retouch.inpaint import (  # noqa: E402
    OurModel, inpaint, load_diffusion, refine_with_diffusion,
)

MAX_SIDE = 2048
DEFAULT_STRENGTH = 0.7
STEPS = 30

our_model = OurModel()
pipe = load_diffusion(DIFFUSION_DEVICE)

NOTES = """
### How it works

1. **Our model (stage 1).** A coarse U-Net and a gated-convolution refiner,
   trained from scratch for this project on landscape and street photos at
   128 × 128. It fills a square crop around the painted area.
2. **Pretrained refinement (stage 2).** [Stable Diffusion 1.5
   Inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)
   (CreativeML OpenRAIL-M license), used as is, not retrained. It starts from
   stage 1's fill with part of it replaced by noise and completes it at
   512 × 512. **Refinement strength** sets how much of stage 1's fill it may
   change: at 0 the result is our model alone.

Outside the painted area and a blend margin of a few pixels, the photo is not
changed. Photos larger than 2048 px are scaled down first.
"""


def photo_and_mask(editor_value):
    if editor_value is None or editor_value.get("background") is None:
        raise gr.Error("Upload a photo first.")
    # RGBA keeps the brush strokes' alpha; flatten any transparency onto white.
    background = np.asarray(editor_value["background"], np.float32)
    if background.shape[2] == 4:
        alpha = background[..., 3:] / 255
        background = background[..., :3] * alpha + 255 * (1 - alpha)
    photo = np.uint8(np.clip(background[..., :3] + 0.5, 0, 255))
    mask = np.zeros(photo.shape[:2], bool)
    for layer in editor_value.get("layers") or []:
        layer = np.asarray(layer)
        if layer.ndim == 3 and layer.shape[2] == 4 and layer.shape[:2] == mask.shape:
            mask |= layer[..., 3] > 0
    if not mask.any():
        raise gr.Error("Paint over the part of the photo you want to remove.")

    scale = MAX_SIDE / max(photo.shape[:2])
    if scale < 1:
        size = (round(photo.shape[1] * scale), round(photo.shape[0] * scale))
        photo = np.asarray(Image.fromarray(photo).resize(size, Image.LANCZOS))
        mask = np.asarray(Image.fromarray(np.uint8(mask) * 255).resize(size)) > 127
    return photo, mask


@gpu
def diffuse(image, mask, strength):
    """Stage 2 only: on ZeroGPU this call alone gets a GPU (and its own process)."""
    return refine_with_diffusion(pipe, image, mask, strength, STEPS, seed=0)


def remove(editor_value, strength):
    photo, mask = photo_and_mask(editor_value)
    refine = (lambda image, hole: diffuse(image, hole, strength)) if strength > 0 else None
    stages = inpaint(photo, mask, our_model, refine)
    path = Path(tempfile.mkdtemp()) / "photo-retouch.png"
    Image.fromarray(stages.result).save(path)
    return (
        stages.result,
        gr.DownloadButton("Download PNG", value=str(path), interactive=True),
        [(stages.crop, "Painted area removed"),
         (stages.ours, "Stage 1: our model (128 × 128)"),
         (stages.refined, f"Stage 2: refined (strength {strength:.2f})")],
    )


with gr.Blocks(title="Photo Retouch") as demo:
    gr.Markdown(
        "# Photo Retouch\n"
        "Upload your photo, paint over the person or object to remove, and "
        "press **Remove**."
    )
    with gr.Row():
        with gr.Column():
            editor = gr.ImageEditor(
                label="Upload your photo",
                type="numpy",
                image_mode="RGBA",
                sources=("upload", "clipboard"),
                brush=gr.Brush(colors=["#ff3b30"], color_mode="fixed"),
                eraser=gr.Eraser(),
                layers=False,
                transforms=(),
                format="png",
                height=520,
            )
            strength = gr.Slider(
                0, 1, value=DEFAULT_STRENGTH, step=0.05,
                label="Refinement strength",
                info="0 uses only our model; higher values let the pretrained "
                     "model change more of its fill.",
            )
            remove_button = gr.Button("Remove", variant="primary")
        with gr.Column():
            result = gr.Image(label="Result", type="numpy", format="png", height=520)
            download = gr.DownloadButton("Download PNG", interactive=False)
    stages_gallery = gr.Gallery(
        label="Stages (crop around the painted area)", columns=3, height=300,
        object_fit="contain",
    )
    gr.Markdown(NOTES)

    remove_button.click(
        remove, [editor, strength], [result, download, stages_gallery],
    )


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
