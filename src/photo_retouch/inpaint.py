"""Chained inpainting for the app: the project's model, then diffusion refinement.

Stage 1 is this project's model: the coarse U-Net and the low-chroma refiner,
trained from scratch at 128 × 128. Stage 2 is a pretrained model, Stable
Diffusion 1.5 Inpainting. It starts from the stage-1 fill with part of it
replaced by noise and denoises it at 512 × 512; ``strength`` sets how much of
the stage-1 fill it may change (0 keeps it, 1 discards it). Outside the mask
and a blend margin of a few pixels, the photo is never changed.

Both stages work on a square crop around the mask with surrounding context,
so the photo keeps its own resolution everywhere else.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

DIFFUSION_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
MODEL_SIZE = 128
DIFFUSION_SIZE = 512
CONTEXT = 2.0  # crop side as a multiple of the mask's bounding box
MIN_CROP = 128  # smallest crop side in photo pixels, if the photo allows it
DIFFUSION_GROW = 8  # px at 512 × 512 the diffusion mask extends past the hole
FEATHER = 4  # px at 512 × 512 over which the refined crop blends in

PROMPT = "a photo, natural continuation of the surrounding background"
NEGATIVE_PROMPT = (
    "person, people, animal, object, text, watermark, logo, frame, "
    "blurry, distorted, artifacts"
)


@dataclass
class Stages:
    """The full result and the crop at every stage, for display."""

    result: np.ndarray  # full photo, uint8
    crop: np.ndarray  # input crop with the hole blanked, uint8
    ours: np.ndarray  # stage 1 at 128 × 128, uint8
    refined: np.ndarray  # final crop, uint8


def crop_box(mask, context=CONTEXT, min_side=MIN_CROP):
    """(top, left, bottom, right) of a crop around the mask, inside the photo.

    The crop is a square ``context`` times the mask's bounding box, centred on
    it, unless the photo is too small in one direction; then it is as large as
    the photo allows there.
    """
    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    side = max(int(np.ceil(context * max(bottom - top, right - left))), min_side)

    def span(start, stop, limit):
        size = min(side, limit)
        begin = int(round((start + stop - size) / 2))
        begin = min(max(begin, 0), limit - size)
        return begin, begin + size

    top, bottom = span(top, bottom, height)
    left, right = span(left, right, width)
    return top, left, bottom, right


def pad_to_square(array, mode):
    """Pad the shorter side symmetrically; returns the array and the padding."""
    height, width = array.shape[:2]
    side = max(height, width)
    rows, cols = side - height, side - width
    padding = ((rows // 2, rows - rows // 2), (cols // 2, cols - cols // 2))
    padding += ((0, 0),) * (array.ndim - 2)
    return np.pad(array, padding, mode=mode), padding


def unpad(array, padding):
    (top, bottom), (left, right) = padding[:2]
    height, width = array.shape[:2]
    return array[top:height - bottom, left:width - right]


def resize(array, size, resample):
    """Resize a float array in [0, 1] (H × W or H × W × C) to size × size."""
    image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255 + 0.5))
    resized = image.resize((size, size), resample)
    return np.asarray(resized, np.float32) / 255


def resize_mask(mask, size):
    """Any coverage of a target pixel by the hole makes it part of the hole."""
    return resize(mask.astype(np.float32), size, Image.BOX) > 0


def to_uint8(array):
    return np.uint8(np.clip(array, 0, 1) * 255 + 0.5)


def grow(mask, pixels):
    image = Image.fromarray(np.uint8(mask) * 255)
    return np.asarray(image.filter(ImageFilter.MaxFilter(2 * pixels + 1))) > 0


def feather(mask, pixels):
    image = Image.fromarray(np.uint8(mask) * 255)
    return np.asarray(image.filter(ImageFilter.GaussianBlur(pixels)), np.float32) / 255


class OurModel:
    """Stage 1: the coarse U-Net and the low-chroma refiner (TensorFlow)."""

    def __init__(self):
        import tensorflow as tf

        from .models import (
            COARSE_MODEL_FILE, SAVED_REFINERS, load_saved_model,
            make_coarse_composite, refine_low_chroma,
        )

        filename, refine_fn = SAVED_REFINERS["Low-chroma refiner"]
        assert refine_fn is refine_low_chroma
        self._tf = tf
        self._coarse = make_coarse_composite(load_saved_model(COARSE_MODEL_FILE))
        self._refiner = load_saved_model(filename)
        self._refine = refine_low_chroma

    def __call__(self, image, mask):
        """image: 128 × 128 × 3 in [0, 1]; mask: 128 × 128 bool (True = hole)."""
        tf = self._tf
        images = tf.constant(image[None], tf.float32)
        masks = tf.constant(mask[None, :, :, None], tf.float32)
        coarse = self._coarse(images, masks)
        return self._refine(self._refiner, coarse, masks).numpy()[0]


def torch_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_diffusion(device=None):
    """Stage 2: the pretrained inpainting pipeline, without the safety checker.

    Downloads about 2.1 GB of fp16 weights from the Hugging Face Hub the first
    time and reads them from the local cache afterwards.
    """
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    device = device or torch_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        DIFFUSION_MODEL,
        variant=None if device == "cpu" else "fp16",
        dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    if device == "mps":
        pipe.enable_attention_slicing()
    return pipe.to(device)


def refine_with_diffusion(pipe, image, mask, strength, steps, seed, guidance=5.0):
    """image: 512 × 512 × 3 in [0, 1]; mask: 512 × 512 bool. Returns the same size."""
    import torch

    generator = torch.Generator("cpu").manual_seed(seed)
    output = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=Image.fromarray(np.uint8(image * 255 + 0.5)),
        mask_image=Image.fromarray(np.uint8(mask) * 255),
        height=DIFFUSION_SIZE,
        width=DIFFUSION_SIZE,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    return np.asarray(output, np.float32) / 255


def inpaint(photo, mask, our_model, refine=None):
    """Remove the masked region of a photo of any size.

    photo: H × W × 3 uint8; mask: H × W bool (True = remove). ``refine`` is
    stage 2, a function (image, mask) -> image at 512 × 512, for example
    ``refine_with_diffusion`` with its other arguments fixed. Without it, the
    result is stage 1 alone, enlarged to the crop's resolution.
    """
    if not mask.any():
        raise ValueError("The mask is empty: paint over the region to remove.")
    image = photo.astype(np.float32) / 255
    top, left, bottom, right = crop_box(mask)
    crop, padding = pad_to_square(image[top:bottom, left:right], "reflect")
    hole, _ = pad_to_square(mask[top:bottom, left:right], "constant")
    side = crop.shape[0]

    # Stage 1 at 128 × 128, then enlarged into the hole at the crop's resolution.
    small_hole = resize_mask(hole, MODEL_SIZE)
    small = resize(crop, MODEL_SIZE, Image.BOX if side > MODEL_SIZE else Image.BICUBIC)
    ours = our_model(small * ~small_hole[..., None], small_hole)
    enlarged = resize(ours, side, Image.BICUBIC)
    filled = np.where(hole[..., None], enlarged, crop)

    if refine is not None:
        # Stage 2 at 512 × 512, starting from the stage-1 fill.
        big_hole = grow(resize_mask(hole, DIFFUSION_SIZE), DIFFUSION_GROW)
        refined_big = refine(resize(filled, DIFFUSION_SIZE, Image.LANCZOS), big_hole)
        alpha_big = np.maximum(feather(big_hole, FEATHER), big_hole)
        refined = resize(refined_big, side, Image.LANCZOS)
        alpha = np.maximum(resize(alpha_big, side, Image.BILINEAR), hole)
        filled = crop * (1 - alpha[..., None]) + refined * alpha[..., None]

    # Outside the hole and the blend margin, the photo is unchanged.
    result = image.copy()
    result[top:bottom, left:right] = unpad(filled, padding)
    return Stages(
        result=to_uint8(result),
        crop=to_uint8(unpad(crop * ~hole[..., None], padding)),
        ours=to_uint8(ours),
        refined=to_uint8(unpad(filled, padding)),
    )
