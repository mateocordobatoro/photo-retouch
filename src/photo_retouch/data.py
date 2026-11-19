"""Image loading, the frozen validation and test sets, masks, and training batches.

Masks use 1 for missing pixels and 0 for visible ones.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw

from .paths import PROJECT_ROOT, latest_record_dir

IMAGE_SIZE = (128, 128)
CATEGORIES = ["sea", "forest", "street", "buildings"]
TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "intel-image-classification" / "seg_test" / "seg_test"
TEST_MANIFEST = "test_400.csv"
TEST_MASKS = "test_mixed_masks.npz"
TEST_MASK_SEED = 2060


def load_image(path):
    """Decode a JPEG, scale to [0, 1], and resize to 128 × 128 (float32 array)."""
    image = tf.io.read_file(str(path))
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, IMAGE_SIZE).numpy()


@dataclass
class ValidationSet:
    """The frozen validation images and their fixed mixed masks.

    ``interior`` is each hole eroded by a 5 × 5 kernel. Metrics that use
    filters (texture, detail energy, grid tests) are measured there, away
    from the seam where a filter mixes both sides.
    """

    record_dir: Path
    manifest: pd.DataFrame
    originals: np.ndarray
    masks: np.ndarray
    visible: np.ndarray = field(init=False, repr=False)
    inputs: np.ndarray = field(init=False, repr=False)
    interior: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.visible = self.originals * (1 - self.masks)
        self.inputs = np.concatenate([self.visible, self.masks], axis=-1)
        self.interior = np.stack([
            cv2.erode(
                mask[..., 0].astype(np.uint8),
                np.ones((5, 5), np.uint8),
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for mask in self.masks
        ]).astype(np.float32)[..., None]

    def __len__(self):
        return len(self.originals)

    def compose(self, fill):
        """Original pixels outside the hole, ``fill`` inside it."""
        return self.visible + fill * self.masks

    def example_indices(self, categories=CATEGORIES):
        """Highest-coverage image of each category: the fixed visual examples."""
        coverage = self.masks.mean(axis=(1, 2, 3))
        indices = []
        for category in categories:
            in_category = (self.manifest["category"] == category).to_numpy()
            indices.append(int(
                self.manifest.index[in_category][np.argmax(coverage[in_category])]
            ))
        return indices


def load_image_set(record_dir, manifest_file, masks_file):
    """Images listed in a saved manifest, with their saved fixed masks."""
    record_dir = Path(record_dir)
    manifest = pd.read_csv(record_dir / manifest_file)
    masks = np.load(record_dir / masks_file)["masks"]
    originals = np.stack([load_image(PROJECT_ROOT / path) for path in manifest["path"]])
    if len(originals) != len(masks):
        raise ValueError("The manifest and the saved masks differ in length.")
    return ValidationSet(record_dir, manifest, originals, masks.astype(np.float32))


def load_validation_set(record_dir=None):
    """Load the 400 validation images and masks saved in a record snapshot."""
    record_dir = Path(record_dir) if record_dir else latest_record_dir()
    return load_image_set(record_dir, "validation_400.csv", "validation_mixed_masks.npz")


def load_test_set(record_dir=None):
    """Load the 400 test images and masks frozen in a snapshot's test/ folder."""
    record_dir = Path(record_dir) if record_dir else latest_record_dir()
    return load_image_set(record_dir / "test", TEST_MANIFEST, TEST_MASKS)


def create_mixed_mask(rng):
    """Random strokes, a rounded block, or a person silhouette (128 × 128 × 1).

    The generator used to train the coarse U-Net and the refiners.
    """
    mask = Image.new("L", IMAGE_SIZE, 0)
    brush = ImageDraw.Draw(mask)
    kind = rng.choice(["strokes", "block", "silhouette"])

    if kind == "strokes":
        for _ in range(rng.integers(1, 4)):
            points = rng.integers(0, 128, size=(3, 2))
            brush.line(
                [tuple(map(int, point)) for point in points],
                fill=255,
                width=int(rng.integers(6, 18)),
            )
    else:
        width = int(rng.integers(24, 55))
        height = int(rng.integers(45, 105))
        x = int(rng.integers(0, 129 - width))
        y = int(rng.integers(0, 129 - height))

        if kind == "block":
            brush.rounded_rectangle(
                (x, y, x + width - 1, y + height - 1), radius=8, fill=255
            )
        else:
            silhouette = Image.new("L", (64, 128), 0)
            body = ImageDraw.Draw(silhouette)
            body.ellipse((23, 0, 41, 22), fill=255)
            body.rounded_rectangle((15, 22, 49, 80), radius=8, fill=255)
            body.line([(17, 30), (5, 77)], fill=255, width=10)
            body.line([(47, 30), (59, 77)], fill=255, width=10)
            body.line([(24, 75), (20, 127)], fill=255, width=13)
            body.line([(40, 75), (44, 127)], fill=255, width=13)
            silhouette = silhouette.resize((width, height), Image.Resampling.NEAREST)
            mask.paste(silhouette, (x, y))

    return np.asarray(mask, dtype=np.float32)[..., None] / 255


def reserved_test_selection(per_category=100, seed=42):
    """The test images reserved when the dataset was split, in the same order.

    The training notebook drew them per category with
    ``pd.Series(sorted seg_test files).sample(n=100, random_state=42)`` but
    never saved the list; the same rule rebuilds it.
    """
    rows = []
    for category in CATEGORIES:
        files = sorted((TEST_IMAGE_DIR / category).glob("*.jpg"))
        picked = pd.Series(files).sample(n=per_category, random_state=seed)
        rows.extend(
            (path.relative_to(PROJECT_ROOT).as_posix(), category) for path in picked
        )
    return pd.DataFrame(rows, columns=["path", "category"])


def fixed_masks(count, seed=TEST_MASK_SEED):
    """One mixed mask per image, each drawn from its own seed (count × 128 × 128 × 1)."""
    return np.stack([
        create_mixed_mask(np.random.default_rng(np.random.SeedSequence([seed, index])))
        for index in range(count)
    ]) > 0.5


class TrainingBatches:
    """Training batches rebuilt deterministically from the global step.

    Each epoch shuffles the 6,000 training images with its own seed, and each
    step draws new masks from its own seed. Two models trained with the same
    step numbers therefore see identical images and masks, and a resumed run
    continues the same sequence. The default seeds are those of the refiner
    experiments.
    """

    def __init__(self, record_dir=None, batch_size=16, order_seed=2040, mask_seed=2041):
        record_dir = Path(record_dir) if record_dir else latest_record_dir()
        manifest = pd.read_csv(record_dir / "train_6000.csv")
        self.paths = [PROJECT_ROOT / path for path in manifest["path"]]
        self.batch_size = batch_size
        self.order_seed = order_seed
        self.mask_seed = mask_seed
        self.steps_per_epoch = -(-len(self.paths) // batch_size)

    def __call__(self, step):
        epoch, batch_index = divmod(step, self.steps_per_epoch)
        order = np.random.default_rng(
            np.random.SeedSequence([self.order_seed, epoch])
        ).permutation(len(self.paths))
        start = batch_index * self.batch_size
        indices = order[start:start + self.batch_size]

        images = np.stack([load_image(self.paths[i]) for i in indices])
        mask_rng = np.random.default_rng(np.random.SeedSequence([self.mask_seed, step]))
        masks = np.stack([create_mixed_mask(mask_rng) for _ in indices])

        return (
            tf.convert_to_tensor(images, dtype=tf.float32),
            tf.convert_to_tensor(masks, dtype=tf.float32),
        )
