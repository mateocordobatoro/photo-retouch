"""Evaluation on the frozen validation set.

Every method is judged on the same 400 composed images (original pixels
outside the hole). Per-image metrics allow paired comparisons; KID uses the
same random subsets for every method, so its estimates are paired too.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf

from .features import DISTANCE_LAYERS, inception_pool, vgg16_features, vgg_preprocess

FREQUENCY_SIGMA = 2.0
DETAIL_SIGMA = 1.0
GRID_PERIOD = 16
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], np.float32)
RGB_TO_YCBCR = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312],
], np.float32)


def gaussian_blur(images, sigma):
    return np.stack([
        cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
        for image in images
    ])


def masked_mae_per_image(results, originals, masks):
    """Mean absolute error over each image's hole pixels."""
    error = np.abs(results - originals) * masks
    return error.sum(axis=(1, 2, 3)) / (3 * masks.sum(axis=(1, 2, 3)))


def kernel_inception_distance(real, fake, subset_size=100, subsets=100, seed=2033):
    """Unbiased KID estimates (cubic polynomial kernel), one per random subset.

    The same seed gives the same subsets for every method, so estimates are
    paired by subset.
    """
    rng = np.random.default_rng(seed)
    dimension = real.shape[1]
    m = subset_size
    estimates = []

    for _ in range(subsets):
        x = real[rng.choice(len(real), m, replace=False)]
        y = fake[rng.choice(len(fake), m, replace=False)]

        # NumPy 2.0 with Apple Accelerate emits spurious floating-point
        # warnings in matmul; the products are checked explicitly instead.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            k_xx = (x @ x.T / dimension + 1) ** 3
            k_yy = (y @ y.T / dimension + 1) ** 3
            k_xy = (x @ y.T / dimension + 1) ** 3
        assert np.isfinite([k_xx, k_yy, k_xy]).all()

        estimates.append(
            (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
            + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
            - 2 * k_xy.mean()
        )
    return np.array(estimates)


def folded_jump_profile(images, interior_mask, axis, period=GRID_PERIOD):
    """Mean |luminance jump| across pixel boundaries, by position mod ``period``.

    Only boundaries between two interior hole pixels count. Normalized to
    mean 1: a flat profile has no grid-aligned component.
    """
    luma = images @ LUMA_WEIGHTS
    jumps = np.abs(np.diff(luma, axis=axis))
    if axis == 2:
        valid = interior_mask[:, :, 1:] & interior_mask[:, :, :-1]
        phase = np.arange(jumps.shape[2])[None, None, :] % period
    else:
        valid = interior_mask[:, 1:, :] & interior_mask[:, :-1, :]
        phase = np.arange(jumps.shape[1])[None, :, None] % period
    phase = np.broadcast_to(phase, jumps.shape)

    totals = np.bincount(phase[valid], weights=jumps[valid], minlength=period)
    counts = np.bincount(phase[valid], minlength=period)
    profile = totals / counts
    return profile / profile.mean()


@dataclass
class Evaluation:
    """Per-image metrics of one method, plus its paired KID estimates."""

    masked_mae: np.ndarray
    feature_distance: np.ndarray
    low_freq_error: np.ndarray
    high_freq_error: np.ndarray
    texture_ratio: np.ndarray
    kid: np.ndarray

    def summary(self):
        return {
            "masked_mae": self.masked_mae.mean(),
            "feature_distance": self.feature_distance.mean(),
            "kid_x1000": 1000 * self.kid.mean(),
            "kid_x1000_std": 1000 * self.kid.std(),
            "low_freq_error": self.low_freq_error.mean(),
            "high_freq_error": self.high_freq_error.mean(),
            "texture_ratio_median": np.nanmedian(self.texture_ratio),
        }


class Evaluator:
    """All metrics for one ValidationSet.

    Building it loads VGG16 and InceptionV3 and computes the true images'
    reference features once.

    - Masked MAE: pixel error inside the hole.
    - Low/high-frequency error: error of the Gaussian-blurred (σ = 2) part
      and of the residual; texture ratio: residual energy of the fill over
      that of the truth, in the eroded hole.
    - Feature distance: unlearned LPIPS-style VGG16 distance (lower is closer).
    - KID: kernel inception distance on 100 paired subsets of 100 images.
    """

    def __init__(self, validation, kid_seed=2033, bootstrap_seed=2034, bootstrap_samples=2000):
        self.validation = validation
        self.originals = validation.originals
        self.masks = validation.masks
        self.interior = validation.interior
        self.interior_pixels = self.interior.sum(axis=(1, 2, 3))
        self.hole_values = 3 * self.masks.sum(axis=(1, 2, 3))
        self.original_low = gaussian_blur(self.originals, FREQUENCY_SIGMA)
        self.original_high = self.originals - self.original_low
        self.kid_seed = kid_seed

        self._distance_features = vgg16_features(DISTANCE_LAYERS)
        self._inception = inception_pool()
        self._distance_batch = tf.function(self._distance_batch_eager)
        self._inception_batch = tf.function(self._inception_batch_eager)
        self.real_inception = self.inception_features(self.originals)

        self.truth_luma_detail, self.truth_chroma_detail = self.detail_energy(self.originals)
        self.detail_valid = (self.interior_pixels >= 20) & (self.truth_chroma_detail > 1e-8)

        self.bootstrap_index = np.random.default_rng(bootstrap_seed).integers(
            0, len(validation), size=(bootstrap_samples, len(validation))
        )

    # Pixel error -----------------------------------------------------------

    def masked_mae(self, results):
        return masked_mae_per_image(results, self.originals, self.masks)

    def frequency_errors(self, results):
        low = gaussian_blur(results, FREQUENCY_SIGMA)
        high = results - low
        low_error = (
            np.abs(low - self.original_low) * self.masks
        ).sum(axis=(1, 2, 3)) / self.hole_values
        high_error = (
            np.abs(high - self.original_high) * self.masks
        ).sum(axis=(1, 2, 3)) / self.hole_values

        with np.errstate(divide="ignore", invalid="ignore"):
            texture_ratio = (
                (np.abs(high) * self.interior).sum(axis=(1, 2, 3))
                / (np.abs(self.original_high) * self.interior).sum(axis=(1, 2, 3))
            )
        texture_ratio[self.interior_pixels < 20] = np.nan
        return low_error, high_error, texture_ratio

    # Perceptual distance ----------------------------------------------------

    def _distance_batch_eager(self, results, originals):
        result_features = self._distance_features(vgg_preprocess(results), training=False)
        original_features = self._distance_features(vgg_preprocess(originals), training=False)
        total = 0.0
        for a, b in zip(result_features, original_features):
            a = a / (tf.norm(a, axis=-1, keepdims=True) + 1e-10)
            b = b / (tf.norm(b, axis=-1, keepdims=True) + 1e-10)
            total += tf.reduce_mean(tf.reduce_sum(tf.square(a - b), axis=-1), axis=[1, 2])
        return total

    def feature_distance(self, results, batch_size=50):
        return np.concatenate([
            self._distance_batch(
                tf.constant(results[start:start + batch_size]),
                tf.constant(self.originals[start:start + batch_size]),
            ).numpy()
            for start in range(0, len(results), batch_size)
        ])

    # KID --------------------------------------------------------------------

    def _inception_batch_eager(self, images):
        resized = tf.image.resize(images, (299, 299), method="bilinear")
        return self._inception(resized * 2.0 - 1.0, training=False)

    def inception_features(self, images, batch_size=50):
        return np.concatenate([
            self._inception_batch(tf.constant(images[start:start + batch_size])).numpy()
            for start in range(0, len(images), batch_size)
        ]).astype(np.float64)

    def kid(self, results):
        return kernel_inception_distance(
            self.real_inception, self.inception_features(results), seed=self.kid_seed
        )

    # All metrics -------------------------------------------------------------

    def evaluate(self, results):
        low_error, high_error, texture_ratio = self.frequency_errors(results)
        return Evaluation(
            masked_mae=self.masked_mae(results),
            feature_distance=self.feature_distance(results),
            low_freq_error=low_error,
            high_freq_error=high_error,
            texture_ratio=texture_ratio,
            kid=self.kid(results),
        )

    def paired(self, method, baseline):
        """Image-by-image differences, method minus baseline (negative is better).

        Mean difference with a 95% bootstrap interval over images, the share
        of images where the method is better, and for KID the mean paired
        difference across subsets and the share of subsets where it is lower.
        Subsets overlap, so that share is descriptive, not a significance test.
        """
        row = {}
        for metric in ["masked_mae", "feature_distance"]:
            difference = getattr(method, metric) - getattr(baseline, metric)
            low, high = np.percentile(
                difference[self.bootstrap_index].mean(axis=1), [2.5, 97.5]
            )
            row[f"{metric} Δ"] = difference.mean()
            row[f"{metric} 95% CI"] = f"[{low:+.4f}, {high:+.4f}]"
            row[f"{metric} images better"] = f"{(difference < 0).mean():.0%}"
        kid_difference = 1000 * (method.kid - baseline.kid)
        row["KID×1000 Δ"] = kid_difference.mean()
        row["KID subsets lower"] = f"{(kid_difference < 0).mean():.0%}"
        return row

    # Artifact diagnostics -----------------------------------------------------

    def detail_energy(self, images):
        """Fine-detail energy (image minus Gaussian σ = 1) in luma and chroma.

        Measured per image in the eroded hole, in YCbCr.
        """
        ycbcr = images @ RGB_TO_YCBCR.T
        detail = (ycbcr - gaussian_blur(ycbcr, DETAIL_SIGMA)) ** 2
        energy = (
            (detail * self.interior).sum(axis=(1, 2))
            / np.maximum(self.interior_pixels, 1)[:, None]
        )
        return energy[:, 0], energy[:, 1:].sum(axis=1)

    def detail_summary(self, images):
        """Median detail ratios to the truth; chroma well above 1 means colour fringing."""
        luma, chroma = self.detail_energy(images)
        valid = self.detail_valid
        luma_ratio = luma[valid] / self.truth_luma_detail[valid]
        chroma_ratio = chroma[valid] / self.truth_chroma_detail[valid]
        return {
            "luma_detail_ratio": np.median(luma_ratio),
            "chroma_detail_ratio": np.median(chroma_ratio),
            "chroma_share": np.median((chroma / (luma + chroma))[valid]),
            "images_chroma_over_2x": (chroma_ratio > 2).mean(),
        }

    def grid_periodicity(self, images):
        """Strength of grid-aligned jumps at periods 16, 8, 4, and 2 px.

        One row per direction. Compare with the true images' level.
        """
        interior_mask = self.interior[..., 0] > 0.5
        rows = []
        for direction, axis in [("horizontal", 2), ("vertical", 1)]:
            profile = folded_jump_profile(images, interior_mask, axis)
            harmonics = np.abs(np.fft.rfft(profile - 1.0)) / GRID_PERIOD
            rows.append({
                "direction": direction,
                **{f"period {GRID_PERIOD // k} px": harmonics[k] for k in [1, 2, 4, 8]},
            })
        return rows
