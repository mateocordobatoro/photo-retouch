"""The coarse U-Net, the gated-convolution refiner, and the patch discriminator."""
from functools import lru_cache

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .paths import MODEL_DIR

COARSE_MODEL_FILE = "best_deep_unet_6000_tuned.keras"
CHROMA_SIGMA = 2.0

# Rows map the Y, Cb, and Cr residuals to RGB.
YCBCR_TO_RGB = np.array([
    [1.0, 1.0, 1.0],
    [0.0, -0.344136, 1.772],
    [1.402, -0.714136, 0.0],
], np.float32)


def load_saved_model(filename):
    """Load a model saved in models/ for inference."""
    return keras.models.load_model(MODEL_DIR / filename, compile=False)


def make_coarse_composite(coarse_model):
    """Freeze the coarse model and return a function images, masks -> composite."""
    coarse_model.trainable = False

    @tf.function
    def coarse_composite(images, masks):
        visible = images * (1 - masks)
        prediction = coarse_model(tf.concat([visible, masks], axis=-1), training=False)
        return visible + prediction * masks

    return coarse_composite


def predict_coarse(coarse_model, validation, batch_size=16):
    """Composed coarse results for a ValidationSet."""
    return validation.compose(
        coarse_model.predict(validation.inputs, batch_size=batch_size, verbose=0)
    )


def gated_conv(x, filters, kernel_size=3, strides=1, dilation_rate=1, name=None):
    """Features times a learned sigmoid gate (Yu et al., 2019)."""
    features = layers.Conv2D(
        filters, kernel_size, strides=strides, dilation_rate=dilation_rate,
        padding="same", name=f"{name}_features",
    )(x)
    features = layers.ELU(name=f"{name}_elu")(features)
    gate = layers.Conv2D(
        filters, kernel_size, strides=strides, dilation_rate=dilation_rate,
        padding="same", activation="sigmoid", name=f"{name}_gate",
    )(x)
    return layers.Multiply(name=name)([features, gate])


def build_refiner(name):
    """Gated-conv encoder-decoder: 128 → 32 px, dilations 2/4/8, skip connections.

    Input: coarse composite plus mask (4 channels). Output: a 3-channel
    residual. The last layer starts at zero, so a new refiner reproduces
    the coarse result exactly.
    """
    inputs = keras.Input(shape=(128, 128, 4))

    x128 = gated_conv(inputs, 32, kernel_size=5, name="enc128")
    x = gated_conv(x128, 64, strides=2, name="down64")
    x64 = gated_conv(x, 64, name="enc64")
    x = gated_conv(x64, 128, strides=2, name="down32")
    x = gated_conv(x, 128, name="enc32")

    for rate in [2, 4, 8]:
        x = gated_conv(x, 128, dilation_rate=rate, name=f"dilated{rate}")

    x = gated_conv(x, 128, name="dec32")
    x = layers.UpSampling2D(name="up64")(x)
    x = layers.Concatenate(name="skip64")([x, x64])
    x = gated_conv(x, 64, name="dec64")
    x = layers.UpSampling2D(name="up128")(x)
    x = layers.Concatenate(name="skip128")([x, x128])
    x = gated_conv(x, 32, name="dec128")

    residual = layers.Conv2D(
        3, 3, padding="same",
        kernel_initializer="zeros", bias_initializer="zeros", name="residual",
    )(x)
    return keras.Model(inputs, residual, name=name)


def _coarse_logits(coarse):
    clipped = tf.clip_by_value(coarse, 1e-3, 1 - 1e-3)
    return tf.math.log(clipped) - tf.math.log1p(-clipped)


def refine(refiner, coarse, masks, training=False):
    """refined = sigmoid(logit(coarse) + residual) inside the hole."""
    residual = refiner(tf.concat([coarse, masks], axis=-1), training=training)
    refined = tf.sigmoid(_coarse_logits(coarse) + residual)
    # Original pixels outside the hole are always preserved.
    return coarse * (1 - masks) + refined * masks


@lru_cache
def chroma_kernel(sigma=CHROMA_SIGMA):
    """Normalized 2-D Gaussian for the two chroma channels (depthwise)."""
    radius = int(3 * sigma)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    taps = np.exp(-offsets**2 / (2 * sigma**2))
    taps /= taps.sum()
    kernel = np.tile(np.outer(taps, taps)[:, :, None, None], (1, 1, 2, 1))
    kernel.setflags(write=False)
    return kernel


def low_chroma_residual(raw, sigma=CHROMA_SIGMA):
    """Read the raw residual as (Y, Cb, Cr), blur Cb and Cr, convert to RGB."""
    luma = raw[..., :1]
    chroma = tf.nn.depthwise_conv2d(
        raw[..., 1:], chroma_kernel(sigma), strides=[1, 1, 1, 1], padding="SAME"
    )
    return tf.tensordot(tf.concat([luma, chroma], axis=-1), YCBCR_TO_RGB, axes=1)


def refine_low_chroma(refiner, coarse, masks, training=False):
    """As refine(), but fine detail can be added only in luminance.

    The residual is added in logit space, so the sigmoid can still add a
    little chroma where the coarse colour channels differ.
    """
    raw = refiner(tf.concat([coarse, masks], axis=-1), training=training)
    refined = tf.sigmoid(_coarse_logits(coarse) + low_chroma_residual(raw))
    return coarse * (1 - masks) + refined * masks


# Saved refiners and the refine function each was trained with. A low-chroma
# refiner applied with refine() gives wrong colours, so always look it up here.
SAVED_REFINERS = {
    "L1 refiner": ("refiner_l1_control.keras", refine),
    "Texture refiner": ("refiner_texture.keras", refine),
    "Low-chroma refiner": ("refiner_low_chroma.keras", refine_low_chroma),
    "Low-chroma, no style": ("refiner_no_style.keras", refine_low_chroma),
}


def refine_images(refiner, coarse_results, masks, refine_fn=refine, batch_size=50):
    """Apply a refiner to arrays of coarse composites in batches."""
    return np.concatenate([
        refine_fn(
            refiner,
            tf.constant(coarse_results[start:start + batch_size]),
            tf.constant(masks[start:start + batch_size]),
        ).numpy()
        for start in range(0, len(coarse_results), batch_size)
    ])


def build_patch_discriminator():
    """SN-PatchGAN: four stride-2 spectrally normalized convs, 8 × 8 score grid."""
    inputs = keras.Input(shape=(128, 128, 4))
    x = inputs

    for filters in [64, 128, 256, 256]:
        x = layers.SpectralNormalization(
            layers.Conv2D(filters, 5, strides=2, padding="same")
        )(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    scores = layers.SpectralNormalization(layers.Conv2D(1, 3, padding="same"))(x)
    return keras.Model(inputs, scores, name="sn_patch_discriminator")


def patch_weights_from_mask(masks):
    """Score-grid weights: 1 where a patch's receptive field overlaps the hole."""
    weights = masks
    for _ in range(4):
        weights = tf.nn.max_pool2d(weights, ksize=5, strides=2, padding="SAME")
    return tf.nn.max_pool2d(weights, ksize=3, strides=1, padding="SAME")
