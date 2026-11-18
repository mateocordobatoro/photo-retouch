"""Training losses for the refiners."""
import tensorflow as tf


def hole_l1(images, results, masks):
    """Mean absolute error over hole pixels."""
    return tf.math.divide_no_nan(
        tf.reduce_sum(tf.abs(images - results) * masks), 3.0 * tf.reduce_sum(masks)
    )


def perceptual_and_style(truth_features, result_features, masks):
    """Hole-weighted VGG feature L1 and Gram-matrix (style) L1, averaged over layers.

    A Gram matrix records which feature channels occur together, regardless
    of position, so the style term measures texture character rather than
    pixel alignment.
    """
    perceptual = 0.0
    style = 0.0

    for truth, result in zip(truth_features, result_features):
        weights = tf.image.resize(masks, tf.shape(truth)[1:3], method="area")
        channels = tf.cast(tf.shape(truth)[-1], tf.float32)

        perceptual += tf.math.divide_no_nan(
            tf.reduce_sum(tf.abs(truth - result) * weights),
            tf.reduce_sum(weights) * channels,
        )

        # Gram matrices over the hole region of each image.
        area = tf.reduce_sum(weights, axis=[1, 2, 3])[:, None, None]
        normalizer = area * channels + 1e-6
        truth_gram = tf.einsum("bhwc,bhwd->bcd", truth * weights, truth) / normalizer
        result_gram = tf.einsum("bhwc,bhwd->bcd", result * weights, result) / normalizer
        style += tf.reduce_mean(tf.abs(truth_gram - result_gram))

    layers_count = len(truth_features)
    return perceptual / layers_count, style / layers_count


def patch_mean(values, weights):
    """Weighted mean over discriminator patches."""
    return tf.math.divide_no_nan(tf.reduce_sum(values * weights), tf.reduce_sum(weights))


def hinge_discriminator_loss(real_scores, fake_scores, weights):
    return (
        patch_mean(tf.nn.relu(1.0 - real_scores), weights)
        + patch_mean(tf.nn.relu(1.0 + fake_scores), weights)
    )
