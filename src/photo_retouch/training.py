"""Loss-weight calibration and the adversarial refiner update."""
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .features import vgg_preprocess
from .losses import hinge_discriminator_loss, hole_l1, patch_mean, perceptual_and_style
from .models import build_patch_discriminator, build_refiner, patch_weights_from_mask, refine
from .paths import MODEL_DIR

LEARNING_RATE = 1e-4
ADV_RATIO = 0.5


def saved_loss_weights(checkpoint_dir=MODEL_DIR / "refinement"):
    """(perceptual weight, style weight, adversarial ratio) stored by the refiner run."""
    path = tf.train.latest_checkpoint(str(checkpoint_dir))
    if path is None:
        raise FileNotFoundError(f"No checkpoint in {checkpoint_dir}.")
    values = tf.train.load_variable(path, "loss_weights/.ATTRIBUTES/VARIABLE_VALUE")
    return tuple(float(value) for value in values)


def calibrate_loss_weights(refiner, batches, coarse_composite, loss_features,
                           steps=4, refine_fn=refine):
    """Weights that equalize each term's gradient norm with masked L1's.

    Gradients are taken with respect to the refined image, from an identity
    (newly built) refiner, on the first ``steps`` training batches. Returns
    the perceptual weight, the style weight, and the per-batch table.
    """
    rows = []
    for step in range(steps):
        images, masks = batches(step)
        coarse = coarse_composite(images, masks)
        truth_features = loss_features(vgg_preprocess(images), training=False)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(coarse)
            result = refine_fn(refiner, coarse, masks)
            l1 = hole_l1(images, result, masks)
            perceptual, style = perceptual_and_style(
                truth_features,
                loss_features(vgg_preprocess(result), training=False),
                masks,
            )

        rows.append({
            "l1": float(l1),
            "perceptual": float(perceptual),
            "style": float(style),
            "l1_grad": float(tf.norm(tape.gradient(l1, coarse))),
            "perceptual_grad": float(tf.norm(tape.gradient(perceptual, coarse))),
            "style_grad": float(tf.norm(tape.gradient(style, coarse))),
        })
        del tape

    table = pd.DataFrame(rows)
    if not (table[["l1_grad", "perceptual_grad", "style_grad"]] > 0).all().all():
        raise ValueError("A loss term has zero gradient; calibration is undefined.")
    perceptual_weight = float((table["l1_grad"] / table["perceptual_grad"]).mean())
    style_weight = float((table["l1_grad"] / table["style_grad"]).mean())
    return perceptual_weight, style_weight, table


@dataclass
class RefinerArm:
    """One refiner trained adversarially, with its own discriminator and optimizers."""

    key: str
    refiner: keras.Model
    discriminator: keras.Model
    refine_fn: Callable
    style_weight: float
    refiner_optimizer: keras.optimizers.Optimizer
    discriminator_optimizer: keras.optimizers.Optimizer


def new_arm(key, refine_fn, style_weight, initial_refiner, initial_discriminator,
            learning_rate=LEARNING_RATE):
    """Copy shared initial weights into a fresh refiner and discriminator."""
    refiner = build_refiner(f"refiner_{key}")
    refiner.set_weights(initial_refiner.get_weights())
    discriminator = build_patch_discriminator()
    discriminator.set_weights(initial_discriminator.get_weights())
    return RefinerArm(
        key=key,
        refiner=refiner,
        discriminator=discriminator,
        refine_fn=refine_fn,
        style_weight=style_weight,
        refiner_optimizer=keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.999),
        discriminator_optimizer=keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.999),
    )


def make_train_step(arms, coarse_composite, loss_features, perceptual_weight,
                    adv_ratio=ADV_RATIO):
    """Compiled update for several arms on one shared batch.

    For each arm: update its discriminator (hinge loss on hole patches), then
    its refiner with masked L1 + perceptual + style + an adaptive adversarial
    term. The adversarial coefficient is set each step so that its gradient
    norm with respect to the composed output is ``adv_ratio`` times the
    reconstruction gradient norm. Both gradients are combined before one
    backward pass through the refiner. The style distance is logged even for
    arms whose style weight is zero.
    """

    @tf.function
    def train_step(images, masks):
        coarse = coarse_composite(images, masks)
        weights = patch_weights_from_mask(masks)
        truth_features = loss_features(vgg_preprocess(images), training=False)
        real_input = tf.concat([images, masks], axis=-1)
        values = {}

        for arm in arms:
            fake = tf.stop_gradient(arm.refine_fn(arm.refiner, coarse, masks))
            with tf.GradientTape() as tape:
                real_scores = arm.discriminator(real_input, training=True)
                fake_scores = arm.discriminator(
                    tf.concat([fake, masks], axis=-1), training=True
                )
                d_loss = hinge_discriminator_loss(real_scores, fake_scores, weights)
            gradients = tape.gradient(d_loss, arm.discriminator.trainable_variables)
            arm.discriminator_optimizer.apply_gradients(
                zip(gradients, arm.discriminator.trainable_variables)
            )

            with tf.GradientTape(persistent=True) as tape:
                result = arm.refine_fn(arm.refiner, coarse, masks, training=True)
                l1 = hole_l1(images, result, masks)
                perceptual, style = perceptual_and_style(
                    truth_features,
                    loss_features(vgg_preprocess(result), training=False),
                    masks,
                )
                reconstruction = (
                    l1 + perceptual_weight * perceptual + arm.style_weight * style
                )
                generated_scores = arm.discriminator(
                    tf.concat([result, masks], axis=-1), training=False
                )
                adversarial = -patch_mean(generated_scores, weights)

            reconstruction_gradient = tape.gradient(reconstruction, result)
            adversarial_gradient = tape.gradient(adversarial, result)
            adv_weight = tf.clip_by_value(
                adv_ratio * tf.norm(reconstruction_gradient)
                / (tf.norm(adversarial_gradient) + 1e-12),
                0.0, 1e4,
            )
            variables = arm.refiner.trainable_variables
            gradients = tape.gradient(
                result, variables,
                output_gradients=reconstruction_gradient + adv_weight * adversarial_gradient,
            )
            del tape
            arm.refiner_optimizer.apply_gradients(zip(gradients, variables))

            values.update({
                f"{arm.key}_l1": l1,
                f"{arm.key}_perceptual": perceptual,
                f"{arm.key}_style": style,
                f"{arm.key}_adv_weight": adv_weight,
                f"{arm.key}_discriminator_loss": d_loss,
            })

        return values

    return train_step
