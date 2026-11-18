"""Frozen ImageNet feature extractors shared by the losses and the metrics."""
import numpy as np
from tensorflow import keras

VGG_MEAN_BGR = np.array([103.939, 116.779, 123.68], np.float32)

# Training losses use pooling outputs; the evaluation distance uses
# convolution outputs. They are related, so the perceptual distance is not
# fully independent of models trained with these losses. KID is.
LOSS_LAYERS = ["block1_pool", "block2_pool", "block3_pool"]
DISTANCE_LAYERS = [
    "block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3",
]


def vgg_preprocess(images):
    """RGB in [0, 1] to the Keras VGG16 input (BGR, 0–255, mean-subtracted)."""
    return images[..., ::-1] * 255.0 - VGG_MEAN_BGR


def vgg16_features(layer_names):
    """Frozen VGG16 model returning the named layers' activations."""
    vgg = keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(128, 128, 3)
    )
    model = keras.Model(vgg.input, [vgg.get_layer(name).output for name in layer_names])
    model.trainable = False
    return model


def inception_pool():
    """InceptionV3 with global average pooling, for KID."""
    return keras.applications.InceptionV3(
        include_top=False, weights="imagenet", pooling="avg", input_shape=(299, 299, 3)
    )
