"""Apple Metal / CPU device selection."""
import tensorflow as tf


def configure_device():
    """Enable soft placement and check that a detected GPU actually runs ops.

    Returns the preferred device string, "/GPU:0" or "/CPU:0". Supported
    operations use Metal automatically; auxiliary operations without Metal
    kernels fall back to the CPU.
    """
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    device = "/GPU:0" if gpus else "/CPU:0"

    with tf.device(device):
        probe = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
    if gpus and "GPU:0" not in probe.device:
        raise RuntimeError("GPU detected, but the probe did not execute on it.")
    return device
