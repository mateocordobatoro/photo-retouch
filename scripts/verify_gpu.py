"""Verify actual Metal GPU execution with a synthetic inpainting-style batch."""
import json
import numpy as np
import tensorflow as tf

assert tf.config.list_physical_devices('GPU'), 'No GPU detected'
tf.config.set_soft_device_placement(True)
with tf.device('/GPU:0'):
    image = tf.random.uniform((2, 128, 128, 3))
    mask = tf.cast(tf.random.uniform((2, 128, 128, 1)) > 0.7, tf.float32)
    inputs = tf.concat([image * (1 - mask), mask], axis=-1)
    model = tf.keras.Sequential([
        tf.keras.Input((128, 128, 4)),
        tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(3, 1, activation='sigmoid'),
    ])
    optimizer = tf.keras.optimizers.Adam()
    with tf.GradientTape() as tape:
        output = model(inputs, training=True)
        loss = tf.reduce_sum(tf.abs(image - output) * mask) / (3 * tf.reduce_sum(mask))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    assert 'GPU:0' in output.device, output.device
    assert all('GPU:0' in g.device for g in gradients), [g.device for g in gradients]
    assert all(g is not None and np.all(np.isfinite(g.numpy())) for g in gradients)
    assert np.isfinite(loss.numpy())
    print(json.dumps({'tensorflow':tf.__version__, 'output_device':output.device,
                      'loss_device':loss.device, 'gradient_devices':[g.device for g in gradients],
                      'masked_mae':float(loss.numpy()), 'optimizer_steps':int(optimizer.iterations.numpy())},indent=2))
# Also exercise Keras graph-mode training, as used by model.fit.
tf.config.set_soft_device_placement(True)
model.compile(optimizer='adam', loss='mae', jit_compile=False)
value = float(model.train_on_batch(inputs, image))
assert np.isfinite(value)
print('Keras graph-mode train_on_batch passed:', value)
