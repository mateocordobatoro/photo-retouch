"""Reusable code for the Photo Retouch inpainting notebooks.

Submodules:

- ``paths``: project, model, and record locations.
- ``device``: Apple Metal / CPU device selection.
- ``data``: image loading, the frozen validation and test sets, masks,
  training batches.
- ``models``: the coarse U-Net, the gated-convolution refiner, the patch
  discriminator, the refine functions, and the OpenCV baseline.
- ``features``: frozen VGG16 and InceptionV3 feature extractors.
- ``losses``: masked L1, perceptual, style, and hinge losses.
- ``training``: loss-weight calibration and the adversarial refiner update.
- ``metrics``: masked MAE, frequency errors, perceptual distance, KID,
  detail energy, grid periodicity, and paired comparisons.
- ``plotting``: side-by-side visual comparisons.
- ``records``: write-once result tables that later executions must reproduce.
- ``inpaint``: the app's pipeline, our model followed by a pretrained
  diffusion model, on photos of any size.

The package does not import TensorFlow here, so ``import photo_retouch``
stays cheap; import the submodules you need.
"""

__version__ = "0.1.0"
