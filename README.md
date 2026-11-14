# Photo Retouch

Final project for **Advanced Mathematics for Deep Learning**.

Photo Retouch explores image inpainting: a user uploads a photo, paints a mask over an unwanted person or object, and reconstructs the selected region with a plausible background. The planned interface displays the original and result side by side and exports the edited image as PNG.

## Current status

Initial project setup. The training notebook contains library imports only. Data preparation, training, evaluation, and the application are planned work; no trained model or experimental results are included yet.

## Dataset and experiment plan

Use [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), initially restricting the experiment to `sea`, `forest`, `street`, and `buildings`.

- Resize images to 128 × 128 RGB and normalize pixel values to [0, 1].
- Sample 2,000 training images (500 per class) and 400 validation images (100 per class) from disjoint subsets of `seg_train`.
- Reserve 400 test images (100 per class) from `seg_test`.
- Split images before generating masks. Use random brush-like masks during training and fixed, reproducible masks for validation and testing.
- Train a small convolutional autoencoder from scratch with TensorFlow/Keras. Input: masked RGB image plus a binary mask channel. Target: original RGB image.
- Optimize mean absolute error (MAE) inside the masked region and preserve original pixels outside the mask when composing the result.

Begin with a one-epoch CPU trial and batch size 16, reducing to 8 if memory is constrained. Record runtime, hardware, available memory, and observed memory use before deciding the full training schedule. Hardware acceleration must be checked in the actual environment.

## Evaluation plan

Compare the autoencoder with an OpenCV inpainting baseline using identical test images and masks. Report masked-region MAE, training and validation loss curves, and visual comparisons of the original, masked input, baseline, and model output.

Personal photographs are reserved for private qualitative inference tests. Their hidden backgrounds are unknown, so these examples assess visual plausibility rather than reconstruction error. At 128 × 128, fine structures and large masked regions may be difficult to reconstruct.

## Planned application

Upload a JPG or PNG, paint a mask with an adjustable brush, clear the mask when needed, run inpainting, inspect the result, and download a PNG. The interface will use the labels **Upload your photo**, **Choose photo**, **Original**, **Brush size**, **Clear mask**, **Result**, **Remove**, and **Download PNG**. Local inference will be validated before choosing public demo hosting.

## Notebook environment

Use a Python virtual environment with a compatible TensorFlow installation. The initial notebook uses these packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install tensorflow numpy pandas matplotlib pillow opencv-python scikit-learn psutil tqdm jupyterlab ipykernel
python -m jupyterlab
```

Open `notebooks/photo_retouch_training.ipynb` and select the virtual environment's Python kernel. Package compatibility and versions will be recorded once the training environment has been validated. The notebook does not download data, install packages, or start training.

## Academic deliverables

The completed project will include a Python notebook organized with functions and/or classes, a PDF export of that notebook, and a group presentation. The notebook will cover dataset provenance, the practical problem, visual exploration, preprocessing, model design, training and validation, prediction, evaluation, interpretation, compute and memory usage, a final report, individual contributions, lessons learned, and next steps. Hyperparameter tuning is an optional extension. The course requires a group of 5–7 students.

MAE is the planned task-specific metric because the model reconstructs pixels rather than predicting class labels. Team membership and contributions will be documented as work progresses.

## Repository hygiene

Keep datasets, personal photographs, model checkpoints, generated outputs, credentials, and local environments outside version control. Dataset use must follow its source terms. Reference mockups are design references, not model outputs.
