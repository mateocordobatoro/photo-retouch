# Development Environment

## Open the project

Open the repository folder in VS Code. The workspace settings point to `.venv/bin/python`. In the notebook kernel picker, select **photo-retouch (Python 3.12)**. If needed, install the Microsoft Python and Jupyter extensions in VS Code.

```bash
source .venv/bin/activate
python -m jupyterlab
```

The environment is named `photo-retouch`; activating it displays `(photo-retouch)` in the terminal. Its directory remains `.venv`. The local virtual environment is ignored by Git. `requirements.txt` lists direct dependencies; `requirements-lock.txt` records the exact installed direct and transitive versions for this macOS ARM64 / Python 3.12 environment. Other operating systems may need a fresh dependency resolution.

## Recreate the environment

```bash
python3.12 -m venv --prompt photo-retouch .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m ipykernel install --user --name photo-retouch --display-name "photo-retouch (Python 3.12)"
python -m playwright install chromium
python -m pip check
```

## Notebooks and shared code

- `notebooks/photo_retouch_training.ipynb` records every experiment up to the low-chroma refiner, with outputs. It is no longer extended.
- `notebooks/photo_retouch_refinement.ipynb` is the current notebook. It loads the saved models and the frozen validation set and checks them against the tables in `records/` before any new experiment.
- `src/photo_retouch/` holds the reusable code imported by the current notebook: data loading, models, losses, training updates, metrics, and plotting. `pyproject.toml` declares it as the `photo-retouch` package. The editable install (`pip install -e .`) means changes take effect after a kernel restart, without reinstalling.

Trained models in `models/` are ignored by Git. Back them up separately; without them, the earlier experiments must be retrained.

## Libraries by workflow stage

| Package | Purpose |
| --- | --- |
| TensorFlow / Keras | Image input pipelines, convolutional autoencoder, automatic differentiation, optimization, callbacks, and model saving/loading. Keras is installed with TensorFlow and imported through `tf.keras`. |
| NumPy | Image arrays, masks, normalization, numerical operations, and masked MAE calculations. |
| SciPy | Scientific numerical routines and supporting image/signal operations when needed. |
| pandas | Dataset manifests, experiment records, and per-class evaluation tables. |
| scikit-learn | Reproducible train/validation splitting and supporting evaluation utilities. |
| Pillow | Read/write images, RGB conversion, resize operations, and drawing synthetic brush masks. |
| OpenCV (`cv2`) | Classical inpainting baseline and image/mask processing. OpenCV commonly uses BGR; convert explicitly when comparing with RGB pipelines. |
| scikit-image | Optional PSNR and SSIM reconstruction metrics. These supplement, not replace, masked MAE; full-image scores can be inflated by unchanged pixels. |
| Matplotlib | Static image grids, masks, before/after comparisons, and training curves suitable for the report and PDF. |
| Seaborn | Statistical plots for class distributions and evaluation summaries. |
| Plotly | Interactive training curves and exploratory result comparisons in notebooks or the app. Use static plots for reliable PDF presentation. |
| KaggleHub | Download the selected Kaggle dataset when the data stage begins. Access may require account authentication or acceptance of dataset terms. Never commit credentials. |
| psutil | Measure process memory, system memory, and CPU resources. |
| tqdm | Progress bars for loading, preprocessing, and evaluation loops. |
| JupyterLab | Browser-based notebook editor. |
| ipykernel | Python kernel shared by Jupyter and VS Code notebooks. |
| ipywidgets | Notebook controls and interactive progress widgets. |
| nbformat | Read and validate notebook files. |
| nbclient | Execute notebooks programmatically for verification. |
| nbconvert[webpdf] | Export notebooks to HTML/PDF. The WebPDF exporter uses Playwright and a separately installed Chromium browser. |
| Streamlit | Planned Python app framework for upload, controls, plots, and downloads. The brush integration still needs implementation and validation. |
| Gradio | Alternative demo framework with an ImageEditor brush component; installed for prototyping if Streamlit's brush integration is unsuitable. |

Python's standard library also provides `pathlib`, `json`, `platform`, `random`, `time`, and `io`; these do not require pip installation. Not every installed package needs to be imported in the training notebook: application and export tools belong in their respective stages.

## Export the notebook

```bash
python -m jupyter nbconvert --to webpdf notebooks/photo_retouch_training.ipynb --output-dir outputs
```

The output directory is ignored by Git. Interactive figures may need static equivalents in the final report.

## Scope

The environment prepares the tools for the workflow. The dataset has not been downloaded, and the actual model, full training run, and application have not been implemented. The environment check uses synthetic data only. Apple Metal GPU acceleration is configured for this Mac. CPU remains available as a fallback and for baseline timing.

Official references: [TensorFlow installation](https://www.tensorflow.org/install/pip), [Gradio ImageEditor](https://www.gradio.app/docs/gradio/imageeditor), [nbconvert WebPDF](https://nbconvert.readthedocs.io/en/latest/install.html#installing-chromium).

## Verified local setup

- Python 3.12.13, macOS ARM64.
- TensorFlow 2.18.1 with the official tensorflow-metal 1.2.0 plugin; Apple M5 GPU detected. TensorFlow 2.21.0 could not load this plugin, so the compatible TensorFlow version is pinned.
- `pip check`: no broken requirements.
- All notebook cells executed successfully through the registered kernel.
- A synthetic batch completed an optimizer update with a finite loss; model saving/loading passed.
- OpenCV inpainting, SSIM, PNG writing, Matplotlib rendering, Plotly HTML output, and construction of the Gradio brush component passed.

These checks validate the environment, not the quality of the future inpainting model or an end-to-end app.

## Apple M5 GPU workflow

The notebook includes device selection and an actual matrix-multiplication placement check. Restart the kernel, then run all cells. Supported TensorFlow operations use Metal automatically; auxiliary operations without Metal kernels (such as integer seed arithmetic) may execute on CPU. This is normal and soft device placement stays enabled.

Compile the future Keras model with `jit_compile=False` to avoid requiring XLA support from the Metal plugin. Keep float32 for the initial experiment. Use `with tf.device(TRAINING_DEVICE):` around model construction/training when explicitly selecting the preferred device. CPU-only comparisons must hide GPUs before TensorFlow initializes them, so use a fresh kernel/process.

```bash
source .venv/bin/activate
python scripts/verify_gpu.py
```

This verification asserts GPU placement of convolution output and gradients, applies an Adam update with masked MAE to a synthetic 128 × 128 batch, and exercises Keras graph-mode training. It is an environment check, not an experiment result. Small workloads may still run faster on CPU; benchmark the actual model before choosing batch sizes and epochs.

The Mac has 16 GB of shared system memory. TensorFlow's Metal startup message reporting a device with 0 MB is not a measurement of physical GPU memory. Monitor system/process memory with psutil during actual training.

Reference: [Apple's TensorFlow Metal plugin](https://developer.apple.com/metal/tensorflow-plugin/).
