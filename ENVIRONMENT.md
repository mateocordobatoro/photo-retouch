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

- `notebooks/01_training.ipynb` records every experiment up to the low-chroma refiner, with its outputs. It is no longer extended.
- `notebooks/02_refinement.ipynb` holds the refinement experiments. Before any new experiment, it loads the saved models and the frozen validation set and checks them against the tables in `records/`.
- `notebooks/03_evaluation.ipynb` is the final evaluation on the reserved test set. Its diffusion results are cached in `models/test_evaluation/`. Without that cache, re-running it takes about two hours on the Apple M5.
- `src/photo_retouch/` holds the reusable code imported by the notebooks and the app: data loading, models, losses, training updates, metrics, plotting, write-once records, and the app's inpainting pipeline. `pyproject.toml` declares it as the `photo-retouch` package. The editable install (`pip install -e .`) means changes take effect after a kernel restart, without reinstalling.

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
| Gradio | The app's interface (`app/app.py`): ImageEditor brush, result, stage gallery, and PNG download. |
| PyTorch | Runs the app's pretrained diffusion model (stage 2) on Apple Metal (MPS), CUDA, or CPU. Not used for training. |
| diffusers, transformers, accelerate, safetensors | Load and run Stable Diffusion 1.5 Inpainting for the app's stage 2. `importlib_metadata` is pinned below 10: version 10 pulls in `lazy-importer`, a new package with no author or project links on PyPI. |

Python's standard library also provides `pathlib`, `json`, `platform`, `random`, `time`, and `io`; these do not require pip installation. Not every installed package is imported in every notebook: the application and export tools belong to their own stages.

## Export a notebook

```bash
python -m jupyter nbconvert --to webpdf notebooks/03_evaluation.ipynb --output-dir outputs
```

To export another notebook, change the name. The output directory is ignored by Git. Interactive figures may need static equivalents in the final report.

## Scope

This file covers the tools, not the results: the notebooks hold the experiments and their results. The environment checks below use synthetic data only. Apple Metal GPU acceleration is configured for this Mac. The CPU remains available as a fallback and for baseline timing.

Official references: [TensorFlow installation](https://www.tensorflow.org/install/pip), [Gradio ImageEditor](https://www.gradio.app/docs/gradio/imageeditor), [nbconvert WebPDF](https://nbconvert.readthedocs.io/en/latest/install.html#installing-chromium).

## Verified local setup

- Python 3.12.13, macOS ARM64.
- TensorFlow 2.18.1 with the official tensorflow-metal 1.2.0 plugin; Apple M5 GPU detected. TensorFlow 2.21.0 could not load this plugin, so the compatible TensorFlow version is pinned.
- `pip check`: no broken requirements.
- All notebook cells executed successfully through the registered kernel.
- A synthetic batch completed an optimizer update with a finite loss; model saving/loading passed.
- OpenCV inpainting, SSIM, PNG writing, Matplotlib rendering, Plotly HTML output, and construction of the Gradio brush component passed.

- The app ran end to end locally (PyTorch on Metal) and on its Hugging Face Space (ZeroGPU).

These checks validate the environment, not the quality of the inpainting results.

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
