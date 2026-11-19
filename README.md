# Photo Retouch

Final project for **Advanced Mathematics for Deep Learning**.

Photo Retouch explores image inpainting: a user uploads a photo, paints a mask over an unwanted person or object, and reconstructs the selected region with a plausible background. The planned interface displays the original and result side by side and exports the edited image as PNG.

## Current status

- `notebooks/photo_retouch_training.ipynb`: data preparation, the coarse U-Net, the OpenCV comparison, the adversarial experiments, and the two-stage refiner, up to the low-chroma refiner (the current reference model).
- `notebooks/photo_retouch_refinement.ipynb`: follow-up experiments on the refiner's texture artifacts (style weight, source of the grid, a second-seed replicate).
- `notebooks/photo_retouch_evaluation.ipynb`: the final evaluation on the 400 reserved test images, used once: OpenCV, our models, the pretrained diffusion model alone, and the app's chain. The test list and masks are frozen in `records/<snapshot>/test/`.
- `app/`: the Gradio app, which chains our model with a pretrained diffusion model (see [Application](#application)).
- Not done yet: the final report sections, the PDF export, and the presentation.

## Dataset and experiment plan

Use [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), initially restricting the experiment to `sea`, `forest`, `street`, and `buildings`.

- Resize images to 128 × 128 RGB and normalize pixel values to [0, 1].
- Sample 2,000 training images (500 per class) and 400 validation images (100 per class) from disjoint subsets of `seg_train`.
- Reserve 400 test images (100 per class) from `seg_test`.
- Split images before generating masks. Use random brush-like masks during training and fixed, reproducible masks for validation and testing.
- Train a small convolutional autoencoder from scratch with TensorFlow/Keras. Input: masked RGB image plus a binary mask channel. Target: original RGB image.
- Optimize mean absolute error (MAE) inside the masked region and preserve original pixels outside the mask when composing the result.

Begin with a one-epoch CPU trial and batch size 16, reducing to 8 if memory is constrained. Record runtime, hardware, available memory, and observed memory use before deciding the full training schedule. Apple M5 Metal acceleration is configured and checked in the notebook; compare CPU/GPU timing on the actual model.

## Evaluation plan

Compare the autoencoder with an OpenCV inpainting baseline using identical test images and masks. Report masked-region MAE, training and validation loss curves, and visual comparisons of the original, masked input, baseline, and model output.

Personal photographs are reserved for private qualitative inference tests. Their hidden backgrounds are unknown, so these examples assess visual plausibility rather than reconstruction error. At 128 × 128, fine structures and large masked regions may be difficult to reconstruct.

## Application

`app/app.py` removes a painted region from a photo of any size: upload a JPG or PNG, paint over the region with the brush (adjustable size, eraser, clear), press **Remove**, and download the result as a PNG. The fill comes from two models in sequence:

1. **Our model (stage 1):** the coarse U-Net and the low-chroma refiner from the notebooks, at 128 × 128, on a square crop around the painted area.
2. **Pretrained refinement (stage 2):** Stable Diffusion 1.5 Inpainting ([`stable-diffusion-v1-5/stable-diffusion-inpainting`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting), CreativeML OpenRAIL-M license), used as is and not trained in this project. It starts from stage 1's fill with part of it replaced by noise and completes it at 512 × 512. **Refinement strength** sets how much of stage 1's fill it may change: 0 shows our model alone; the default is 0.7.

Outside the painted area and a blend margin of a few pixels, the photo is unchanged. A gallery shows the crop at each stage. The pipeline is in `src/photo_retouch/inpaint.py`.

Run it locally from the project root, then open http://127.0.0.1:7860:

```bash
python app/app.py
```

The first run downloads about 2.1 GB of fp16 weights for stage 2 into the Hugging Face cache (`~/.cache/huggingface/`), outside the repository. On the Apple M5, removing a person from an 800 × 600 photo took 0.4 s with stage 1 alone and about 8 s at strength 0.7.

### Publish the app

The same app runs on a Hugging Face Space with ZeroGPU hardware: stage 2 runs on a GPU attached only for that call, and stage 1 on the CPU. The Space downloads the stage-2 weights itself; only our two models (48 MB) are uploaded.

1. On Hugging Face, create a Space with SDK **Gradio** and hardware **ZeroGPU**.
2. Assemble the Space folder: the app, the package modules it imports, the two stage-1 models, `app/requirements-space.txt` as its requirements, and `app/SPACE_README.md` as its card.

   ```bash
   python scripts/build_space.py
   ```

3. Log in and upload `outputs/space/` to the Space:

   ```bash
   hf auth login
   hf upload <user>/<space> outputs/space . --repo-type space
   ```

## Notebook environment

The local environment uses Python 3.12 and `.venv`. See [ENVIRONMENT.md](ENVIRONMENT.md) for package roles, validation, and setup instructions.

```bash
source .venv/bin/activate
python -m jupyterlab
```

In VS Code, select **photo-retouch (Python 3.12)** as the notebook kernel. Current work is in `notebooks/photo_retouch_refinement.ipynb`, which imports the shared code in `src/photo_retouch/`. `notebooks/photo_retouch_training.ipynb` records the earlier experiments. See [ENVIRONMENT.md](ENVIRONMENT.md#notebooks-and-shared-code) for the layout.

## Academic deliverables

The completed project will include a Python notebook organized with functions and/or classes, a PDF export of that notebook, and a group presentation. The notebook will cover dataset provenance, the practical problem, visual exploration, preprocessing, model design, training and validation, prediction, evaluation, interpretation, compute and memory usage, a final report, individual contributions, lessons learned, and next steps. Hyperparameter tuning is an optional extension. The course requires a group of 5–7 students.

MAE is the planned task-specific metric because the model reconstructs pixels rather than predicting class labels. Team membership and contributions will be documented as work progresses.

## Repository hygiene

Keep datasets, personal photographs, model checkpoints, generated outputs, credentials, and local environments outside version control. Dataset use must follow its source terms. Reference mockups are design references, not model outputs.
