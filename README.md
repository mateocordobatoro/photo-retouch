# Photo Retouch

Final project for **Advanced Mathematics for Deep Learning**.

Photo Retouch removes an unwanted person or object from a photo: paint over it, and the region is filled with a plausible background. The fill comes from our own model, trained from scratch on 128 × 128 landscapes and street scenes. In the app, a pretrained diffusion model then refines that fill at higher resolution.

Try it: [Photo Retouch on Hugging Face Spaces](https://huggingface.co/spaces/matcordto/photo-retouch).

## Results

The final evaluation used 400 reserved test images, 100 per category, and was run once. Lower is better for all three measures:

- **Masked MAE:** pixel error inside the hole.
- **Perceptual distance:** distance between VGG16 features.
- **KID × 1000:** how far the results' distribution is from the true images', measured with InceptionV3 features.

| Method | Masked MAE | Perceptual distance | KID × 1000 |
| --- | --- | --- | --- |
| OpenCV Telea (classical baseline) | 0.105 | 1.085 | 15.2 |
| Coarse U-Net (ours) | **0.092** | 0.884 | 6.0 |
| Low-chroma refiner (ours, final model) | 0.104 | 0.714 | 0.4 |
| Pretrained diffusion alone | 0.138 | 0.822 | −0.6 |
| Our model + pretrained diffusion (the app) | 0.103 | **0.596** | −1.5 |

- **Against OpenCV.** Both of our models beat OpenCV on perceptual distance and KID. The U-Net also has lower pixel error.
- **Refiner against U-Net.** The refiner has higher pixel error than the U-Net, but lower perceptual distance and more realistic texture.
- **The app's chain:**
  - It has the best perceptual distance of the five methods.
  - Its pixel error is the same as our refiner's.
  - It beats the pretrained model used alone on pixel error and on perceptual distance.

[`notebooks/03_evaluation.ipynb`](notebooks/03_evaluation.ipynb) has the confidence intervals, the results by category, the texture measures, and the figures.

## Repository layout

| Path | Contents |
| --- | --- |
| [`notebooks/01_training.ipynb`](notebooks/01_training.ipynb) | Data download, exploration, and splits; masks; the U-Net experiments and the OpenCV comparison; the adversarial experiments; the two-stage refiner, up to the low-chroma refiner. |
| [`notebooks/02_refinement.ipynb`](notebooks/02_refinement.ipynb) | Follow-up experiments on the refiner's texture artifacts: style weight, the source of the grid pattern, and a second training run. |
| [`notebooks/03_evaluation.ipynb`](notebooks/03_evaluation.ipynb) | The final evaluation on the reserved test set. |
| `src/photo_retouch/` | Code shared by the notebooks and the app, installed as the `photo-retouch` package. |
| `app/` | The Gradio app and the files for its Hugging Face Space. |
| `records/<snapshot>/` | Frozen data splits, masks, and result tables. Later executions must reproduce them. |
| `scripts/` | `verify_gpu.py` checks the Apple Metal setup. `build_space.py` assembles the Space folder. |

These folders are not in Git:

- `data/`: the dataset.
- `models/`: the trained models.
- `outputs/`: generated files.

## Data

The data come from [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), restricted to the `sea`, `forest`, `street`, and `buildings` categories. Images are resized to 128 × 128 RGB, with values in [0, 1].

| Split | Images | Source | Use |
| --- | --- | --- | --- |
| Training | 6,000 (1,500 per category) | `seg_train` | Training |
| Validation | 400 (100 per category) | `seg_train`, disjoint from training | Choosing every model and setting |
| Test | 400 (100 per category) | `seg_test` | Used once, in `03_evaluation.ipynb` |

Each mask is brush strokes, a rounded block, or a person silhouette (1 marks a missing pixel):

- Training draws random masks from that generator.
- Validation and test use fixed masks, saved in `records/`.
- The holes cover about 14% of each image on average.

## Models

1. **Coarse U-Net.** A deep U-Net trained from scratch with masked MAE. Its input is the masked image plus a mask channel.
2. **Low-chroma refiner.** A gated-convolution network (Yu et al., 2019) that adds detail to the U-Net's fill.
   - It is trained with masked L1, perceptual, style, and adversarial (PatchGAN) losses.
   - The colour part of its correction is blurred, so the detail it adds is mostly in brightness. This suppresses colour noise.

Both models keep the original pixels outside the hole.

## Application

`app/app.py` removes a painted region from a photo of any size:

1. Upload a JPG or PNG.
2. Paint over the region with the brush, which has an adjustable size, an eraser, and a clear button.
3. Press **Remove**.
4. Download the result as a PNG.

The fill comes from two models in sequence:

1. **Our model (stage 1):** the coarse U-Net and the low-chroma refiner, at 128 × 128, on a square crop around the painted area.
2. **Pretrained refinement (stage 2):** [Stable Diffusion 1.5 Inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting), licensed under CreativeML OpenRAIL-M.
   - It is used as is and was not trained in this project.
   - It starts from stage 1's fill, with part of that fill replaced by noise, and completes it at 512 × 512.
   - **Refinement strength** sets how much of stage 1's fill it may change. At 0 you see our model alone; the default is 0.7.

The photo is unchanged outside the painted area and a blend margin of a few pixels. A gallery shows the crop at each stage. The pipeline is in `src/photo_retouch/inpaint.py`.

Run the app locally from the project root, then open http://127.0.0.1:7860:

```bash
python app/app.py
```

The first run downloads about 2.1 GB of fp16 weights for stage 2 into the Hugging Face cache (`~/.cache/huggingface/`), outside the repository. On the Apple M5, removing a person from an 800 × 600 photo took 0.4 s with stage 1 alone, and about 8 s at strength 0.7.

### Publish the app

The same app runs on a Hugging Face Space with ZeroGPU hardware:

- Stage 2 runs on a GPU that is attached only for that call.
- Stage 1 runs on the CPU.
- The Space downloads the stage-2 weights itself; only our two models (48 MB) are uploaded.

To publish it:

1. On Hugging Face, create a Space with the SDK **Gradio** and the hardware **ZeroGPU**.
2. Assemble the Space folder. It contains the app, the package modules the app imports, and the two stage-1 models. `app/requirements-space.txt` becomes its requirements, and `app/SPACE_README.md` its card.

   ```bash
   python scripts/build_space.py
   ```

3. Log in to Hugging Face:

   ```bash
   hf auth login
   ```

4. Upload `outputs/space/` to the Space:

   ```bash
   hf upload <user>/<space> outputs/space . --repo-type space
   ```

## Reproduce

The environment uses Python 3.12 in `.venv`. [ENVIRONMENT.md](ENVIRONMENT.md) has the setup steps, the role of each package, and the Apple Metal notes.

```bash
source .venv/bin/activate
python -m jupyterlab
```

In VS Code, select **photo-retouch (Python 3.12)** as the notebook kernel.

- **Run the notebooks in order.** The second and third notebooks load the saved models and the frozen records. They check those against `records/` before doing anything new.
- **Trained models.** These are not in Git, so keep a copy of `models/`. Without it, the experiments must be retrained.
- **Diffusion cache.** Without `models/test_evaluation/`, rerunning the evaluation takes about two hours on the Apple M5.

## Academic deliverables

**Required:**

- A Python notebook organized with functions and/or classes, covering:
  - dataset provenance and the practical problem;
  - visual exploration and preprocessing;
  - model design, training, and validation;
  - prediction, evaluation, and interpretation;
  - compute and memory usage;
  - the final report;
  - individual contributions;
  - lessons learned and next steps.
- A PDF export of that notebook.
- A group presentation. The course requires a group of 5–7 students.

**Status.** The experiments, the test evaluation, and the app are done. Still to do: the final report, its PDF export, and the presentation.

## Repository hygiene

Keep these outside version control:

- datasets and personal photographs;
- model checkpoints and generated outputs;
- credentials;
- local environments and local tool settings.

Use the dataset according to its source terms.
