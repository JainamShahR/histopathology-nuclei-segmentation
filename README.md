# Histopathology Nuclei Segmentation — PyTorch U-Net Starter
This repository contains a working **binary segmentation baseline** and dataset preparation tools. It does **not** yet contain results from training on the real MoNuSeg dataset, clinically validated predictions, nucleus instance counting, or whole-slide OpenSlide processing. The demo uses **synthetic circles, not patient tissue**.

## Goal

Develop an educational computer-vision pipeline to segment cell nuclei in H&E-stained histopathology images, evaluate binary segmentation quality, and eventually support high-resolution tiled inference. This is a software-learning project, **not a clinical diagnostic tool**.

### What works now

- [x] A compact U-Net implemented in PyTorch (binary foreground/background segmentation)
- [x] Image + mask data loading, with correct interpolation for binary masks
- [x] MoNuSeg XML polygon annotations → binary masks
- [x] Split original images into training/validation sets **before** making patches (avoids tile leakage)
- [x] Patch extraction and overlapping tiled model inference on regular RGB images
- [x] Training script, checkpoint saving, validation Dice/IoU, prediction overlays
- [x] Generated **synthetic** smoke-test dataset and unit tests

### Still to implement / verify

- [ ] Download MoNuSeg and successfully train/evaluate the baseline on **real** tissue images
- [ ] Augmentations and stain normalization experiments
- [ ] Instance separation via distance transform / watershed and nucleus-count evaluation
- [ ] Whole-slide SVS/WSI reading with **OpenSlide** (regular tiled RGB inference already works)
- [ ] Compare models and report reproducible real-data validation scores

## Repository layout

```text
src/histoseg/
  model.py             # Small U-Net model
  metrics.py           # Binary Dice, IoU and loss
  data.py              # Paired RGB image / binary mask dataset
  monuseg.py           # XML polygon annotation to mask
  tiles.py             # Patch positions and tile extraction
scripts/
  make_demo_data.py    # Creates synthetic images for smoke tests
  prepare_monuseg.py   # Prepare real MoNuSeg image/XML pairs as patches
  train.py             # Baseline training
  evaluate.py          # Held-out-patch metrics
  predict.py           # Tiled inference and visual outputs
tests/                 # Unit tests
docs/synthetic_example.png # Left: synthetic input; right: synthetic mask
```

## Quick start (no real medical data required)

Requirements: Python **3.11+**. CPU works for the demo; CUDA-enabled Kaggle notebooks can accelerate real training.

```bash
# clone or extract this repository, then from its root:
python3 -m venv .venv
source .venv/bin/activate             # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# 1. Verify the core code
pytest -q

# 2. Generate purely SYNTHETIC demo images and masks
python scripts/make_demo_data.py

# 3. Run a short end-to-end training smoke test
python scripts/train.py --data data/demo --epochs 8 --batch-size 2 \
  --image-size 128 --base-channels 8 --out-dir runs/demo

# 4. Recalculate metrics using the best saved model
python scripts/evaluate.py --data data/demo \
  --checkpoint runs/demo/best.pt --output runs/demo/eval.json

# 5. Run tiled inference on a synthetic example
python scripts/predict.py \
  --image data/demo/val/images/toy_val_000.png \
  --checkpoint runs/demo/best.pt \
  --tile-size 128 --overlap 32 \
  --output-dir runs/demo/prediction
```

On Windows PowerShell use `.venv\Scripts\Activate.ps1` for activation. If you need a specific CUDA PyTorch wheel, follow the [official installation selector](https://pytorch.org/get-started/locally/) rather than guessing a CUDA version.

Check `runs/demo/prediction/predicted_mask.png`, `overlay.png`, and `runs/demo/eval.json`. **Numbers from this tiny synthetic set are not medical image results, and short training is not expected to produce useful nuclei predictions.**

## Working with the actual MoNuSeg dataset

1. Read the official dataset page: https://monuseg.grand-challenge.org/Data/
2. Download and extract the **training** tissue images and their matching XML annotations. The dataset is not redistributed in this repo.
3. Point the conversion script at the extracted image and XML directories. Each image stem should match an XML stem, e.g. `TCGA_example.tif` and `TCGA_example.xml`. Directory naming varies with archive extraction.

```bash
python scripts/prepare_monuseg.py \
  --images /path/to/MoNuSegTrainingData/images \
  --annotations /path/to/MoNuSegTrainingData/annotations \
  --output data/monuseg --tile-size 256 --stride 256 \
  --val-ratio 0.2 --seed 42

# Train on the extracted original-image-level split
python scripts/train.py --data data/monuseg \
  --image-size 256 --base-channels 16 --epochs 20 \
  --batch-size 4 --out-dir runs/monuseg

python scripts/evaluate.py --data data/monuseg \
  --checkpoint runs/monuseg/best.pt --output runs/monuseg/val_metrics.json
```

**Evaluation warning:** The script creates a reproducible **original-image-level holdout** from the training images. It is **not** the official challenge test evaluation. Do not report a result from `train` patches or the synthetic demo as generalization performance. Validation metrics are computed on patches, not whole-slide/instance masks. Do not compare them directly with published challenge metrics without matching protocols.

**Model warning:** Binary segmentation does not automatically separate or count individual touching nuclei. That later stage requires instance postprocessing and its own evaluation.

## Reproducibility and next improvements

- Random seed in training and data split defaults to `42`.
- The dataset split names are recorded at `data/monuseg/split_manifest.json`.
- Training logs go to `runs/<experiment>/history.json`; the best checkpoint to `best.pt`.
- Future steps: evaluate on real MoNuSeg images, augmentations, watershed-based instance separation, true whole-slide IO with OpenSlide, performance profiling.

## Dataset attribution and license notes

MoNuSeg was developed by Kumar *et al.*, including *A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology* (IEEE TMI, 2017). The [official dataset page](https://monuseg.grand-challenge.org/Data/) provides download, citation, and license information. The dataset is licensed **CC BY-NC-SA 4.0**; **do not commit the source images or XML annotations to this repository** without confirming redistribution permissions. Code in this starter is MIT licensed; **that does not change the MoNuSeg dataset's terms**.

## Ethical use

For research and education only. No clinical decisions, pathology diagnoses, or medical claims should be made from this baseline. The generated demo images are synthetic and do not represent real tissue morphology.
