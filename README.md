<div align="center">

# Affectnet Facial Expression Recognition

**Multi-task Facial Expression Recognition using VGGFace2Net**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)](https://docs.astral.sh/uv/)

Jointly predicts **facial expression class** and **continuous valence/arousal** from a single image using a pretrained VGG-16 backbone with dual task-specific heads.

</div>

---

## Table of Contents

- [Results](#results)
- [Hardware](#hardware)
- [Overview](#overview)
- [Architecture](#architecture)
- [Setup](#setup)
- [Dataset](#dataset)
- [Usage](#usage)
- [Outputs](#outputs)
- [Metrics](#metrics)
- [Project Structure](#project-structure)

---

## Results

Trained on [AffectNet](https://mohammadmahoor.com/pages/databases/affectnetplus/) (8 expression classes):

| Metric | Train | Val | Test |
|:---|:---:|:---:|:---:|
| Accuracy | 96.4% | 69.2% | **71.1%** |
| Weighted F1 | 0.964 | 0.692 | **0.711** |
| Cohen's Kappa | 0.959 | 0.648 | 0.670 |
| CCC (Valence/Arousal) | 0.846 | 0.730 | **0.715** |
| RMSE (Valence/Arousal) | 0.265 | 0.312 | 0.319 |

---

## Hardware

Trained on AWS EC2:

| | |
|:---|:---|
| Instance | g4dn.xlarge |
| GPU | NVIDIA T4 (16 GB VRAM) |
| vCPUs | 4 |
| RAM | 16 GB |
| Storage | 100 GB gp3 EBS |
| AMI | Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04) |
| CUDA | 12.4 |
| Python | 3.12 |

---

## Overview

This project implements a multi-task deep learning model for **Facial Expression Recognition (FER)** on the AffectNet dataset. The model is trained to simultaneously solve two tasks:

- **Expression Classification:** 8 discrete categories: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
- **Valence/Arousal Regression:** continuous dimensional emotion representation

---

## Architecture

Pretrained VGG-16 with two task heads attached after the feature extractor:

```
Input (224x224x3)
      |
  VGG-16 Features     <- frozen for first 4 epochs
      |                  last 10 layers unfrozen at epoch 4
  Flatten (25088)
      |-- Expression Head: FC(256) -> ReLU -> Dropout(0.6) -> FC(8)
      |-- VA Head:         FC(256) -> ReLU -> Dropout(0.6) -> FC(2)
```

**Training:**
- Loss: `CrossEntropy(expression) + MSE(valence, arousal)`
- Optimizer: Adam, lr=1e-4, weight_decay=1e-4
- LR scheduler: ReduceLROnPlateau on validation F1
- Early stopping: patience=5

---

## Setup

**Requirements:**
- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- NVIDIA GPU with CUDA 12.4 (CPU also works)

```bash
git clone https://github.com/muhammadhaider02/Affectnet-Facial-Expression-Recognition.git
cd Affectnet-Facial-Expression-Recognition
uv sync
```

PyTorch installs with CUDA 12.4 automatically via the configured index.

---

## Dataset

Put the AffectNet dataset in `data/`:

```
data/
├── images/
│   ├── 1.jpg
│   └── ...
└── annotations/
    ├── 1_exp.npy    <- expression label (0-7)
    ├── 1_val.npy    <- valence (-1 to 1)
    ├── 1_aro.npy    <- arousal (-1 to 1)
    └── ...
```

Data split (stratified, fixed by `--seed`):

| Split | Size |
|:---|:---:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

*After 3x augmentation expansion. Augmentations: horizontal flip, rotation, color jitter, Gaussian blur, random crop.*

---

## Usage

**Full pipeline:**

```bash
# Windows
.\run.ps1

# Linux / macOS
chmod +x run.sh && ./run.sh
```

**Train only:**

```bash
uv run fer-train
```

**Evaluate only:**

```bash
uv run fer-evaluate
uv run fer-evaluate --checkpoint outputs/best_model.pt
```

**With custom args:**

```bash
uv run fer-train --epochs 20 --batch-size 32 --lr 5e-5
```

**All options:**

```bash
uv run fer-train --help
uv run fer-evaluate --help
```

| Argument | Default | Description |
|:---|:---:|:---|
| `--img-dir` | `data/images` | Images directory |
| `--ann-dir` | `data/annotations` | Annotations directory |
| `--output-dir` | `outputs` | Output directory |
| `--epochs` | `10` | Max training epochs |
| `--batch-size` | `64` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--patience` | `5` | Early stopping patience |
| `--lambda-reg` | `1.0` | VA loss weight |
| `--expand-factor` | `3` | Augmentation multiplier |
| `--val-split` | `0.15` | Validation fraction |
| `--test-split` | `0.15` | Test fraction |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | `cuda` or `cpu` |

---

## Outputs

All files are saved to `outputs/`:

| File | Stage | Description |
|:---|:---:|:---|
| `best_model.pt` | Train | Best checkpoint by val F1 (overwritten on improvement) |
| `final_model.pt` | Train | Last epoch checkpoint |
| `training_history.png` | Train | Train/val metric curves |
| `predictions.png` | Evaluate | Test set sample predictions |
| `confusion_matrix.png` | Evaluate | 8-class confusion matrix |
| `eval_results.json` | Evaluate | Test set metrics |

---

## Metrics

| Category | Metrics |
|:---|:---|
| Classification | Accuracy, Weighted F1, Cohen's Kappa, Krippendorff's Alpha, AUC, PR-AUC |
| Regression | RMSE, Pearson r, CCC, Sign Agreement |

---

## Project Structure

```
Affectnet-Facial-Expression-Recognition/
├── src/fer/
│   ├── cli.py          <- train and evaluate entry points
│   ├── dataset.py      <- AffectNet dataset + transforms
│   ├── metrics.py      <- evaluation metrics
│   ├── model.py        <- VGGFace2Net
│   ├── trainer.py      <- training loop
│   └── visualize.py    <- plots
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── data/
├── outputs/
├── run.ps1
├── run.sh
└── pyproject.toml
```
