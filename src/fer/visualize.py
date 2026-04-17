"""Visualization utilities for training history and predictions."""

import os
import random

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — save to file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import Dataset

# AffectNet 8-class label mapping
EXPRESSION_LABELS: dict[int, str] = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
}


def plot_history(
    history: dict[str, list[float]],
    save_path: str = "outputs/training_history.png",
) -> None:
    """Plot train/val metric curves and save to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history["train_acc"]) + 1)
    metrics = [
        ("acc",   "Accuracy"),
        ("f1",    "F1-Score"),
        ("kappa", "Cohen Kappa"),
        ("rmse",  "RMSE (Val/Aro)"),
        ("ccc",   "CCC (Val/Aro)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):
        ax.plot(epochs, history[f"train_{key}"], "b-o", label="Train", markersize=4)
        ax.plot(epochs, history[f"val_{key}"],   "r-o", label="Val",   markersize=4)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training history plot saved → {save_path}")


def plot_confusion_matrix(
    labels: list,
    preds: list,
    num_classes: int = 8,
    save_path: str = "outputs/confusion_matrix.png",
) -> None:
    """Compute and save a confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    class_names = [EXPRESSION_LABELS.get(i, str(i)) for i in range(num_classes)]
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix (Test Set)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


def show_random_predictions(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    num_images: int = 10,
    save_path: str = "outputs/predictions.png",
) -> None:
    """Display a random sample of images with GT vs predicted labels, saved to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.eval()
    indices = random.sample(range(len(dataset)), num_images)

    cols = 5
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        img, label, _ = dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            exp_out, _ = model(img_tensor)
            pred_label = torch.argmax(exp_out, dim=1).item()

        img_disp = img.permute(1, 2, 0).cpu().numpy()
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

        gt_name   = EXPRESSION_LABELS.get(int(label), str(label))
        pred_name = EXPRESSION_LABELS.get(int(pred_label), str(pred_label))
        color = "green" if label == pred_label else "red"

        ax.imshow(img_disp)
        ax.axis("off")
        ax.set_title(f"GT: {gt_name}\nPred: {pred_name}", fontsize=8, color=color)

    for ax in axes[len(indices):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Predictions plot saved → {save_path}")
