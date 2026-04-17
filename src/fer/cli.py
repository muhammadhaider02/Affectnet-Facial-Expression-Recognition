"""CLI entry points — train and evaluate VGGFace2Net on AffectNet."""

import argparse
import json
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .dataset import AffectNet, get_augmentation_transforms, get_base_transform
from .metrics import compute_metrics
from .model import VGGFace2Net
from .trainer import train as run_training
from .visualize import plot_confusion_matrix, plot_history, show_random_predictions


# ── Shared helpers ─────────────────────────────────────────────────────────────

def get_splits(
    dataset: AffectNet,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified 3-way split → (train_idx, val_idx, test_idx).

    The same seed always produces the same partition, so evaluate_main()
    can reconstruct the identical test set without any index file.
    """
    all_indices = list(range(len(dataset)))
    all_labels  = [dataset[i][1] for i in all_indices]

    # Step 1: carve out test set
    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_split,
        stratify=all_labels,
        random_state=seed,
    )
    # Step 2: split remainder into train + val
    trainval_labels  = [all_labels[i] for i in trainval_idx]
    effective_val    = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=effective_val,
        stratify=trainval_labels,
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def _build_dataset(args: argparse.Namespace) -> AffectNet:
    return AffectNet(
        img_directory=args.img_dir,
        ann_directory=args.ann_dir,
        base_transform=get_base_transform(),
        aug_transforms=get_augmentation_transforms(),
        expand_factor=args.expand_factor,
    )


def _common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared between train and evaluate."""
    parser.add_argument("--img-dir",       default="data/images")
    parser.add_argument("--ann-dir",       default="data/annotations")
    parser.add_argument("--output-dir",    default="outputs")
    parser.add_argument("--expand-factor", type=int,   default=3)
    parser.add_argument("--num-classes",   type=int,   default=8)
    parser.add_argument("--val-split",     type=float, default=0.15)
    parser.add_argument("--test-split",    type=float, default=0.15)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )


# ── Training entry point ───────────────────────────────────────────────────────

def _parse_train_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VGGFace2Net on AffectNet")
    _common_args(p)
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=5)
    p.add_argument("--lambda-reg", type=float, default=1.0)
    return p.parse_args()


def train_main() -> None:
    args = _parse_train_args()
    print(f"Device: {args.device}")

    dataset = _build_dataset(args)
    print(
        f"Dataset: {len(dataset.indices)} images → "
        f"{len(dataset)} samples (×{args.expand_factor})"
    )

    train_idx, val_idx, test_idx = get_splits(
        dataset,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    print(
        f"Split  Train: {len(train_idx)} | "
        f"Val: {len(val_idx)} | Test: {len(test_idx)}"
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)

    model = VGGFace2Net(num_classes=args.num_classes)

    history = run_training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        lambda_reg=args.lambda_reg,
        device=args.device,
        patience=args.patience,
        checkpoint_dir=args.output_dir,
    )

    # Final metrics summary
    print("\nFinal metrics (last epoch):")
    last = len(history["val_acc"]) - 1
    for k, v in history.items():
        print(f"  {k}: {v[last]:.4f}")

    plot_history(
        history,
        save_path=os.path.join(args.output_dir, "training_history.png"),
    )


# ── Evaluation entry point ─────────────────────────────────────────────────────

def _parse_eval_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate VGGFace2Net on the test set")
    _common_args(p)
    p.add_argument(
        "--checkpoint",
        default="outputs/best_model.pt",
        help="Path to model checkpoint (.pt file)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def evaluate_main() -> None:
    args = _parse_eval_args()
    print(f"Device: {args.device}")

    # Reconstruct the same test split using the same seed
    dataset = _build_dataset(args)
    _, _, test_idx = get_splits(
        dataset,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    test_dataset = Subset(dataset, test_idx)
    print(f"Test set: {len(test_dataset)} samples")

    # Load model from checkpoint
    model = VGGFace2Net(num_classes=args.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()
    print(
        f"Loaded checkpoint: {args.checkpoint}  "
        f"(epoch {checkpoint.get('epoch', '?')}, "
        f"val F1: {checkpoint.get('val_f1', '?'):.4f})"
    )

    # Run inference
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    all_preds, all_labels = [], []
    va_true_list: list[torch.Tensor] = []
    va_pred_list: list[torch.Tensor] = []

    with torch.no_grad():
        for imgs, exps, vas in test_loader:
            imgs = imgs.to(args.device)
            exps = exps.to(args.device)
            vas  = vas.to(args.device)
            exp_out, va_out = model(imgs)
            all_preds.extend(torch.argmax(exp_out, dim=1).cpu().numpy())
            all_labels.extend(exps.cpu().numpy())
            va_true_list.append(vas)
            va_pred_list.append(va_out)

    va_true = torch.cat(va_true_list, dim=0)
    va_pred = torch.cat(va_pred_list, dim=0)

    metrics = compute_metrics(all_labels, all_preds, va_true, va_pred, args.num_classes)

    print("\nTest Set Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics JSON
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"Metrics saved → {results_path}")

    # Save plots
    plot_confusion_matrix(
        all_labels, all_preds,
        num_classes=args.num_classes,
        save_path=os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    show_random_predictions(
        model, test_dataset, device=args.device, num_images=10,
        save_path=os.path.join(args.output_dir, "predictions.png"),
    )
