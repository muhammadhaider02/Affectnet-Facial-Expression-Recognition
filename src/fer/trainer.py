"""Training loop for VGGFace2Net."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .metrics import compute_metrics


def train(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_classes: int = 8,
    batch_size: int = 64,
    lr: float = 1e-4,
    epochs: int = 30,
    lambda_reg: float = 1.0,
    device: str = "cuda",
    patience: int = 5,
    checkpoint_dir: str = "outputs",
) -> dict[str, list[float]]:
    """Train VGGFace2Net with early stopping, LR scheduling, and best-model checkpointing.

    Checkpointing strategy (standard best-practice):
        - After each epoch, if validation F1 improves, overwrite ``best_model.pt``.
        - This means ``best_model.pt`` always contains the single best weights seen
          so far, avoiding wasted disk space from saving every epoch.
        - At the end of training, a separate ``final_model.pt`` is also saved for
         reference (the last epoch, which may not be the best).

    Returns:
        history dict with train_* / val_* lists for every tracked metric.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = model.to(device)

    # ── Freeze backbone initially ──────────────────────────────────────────
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    metric_keys = [
        "acc", "f1", "kappa", "alpha",
        "auc", "pr_auc",
        "rmse", "corr", "sam", "ccc",
    ]
    history: dict[str, list[float]] = {
        f"{split}_{k}": []
        for split in ("train", "val")
        for k in metric_keys
    }

    best_f1: float = 0.0
    patience_counter: int = 0

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        all_preds, all_labels = [], []
        va_true_list: list[torch.Tensor] = []
        va_pred_list: list[torch.Tensor] = []

        for imgs, exps, vas in train_loader:
            imgs = imgs.to(device)
            exps = exps.to(device)
            vas = vas.to(device)

            optimizer.zero_grad()
            exp_out, va_out = model(imgs)
            loss = ce_loss(exp_out, exps) + lambda_reg * mse_loss(va_out, vas)
            loss.backward()
            optimizer.step()

            all_preds.extend(torch.argmax(exp_out, dim=1).cpu().numpy())
            all_labels.extend(exps.cpu().numpy())
            va_true_list.append(vas.detach())
            va_pred_list.append(va_out.detach())

        va_true = torch.cat(va_true_list, dim=0)
        va_pred = torch.cat(va_pred_list, dim=0)
        train_metrics = compute_metrics(all_labels, all_preds, va_true, va_pred, num_classes)
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_preds, val_labels = [], []
        val_va_true_list: list[torch.Tensor] = []
        val_va_pred_list: list[torch.Tensor] = []

        with torch.no_grad():
            for imgs, exps, vas in val_loader:
                imgs = imgs.to(device)
                exps = exps.to(device)
                vas = vas.to(device)
                exp_out, va_out = model(imgs)
                val_preds.extend(torch.argmax(exp_out, dim=1).cpu().numpy())
                val_labels.extend(exps.cpu().numpy())
                val_va_true_list.append(vas)
                val_va_pred_list.append(va_out)

        val_va_true = torch.cat(val_va_true_list, dim=0)
        val_va_pred = torch.cat(val_va_pred_list, dim=0)
        val_metrics = compute_metrics(val_labels, val_preds, val_va_true, val_va_pred, num_classes)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)

        # ── Logging ────────────────────────────────────────────────────────
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train  Acc: {train_metrics['acc']:.3f}  F1: {train_metrics['f1']:.3f}  CCC: {train_metrics['ccc']:.3f} | "
            f"Val    Acc: {val_metrics['acc']:.3f}  F1: {val_metrics['f1']:.3f}  CCC: {val_metrics['ccc']:.3f}"
        )

        scheduler.step(val_metrics["f1"])

        # ── Best-model checkpoint ──────────────────────────────────────────
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_f1,
                    "val_acc": val_metrics["acc"],
                },
                best_model_path,
            )
            print(f"  ✓ New best model saved (val F1: {best_f1:.4f}) → {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        # ── Progressive unfreezing (epoch 4) ──────────────────────────────
        if epoch == 3:
            for param in model.backbone.features[-10:].parameters():
                param.requires_grad = True
            print("  Unfroze last 10 backbone feature layers.")

    # ── Save final model ───────────────────────────────────────────────────
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_model_path,
    )
    print(f"Final model saved → {final_model_path}")

    return history
