"""Evaluation metrics for facial expression recognition."""

import numpy as np
import torch
import krippendorff
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Root Mean Squared Error over valence/arousal predictions."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def concordance_ccc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Concordance Correlation Coefficient (CCC)."""
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return float((2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8))


def krippendorffs_alpha(labels: np.ndarray, preds: np.ndarray) -> float:
    """Krippendorff's alpha for nominal (categorical) data."""
    data = np.array([labels, preds])
    return float(krippendorff.alpha(reliability_data=data, level_of_measurement="nominal"))


def correlation(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Pearson correlation between flattened predictions and targets."""
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_pred_np = y_pred.detach().cpu().numpy().flatten()
    corr, _ = pearsonr(y_true_np, y_pred_np)
    return float(corr)


def sign_agreement(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Fraction of samples where prediction and target share the same sign deviation."""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    return float(
        np.mean(np.sign(y_true_np - np.mean(y_true_np)) == np.sign(y_pred_np - np.mean(y_pred_np)))
    )


def compute_metrics(
    labels: list,
    preds: list,
    va_true: torch.Tensor,
    va_pred: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Aggregate all metrics for one epoch pass.

    Args:
        labels:      Ground-truth expression class indices.
        preds:       Predicted expression class indices.
        va_true:     Ground-truth valence/arousal tensor (N, 2).
        va_pred:     Predicted valence/arousal tensor (N, 2).
        num_classes: Number of expression classes.

    Returns:
        Dictionary with keys: acc, f1, kappa, alpha, auc, pr_auc,
        rmse, corr, sam, ccc.
    """
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)

    acc = accuracy_score(labels_arr, preds_arr)
    f1 = f1_score(labels_arr, preds_arr, average="weighted")
    kappa = cohen_kappa_score(labels_arr, preds_arr)
    alpha = krippendorffs_alpha(labels_arr, preds_arr)

    probs = np.eye(num_classes)[preds_arr]
    try:
        auc = roc_auc_score(labels_arr, probs, multi_class="ovr")
        pr_auc = average_precision_score(
            np.eye(num_classes)[labels_arr], probs, average="macro"
        )
    except ValueError:
        auc, pr_auc = float("nan"), float("nan")

    return {
        "acc": acc,
        "f1": f1,
        "kappa": kappa,
        "alpha": alpha,
        "auc": auc,
        "pr_auc": pr_auc,
        "rmse": rmse(va_true, va_pred),
        "corr": correlation(va_true, va_pred),
        "sam": sign_agreement(va_true, va_pred),
        "ccc": concordance_ccc(va_true, va_pred),
    }
