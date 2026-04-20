"""
Classification metrics for binary temporal contact (extends frame-wise collision metrics).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Parameters
    ----------
    labels : (N,) int {0,1}
    preds : (N,) int {0,1}
    probs : (N,) float optional
        Predicted probability of class 1 (contact). Used for ROC-AUC / PR-AUC.
    """
    labels = np.asarray(labels).ravel()
    preds = np.asarray(preds).ravel()

    out: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }

    if probs is not None:
        probs = np.asarray(probs).ravel()
        try:
            if len(np.unique(labels)) > 1:
                out["roc_auc"] = float(roc_auc_score(labels, probs))
                out["pr_auc"] = float(average_precision_score(labels, probs))
            else:
                out["roc_auc"] = float("nan")
                out["pr_auc"] = float("nan")
        except ValueError:
            out["roc_auc"] = float("nan")
            out["pr_auc"] = float("nan")

    return out


def confusion_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Sensitivity, specificity, NPV, PPV from a 2×2 confusion matrix."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        "ppv": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
    }
