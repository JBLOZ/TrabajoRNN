from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_global_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, Any]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        try:
            result["macro_ovr_roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            result["macro_ovr_roc_auc"] = None
    return result


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: dict[int, str]) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=[class_names[i] for i in sorted(class_names)],
        output_dict=True,
        zero_division=0,
    )
    return report


def multiclass_curve_data(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    result: dict[str, Any] = {"roc_auc": {}, "pr_auc": {}}
    for cls in range(num_classes):
        mask = (y_true == cls).astype(int)
        if mask.sum() == 0:
            result["roc_auc"][cls] = None
            result["pr_auc"][cls] = None
            continue
        try:
            result["roc_auc"][cls] = float(roc_auc_score(mask, y_prob[:, cls]))
        except Exception:
            result["roc_auc"][cls] = None
        try:
            result["pr_auc"][cls] = float(average_precision_score(mask, y_prob[:, cls]))
        except Exception:
            result["pr_auc"][cls] = None
    return result


def build_evaluation_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: dict[int, str],
) -> dict[str, Any]:
    num_classes = len(class_names)
    return {
        "global_metrics": compute_global_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob),
        "classification_report": per_class_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "curves": multiclass_curve_data(y_true=y_true, y_prob=y_prob, num_classes=num_classes),
    }
