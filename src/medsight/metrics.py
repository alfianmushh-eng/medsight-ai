"""Evaluation metrics for classification and segmentation tasks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    support_pos: int
    support_neg: int


def binary_classification(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    """Compute standard binary classification metrics.

    ``y_true`` and ``y_pred`` are 0/1 integer arrays of equal length.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must share the same shape")

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    eps = 1e-12
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        support_pos=int((y_true == 1).sum()),
        support_neg=int((y_true == 0).sum()),
    )


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, *, eps: float = 1e-8) -> float:
    """Compute the Dice (F1) score between two boolean masks."""
    a = pred_mask.astype(bool).flatten()
    b = true_mask.astype(bool).flatten()
    inter = float((a & b).sum())
    return (2 * inter) / (a.sum() + b.sum() + eps)


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray, *, eps: float = 1e-8) -> float:
    """Compute the Jaccard / IoU score between two boolean masks."""
    a = pred_mask.astype(bool).flatten()
    b = true_mask.astype(bool).flatten()
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / (union + eps)
