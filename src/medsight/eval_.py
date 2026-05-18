Extended evaluation metrics: per-class, confusion matrix, ROC.
from __future__ import annotations
import numpy as np
import pandas as pd

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[str] | None = None) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(y_true, y_pred)
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    return pd.DataFrame(cm, index=labels, columns=labels)

def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[str] | None = None) -> pd.DataFrame:
    from sklearn.metrics import precision_score, recall_score, f1_score
    rows = []
    for i, cls in enumerate(labels or [str(i) for i in range(len(np.unique(y_true)))]):
        rows.append({"class": cls, "precision": precision_score(y_true, y_pred, labels=[i], average="micro"), "recall": recall_score(y_true, y_pred, labels=[i], average="micro"), "f1": f1_score(y_true, y_pred, labels=[i], average="micro")})
    return pd.DataFrame(rows)

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {"auc": auc, "fpr": fpr.tolist(), "tpr": tpr.tolist()}
