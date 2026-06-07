"""Tính các metric phân loại dùng chung."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    """Gói các metric phân loại tổng hợp."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    top3_accuracy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "top3_accuracy": self.top3_accuracy,
        }


def top_k_accuracy(
    y_true: np.ndarray, probs: np.ndarray, k: int = 3
) -> float:
    """Tính top-k accuracy.

    Args:
        y_true: Mảng nhãn thật shape (N,).
        probs: Ma trận xác suất shape (N, num_classes).
        k: Số lớp xét top-k.
    """
    if probs.shape[1] < k:
        k = probs.shape[1]
    topk = np.argsort(probs, axis=1)[:, -k:]
    hits = [y_true[i] in topk[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    top_k: int = 3,
) -> ClassificationMetrics:
    """Tính toàn bộ metric phân loại từ nhãn thật, dự đoán và xác suất."""
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        macro_recall=float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        top3_accuracy=top_k_accuracy(y_true, probs, k=top_k),
    )
