"""Sinh classification report theo từng lớp và lưu ra CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> pd.DataFrame:
    """Tạo classification report dạng bảng và lưu CSV.

    Returns:
        DataFrame report (precision/recall/f1/support theo từng lớp).
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).transpose()
    df.index.name = "class_name"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, encoding="utf-8-sig")
    return df
