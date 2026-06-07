"""Tạo và lưu confusion matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
    model_name: str,
    normalize: bool = True,
) -> None:
    """Vẽ và lưu confusion matrix.

    Args:
        y_true: Nhãn thật.
        y_pred: Nhãn dự đoán.
        class_names: Danh sách tên lớp theo index.
        out_path: Đường dẫn lưu hình.
        model_name: Tên mô hình (cho tiêu đề).
        normalize: Chuẩn hóa theo hàng (tỉ lệ) hay đếm tuyệt đối.
    """
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    if normalize:
        with np.errstate(all="ignore"):
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_display = np.nan_to_num(cm_display)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    size = max(10, num_classes * 0.45)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        annot_kws={"size": 6},
        ax=ax,
    )
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thật")
    title = "Confusion Matrix" + (" (chuẩn hóa)" if normalize else "")
    ax.set_title(f"{model_name} - {title}")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
