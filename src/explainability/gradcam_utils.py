"""Tiện ích trực quan hóa cho Grad-CAM: tạo overlay, lưu hình."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.transforms import denormalize


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Chuyển tensor ảnh đã normalize thành mảng uint8 HxWx3 [0,255]."""
    img = denormalize(tensor).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def heatmap_to_color(heatmap: np.ndarray) -> np.ndarray:
    """Chuyển heatmap [0,1] thành ảnh màu (JET), trả về RGB uint8."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Chồng heatmap lên ảnh gốc.

    Args:
        image: Ảnh gốc RGB uint8 (HxWx3).
        heatmap: Heatmap [0,1] cùng kích thước HxW.
        alpha: Độ trong suốt của heatmap.
    """
    colored = heatmap_to_color(heatmap)
    overlay = cv2.addWeighted(colored, alpha, image, 1 - alpha, 0)
    return overlay


def save_gradcam_figure(
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    out_path: Path,
    true_label: str,
    pred_label: str,
    confidence: float,
) -> None:
    """Lưu hình gồm 3 panel: ảnh gốc, heatmap, overlay + thông tin nhãn."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    axes[0].imshow(image)
    axes[0].set_title("Ảnh gốc")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    correct = true_label == pred_label
    color = "green" if correct else "red"
    fig.suptitle(
        f"Thật: {true_label}  |  Dự đoán: {pred_label}  |  "
        f"Confidence: {confidence:.2%}",
        color=color,
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
