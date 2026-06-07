"""Đo thời gian inference và các metric triển khai của mô hình."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn


def measure_model_size_mb(checkpoint_path: str | Path) -> float:
    """Đo kích thước file checkpoint (MB)."""
    path = Path(checkpoint_path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    runs: int = 100,
    warmup: int = 10,
) -> float:
    """Đo thời gian inference trung bình cho một ảnh (ms).

    Args:
        model: Mô hình ở chế độ eval.
        device: Thiết bị chạy.
        image_size: Kích thước cạnh ảnh đầu vào.
        runs: Số lần đo.
        warmup: Số lần chạy làm nóng (không tính giờ).

    Returns:
        Thời gian trung bình mỗi ảnh (mili-giây).
    """
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    return (elapsed / runs) * 1000.0
