"""Callback hỗ trợ huấn luyện: early stopping và lưu checkpoint tốt nhất."""

from __future__ import annotations

from pathlib import Path

import torch


class EarlyStopping:
    """Dừng sớm khi metric theo dõi không cải thiện sau ``patience`` epoch.

    Hỗ trợ hai chế độ:
        - mode="min": metric càng nhỏ càng tốt (vd: validation loss).
        - mode="max": metric càng lớn càng tốt (vd: macro F1).
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def step(self, value: float) -> bool:
        """Cập nhật trạng thái với giá trị metric mới.

        Returns:
            True nếu đây là giá trị tốt nhất từ trước tới nay.
        """
        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class CheckpointSaver:
    """Lưu checkpoint tốt nhất dựa trên metric theo dõi."""

    def __init__(self, checkpoint_dir: str | Path, model_name: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.best_path = self.checkpoint_dir / "best.pt"

    def save(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: dict[str, float],
        class_mapping: dict[str, int],
    ) -> Path:
        """Lưu trạng thái mô hình tốt nhất."""
        payload = {
            "model_name": self.model_name,
            "model_state": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "class_mapping": class_mapping,
        }
        torch.save(payload, self.best_path)
        return self.best_path
