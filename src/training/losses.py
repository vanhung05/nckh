"""Hàm loss dùng cho huấn luyện.

Mặc định CrossEntropyLoss (có thể kèm class weights khi mất cân bằng).
Bổ sung Focal Loss như một thí nghiệm tùy chọn.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss cho phân loại đa lớp (thí nghiệm bổ sung).

    L = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, target, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    name: str = "cross_entropy",
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Tạo hàm loss theo tên.

    Args:
        name: "cross_entropy" hoặc "focal".
        class_weights: Trọng số lớp (tùy chọn).
        focal_gamma: Tham số gamma cho focal loss.
    """
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    raise ValueError(f"Loss không hỗ trợ: {name}")
