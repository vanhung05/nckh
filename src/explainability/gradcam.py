"""Cài đặt Grad-CAM thuần PyTorch (không phụ thuộc package ngoài).

Grad-CAM tính bản đồ nhiệt thể hiện vùng ảnh ảnh hưởng nhiều nhất tới dự đoán
của mô hình, bằng cách lấy gradient của lớp mục tiêu theo feature map của
layer convolution cuối.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM cho một layer convolution mục tiêu.

    Cách dùng::

        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(input_tensor, class_idx)
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        # Chỉ dùng forward hook trên module; gradient được bắt bằng tensor hook
        # gắn trực tiếp lên activation. Cách này tránh xung đột view+inplace
        # (lỗi xảy ra với DenseNet do F.relu(..., inplace=True)).
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
        ]

    def _forward_hook(self, module, inputs, output) -> None:
        # Giữ activation kèm graph (không detach) để có thể lấy gradient.
        self._activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def generate(
        self, input_tensor: torch.Tensor, class_idx: int | None = None
    ) -> tuple[np.ndarray, int, float]:
        """Sinh heatmap Grad-CAM cho một ảnh.

        Args:
            input_tensor: Tensor shape (1, C, H, W).
            class_idx: Lớp cần giải thích. Nếu None, dùng lớp dự đoán cao nhất.

        Returns:
            (heatmap 2D đã chuẩn hóa [0,1], class_idx dùng, confidence).
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, class_idx].item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # weights: trung bình gradient trên không gian (global average pooling).
        gradients = self._gradients  # (1, C, h, w)
        activations = (
            self._activations.detach() if self._activations is not None else None
        )  # (1, C, h, w)
        if gradients is None or activations is None:
            raise RuntimeError("Không bắt được activation/gradient từ target layer.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Resize về kích thước ảnh đầu vào.
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Chuẩn hóa về [0, 1].
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, confidence

    def remove_hooks(self) -> None:
        """Gỡ các hook đã đăng ký."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove_hooks()
