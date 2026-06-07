"""Model factory cho 4 kiến trúc CNN.

Hỗ trợ:
    - resnet18
    - mobilenet_v3_large
    - densenet121
    - efficientnet_b0

Tất cả dùng pretrained ImageNet, thay classification head bằng head mới có
số neuron bằng số lớp thực tế (đọc từ class_mapping, không hard-code).

Mỗi mô hình cũng cung cấp tên layer mục tiêu cho Grad-CAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = (
    "resnet18",
    "mobilenet_v3_large",
    "densenet121",
    "efficientnet_b0",
)


@dataclass
class ModelBundle:
    """Gói mô hình kèm metadata phục vụ train và Grad-CAM."""

    model: nn.Module
    name: str
    # Tên thuộc tính (dotted) của layer conv cuối, dùng cho Grad-CAM.
    target_layer_name: str


def _replace_classifier(model: nn.Module, name: str, num_classes: int) -> None:
    """Thay head phân loại của model theo số lớp thực tế."""
    if name == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "mobilenet_v3_large":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:  # pragma: no cover - đã kiểm tra ở create_model
        raise ValueError(f"Mô hình không hỗ trợ: {name}")


def _target_layer_name(name: str) -> str:
    """Trả về tên layer mục tiêu cho Grad-CAM của từng kiến trúc."""
    return {
        "resnet18": "layer4",
        "mobilenet_v3_large": "features",
        "densenet121": "features",
        "efficientnet_b0": "features",
    }[name]


def create_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
) -> ModelBundle:
    """Tạo mô hình theo tên.

    Args:
        name: Một trong SUPPORTED_MODELS.
        num_classes: Số lớp (suy ra từ class_mapping).
        pretrained: Có dùng pretrained ImageNet hay không.

    Returns:
        ModelBundle chứa model và metadata.
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Mô hình '{name}' không hỗ trợ. Chọn một trong: {SUPPORTED_MODELS}"
        )

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "mobilenet_v3_large":
        weights = (
            models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model = models.mobilenet_v3_large(weights=weights)
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
    else:  # efficientnet_b0
        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model = models.efficientnet_b0(weights=weights)

    _replace_classifier(model, name, num_classes)

    return ModelBundle(
        model=model,
        name=name,
        target_layer_name=_target_layer_name(name),
    )


def get_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Lấy module layer mục tiêu cho Grad-CAM từ một model đã tạo."""
    if name == "resnet18":
        return model.layer4[-1]
    if name == "mobilenet_v3_large":
        return model.features[-1]
    if name == "densenet121":
        return model.features[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Mô hình không hỗ trợ: {name}")


def freeze_backbone(model: nn.Module, name: str) -> None:
    """Đóng băng toàn bộ backbone, chỉ để head phân loại train được.

    Dùng cho giai đoạn baseline (chỉ train classifier head).
    """
    for param in model.parameters():
        param.requires_grad = False

    # Mở lại tham số của head phân loại.
    if name == "resnet18":
        head = model.fc
    elif name in ("mobilenet_v3_large", "efficientnet_b0"):
        head = model.classifier
    elif name == "densenet121":
        head = model.classifier
    else:  # pragma: no cover
        raise ValueError(f"Mô hình không hỗ trợ: {name}")

    for param in head.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Mở băng toàn bộ tham số cho giai đoạn fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> int:
    """Đếm tổng số tham số của mô hình."""
    return sum(p.numel() for p in model.parameters())
