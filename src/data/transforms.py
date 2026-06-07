"""Pipeline tiền xử lý và augmentation dùng chung cho cả 4 mô hình.

Train: augmentation mức vừa phải (không làm méo đặc điểm bệnh da liễu).
Val/Test: chỉ resize + center crop + normalize (không augmentation ngẫu nhiên).

Dùng chuẩn hóa theo thống kê ImageNet vì các backbone đều pretrained ImageNet.
"""

from __future__ import annotations

from torchvision import transforms

# Thống kê chuẩn hóa ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Tạo transform cho tập train (có augmentation vừa phải)."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Tạo transform cho val/test (không augmentation ngẫu nhiên)."""
    resize_size = int(round(image_size * 1.14))  # ~256 khi image_size=224
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def denormalize(tensor):
    """Đảo chuẩn hóa ImageNet để hiển thị/đưa vào Grad-CAM overlay.

    Args:
        tensor: Tensor ảnh shape (C, H, W) đã normalize.

    Returns:
        Tensor cùng shape với giá trị trong khoảng [0, 1].
    """
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)
