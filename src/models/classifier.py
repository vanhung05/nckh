"""Tiện ích cấp cao để nạp mô hình đã train phục vụ inference/Grad-CAM."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.models.model_factory import ModelBundle, create_model


def load_class_names(output_root: Path) -> list[str]:
    """Đọc danh sách tên lớp theo đúng thứ tự index từ class_mapping.json."""
    path = output_root / "dataset" / "class_mapping.json"
    with path.open("r", encoding="utf-8") as fh:
        mapping: dict[str, int] = json.load(fh)
    # Sắp xếp theo index để tên[i] tương ứng class_index = i.
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def load_trained_model(
    model_name: str,
    checkpoint_path: str | Path,
    num_classes: int,
    device: torch.device | str = "cpu",
) -> ModelBundle:
    """Tạo mô hình và nạp trọng số đã train từ checkpoint.

    Checkpoint được lưu ở dạng dict có key 'model_state'.
    """
    bundle = create_model(model_name, num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    bundle.model.load_state_dict(state)
    bundle.model.to(device)
    bundle.model.eval()
    return bundle
