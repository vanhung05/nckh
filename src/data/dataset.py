"""Dataset và DataLoader đọc từ split_metadata.csv.

Toàn bộ train/val/test đều lấy danh sách ảnh từ file metadata đã sinh ở
bước create_validation_split, đảm bảo không rò rỉ và split nhất quán giữa
các mô hình.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import build_eval_transforms, build_train_transforms
from src.utils.config import Config


class SkinDiseaseDataset(Dataset):
    """Dataset đọc ảnh theo một split cụ thể từ metadata."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        split: str,
        transform=None,
    ) -> None:
        """
        Args:
            metadata: DataFrame chứa cột image_path, class_index, split.
            split: "train" | "val" | "test".
            transform: torchvision transform áp dụng lên ảnh.
        """
        self._df = metadata[metadata["split"] == split].reset_index(drop=True)
        self._transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self._df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        label = int(row["class_index"])
        return image, label

    @property
    def labels(self) -> list[int]:
        """Danh sách nhãn (class_index) theo thứ tự mẫu."""
        return self._df["class_index"].astype(int).tolist()


def load_class_mapping(output_root: Path) -> dict[str, int]:
    """Đọc class_mapping.json từ thư mục outputs/dataset."""
    path = output_root / "dataset" / "class_mapping.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_metadata(output_root: Path) -> pd.DataFrame:
    """Đọc split_metadata.csv."""
    path = output_root / "dataset" / "split_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Thiếu {path}. Hãy chạy 'python -m src.data.create_validation_split'."
        )
    return pd.read_csv(path)


def build_dataloaders(
    config: Config,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    """Tạo DataLoader cho train/val/test.

    Returns:
        (loaders, class_mapping) với loaders gồm key "train", "val", "test".
    """
    output_root = config.output_root
    metadata = load_metadata(output_root)
    class_mapping = load_class_mapping(output_root)

    image_size = int(config.get("data.image_size", 224))
    batch_size = int(config.get("data.batch_size", 32))
    num_workers = int(config.get("data.num_workers", 2))

    train_tf = build_train_transforms(image_size)
    eval_tf = build_eval_transforms(image_size)

    datasets = {
        "train": SkinDiseaseDataset(metadata, "train", train_tf),
        "val": SkinDiseaseDataset(metadata, "val", eval_tf),
        "test": SkinDiseaseDataset(metadata, "test", eval_tf),
    }

    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        for name, ds in datasets.items()
    }
    return loaders, class_mapping


def compute_class_weights(
    config: Config, num_classes: int
) -> torch.Tensor:
    """Tính trọng số lớp (inverse frequency) từ tập train để xử lý mất cân bằng."""
    metadata = load_metadata(config.output_root)
    train_df = metadata[metadata["split"] == "train"]
    counts = train_df["class_index"].value_counts().sort_index()

    weights = torch.ones(num_classes, dtype=torch.float32)
    total = counts.sum()
    for idx in range(num_classes):
        c = int(counts.get(idx, 0))
        if c > 0:
            # inverse frequency, chuẩn hóa theo trung bình.
            weights[idx] = total / (num_classes * c)
    return weights
