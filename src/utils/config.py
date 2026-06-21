"""Đọc và quản lý file cấu hình YAML.

Toàn bộ giá trị cấu hình (đường dẫn, batch size, epoch, learning rate, ...)
phải được đọc từ file config thay vì hard-code rải rác trong code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Thư mục gốc của project = thư mục cha của "src".
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "base.yaml"


class Config:
    """Bao bọc dictionary cấu hình, hỗ trợ truy cập theo thuộc tính lồng nhau.

    Ví dụ::

        cfg = Config.load()
        cfg.data["image_size"]
        cfg.get("training.classifier_lr")
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    # ----- Truy cập dữ liệu -------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Lấy giá trị theo key dạng "a.b.c". Trả ``default`` nếu không có."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    @property
    def raw(self) -> dict[str, Any]:
        """Trả về dictionary gốc."""
        return self._data

    # ----- Đường dẫn tiện ích ----------------------------------------------
    @property
    def data_root(self) -> Path:
        """Đường dẫn tuyệt đối tới thư mục dữ liệu.

        Ưu tiên biến môi trường SKIN_DATA_ROOT (hữu ích trên Kaggle khi
        dataset nằm ở vị trí khác), sau đó mới tới giá trị trong config.
        """
        env = os.environ.get("SKIN_DATA_ROOT")
        if env:
            return self._resolve_path(env)
        return self._resolve_path(self.get("paths.data_root", "Data"))

    @property
    def output_root(self) -> Path:
        """Đường dẫn tuyệt đối tới thư mục outputs.

        Ưu tiên biến môi trường SKIN_OUTPUT_ROOT (vd: lưu ra Google Drive hoặc
        /kaggle/working để giữ kết quả sau khi phiên kết thúc).
        """
        env = os.environ.get("SKIN_OUTPUT_ROOT")
        if env:
            return self._resolve_path(env)
        return self._resolve_path(self.get("paths.output_root", "outputs"))

    @staticmethod
    def _resolve_path(value: str) -> Path:
        """Chuyển đường dẫn trong config thành đường dẫn tuyệt đối.

        Đường dẫn tương đối được tính từ ``PROJECT_ROOT`` để code chạy được
        bất kể thư mục hiện hành khi gọi script.
        """
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    # ----- Khởi tạo ---------------------------------------------------------
    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Config":
        """Đọc file YAML và trả về đối tượng :class:`Config`."""
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file config: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data)


def load_config(config_path: str | Path | None = None) -> Config:
    """Hàm tiện lợi để đọc config."""
    return Config.load(config_path)
