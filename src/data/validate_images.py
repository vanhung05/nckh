"""Giai đoạn 1 / Bước 3 - Kiểm tra ảnh lỗi.

Script này quét toàn bộ ảnh trong train và test để phát hiện:
    - File có phần mở rộng không hỗ trợ.
    - Ảnh không mở/đọc được (file hỏng).
    - Ảnh bị cắt cụt (truncated).
    - Ảnh có kích thước bất thường (quá nhỏ).

Kết quả được ghi ra outputs/dataset/invalid_images.csv.

Cách chạy::

    python -m src.data.validate_images
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile

from src.utils.config import Config, load_config
from src.utils.file_utils import ensure_dir, list_subdirectories
from src.utils.logger import get_logger

# Cho phép phát hiện ảnh truncated thay vì âm thầm load một phần.
ImageFile.LOAD_TRUNCATED_IMAGES = False

logger = get_logger("validate_images")

MIN_SIZE = 16  # ngưỡng cảnh báo kích thước quá nhỏ (pixel)


@dataclass
class InvalidImage:
    """Bản ghi một ảnh có vấn đề."""

    image_path: str
    split: str
    class_name: str
    issue: str
    detail: str


def _iter_class_dirs(split_dir: Path):
    """Sinh ra (class_name, class_dir) cho từng lớp trong split."""
    for class_dir in list_subdirectories(split_dir):
        yield class_dir.name, class_dir


def _check_image(path: Path, valid_extensions: set[str]) -> tuple[str, str] | None:
    """Kiểm tra một ảnh. Trả về (issue, detail) nếu lỗi, None nếu hợp lệ."""
    if path.suffix.lower() not in valid_extensions:
        return ("unsupported_extension", path.suffix)

    try:
        # verify() phát hiện file hỏng mà không decode toàn bộ.
        with Image.open(path) as img:
            img.verify()
        # Mở lại để lấy kích thước thật (verify làm hỏng đối tượng img).
        with Image.open(path) as img:
            width, height = img.size
            img.load()  # ép decode để bắt lỗi truncated.
    except Exception as exc:  # noqa: BLE001 - cần bắt mọi lỗi đọc ảnh
        return ("unreadable", f"{type(exc).__name__}: {exc}")

    if width < MIN_SIZE or height < MIN_SIZE:
        return ("too_small", f"{width}x{height}")

    return None


def validate(config: Config) -> list[InvalidImage]:
    """Quét toàn bộ ảnh và trả về danh sách ảnh lỗi."""
    data_root = config.data_root
    valid_extensions = {
        ext.lower()
        for ext in config.get("data.valid_extensions", [".jpg", ".jpeg", ".png"])
    }

    invalid: list[InvalidImage] = []
    total = 0

    for split in ("train", "test"):
        split_dir = data_root / split
        if not split_dir.is_dir():
            logger.warning("Bỏ qua split không tồn tại: %s", split_dir)
            continue

        for class_name, class_dir in _iter_class_dirs(split_dir):
            for path in sorted(class_dir.iterdir()):
                if not path.is_file():
                    continue
                total += 1
                result = _check_image(path, valid_extensions)
                if result is not None:
                    issue, detail = result
                    invalid.append(
                        InvalidImage(
                            image_path=str(path),
                            split=split,
                            class_name=class_name,
                            issue=issue,
                            detail=detail,
                        )
                    )
                    logger.warning("[%s] %s -> %s", issue, path, detail)

        logger.info("Đã quét xong split '%s'", split)

    logger.info("Tổng số file đã quét: %d", total)
    return invalid


def save_invalid_csv(invalid: list[InvalidImage], out_path: Path) -> None:
    """Lưu danh sách ảnh lỗi ra CSV (luôn tạo file kể cả khi rỗng)."""
    rows = [vars(item) for item in invalid]
    df = pd.DataFrame(
        rows, columns=["image_path", "split", "class_name", "issue", "detail"]
    )
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu danh sách ảnh lỗi: %s (%d bản ghi)", out_path, len(df))


def main() -> None:
    """Điểm vào của script."""
    config = load_config()
    out_dataset = ensure_dir(config.output_root / "dataset")

    invalid = validate(config)
    save_invalid_csv(invalid, out_dataset / "invalid_images.csv")

    print("=" * 64)
    print("KẾT QUẢ KIỂM TRA ẢNH LỖI")
    print("=" * 64)
    if not invalid:
        print("[OK] Không phát hiện ảnh lỗi nào.")
    else:
        print(f"[CẢNH BÁO] Phát hiện {len(invalid)} ảnh có vấn đề.")
        by_issue: dict[str, int] = {}
        for item in invalid:
            by_issue[item.issue] = by_issue.get(item.issue, 0) + 1
        for issue, count in sorted(by_issue.items()):
            print(f"   - {issue}: {count}")
    print("=" * 64)


if __name__ == "__main__":
    main()
