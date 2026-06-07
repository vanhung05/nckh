"""Giai đoạn 1 / Bước 4 - Phát hiện ảnh trùng và gần trùng.

Hai cấp độ kiểm tra:
    1. Exact duplicate: băm nội dung file bằng MD5. Hai file có cùng MD5 là
       trùng tuyệt đối.
    2. Near-duplicate: dùng perceptual hash (phash) để phát hiện ảnh gần
       giống nhau (khác kích thước, nén, chỉnh sửa nhẹ).

Đặc biệt quan trọng: phát hiện ảnh trùng/gần trùng XUẤT HIỆN Ở CẢ train VÀ
test, vì đây là nguồn rò rỉ dữ liệu nghiêm trọng nhất.

Kết quả ghi ra outputs/dataset/duplicate_images.csv.

Cách chạy::

    python -m src.data.detect_duplicates
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

from src.utils.config import Config, load_config
from src.utils.file_utils import ensure_dir, list_subdirectories
from src.utils.logger import get_logger

logger = get_logger("detect_duplicates")

# Ngưỡng khoảng cách Hamming giữa hai perceptual hash để coi là gần trùng.
PHASH_THRESHOLD = 5


@dataclass
class ImageRecord:
    """Thông tin một ảnh phục vụ đối chiếu trùng lặp."""

    path: Path
    split: str
    class_name: str
    md5: str
    phash: imagehash.ImageHash | None


@dataclass
class DuplicatePair:
    """Một cặp ảnh trùng hoặc gần trùng."""

    image_a: str
    image_b: str
    split_a: str
    split_b: str
    class_a: str
    class_b: str
    dup_type: str  # "exact" hoặc "near"
    distance: int
    cross_split: bool  # True nếu một ảnh ở train, ảnh kia ở test (rò rỉ!)


def _md5(path: Path, chunk_size: int = 65536) -> str:
    """Tính MD5 của nội dung file."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_records(config: Config) -> list[ImageRecord]:
    """Quét toàn bộ ảnh và tính MD5 + perceptual hash."""
    data_root = config.data_root
    valid_extensions = {
        ext.lower()
        for ext in config.get("data.valid_extensions", [".jpg", ".jpeg", ".png"])
    }

    records: list[ImageRecord] = []
    for split in ("train", "test"):
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in list_subdirectories(split_dir):
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in valid_extensions:
                    continue
                md5 = _md5(path)
                phash: imagehash.ImageHash | None
                try:
                    with Image.open(path) as img:
                        phash = imagehash.phash(img.convert("RGB"))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Không tính được phash cho %s: %s", path, exc)
                    phash = None
                records.append(
                    ImageRecord(
                        path=path,
                        split=split,
                        class_name=class_dir.name,
                        md5=md5,
                        phash=phash,
                    )
                )
        logger.info("Đã xử lý split '%s'", split)

    logger.info("Tổng số ảnh đã băm: %d", len(records))
    return records


def _find_exact_duplicates(records: list[ImageRecord]) -> list[DuplicatePair]:
    """Tìm các cặp trùng tuyệt đối dựa trên MD5."""
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        groups[rec.md5].append(rec)

    pairs: list[DuplicatePair] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        # Lập tất cả các cặp trong nhóm trùng.
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                pairs.append(
                    DuplicatePair(
                        image_a=str(a.path),
                        image_b=str(b.path),
                        split_a=a.split,
                        split_b=b.split,
                        class_a=a.class_name,
                        class_b=b.class_name,
                        dup_type="exact",
                        distance=0,
                        cross_split=a.split != b.split,
                    )
                )
    return pairs


def _find_near_duplicates(
    records: list[ImageRecord], exact_pairs: set[frozenset[str]]
) -> list[DuplicatePair]:
    """Tìm các cặp gần trùng dựa trên perceptual hash.

    So sánh O(n^2) các ảnh có phash. Với ~5000 ảnh là chấp nhận được.
    Bỏ qua các cặp đã được đánh dấu trùng tuyệt đối.
    """
    items = [r for r in records if r.phash is not None]
    pairs: list[DuplicatePair] = []

    for i in range(len(items)):
        a = items[i]
        for j in range(i + 1, len(items)):
            b = items[j]
            key = frozenset({str(a.path), str(b.path)})
            if key in exact_pairs:
                continue
            distance = a.phash - b.phash  # type: ignore[operator]
            if distance <= PHASH_THRESHOLD:
                pairs.append(
                    DuplicatePair(
                        image_a=str(a.path),
                        image_b=str(b.path),
                        split_a=a.split,
                        split_b=b.split,
                        class_a=a.class_name,
                        class_b=b.class_name,
                        dup_type="near",
                        distance=distance,
                        cross_split=a.split != b.split,
                    )
                )
    return pairs


def detect(config: Config) -> list[DuplicatePair]:
    """Chạy toàn bộ pipeline phát hiện trùng lặp."""
    records = _collect_records(config)

    exact = _find_exact_duplicates(records)
    exact_keys = {frozenset({p.image_a, p.image_b}) for p in exact}
    logger.info("Số cặp trùng tuyệt đối: %d", len(exact))

    near = _find_near_duplicates(records, exact_keys)
    logger.info("Số cặp gần trùng (phash <= %d): %d", PHASH_THRESHOLD, len(near))

    return exact + near


def save_duplicates_csv(pairs: list[DuplicatePair], out_path: Path) -> None:
    """Lưu danh sách cặp trùng ra CSV."""
    rows = [vars(p) for p in pairs]
    df = pd.DataFrame(
        rows,
        columns=[
            "dup_type",
            "distance",
            "cross_split",
            "image_a",
            "image_b",
            "split_a",
            "split_b",
            "class_a",
            "class_b",
        ],
    )
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu danh sách ảnh trùng: %s (%d cặp)", out_path, len(df))


def main() -> None:
    """Điểm vào của script."""
    config = load_config()
    out_dataset = ensure_dir(config.output_root / "dataset")

    pairs = detect(config)
    save_duplicates_csv(pairs, out_dataset / "duplicate_images.csv")

    exact = [p for p in pairs if p.dup_type == "exact"]
    near = [p for p in pairs if p.dup_type == "near"]
    cross = [p for p in pairs if p.cross_split]

    print("=" * 64)
    print("KẾT QUẢ PHÁT HIỆN ẢNH TRÙNG")
    print("=" * 64)
    print(f"Cặp trùng tuyệt đối (MD5)     : {len(exact)}")
    print(f"Cặp gần trùng (phash <= {PHASH_THRESHOLD})    : {len(near)}")
    print(f"Cặp trùng giữa train và test : {len(cross)}")
    if cross:
        print("\n[CẢNH BÁO RÒ RỈ DỮ LIỆU] Các cặp xuất hiện ở cả train và test:")
        for p in cross[:20]:
            print(f"   [{p.dup_type}, d={p.distance}] {p.image_a}  <->  {p.image_b}")
        if len(cross) > 20:
            print(f"   ... và {len(cross) - 20} cặp khác (xem file CSV).")
    else:
        print("\n[OK] Không có ảnh trùng giữa train và test.")
    print("=" * 64)


if __name__ == "__main__":
    main()
