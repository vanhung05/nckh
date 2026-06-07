"""Giai đoạn 2 / Bước 5 - Tạo validation split (stratified) + làm sạch rò rỉ.

Nguyên tắc:
    - KHÔNG di chuyển hay thay đổi dữ liệu gốc, chỉ ghi file metadata.
    - Validation chỉ tách ra từ thư mục train.
    - Test giữ nguyên.
    - Loại bỏ rò rỉ và ảnh trùng/nhãn mâu thuẫn dựa trên duplicate_images.csv:
        * Ảnh train trùng (exact/near) với một ảnh test  -> loại khỏi train.
        * Nhóm trùng nội bộ train có nhãn mâu thuẫn       -> loại toàn bộ.
        * Nhóm trùng nội bộ train cùng nhãn               -> giữ 1, loại còn lại.

Đầu ra:
    - outputs/dataset/split_metadata.csv   (image_path, class_name, class_index, split)
    - outputs/dataset/excluded_images.csv  (ảnh bị loại + lý do)

Yêu cầu chạy trước:
    - explore_dataset.py  (cần class_mapping.json)
    - detect_duplicates.py (cần duplicate_images.csv)

Cách chạy::

    python -m src.data.create_validation_split
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.config import Config, load_config
from src.utils.file_utils import ensure_dir, list_image_files, list_subdirectories
from src.utils.logger import get_logger

logger = get_logger("create_validation_split")


@dataclass
class SplitRow:
    """Một dòng trong split_metadata.csv."""

    image_path: str
    class_name: str
    class_index: int
    split: str


@dataclass
class ExcludedRow:
    """Một ảnh train bị loại khỏi quá trình train cùng lý do."""

    image_path: str
    class_name: str
    reason: str
    detail: str


# --------------------------------------------------------------------------- #
# Tiện ích chuẩn hóa & ánh xạ lớp
# --------------------------------------------------------------------------- #
def _normalize(name: str) -> str:
    """Chuẩn hóa tên lớp (bỏ khoảng trắng thừa, về chữ thường)."""
    return " ".join(name.split()).lower()


def _load_class_mapping(path: Path) -> dict[str, int]:
    """Đọc class_mapping.json (tên canonical -> index)."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_canonical_lookup(mapping: dict[str, int]) -> dict[str, str]:
    """Tạo map từ tên đã chuẩn hóa -> tên canonical."""
    return {_normalize(name): name for name in mapping}


# --------------------------------------------------------------------------- #
# Union-Find để gom nhóm ảnh trùng
# --------------------------------------------------------------------------- #
class _UnionFind:
    """Cấu trúc union-find đơn giản trên các chuỗi đường dẫn."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Nén đường đi.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for node in self._parent:
            result[self.find(node)].append(node)
        return result


# --------------------------------------------------------------------------- #
# Xác định ảnh train cần loại bỏ
# --------------------------------------------------------------------------- #
def compute_exclusions(
    duplicates_csv: Path,
) -> dict[str, tuple[str, str]]:
    """Tính tập ảnh train cần loại bỏ dựa trên file duplicate_images.csv.

    Returns:
        Map: đường dẫn ảnh train bị loại -> (reason, detail).
    """
    if not duplicates_csv.exists():
        logger.warning(
            "Không tìm thấy %s. Bỏ qua bước làm sạch trùng lặp.", duplicates_csv
        )
        return {}

    df = pd.read_csv(duplicates_csv)
    if df.empty:
        logger.info("Không có cặp trùng nào trong %s.", duplicates_csv)
        return {}

    # Gom nhóm bằng union-find trên tất cả các cặp.
    uf = _UnionFind()
    info: dict[str, tuple[str, str]] = {}  # path -> (split, class_name)
    for row in df.itertuples(index=False):
        uf.union(row.image_a, row.image_b)
        info[row.image_a] = (row.split_a, row.class_a)
        info[row.image_b] = (row.split_b, row.class_b)

    exclusions: dict[str, tuple[str, str]] = {}

    for members in uf.groups().values():
        if len(members) < 2:
            continue
        splits = {info[m][0] for m in members}
        classes = {_normalize(info[m][1]) for m in members}
        train_members = sorted(m for m in members if info[m][0] == "train")

        has_test = "test" in splits
        conflicting = len(classes) > 1

        if has_test:
            # Rò rỉ: loại mọi ảnh train trong nhóm.
            for m in train_members:
                exclusions[m] = (
                    "leakage_train_test",
                    f"trùng với ảnh test trong nhóm {len(members)} ảnh",
                )
        elif conflicting:
            # Trùng nội bộ train nhưng nhãn mâu thuẫn -> loại hết.
            for m in train_members:
                exclusions[m] = (
                    "conflicting_label",
                    f"nhóm trùng có nhãn mâu thuẫn: {sorted(classes)}",
                )
        else:
            # Trùng nội bộ train, cùng nhãn -> giữ ảnh đầu, loại còn lại.
            for m in train_members[1:]:
                exclusions[m] = (
                    "intra_train_duplicate",
                    f"trùng với {train_members[0]}",
                )

    logger.info("Tổng số ảnh train bị loại bỏ: %d", len(exclusions))
    return exclusions


# --------------------------------------------------------------------------- #
# Tạo split
# --------------------------------------------------------------------------- #
def build_split(
    config: Config,
    class_mapping: dict[str, int],
    exclusions: dict[str, tuple[str, str]],
) -> tuple[list[SplitRow], list[ExcludedRow]]:
    """Tạo danh sách dòng split và danh sách ảnh bị loại."""
    data_root = config.data_root
    valid_extensions = config.get(
        "data.valid_extensions", [".jpg", ".jpeg", ".png"]
    )
    val_ratio = float(config.get("data.validation_ratio", 0.15))
    seed = int(config.get("project.seed", 42))

    canonical_lookup = _build_canonical_lookup(class_mapping)
    rng = random.Random(seed)

    split_rows: list[SplitRow] = []
    excluded_rows: list[ExcludedRow] = []

    # ----- TRAIN -> train/val (loại bỏ ảnh trong exclusions) ----------------
    train_dir = data_root / "train"
    for class_dir in list_subdirectories(train_dir):
        canonical = canonical_lookup.get(_normalize(class_dir.name))
        if canonical is None:
            logger.warning("Bỏ qua lớp không có trong mapping: %s", class_dir.name)
            continue
        class_index = class_mapping[canonical]

        kept_images: list[Path] = []
        for img in list_image_files(class_dir, valid_extensions):
            key = str(img)
            if key in exclusions:
                reason, detail = exclusions[key]
                excluded_rows.append(
                    ExcludedRow(
                        image_path=key,
                        class_name=canonical,
                        reason=reason,
                        detail=detail,
                    )
                )
            else:
                kept_images.append(img)

        # Stratified split trong từng lớp.
        kept_images.sort()
        rng.shuffle(kept_images)
        n_val = round(len(kept_images) * val_ratio)
        val_set = set(kept_images[:n_val])

        for img in kept_images:
            split = "val" if img in val_set else "train"
            split_rows.append(
                SplitRow(
                    image_path=str(img),
                    class_name=canonical,
                    class_index=class_index,
                    split=split,
                )
            )

    # ----- TEST -> giữ nguyên ----------------------------------------------
    test_dir = data_root / "test"
    for class_dir in list_subdirectories(test_dir):
        canonical = canonical_lookup.get(_normalize(class_dir.name))
        if canonical is None:
            logger.warning("Bỏ qua lớp test không có trong mapping: %s", class_dir.name)
            continue
        class_index = class_mapping[canonical]
        for img in list_image_files(class_dir, valid_extensions):
            split_rows.append(
                SplitRow(
                    image_path=str(img),
                    class_name=canonical,
                    class_index=class_index,
                    split="test",
                )
            )

    return split_rows, excluded_rows


def save_split_csv(rows: list[SplitRow], out_path: Path) -> None:
    """Lưu split_metadata.csv."""
    df = pd.DataFrame([vars(r) for r in rows])
    df = df[["image_path", "class_name", "class_index", "split"]]
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu split metadata: %s (%d dòng)", out_path, len(df))


def save_excluded_csv(rows: list[ExcludedRow], out_path: Path) -> None:
    """Lưu excluded_images.csv."""
    df = pd.DataFrame(
        [vars(r) for r in rows],
        columns=["image_path", "class_name", "reason", "detail"],
    )
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu danh sách ảnh bị loại: %s (%d dòng)", out_path, len(df))


def _print_summary(rows: list[SplitRow], excluded: list[ExcludedRow]) -> None:
    """In tóm tắt kết quả split."""
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r.split] += 1

    reason_counts: dict[str, int] = defaultdict(int)
    for r in excluded:
        reason_counts[r.reason] += 1

    line = "=" * 64
    print(line)
    print("KẾT QUẢ TẠO VALIDATION SPLIT")
    print(line)
    print(f"Số ảnh train : {counts['train']}")
    print(f"Số ảnh val   : {counts['val']}")
    print(f"Số ảnh test  : {counts['test']}")
    print(f"Tổng dòng    : {len(rows)}")
    print("-" * 64)
    print(f"Số ảnh train bị loại bỏ: {len(excluded)}")
    for reason, count in sorted(reason_counts.items()):
        print(f"   - {reason}: {count}")
    print(line)


def main() -> None:
    """Điểm vào của script."""
    config = load_config()
    out_dataset = ensure_dir(config.output_root / "dataset")

    mapping_path = out_dataset / "class_mapping.json"
    if not mapping_path.exists():
        logger.error(
            "Thiếu %s. Hãy chạy 'python -m src.data.explore_dataset' trước.",
            mapping_path,
        )
        return
    class_mapping = _load_class_mapping(mapping_path)

    exclusions = compute_exclusions(out_dataset / "duplicate_images.csv")

    rows, excluded = build_split(config, class_mapping, exclusions)

    save_split_csv(rows, out_dataset / "split_metadata.csv")
    save_excluded_csv(excluded, out_dataset / "excluded_images.csv")

    _print_summary(rows, excluded)


if __name__ == "__main__":
    main()
