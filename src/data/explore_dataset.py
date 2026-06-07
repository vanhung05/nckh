"""Giai đoạn 1 - Khảo sát dữ liệu.

Script này:
    - Đọc dữ liệu thật từ thư mục Data (train, test).
    - Liệt kê các lớp.
    - Thống kê số lượng ảnh train/test theo từng lớp.
    - Phát hiện lớp chỉ có ở train hoặc chỉ có ở test.
    - Phát hiện khác biệt cách viết tên lớp (hoa/thường) giữa train và test.
    - Xuất class_distribution.csv và class_mapping.json.
    - Xuất biểu đồ phân bố train và test.
    - In tóm tắt ra terminal.

Cách chạy::

    python -m src.data.explore_dataset

Lưu ý: không thay đổi dữ liệu gốc, chỉ đọc và ghi metadata ra thư mục outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend không cần màn hình, an toàn khi chạy script
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import Config, load_config
from src.utils.file_utils import ensure_dir, list_image_files, list_subdirectories
from src.utils.logger import get_logger

logger = get_logger("explore_dataset")


@dataclass
class ClassInfo:
    """Thông tin gộp của một lớp bệnh sau khi đối chiếu train và test."""

    canonical_name: str
    train_name: str | None = None
    test_name: str | None = None
    train_count: int = 0
    test_count: int = 0

    @property
    def total_count(self) -> int:
        return self.train_count + self.test_count

    @property
    def name_mismatch(self) -> bool:
        """True nếu tên ở train và test khác nhau (kể cả hoa/thường)."""
        if self.train_name is None or self.test_name is None:
            return False
        return self.train_name != self.test_name


@dataclass
class DatasetSummary:
    """Kết quả tổng hợp của quá trình khảo sát."""

    classes: list[ClassInfo] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def total_train(self) -> int:
        return sum(c.train_count for c in self.classes)

    @property
    def total_test(self) -> int:
        return sum(c.test_count for c in self.classes)

    @property
    def only_in_train(self) -> list[ClassInfo]:
        return [c for c in self.classes if c.train_name and not c.test_name]

    @property
    def only_in_test(self) -> list[ClassInfo]:
        return [c for c in self.classes if c.test_name and not c.train_name]

    @property
    def name_mismatches(self) -> list[ClassInfo]:
        return [c for c in self.classes if c.name_mismatch]


def _normalize(name: str) -> str:
    """Chuẩn hóa tên lớp để đối chiếu giữa train và test.

    Bỏ khoảng trắng thừa và đưa về chữ thường để không bị lệch do
    khác biệt hoa/thường (vd: "Actinic keratosis" vs "Actinic Keratosis").
    """
    return " ".join(name.split()).lower()


def _count_classes(
    split_dir: Path, valid_extensions: list[str]
) -> dict[str, tuple[str, int]]:
    """Đếm số ảnh theo từng lớp trong một split.

    Returns:
        Map từ tên đã chuẩn hóa -> (tên gốc trên đĩa, số ảnh).
    """
    result: dict[str, tuple[str, int]] = {}
    if not split_dir.is_dir():
        logger.warning("Không tìm thấy thư mục split: %s", split_dir)
        return result

    for class_dir in list_subdirectories(split_dir):
        original_name = class_dir.name
        key = _normalize(original_name)
        count = len(list_image_files(class_dir, valid_extensions))
        result[key] = (original_name, count)
    return result


def explore(config: Config) -> DatasetSummary:
    """Thực hiện khảo sát dữ liệu và trả về kết quả tổng hợp."""
    data_root = config.data_root
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    valid_extensions = config.get("data.valid_extensions", [".jpg", ".jpeg", ".png"])

    logger.info("Đường dẫn dữ liệu: %s", data_root)
    logger.info("Đọc thư mục train: %s", train_dir)
    logger.info("Đọc thư mục test : %s", test_dir)

    train_counts = _count_classes(train_dir, valid_extensions)
    test_counts = _count_classes(test_dir, valid_extensions)

    # Tập hợp tất cả key đã chuẩn hóa từ cả hai split.
    all_keys = sorted(set(train_counts) | set(test_counts))

    classes: list[ClassInfo] = []
    for key in all_keys:
        train_entry = train_counts.get(key)
        test_entry = test_counts.get(key)
        # Ưu tiên tên ở train làm tên canonical; nếu không có thì dùng test.
        canonical = train_entry[0] if train_entry else test_entry[0]  # type: ignore[index]
        classes.append(
            ClassInfo(
                canonical_name=canonical,
                train_name=train_entry[0] if train_entry else None,
                test_name=test_entry[0] if test_entry else None,
                train_count=train_entry[1] if train_entry else 0,
                test_count=test_entry[1] if test_entry else 0,
            )
        )

    return DatasetSummary(classes=classes)


def save_class_distribution_csv(summary: DatasetSummary, out_path: Path) -> None:
    """Lưu file class_distribution.csv."""
    rows = [
        {
            "class_name": c.canonical_name,
            "train_name": c.train_name or "",
            "test_name": c.test_name or "",
            "train_count": c.train_count,
            "test_count": c.test_count,
            "total_count": c.total_count,
            "name_mismatch": c.name_mismatch,
        }
        for c in summary.classes
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu phân bố lớp: %s", out_path)


def save_class_mapping_json(summary: DatasetSummary, out_path: Path) -> None:
    """Lưu class_mapping.json: tên lớp canonical -> chỉ số (0..N-1).

    Chỉ số được gán theo thứ tự alphabet của tên canonical để ổn định và
    tái lập được giữa các lần chạy.
    """
    sorted_classes = sorted(summary.classes, key=lambda c: c.canonical_name.lower())
    mapping = {c.canonical_name: idx for idx, c in enumerate(sorted_classes)}
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)
    logger.info("Đã lưu class mapping: %s (%d lớp)", out_path, len(mapping))


def _plot_distribution(
    summary: DatasetSummary, split: str, out_path: Path
) -> None:
    """Vẽ biểu đồ phân bố số ảnh theo lớp cho một split."""
    names = [c.canonical_name for c in summary.classes]
    counts = [getattr(c, f"{split}_count") for c in summary.classes]

    # Sắp xếp giảm dần để dễ quan sát mất cân bằng.
    order = sorted(range(len(names)), key=lambda i: counts[i], reverse=True)
    names = [names[i] for i in order]
    counts = [counts[i] for i in order]

    fig_height = max(6, len(names) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(names, counts, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Số ảnh")
    ax.set_title(f"Phân bố số ảnh theo lớp - tập {split}")
    for i, v in enumerate(counts):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Đã lưu biểu đồ: %s", out_path)


def print_summary(summary: DatasetSummary) -> None:
    """In tóm tắt kết quả khảo sát ra terminal."""
    line = "=" * 64
    print(line)
    print("TÓM TẮT KHẢO SÁT DỮ LIỆU")
    print(line)
    print(f"Tổng số lớp (sau khi đối chiếu train/test): {summary.num_classes}")
    print(f"Tổng số ảnh train: {summary.total_train}")
    print(f"Tổng số ảnh test : {summary.total_test}")
    print(f"Tổng số ảnh      : {summary.total_train + summary.total_test}")
    print(line)

    print(f"{'Tên lớp (canonical)':40s} {'train':>7s} {'test':>7s} {'total':>7s}")
    print("-" * 64)
    for c in summary.classes:
        flag = "  <-- tên khác nhau" if c.name_mismatch else ""
        print(
            f"{c.canonical_name[:40]:40s} "
            f"{c.train_count:>7d} {c.test_count:>7d} {c.total_count:>7d}{flag}"
        )
    print(line)

    only_train = summary.only_in_train
    only_test = summary.only_in_test
    mismatches = summary.name_mismatches

    if only_train:
        print(f"[CẢNH BÁO] {len(only_train)} lớp chỉ có ở TRAIN, không có ở TEST:")
        for c in only_train:
            print(f"   - {c.canonical_name}")
    else:
        print("[OK] Mọi lớp ở train đều có ở test.")

    if only_test:
        print(f"[CẢNH BÁO] {len(only_test)} lớp chỉ có ở TEST, không có ở TRAIN:")
        for c in only_test:
            print(f"   - {c.canonical_name}")
    else:
        print("[OK] Mọi lớp ở test đều có ở train.")

    if mismatches:
        print(
            f"[LƯU Ý] {len(mismatches)} lớp có tên viết khác nhau giữa train/test "
            "(chỉ khác hoa/thường - đã được đối chiếu tự động):"
        )
        for c in mismatches:
            print(f"   - train: '{c.train_name}'  |  test: '{c.test_name}'")
    else:
        print("[OK] Tên lớp ở train và test khớp nhau hoàn toàn.")

    # Đánh giá mất cân bằng đơn giản dựa trên train.
    train_counts = [c.train_count for c in summary.classes if c.train_count > 0]
    if train_counts:
        max_c, min_c = max(train_counts), min(train_counts)
        ratio = max_c / min_c if min_c else float("inf")
        print("-" * 64)
        print(
            f"Mất cân bằng (train): lớp nhiều nhất={max_c}, ít nhất={min_c}, "
            f"tỉ lệ max/min={ratio:.2f}"
        )
    print(line)


def main() -> None:
    """Điểm vào của script."""
    config = load_config()

    out_dataset = ensure_dir(config.output_root / "dataset")
    out_figures = ensure_dir(config.output_root / "figures" / "dataset")

    summary = explore(config)

    if summary.num_classes == 0:
        logger.error(
            "Không tìm thấy lớp nào. Kiểm tra lại đường dẫn dữ liệu: %s",
            config.data_root,
        )
        return

    save_class_distribution_csv(
        summary, out_dataset / "class_distribution.csv"
    )
    save_class_mapping_json(summary, out_dataset / "class_mapping.json")

    _plot_distribution(
        summary, "train", out_figures / "class_distribution_train.png"
    )
    _plot_distribution(
        summary, "test", out_figures / "class_distribution_test.png"
    )

    print_summary(summary)


if __name__ == "__main__":
    main()
