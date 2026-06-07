"""Sinh báo cáo tổng hợp cuối cùng (final_summary.md) từ các kết quả thực tế.

Tổng hợp:
    - Thông tin dataset (số lớp, số ảnh, mất cân bằng).
    - Kết quả làm sạch dữ liệu (ảnh trùng, rò rỉ, loại bỏ).
    - Bảng so sánh mô hình.
    - Mô hình tốt nhất theo macro-F1.

Chỉ tổng hợp số liệu CÓ THẬT từ các file output; không bịa số.

Cách chạy::

    python -m src.evaluation.generate_summary
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("generate_summary")


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Đọc CSV nếu tồn tại, trả None nếu không."""
    if path.exists():
        return pd.read_csv(path)
    return None


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Chuyển DataFrame thành bảng markdown (không cần thư viện tabulate)."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def build_summary(output_root: Path) -> str:
    """Tạo nội dung markdown báo cáo tổng hợp."""
    dataset_dir = output_root / "dataset"
    report_dir = output_root / "reports"

    lines: list[str] = []
    lines.append("# Báo cáo tổng hợp kết quả\n")
    lines.append(
        "> Tự động sinh từ các file output thực tế. "
        "Kết quả chỉ mang tính hỗ trợ tham khảo, không thay thế chẩn đoán "
        "của bác sĩ.\n"
    )

    # ----- Dataset ----------------------------------------------------------
    lines.append("## 1. Dữ liệu\n")
    dist = _read_csv_safe(dataset_dir / "class_distribution.csv")
    if dist is not None:
        n_classes = len(dist)
        total_train = int(dist["train_count"].sum())
        total_test = int(dist["test_count"].sum())
        lines.append(f"- Số lớp: **{n_classes}**")
        lines.append(f"- Tổng ảnh train (gốc): **{total_train}**")
        lines.append(f"- Tổng ảnh test: **{total_test}**")
        max_c = int(dist["train_count"].max())
        min_c = int(dist["train_count"][dist["train_count"] > 0].min())
        lines.append(
            f"- Mất cân bằng train: nhiều nhất {max_c}, ít nhất {min_c}, "
            f"tỉ lệ {max_c / min_c:.2f}\n"
        )
    else:
        lines.append("- (Chưa có class_distribution.csv)\n")

    # ----- Làm sạch dữ liệu -------------------------------------------------
    lines.append("## 2. Làm sạch dữ liệu\n")
    excluded = _read_csv_safe(dataset_dir / "excluded_images.csv")
    if excluded is not None and not excluded.empty:
        lines.append(f"- Tổng ảnh train bị loại: **{len(excluded)}**")
        for reason, count in excluded["reason"].value_counts().items():
            lines.append(f"  - {reason}: {count}")
        lines.append("")
    else:
        lines.append("- Không loại ảnh nào (hoặc chưa chạy split).\n")

    split = _read_csv_safe(dataset_dir / "split_metadata.csv")
    if split is not None:
        counts = split["split"].value_counts().to_dict()
        lines.append(
            f"- Sau làm sạch: train={counts.get('train', 0)}, "
            f"val={counts.get('val', 0)}, test={counts.get('test', 0)}\n"
        )

    # ----- So sánh mô hình --------------------------------------------------
    lines.append("## 3. So sánh mô hình\n")
    comparison = _read_csv_safe(report_dir / "model_comparison.csv")
    if comparison is not None and not comparison.empty:
        lines.append(_df_to_markdown(comparison))
        lines.append("")
        best = comparison.sort_values("macro_f1", ascending=False).iloc[0]
        lines.append(
            f"- Mô hình tốt nhất theo macro-F1: **{best['model_name']}** "
            f"(macro-F1 = {best['macro_f1']}, accuracy = {best['accuracy']})\n"
        )
    else:
        lines.append("- (Chưa có model_comparison.csv. Hãy chạy evaluate.)\n")

    return "\n".join(lines)


def main() -> None:
    config = load_config()
    output_root = config.output_root
    report_dir = ensure_dir(output_root / "reports")

    content = build_summary(output_root)
    out_path = report_dir / "final_summary.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("Đã sinh báo cáo tổng hợp: %s", out_path)
    print(content)


if __name__ == "__main__":
    main()
