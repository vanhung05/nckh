"""CLI đánh giá mô hình trên tập test.

Với mỗi mô hình có checkpoint:
    - Chạy inference trên test set.
    - Tính metric (accuracy, macro P/R/F1, weighted F1, top-3).
    - Vẽ confusion matrix.
    - Sinh classification report theo lớp.
    - Đo số tham số, kích thước checkpoint, thời gian inference.

Tổng hợp tất cả vào outputs/reports/model_comparison.csv.

Cách chạy::

    python -m src.evaluation.evaluate --model resnet18
    python -m src.evaluation.evaluate --all-models
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.dataset import build_dataloaders
from src.evaluation.benchmark_inference import (
    benchmark_inference,
    measure_model_size_mb,
)
from src.evaluation.classification_report import save_classification_report
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.metrics import compute_metrics
from src.models.classifier import load_class_names, load_trained_model
from src.models.model_factory import SUPPORTED_MODELS, count_parameters
from src.utils.config import Config, load_config
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("evaluate")


@torch.no_grad()
def _run_inference(
    model, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chạy inference, trả về (y_true, y_pred, probs)."""
    all_true: list[int] = []
    all_pred: list[int] = []
    all_probs: list[np.ndarray] = []

    model.eval()
    for images, targets in tqdm(loader, desc="test", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_true.extend(targets.tolist())
        all_pred.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu().numpy())

    return (
        np.array(all_true),
        np.array(all_pred),
        np.concatenate(all_probs, axis=0),
    )


def evaluate_one_model(
    model_name: str, config: Config, device: torch.device
) -> dict[str, float] | None:
    """Đánh giá một mô hình. Trả về dòng kết quả cho bảng so sánh."""
    checkpoint_path = (
        config.output_root / "checkpoints" / model_name / "best.pt"
    )
    if not checkpoint_path.exists():
        logger.warning(
            "Bỏ qua %s: chưa có checkpoint tại %s", model_name, checkpoint_path
        )
        return None

    loaders, class_mapping = build_dataloaders(config)
    num_classes = len(class_mapping)
    class_names = load_class_names(config.output_root)

    bundle = load_trained_model(
        model_name, checkpoint_path, num_classes, device=device
    )

    y_true, y_pred, probs = _run_inference(bundle.model, loaders["test"], device)

    top_k = int(config.get("evaluation.top_k", 3))
    metrics = compute_metrics(y_true, y_pred, probs, top_k=top_k)

    # Confusion matrix + classification report.
    fig_dir = ensure_dir(config.output_root / "figures" / "evaluation")
    report_dir = ensure_dir(config.output_root / "reports")

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        fig_dir / f"confusion_matrix_{model_name}.png",
        model_name,
        normalize=True,
    )
    report_df = save_classification_report(
        y_true,
        y_pred,
        class_names,
        report_dir / f"classification_report_{model_name}.csv",
    )

    # Metric triển khai.
    num_params = count_parameters(bundle.model)
    size_mb = measure_model_size_mb(checkpoint_path)
    runs = int(config.get("evaluation.benchmark_runs", 100))
    image_size = int(config.get("data.image_size", 224))
    inference_ms = benchmark_inference(
        bundle.model, device, image_size=image_size, runs=runs
    )

    logger.info(
        "[%s] acc=%.4f macroF1=%.4f weightedF1=%.4f top3=%.4f "
        "params=%s size=%.1fMB inf=%.2fms",
        model_name,
        metrics.accuracy,
        metrics.macro_f1,
        metrics.weighted_f1,
        metrics.top3_accuracy,
        f"{num_params:,}",
        size_mb,
        inference_ms,
    )

    row = {
        "model_name": model_name,
        "accuracy": round(metrics.accuracy, 4),
        "macro_precision": round(metrics.macro_precision, 4),
        "macro_recall": round(metrics.macro_recall, 4),
        "macro_f1": round(metrics.macro_f1, 4),
        "weighted_f1": round(metrics.weighted_f1, 4),
        "top3_accuracy": round(metrics.top3_accuracy, 4),
        "num_parameters": num_params,
        "model_size_mb": round(size_mb, 2),
        "inference_time_ms": round(inference_ms, 3),
    }

    # Lưu per-class metrics (thêm cột model).
    per_class = report_df.copy()
    per_class.insert(0, "model_name", model_name)
    per_class_path = report_dir / "per_class_metrics.csv"
    header = not per_class_path.exists()
    per_class.to_csv(
        per_class_path, mode="a", header=header, encoding="utf-8-sig"
    )

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên test set.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=SUPPORTED_MODELS)
    group.add_argument("--all-models", action="store_true")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all_models:
        models_to_eval = list(config.get("models", list(SUPPORTED_MODELS)))
    else:
        models_to_eval = [args.model]

    report_dir = ensure_dir(config.output_root / "reports")
    # Xóa per_class cũ để tránh nối chồng khi chạy lại.
    per_class_path = report_dir / "per_class_metrics.csv"
    if per_class_path.exists():
        per_class_path.unlink()

    rows = []
    for name in models_to_eval:
        row = evaluate_one_model(name, config, device)
        if row is not None:
            rows.append(row)

    if not rows:
        logger.error("Không có mô hình nào được đánh giá (thiếu checkpoint).")
        return

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("macro_f1", ascending=False)
    out_path = report_dir / "model_comparison.csv"
    comparison.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu bảng so sánh: %s", out_path)

    print("=" * 80)
    print("BẢNG SO SÁNH MÔ HÌNH (sắp theo macro F1)")
    print("=" * 80)
    print(comparison.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
