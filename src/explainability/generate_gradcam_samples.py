"""CLI sinh các mẫu Grad-CAM cho mô hình đã train.

Với mỗi mô hình, chọn các ảnh test thuộc các nhóm:
    - Dự đoán đúng, confidence cao.
    - Dự đoán đúng, confidence thấp.
    - Dự đoán sai.
Sau đó sinh ảnh gốc + heatmap + overlay và lưu kèm thông tin nhãn.

Cách chạy::

    python -m src.explainability.generate_gradcam_samples --model resnet18
    python -m src.explainability.generate_gradcam_samples --all-models
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import build_eval_transforms
from src.explainability.gradcam import GradCAM
from src.explainability.gradcam_utils import (
    overlay_heatmap,
    save_gradcam_figure,
    tensor_to_image,
)
from src.models.classifier import load_class_names, load_trained_model
from src.models.model_factory import SUPPORTED_MODELS, get_target_layer
from src.utils.config import Config, load_config
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("gradcam_samples")


def _load_test_metadata(config: Config) -> pd.DataFrame:
    """Đọc danh sách ảnh test từ split_metadata.csv."""
    path = config.output_root / "dataset" / "split_metadata.csv"
    df = pd.read_csv(path)
    return df[df["split"] == "test"].reset_index(drop=True)


@torch.no_grad()
def _predict_all(
    model, df: pd.DataFrame, transform, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Dự đoán toàn bộ ảnh test, trả về (preds, confidences)."""
    preds, confs = [], []
    for path in df["image_path"]:
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        probs = F.softmax(model(tensor), dim=1)
        conf, pred = probs.max(dim=1)
        preds.append(int(pred.item()))
        confs.append(float(conf.item()))
    return np.array(preds), np.array(confs)


def _select_samples(
    df: pd.DataFrame,
    preds: np.ndarray,
    confs: np.ndarray,
    n_per_group: int,
) -> dict[str, list[int]]:
    """Chọn chỉ số mẫu cho từng nhóm phân tích."""
    y_true = df["class_index"].to_numpy()
    correct = preds == y_true
    incorrect = ~correct

    correct_idx = np.where(correct)[0]
    incorrect_idx = np.where(incorrect)[0]

    # Đúng & confidence cao: sort giảm dần theo conf.
    high_conf = correct_idx[np.argsort(-confs[correct_idx])][:n_per_group]
    # Đúng & confidence thấp: sort tăng dần theo conf.
    low_conf = correct_idx[np.argsort(confs[correct_idx])][:n_per_group]
    # Sai: ưu tiên các trường hợp sai nhưng confidence cao (mô hình "tự tin sai").
    wrong = incorrect_idx[np.argsort(-confs[incorrect_idx])][:n_per_group]

    return {
        "correct_high_conf": high_conf.tolist(),
        "correct_low_conf": low_conf.tolist(),
        "wrong": wrong.tolist(),
    }


def generate_for_model(
    model_name: str, config: Config, device: torch.device, n_per_group: int
) -> None:
    """Sinh mẫu Grad-CAM cho một mô hình."""
    checkpoint_path = config.output_root / "checkpoints" / model_name / "best.pt"
    if not checkpoint_path.exists():
        logger.warning("Bỏ qua %s: chưa có checkpoint.", model_name)
        return

    class_names = load_class_names(config.output_root)
    num_classes = len(class_names)
    bundle = load_trained_model(
        model_name, checkpoint_path, num_classes, device=device
    )

    image_size = int(config.get("data.image_size", 224))
    transform = build_eval_transforms(image_size)

    df = _load_test_metadata(config)
    logger.info("[%s] Dự đoán %d ảnh test để chọn mẫu...", model_name, len(df))
    preds, confs = _predict_all(bundle.model, df, transform, device)

    groups = _select_samples(df, preds, confs, n_per_group)

    out_dir = ensure_dir(
        config.output_root / "figures" / "gradcam" / model_name
    )
    target_layer = get_target_layer(bundle.model, model_name)

    cam = GradCAM(bundle.model, target_layer)
    try:
        for group_name, indices in groups.items():
            for rank, idx in enumerate(indices):
                row = df.iloc[idx]
                image = Image.open(row["image_path"]).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)

                heatmap, used_class, confidence = cam.generate(tensor)

                orig = tensor_to_image(tensor.squeeze(0))
                overlay = overlay_heatmap(orig, heatmap)

                true_label = class_names[int(row["class_index"])]
                pred_label = class_names[used_class]

                out_path = out_dir / f"{group_name}_{rank+1:02d}.png"
                save_gradcam_figure(
                    orig,
                    heatmap,
                    overlay,
                    out_path,
                    true_label,
                    pred_label,
                    confidence,
                )
        logger.info("[%s] Đã lưu mẫu Grad-CAM vào %s", model_name, out_dir)
    finally:
        cam.remove_hooks()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sinh mẫu Grad-CAM.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=SUPPORTED_MODELS)
    group.add_argument("--all-models", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--n-per-group",
        type=int,
        default=4,
        help="Số ảnh mỗi nhóm (đúng-cao, đúng-thấp, sai).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("project.seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all_models:
        models = list(config.get("models", list(SUPPORTED_MODELS)))
    else:
        models = [args.model]

    for name in models:
        generate_for_model(name, config, device, args.n_per_group)


if __name__ == "__main__":
    main()
