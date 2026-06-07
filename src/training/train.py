"""CLI huấn luyện cho 4 mô hình.

Quy trình cho mỗi mô hình:
    1. Tạo model pretrained, thay head.
    2. Giai đoạn baseline: đóng băng backbone, train head với classifier_lr.
    3. Giai đoạn fine-tune: mở băng toàn bộ, train với finetune_lr.
    4. Lưu checkpoint tốt nhất (theo macro-F1) + lịch sử + biểu đồ.

Cách chạy::

    python -m src.training.train --model resnet18
    python -m src.training.train --all-models
    python -m src.training.train --model resnet18 --quick-test   # chạy thử nhanh
"""

from __future__ import annotations

import argparse

import torch

from src.data.dataset import build_dataloaders, compute_class_weights
from src.models.model_factory import (
    SUPPORTED_MODELS,
    count_parameters,
    create_model,
    freeze_backbone,
    unfreeze_all,
)
from src.training.callbacks import EarlyStopping
from src.training.losses import build_loss
from src.training.trainer import Trainer
from src.utils.config import Config, load_config
from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("train")


def train_one_model(
    model_name: str,
    config: Config,
    device: torch.device,
    quick_test: bool = False,
) -> None:
    """Huấn luyện một mô hình theo quy trình 2 giai đoạn."""
    set_seed(int(config.get("project.seed", 42)))

    loaders, class_mapping = build_dataloaders(config)
    num_classes = len(class_mapping)
    logger.info("Mô hình: %s | số lớp: %d | device: %s", model_name, num_classes, device)

    # Loss (có class weights nếu cấu hình bật).
    class_weights = None
    if bool(config.get("data.use_class_weights", True)):
        class_weights = compute_class_weights(config, num_classes).to(device)
    criterion = build_loss("cross_entropy", class_weights=class_weights)

    bundle = create_model(
        model_name,
        num_classes,
        pretrained=bool(config.get("training.pretrained", True)),
    )
    logger.info("Số tham số: %s", f"{count_parameters(bundle.model):,}")

    checkpoint_dir = ensure_dir(config.output_root / "checkpoints" / model_name)
    log_dir = ensure_dir(config.output_root / "logs")
    fig_dir = ensure_dir(config.output_root / "figures" / "training")
    report_dir = ensure_dir(config.output_root / "reports")

    trainer = Trainer(
        model=bundle.model,
        model_name=model_name,
        loaders=loaders,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        class_mapping=class_mapping,
        log_dir=log_dir,
    )

    weight_decay = float(config.get("training.weight_decay", 1e-4))
    classifier_lr = float(config.get("training.classifier_lr", 1e-3))
    finetune_lr = float(config.get("training.finetune_lr", 1e-4))
    baseline_epochs = int(config.get("training.baseline_epochs", 8))
    finetune_epochs = int(config.get("training.finetune_epochs", 20))
    patience = int(config.get("training.early_stopping_patience", 5))

    if quick_test:
        baseline_epochs, finetune_epochs, patience = 1, 1, 10
        logger.info("Chế độ quick-test: chỉ chạy 1+1 epoch để kiểm tra pipeline.")

    # ----- Giai đoạn 1: baseline (đóng băng backbone) ----------------------
    freeze_backbone(bundle.model, model_name)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, bundle.model.parameters()),
        lr=classifier_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    es_baseline = EarlyStopping(patience=patience, mode="max")
    done = trainer.train_phase(
        "baseline", baseline_epochs, optimizer, scheduler, es_baseline
    )

    # ----- Giai đoạn 2: fine-tuning (mở băng toàn bộ) ----------------------
    unfreeze_all(bundle.model)
    optimizer = torch.optim.AdamW(
        bundle.model.parameters(), lr=finetune_lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    es_finetune = EarlyStopping(patience=patience, mode="max")
    trainer.train_phase(
        "finetune", finetune_epochs, optimizer, scheduler, es_finetune, start_epoch=done
    )

    # ----- Lưu lịch sử -----------------------------------------------------
    trainer.save_history(
        report_dir / f"training_history_{model_name}.csv",
        fig_dir / f"training_curves_{model_name}.png",
    )
    logger.info("Hoàn tất huấn luyện %s.", model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại da liễu.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", choices=SUPPORTED_MODELS, help="Tên mô hình cần train."
    )
    group.add_argument(
        "--all-models", action="store_true", help="Train toàn bộ 4 mô hình."
    )
    parser.add_argument(
        "--config", default=None, help="Đường dẫn file config (mặc định base.yaml)."
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Chạy thử nhanh (1+1 epoch) để kiểm tra pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all_models:
        models_to_train = list(config.get("models", list(SUPPORTED_MODELS)))
    else:
        models_to_train = [args.model]

    for name in models_to_train:
        train_one_model(name, config, device, quick_test=args.quick_test)


if __name__ == "__main__":
    main()
