"""Trainer dùng chung cho cả 4 mô hình.

Hỗ trợ huấn luyện 2 giai đoạn:
    1. Baseline: đóng băng backbone, chỉ train classifier head.
    2. Fine-tuning: mở băng toàn bộ, learning rate nhỏ hơn.

Theo dõi loss/accuracy/macro-F1, early stopping theo macro-F1, lưu checkpoint
tốt nhất và ghi lịch sử từng epoch ra CSV + biểu đồ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.callbacks import CheckpointSaver, EarlyStopping
from src.utils.logger import get_logger


@dataclass
class EpochResult:
    """Kết quả một epoch."""

    epoch: int
    phase: str
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_macro_f1: float
    lr: float
    seconds: float


@dataclass
class TrainHistory:
    """Lịch sử huấn luyện toàn bộ các epoch."""

    results: list[EpochResult] = field(default_factory=list)

    def add(self, result: EpochResult) -> None:
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.results])


class Trainer:
    """Bao bọc vòng lặp huấn luyện và đánh giá."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        loaders: dict[str, DataLoader],
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str | Path,
        class_mapping: dict[str, int],
        log_dir: str | Path,
    ) -> None:
        self.model = model.to(device)
        self.model_name = model_name
        self.loaders = loaders
        self.criterion = criterion
        self.device = device
        self.class_mapping = class_mapping
        self.history = TrainHistory()
        self.saver = CheckpointSaver(checkpoint_dir, model_name)

        log_path = Path(log_dir) / f"{model_name}.log"
        self.logger = get_logger(f"trainer.{model_name}", log_file=log_path)

    # ----- vòng lặp cơ bản --------------------------------------------------
    def _run_epoch(
        self, loader: DataLoader, train: bool, optimizer=None
    ) -> tuple[float, float, float]:
        """Chạy một epoch. Trả về (loss, accuracy, macro_f1)."""
        self.model.train(train)
        total_loss = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []

        context = torch.enable_grad() if train else torch.no_grad()
        desc = "train" if train else "eval"
        with context:
            for images, targets in tqdm(loader, desc=desc, leave=False):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if train:
                    optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                if train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        n = len(loader.dataset)  # type: ignore[arg-type]
        avg_loss = total_loss / n
        accuracy = float(np.mean(np.array(all_preds) == np.array(all_targets)))
        macro_f1 = f1_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        return avg_loss, accuracy, macro_f1

    # ----- một giai đoạn (baseline hoặc finetune) --------------------------
    def train_phase(
        self,
        phase: str,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler,
        early_stopping: EarlyStopping,
        start_epoch: int = 0,
    ) -> int:
        """Huấn luyện một giai đoạn. Trả về số epoch đã chạy thực tế."""
        self.logger.info(
            "=== Bắt đầu giai đoạn '%s' (%d epoch) cho %s ===",
            phase,
            epochs,
            self.model_name,
        )
        epochs_done = 0
        for e in range(epochs):
            epoch = start_epoch + e + 1
            t0 = time.time()

            train_loss, train_acc, _ = self._run_epoch(
                self.loaders["train"], train=True, optimizer=optimizer
            )
            val_loss, val_acc, val_f1 = self._run_epoch(
                self.loaders["val"], train=False
            )

            lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                if isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            elapsed = time.time() - t0
            result = EpochResult(
                epoch=epoch,
                phase=phase,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_macro_f1=val_f1,
                lr=lr,
                seconds=elapsed,
            )
            self.history.add(result)

            self.logger.info(
                "[%s][epoch %d] train_loss=%.4f train_acc=%.4f "
                "val_loss=%.4f val_acc=%.4f val_macroF1=%.4f lr=%.2e (%.1fs)",
                phase,
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_f1,
                lr,
                elapsed,
            )

            # Early stopping + checkpoint theo macro F1 (mode=max).
            is_best = early_stopping.step(val_f1)
            if is_best:
                path = self.saver.save(
                    self.model,
                    epoch,
                    {
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_macro_f1": val_f1,
                    },
                    self.class_mapping,
                )
                self.logger.info("  -> Lưu checkpoint tốt nhất: %s", path)

            epochs_done += 1
            if early_stopping.should_stop:
                self.logger.info(
                    "Early stopping ở epoch %d (không cải thiện %d epoch).",
                    epoch,
                    early_stopping.patience,
                )
                break

        return epochs_done

    # ----- lưu lịch sử ------------------------------------------------------
    def save_history(self, out_csv: Path, fig_path: Path) -> None:
        """Lưu lịch sử train ra CSV và vẽ biểu đồ loss/accuracy."""
        df = self.history.to_dataframe()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        self.logger.info("Đã lưu lịch sử train: %s", out_csv)

        if df.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(df["epoch"], df["train_loss"], label="train_loss", marker="o")
        axes[0].plot(df["epoch"], df["val_loss"], label="val_loss", marker="o")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{self.model_name} - Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df["epoch"], df["train_acc"], label="train_acc", marker="o")
        axes[1].plot(df["epoch"], df["val_acc"], label="val_acc", marker="o")
        axes[1].plot(
            df["epoch"], df["val_macro_f1"], label="val_macro_f1", marker="o"
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title(f"{self.model_name} - Accuracy / F1")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        self.logger.info("Đã lưu biểu đồ train: %s", fig_path)
