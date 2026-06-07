"""Cấu hình logger dùng chung cho toàn dự án."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "skin_disease",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Tạo (hoặc lấy lại) logger đã cấu hình sẵn handler.

    Args:
        name: Tên logger.
        level: Mức log.
        log_file: Nếu được cung cấp, log sẽ được ghi thêm vào file này.

    Returns:
        Đối tượng :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Tránh thêm handler trùng lặp khi gọi nhiều lần.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
