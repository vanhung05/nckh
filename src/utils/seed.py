"""Đặt seed cố định để đảm bảo khả năng tái lập kết quả."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Đặt seed cho toàn bộ nguồn ngẫu nhiên thường dùng.

    Args:
        seed: Giá trị seed.
        deterministic: Nếu True, cấu hình PyTorch chạy ở chế độ xác định
            (chậm hơn nhưng tái lập tốt hơn).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch chưa cần thiết ở các bước khảo sát dữ liệu.
        pass
