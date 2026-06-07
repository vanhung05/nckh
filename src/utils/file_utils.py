"""Các tiện ích thao tác file và thư mục."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Tạo thư mục (kể cả thư mục cha) nếu chưa tồn tại.

    Args:
        path: Đường dẫn thư mục cần đảm bảo tồn tại.

    Returns:
        Đối tượng :class:`pathlib.Path` của thư mục.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_image_files(
    directory: str | Path,
    valid_extensions: tuple[str, ...] | list[str],
) -> list[Path]:
    """Liệt kê tất cả file ảnh hợp lệ trong một thư mục (không đệ quy).

    Args:
        directory: Thư mục chứa ảnh.
        valid_extensions: Danh sách phần mở rộng hợp lệ (đã gồm dấu chấm).

    Returns:
        Danh sách đường dẫn ảnh, đã sắp xếp theo tên.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    exts = {ext.lower() for ext in valid_extensions}
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(files)


def list_subdirectories(directory: str | Path) -> list[Path]:
    """Liệt kê các thư mục con trực tiếp, đã sắp xếp theo tên."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_dir())
