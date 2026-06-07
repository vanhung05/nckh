"""Skin disease classification with CNN + Grad-CAM.

Source package gốc của dự án nghiên cứu.
"""

import sys as _sys

# Ép stdout/stderr dùng UTF-8 để in được tiếng Việt trên console Windows
# (mặc định cp1252 sẽ lỗi UnicodeEncodeError). errors="replace" để an toàn.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(_sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - môi trường hiếm
            pass
