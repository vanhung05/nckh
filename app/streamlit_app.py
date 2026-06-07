"""Hệ thống demo phân loại bệnh da liễu bằng Streamlit.

Chức năng:
    - Upload ảnh.
    - Chọn mô hình (hoặc dùng mô hình tốt nhất mặc định).
    - Hiển thị top-3 dự đoán + xác suất từng lớp.
    - Sinh và hiển thị overlay Grad-CAM.
    - Hiển thị cảnh báo y khoa.

Cách chạy::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Cho phép import package src khi chạy bằng streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.transforms import build_eval_transforms  # noqa: E402
from src.explainability.gradcam import GradCAM  # noqa: E402
from src.explainability.gradcam_utils import (  # noqa: E402
    overlay_heatmap,
    tensor_to_image,
)
from src.models.classifier import load_class_names, load_trained_model  # noqa: E402
from src.models.model_factory import SUPPORTED_MODELS, get_target_layer  # noqa: E402
from src.utils.config import load_config  # noqa: E402

MEDICAL_DISCLAIMER = (
    "Kết quả chỉ mang tính hỗ trợ tham khảo và không thay thế "
    "chẩn đoán của bác sĩ."
)


@st.cache_resource
def _load_config_cached():
    return load_config()


@st.cache_data
def _available_models(output_root_str: str) -> list[str]:
    """Liệt kê các mô hình đã có checkpoint."""
    output_root = Path(output_root_str)
    available = []
    for name in SUPPORTED_MODELS:
        if (output_root / "checkpoints" / name / "best.pt").exists():
            available.append(name)
    return available


@st.cache_resource
def _load_model_cached(model_name: str, output_root_str: str, num_classes: int):
    output_root = Path(output_root_str)
    checkpoint = output_root / "checkpoints" / model_name / "best.pt"
    device = torch.device("cpu")
    bundle = load_trained_model(model_name, checkpoint, num_classes, device=device)
    return bundle


def _best_model_name(output_root: Path, available: list[str]) -> str:
    """Chọn mô hình tốt nhất theo macro_f1 trong model_comparison.csv."""
    comparison = output_root / "reports" / "model_comparison.csv"
    if comparison.exists():
        df = pd.read_csv(comparison)
        df = df[df["model_name"].isin(available)]
        if not df.empty:
            return df.sort_values("macro_f1", ascending=False).iloc[0]["model_name"]
    return available[0]


def main() -> None:
    st.set_page_config(page_title="Phân loại bệnh da liễu", layout="wide")
    st.title("Hệ thống hỗ trợ phân loại bệnh da liễu (CNN + Grad-CAM)")
    st.warning(MEDICAL_DISCLAIMER)

    config = _load_config_cached()
    output_root = config.output_root

    try:
        class_names = load_class_names(output_root)
    except FileNotFoundError:
        st.error(
            "Chưa có class_mapping.json. Hãy chạy "
            "`python -m src.data.explore_dataset` trước."
        )
        return
    num_classes = len(class_names)

    available = _available_models(str(output_root))
    if not available:
        st.error(
            "Chưa có mô hình nào được huấn luyện. Hãy chạy "
            "`python -m src.training.train --all-models` trước."
        )
        return

    # ----- Sidebar: chọn mô hình -------------------------------------------
    st.sidebar.header("Cấu hình")
    default_model = _best_model_name(output_root, available)
    use_best = st.sidebar.checkbox(
        f"Dùng mô hình tốt nhất ({default_model})", value=True
    )
    if use_best:
        model_name = default_model
    else:
        model_name = st.sidebar.selectbox("Chọn mô hình", available)

    show_gradcam = st.sidebar.checkbox("Hiển thị Grad-CAM", value=True)

    # ----- Upload ảnh -------------------------------------------------------
    uploaded = st.file_uploader(
        "Tải lên ảnh da liễu", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is None:
        st.info("Vui lòng tải lên một ảnh để bắt đầu.")
        return

    image = Image.open(uploaded).convert("RGB")
    image_size = int(config.get("data.image_size", 224))
    transform = build_eval_transforms(image_size)
    tensor = transform(image).unsqueeze(0)

    bundle = _load_model_cached(model_name, str(output_root), num_classes)

    # ----- Dự đoán ----------------------------------------------------------
    with torch.no_grad():
        logits = bundle.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_k = int(config.get("evaluation.top_k", 3))
    top_idx = np.argsort(-probs)[:top_k]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh đầu vào")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Top-{top_k} dự đoán ({model_name})")
        for rank, idx in enumerate(top_idx, start=1):
            st.write(f"{rank}. **{class_names[idx]}** — {probs[idx]:.2%}")
            st.progress(float(probs[idx]))

    # ----- Grad-CAM ---------------------------------------------------------
    if show_gradcam:
        st.subheader("Trực quan hóa Grad-CAM")
        target_layer = get_target_layer(bundle.model, model_name)
        cam = GradCAM(bundle.model, target_layer)
        try:
            heatmap, used_class, confidence = cam.generate(tensor)
        finally:
            cam.remove_hooks()
        orig = tensor_to_image(tensor.squeeze(0))
        overlay = overlay_heatmap(orig, heatmap)

        gc1, gc2, gc3 = st.columns(3)
        gc1.image(orig, caption="Ảnh (đã tiền xử lý)", use_container_width=True)
        gc2.image(
            (heatmap * 255).astype("uint8"),
            caption="Heatmap",
            use_container_width=True,
            clamp=True,
        )
        gc3.image(overlay, caption="Overlay", use_container_width=True)
        st.caption(
            f"Vùng nóng thể hiện nơi mô hình tập trung khi dự đoán "
            f"'{class_names[used_class]}' ({confidence:.2%})."
        )

    # ----- Bảng xác suất đầy đủ --------------------------------------------
    with st.expander("Xem xác suất tất cả các lớp"):
        prob_df = pd.DataFrame(
            {"class_name": class_names, "probability": probs}
        ).sort_values("probability", ascending=False)
        prob_df["probability"] = (prob_df["probability"] * 100).round(2)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.warning(MEDICAL_DISCLAIMER)


if __name__ == "__main__":
    main()
