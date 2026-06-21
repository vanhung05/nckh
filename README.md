# Phân loại bệnh da liễu bằng CNN kết hợp Grad-CAM

Hệ thống học sâu nhận diện và phân loại bệnh da liễu từ ảnh, so sánh 4 kiến
trúc CNN và tích hợp Grad-CAM để trực quan hóa vùng ảnh ảnh hưởng tới quyết
định của mô hình.

> **Cảnh báo y khoa:** Kết quả chỉ mang tính hỗ trợ tham khảo và không thay
> thế chẩn đoán của bác sĩ.

---

## 1. Giới thiệu đề tài

Đề tài nghiên cứu khoa học: *Ứng dụng học sâu trong nhận diện và phân loại
bệnh da liễu từ hình ảnh sử dụng các mô hình CNN kết hợp Grad-CAM.*

## 2. Mục tiêu

- Xây dựng pipeline phân loại bệnh da liễu thống nhất cho nhiều mô hình.
- So sánh công bằng 4 kiến trúc: ResNet18, MobileNetV3-Large, DenseNet121,
  EfficientNet-B0.
- Tích hợp Grad-CAM để giải thích dự đoán.
- Cung cấp demo Streamlit.

## 3. Dataset

- Nguồn: Kaggle - [31 classes of skin disease](https://www.kaggle.com/datasets/kelixo25/31-classes-of-skin-disease).
- Cấu trúc: `Data/train` và `Data/test`, mỗi thư mục gồm 31 lớp bệnh.
- Dữ liệu gốc **không bị thay đổi**. Validation set được tạo bằng metadata
  (stratified split 85/15 từ `train`), không sao chép ảnh.

### Lưu ý chất lượng dữ liệu (đã phát hiện và xử lý)

Quá trình kiểm tra tự động phát hiện một số vấn đề và đã ghi nhận để chống rò
rỉ dữ liệu:

- Một lớp viết khác hoa/thường giữa train và test (`Actinic keratosis` vs
  `Actinic Keratosis`) — đã đối chiếu tự động.
- Ảnh trùng tuyệt đối giữa `train` và `test` (rò rỉ) — đã loại khỏi train.
- Một số ảnh trùng nhưng gắn nhãn mâu thuẫn (Actinic Keratosis / Nevus) — đã
  loại khỏi train.
- Ảnh trùng nội bộ train — giữ lại một bản.

Chi tiết xem `outputs/dataset/duplicate_images.csv` và
`outputs/dataset/excluded_images.csv`.

---

# PHẦN I — KIẾN TRÚC HỆ THỐNG

## 4. Tổng quan kiến trúc

Hệ thống chia thành các tầng tách biệt rõ ràng, dữ liệu chảy một chiều:

```
            configs/base.yaml  +  biến môi trường (SKIN_DATA_ROOT, SKIN_OUTPUT_ROOT)
                                  │
                                  ▼
        ┌─────────────────── src/utils (nền tảng dùng chung) ───────────────────┐
        │  config.py · logger.py · seed.py · file_utils.py                       │
        └────────────────────────────────────────────────────────────────────────┘
                                  │
   Data/train, Data/test ──► src/data ──► metadata (CSV/JSON) ──► DataLoader
                                  │                                   │
                                  ▼                                   ▼
                            src/models ───────────────────────► src/training
                          (factory 4 CNN)                       (train 2 giai đoạn)
                                  │                                   │
                                  ▼                                   ▼
                            checkpoints/<model>/best.pt  ──► src/evaluation (metric, CM)
                                  │                                   │
                                  ▼                                   ▼
                          src/explainability (Grad-CAM)        reports/ + figures/
                                  │
                                  ▼
                            app/streamlit_app.py (demo)
```

Nguyên tắc thiết kế:

- **Không hard-code**: mọi tham số (đường dẫn, batch size, epoch, lr, số lớp)
  đọc từ `configs/base.yaml` hoặc suy ra từ dữ liệu (`class_mapping.json`).
- **Không rò rỉ dữ liệu**: train/val/test cố định trong `split_metadata.csv`,
  validation chỉ tách từ train, ảnh trùng/rò rỉ bị loại.
- **So sánh công bằng**: 4 mô hình dùng chung split, seed, augmentation, metric.
- **Tái lập**: seed cố định 42 ở mọi script.

## 5. Cây thư mục

```text
NCKH_V2/
├── Data/                         # Dữ liệu gốc (KHÔNG sửa)
│   ├── train/<31 lớp>/*.jpg
│   └── test/<31 lớp>/*.jpg
│
├── configs/
│   └── base.yaml                 # Cấu hình trung tâm
│
├── src/                          # Toàn bộ logic (chi tiết ở mục 6)
│   ├── data/                     # Chuẩn bị & nạp dữ liệu
│   ├── models/                   # Định nghĩa & nạp mô hình
│   ├── training/                 # Huấn luyện
│   ├── evaluation/               # Đánh giá
│   ├── explainability/           # Grad-CAM
│   └── utils/                    # Tiện ích nền tảng
│
├── app/
│   └── streamlit_app.py          # Demo web
│
├── notebooks/                    # EDA + notebook chạy trên Kaggle
│
├── outputs/                      # Mọi kết quả sinh ra
│   ├── dataset/                  # metadata: class_mapping.json, split_metadata.csv, ...
│   ├── checkpoints/<model>/best.pt
│   ├── figures/                  # dataset / training / evaluation / gradcam
│   ├── logs/                     # log train theo mô hình
│   └── reports/                  # model_comparison.csv, classification_report_*, ...
│
├── requirements.txt
└── README.md
```

## 6. Chi tiết nhiệm vụ từng file trong `src/`

### 6.1. `src/utils/` — Nền tảng dùng chung

| File | Nhiệm vụ |
|------|----------|
| `config.py` | Lớp `Config` đọc `base.yaml`, hỗ trợ truy cập key lồng nhau (`config.get("training.classifier_lr")`). Hai property `data_root` / `output_root` **ưu tiên biến môi trường** `SKIN_DATA_ROOT` / `SKIN_OUTPUT_ROOT` (chạy được trên Kaggle không cần sửa file), nếu không có thì lấy từ config. |
| `seed.py` | `set_seed(42)` cố định seed cho `random`, `numpy`, `torch` (+ cuDNN deterministic) để tái lập. |
| `logger.py` | `get_logger()` tạo logger ghi ra cả console và file `.log`, định dạng thống nhất. |
| `file_utils.py` | `ensure_dir()`, `list_image_files()` (ảnh hợp lệ trong 1 thư mục), `list_subdirectories()` (các lớp). |

Đây là tầng đáy: mọi module khác import từ đây, bản thân nó không phụ thuộc
module nào trong `src`.

### 6.2. `src/data/` — Chuẩn bị và nạp dữ liệu

Bốn script CLI chạy **theo thứ tự**, mỗi script ghi ra `outputs/dataset/`:

| File | Là gì | Đầu ra |
|------|-------|--------|
| `explore_dataset.py` | **Bước 1** – Khảo sát: đếm lớp, đếm ảnh train/test từng lớp, đối chiếu chênh lệch hoa/thường, đánh giá mất cân bằng. | `class_distribution.csv`, `class_mapping.json`, `figures/dataset/*.png` |
| `validate_images.py` | **Bước 2** – Kiểm tra ảnh lỗi: file hỏng, không đọc được, truncated, kích thước bất thường, sai phần mở rộng. | `invalid_images.csv` |
| `detect_duplicates.py` | **Bước 3** – Phát hiện trùng: MD5 (trùng tuyệt đối) + perceptual hash (gần trùng), đặc biệt **rò rỉ train↔test**. | `duplicate_images.csv` |
| `create_validation_split.py` | **Bước 4** – Split phân tầng 85/15 từ train, **loại ảnh rò rỉ / trùng / nhãn mâu thuẫn** (union-find). Test giữ nguyên. | `split_metadata.csv`, `excluded_images.csv` |

Hai module **thư viện** (được import bởi tầng train/eval):

| File | Nhiệm vụ |
|------|----------|
| `transforms.py` | `build_train_transforms` (RandomResizedCrop, flip, rotation, color jitter nhẹ + normalize ImageNet); `build_eval_transforms` (resize + center crop + normalize, **không** augmentation ngẫu nhiên); `denormalize` để hiển thị. |
| `dataset.py` | `SkinDiseaseDataset` (đọc ảnh theo split từ `split_metadata.csv`); `build_dataloaders` tạo 3 DataLoader train/val/test; `compute_class_weights` (inverse frequency cho mất cân bằng). |

`class_mapping.json` là **nguồn chân lý** về số lớp và ánh xạ tên→index; số lớp
không bao giờ hard-code.

### 6.3. `src/models/` — Định nghĩa mô hình

| File | Nhiệm vụ |
|------|----------|
| `model_factory.py` | `SUPPORTED_MODELS` = (resnet18, mobilenet_v3_large, densenet121, efficientnet_b0). `create_model()` tải backbone pretrained ImageNet và **thay head** theo số lớp thực tế. `get_target_layer()` trả layer conv cuối cho Grad-CAM. `freeze_backbone()` / `unfreeze_all()` cho train 2 giai đoạn. `count_parameters()`. |
| `classifier.py` | `load_trained_model()` tạo model rồi nạp `best.pt`; `load_class_names()` đọc tên lớp theo đúng thứ tự index. Dùng bởi evaluate, Grad-CAM và app. |

### 6.4. `src/training/` — Huấn luyện

| File | Nhiệm vụ |
|------|----------|
| `train.py` | **CLI điểm vào**. Mỗi mô hình: tạo model → **baseline** (đóng băng backbone, lr cao) → **fine-tune** (mở băng toàn bộ, lr thấp). AdamW + ReduceLROnPlateau. Có cờ `--quick-test`. |
| `trainer.py` | Lớp `Trainer`: vòng lặp train/eval 1 epoch, theo dõi loss/accuracy/macro-F1, gọi early stopping, lưu checkpoint tốt nhất, `save_history()` ghi CSV + vẽ biểu đồ. |
| `callbacks.py` | `EarlyStopping` (theo macro-F1) và `CheckpointSaver` (lưu `best.pt`: model_state, epoch, metrics, class_mapping). |
| `losses.py` | `build_loss()` tạo `CrossEntropyLoss` (kèm class weights) hoặc `FocalLoss` (thí nghiệm bổ sung). |

Tiêu chí chọn checkpoint tốt nhất: **macro-F1 trên validation**.

### 6.5. `src/evaluation/` — Đánh giá

| File | Nhiệm vụ |
|------|----------|
| `evaluate.py` | **CLI điểm vào**. Nạp `best.pt`, inference trên **test set**, tính metric, vẽ confusion matrix, sinh classification report, đo thông số triển khai, tổng hợp `model_comparison.csv`. |
| `metrics.py` | accuracy, macro precision/recall/F1, weighted F1, top-3 accuracy. |
| `confusion_matrix.py` | Vẽ ma trận nhầm lẫn (chuẩn hóa theo hàng). |
| `classification_report.py` | Xuất precision/recall/F1/support theo từng lớp ra CSV. |
| `benchmark_inference.py` | Đo thời gian inference/ảnh (ms) và kích thước checkpoint. |
| `generate_summary.py` | **CLI** tổng hợp các output thực tế thành `reports/final_summary.md` (chỉ dùng số liệu CÓ THẬT). |

### 6.6. `src/explainability/` — Grad-CAM

| File | Nhiệm vụ |
|------|----------|
| `gradcam.py` | Lớp `GradCAM` thuần PyTorch: forward hook bắt activation + tensor hook bắt gradient (tránh xung đột view+inplace với DenseNet). `generate()` tính heatmap = ReLU(Σ weight·activation), nội suy về kích thước ảnh, chuẩn hóa [0,1]. |
| `gradcam_utils.py` | `tensor_to_image`, `heatmap_to_color` (JET), `overlay_heatmap`, `save_gradcam_figure` (hình 3 panel: gốc / heatmap / overlay + nhãn). |
| `generate_gradcam_samples.py` | **CLI điểm vào**. Chọn 3 nhóm ảnh test (đúng-conf cao, đúng-conf thấp, sai) rồi sinh Grad-CAM vào `figures/gradcam/<model>/`. |

## 7. Luồng chạy chi tiết (end-to-end)

### Giai đoạn A — Chuẩn bị dữ liệu (chạy 1 lần)

1. `explore_dataset` quét `Data/train`, `Data/test` → `class_mapping.json` (31 lớp → index 0..30) + thống kê.
2. `validate_images` đánh dấu ảnh hỏng.
3. `detect_duplicates` tìm ảnh trùng và rò rỉ train↔test.
4. `create_validation_split` tạo `split_metadata.csv` (`image_path, class_name, class_index, split`), loại ảnh rò rỉ/trùng/nhãn mâu thuẫn.

→ Mọi bước sau **chỉ đọc metadata**, không quét lại ảnh thô.

### Giai đoạn B — Huấn luyện

```
build_dataloaders → compute_class_weights → create_model
  ├─ baseline:  freeze_backbone → AdamW(lr=1e-3) → train head
  └─ fine-tune: unfreeze_all   → AdamW(lr=1e-4) → train toàn bộ
        mỗi epoch: train → eval(val) → macro-F1 → EarlyStopping + lưu best.pt
  → training_history_<model>.csv + training_curves_<model>.png
```

### Giai đoạn C — Đánh giá

```
load_trained_model → inference(test) → (y_true, y_pred, probs)
  compute_metrics / plot_confusion_matrix / save_classification_report / benchmark_inference
  → gộp thành model_comparison.csv (sắp theo macro-F1)
```

### Giai đoạn D — Giải thích

Chọn ảnh đại diện (đúng cao / đúng thấp / sai) → Grad-CAM → `figures/gradcam/<model>/*.png`.

### Giai đoạn E — Tổng hợp & Demo

`generate_summary` → `reports/final_summary.md`; `streamlit_app.py` đọc
`model_comparison.csv` để chọn mô hình tốt nhất làm mặc định.

## 8. Quan hệ phụ thuộc & hợp đồng dữ liệu

```
utils  ◄── data ◄── (training, evaluation, explainability)
  ▲         ▲              │
  │         │              ▼
  └──── models ◄───────────┘
```

Không có phụ thuộc vòng. Các file trung gian kết nối các tầng:

| File trung gian | Sinh bởi | Tiêu thụ bởi | Cấu trúc |
|-----------------|----------|--------------|----------|
| `class_mapping.json` | explore_dataset | dataset, classifier, evaluate, gradcam | `{ "tên lớp": index }` |
| `split_metadata.csv` | create_validation_split | dataset, gradcam | `image_path, class_name, class_index, split` |
| `best.pt` | trainer | classifier (eval/gradcam/app) | `model_state, epoch, metrics, class_mapping` |
| `model_comparison.csv` | evaluate | generate_summary, app | mỗi dòng 1 mô hình + metric |

---

# PHẦN II — HƯỚNG DẪN SỬ DỤNG

## 9. Cài đặt môi trường

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Lưu ý: torch bản CPU huấn luyện chậm. Nếu có GPU NVIDIA, cài bản CUDA tương
> ứng theo hướng dẫn tại pytorch.org để tăng tốc.

### Train trên GPU miễn phí (Kaggle)

Nếu máy chỉ có CPU, dùng notebook có sẵn để train trên GPU miễn phí:
`notebooks/run_on_kaggle.ipynb` (dataset có sẵn trên Kaggle nên gắn trực tiếp,
không cần upload).

Code hỗ trợ 2 biến môi trường để khỏi sửa config khi đổi môi trường:
`SKIN_DATA_ROOT` (thư mục chứa train/test) và `SKIN_OUTPUT_ROOT` (nơi lưu kết quả).

## 10. Khảo sát và chuẩn bị dữ liệu

```powershell
python -m src.data.explore_dataset
python -m src.data.validate_images
python -m src.data.detect_duplicates
python -m src.data.create_validation_split
```

Kết quả nằm trong `outputs/dataset/` và `outputs/figures/dataset/`.

## 11. Huấn luyện

```powershell
# Một mô hình
python -m src.training.train --model resnet18

# Kiểm tra nhanh pipeline (1+1 epoch)
python -m src.training.train --model resnet18 --quick-test

# Toàn bộ 4 mô hình
python -m src.training.train --all-models
```

Checkpoint tốt nhất theo macro-F1 lưu tại `outputs/checkpoints/<model>/best.pt`.

## 12. Đánh giá

```powershell
python -m src.evaluation.evaluate --all-models
```

Sinh ra: `model_comparison.csv`, `per_class_metrics.csv`,
`classification_report_<model>.csv`, `confusion_matrix_<model>.png`.

## 13. Sinh Grad-CAM

```powershell
python -m src.explainability.generate_gradcam_samples --all-models
```

Kết quả tại `outputs/figures/gradcam/<model>/`: dự đoán đúng-confidence cao,
đúng-confidence thấp, và dự đoán sai.

## 14. Chạy demo Streamlit

```powershell
python -m streamlit run app/streamlit_app.py
```

Upload ảnh → top-3 dự đoán + xác suất từng lớp → overlay Grad-CAM. App tự dùng
mô hình tốt nhất (theo `model_comparison.csv`) làm mặc định.

---

## 15. Kết quả thực nghiệm

Kết quả huấn luyện thực tế trên GPU Tesla T4 (test set, sắp theo macro-F1):

| Mô hình | Accuracy | Macro-F1 | Weighted-F1 | Top-3 Acc | Tham số | Size (MB) | Inference (ms) |
|---|---|---|---|---|---|---|---|
| **densenet121** ⭐ | 0.793 | **0.807** | 0.793 | 0.939 | 6.99M | 27.2 | 15.1 |
| efficientnet_b0 | 0.795 | 0.802 | 0.796 | 0.915 | 4.05M | 15.7 | 8.2 |
| resnet18 | 0.781 | 0.794 | 0.783 | 0.925 | 11.19M | 42.8 | 2.9 |
| mobilenet_v3_large | 0.758 | 0.769 | 0.756 | 0.914 | 4.24M | 16.4 | 6.6 |

- Mô hình tốt nhất theo macro-F1: **DenseNet121** (macro-F1 = 0.807).
- Bảng đầy đủ luôn được sinh lại tại `outputs/reports/model_comparison.csv`
  sau mỗi lần đánh giá. Số liệu trên cập nhật theo lần train gần nhất; chạy lại
  pipeline để tái lập.

## 16. Cảnh báo y khoa

Kết quả chỉ mang tính hỗ trợ tham khảo và không thay thế chẩn đoán của bác sĩ.

## 17. Hướng phát triển

- Kiểm định với bác sĩ da liễu, bổ sung thông tin lâm sàng.
- Phân tích fairness theo màu da.
- Dùng segmentation khoanh vùng tổn thương.
- Tối ưu MobileNetV3 cho thiết bị tài nguyên thấp.
- So sánh thêm Vision Transformer / ConvNeXt.

---

## Tái lập kết quả

- Seed cố định (`42`) trong `configs/base.yaml`.
- Mọi cấu hình đọc từ file config, không hard-code.
- Cùng split, augmentation, metric cho cả 4 mô hình.
