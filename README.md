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

## 4. Cấu trúc thư mục

```text
.
├── Data/                  # dữ liệu gốc (train/test)
├── configs/               # file cấu hình YAML
├── src/
│   ├── data/              # khảo sát, kiểm tra, split, dataset
│   ├── models/            # model factory, classifier
│   ├── training/          # trainer, losses, callbacks, train CLI
│   ├── evaluation/        # metrics, confusion matrix, report, benchmark
│   ├── explainability/    # Grad-CAM
│   └── utils/             # config, logger, seed, file utils
├── app/                   # Streamlit demo
├── scripts/               # script PowerShell chạy nhanh
└── outputs/               # CSV, hình, log, checkpoint, report
```

## 5. Cài đặt môi trường

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Lưu ý: torch bản CPU sẽ huấn luyện chậm. Nếu có GPU NVIDIA, hãy cài bản
> CUDA tương ứng theo hướng dẫn tại pytorch.org để tăng tốc.

### Train trên GPU miễn phí (Colab / Kaggle)

Nếu máy chỉ có CPU, xem **[CLOUD_SETUP.md](CLOUD_SETUP.md)** để chạy trên
Google Colab hoặc Kaggle (GPU T4/P100 miễn phí). Có sẵn notebook:
`notebooks/run_on_kaggle.ipynb` và `notebooks/run_on_colab.ipynb`.

Code hỗ trợ 2 biến môi trường để khỏi sửa config khi đổi môi trường:
`SKIN_DATA_ROOT` (thư mục chứa train/test) và `SKIN_OUTPUT_ROOT` (nơi lưu kết quả).

## 6. Khảo sát và chuẩn bị dữ liệu

Chạy lần lượt hoặc dùng script gộp:

```powershell
python -m src.data.explore_dataset
python -m src.data.validate_images
python -m src.data.detect_duplicates
python -m src.data.create_validation_split
# hoặc:
.\scripts\run_dataset_analysis.ps1
```

Kết quả nằm trong `outputs/dataset/` và `outputs/figures/dataset/`.

## 7. Huấn luyện một mô hình

```powershell
python -m src.training.train --model resnet18
```

Có thể kiểm tra nhanh pipeline (1+1 epoch):

```powershell
python -m src.training.train --model resnet18 --quick-test
```

## 8. Huấn luyện toàn bộ mô hình

```powershell
python -m src.training.train --all-models
# hoặc:
.\scripts\run_train_all.ps1
```

Mỗi mô hình train 2 giai đoạn: baseline (đóng băng backbone) rồi fine-tune
(mở băng toàn bộ). Checkpoint tốt nhất theo macro-F1 lưu tại
`outputs/checkpoints/<model>/best.pt`.

## 9. Đánh giá

```powershell
python -m src.evaluation.evaluate --all-models
# hoặc:
.\scripts\run_evaluate_all.ps1
```

Sinh ra:

- `outputs/reports/model_comparison.csv` — bảng so sánh 4 mô hình.
- `outputs/reports/per_class_metrics.csv` — metric theo lớp.
- `outputs/reports/classification_report_<model>.csv`.
- `outputs/figures/evaluation/confusion_matrix_<model>.png`.

## 10. Sinh Grad-CAM

```powershell
python -m src.explainability.generate_gradcam_samples --all-models
# hoặc:
.\scripts\run_gradcam_all.ps1
```

Kết quả tại `outputs/figures/gradcam/<model>/` gồm các nhóm: dự đoán đúng
confidence cao, đúng confidence thấp, và dự đoán sai.

## 11. Chạy demo Streamlit

```powershell
streamlit run app/streamlit_app.py
```

Giao diện cho phép upload ảnh, chọn mô hình, xem top-3 dự đoán, xác suất từng
lớp và overlay Grad-CAM.

## 12. Kết quả thực nghiệm

Bảng so sánh đầy đủ được sinh tại `outputs/reports/model_comparison.csv` sau
khi huấn luyện và đánh giá thực tế. **Không điền số liệu giả** — hãy chạy
pipeline để có kết quả.

## 13. Cảnh báo y khoa

Kết quả chỉ mang tính hỗ trợ tham khảo và không thay thế chẩn đoán của bác sĩ.

## 14. Hướng phát triển

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
