# PROJECT_SPEC.md  
## Đặc tả dự án nghiên cứu khoa học: Phân loại bệnh da liễu bằng CNN kết hợp Grad-CAM

---

# 1. Thông tin tổng quan

## 1.1. Tên đề tài

**Ứng dụng học sâu trong nhận diện và phân loại bệnh da liễu từ hình ảnh sử dụng các mô hình CNN kết hợp Grad-CAM**

## 1.2. Mục tiêu

Xây dựng một hệ thống học sâu có khả năng nhận diện và phân loại bệnh da liễu từ hình ảnh, đồng thời so sánh hiệu quả của nhiều kiến trúc CNN khác nhau.

Các mô hình cần triển khai:

1. `ResNet18`
2. `MobileNetV3`
3. `DenseNet121`
4. `EfficientNet-B0`

Sau khi huấn luyện và đánh giá, cần chọn ra mô hình phù hợp nhất để tích hợp vào hệ thống demo.

Kỹ thuật `Grad-CAM` phải được tích hợp để trực quan hóa vùng ảnh ảnh hưởng nhiều nhất đến quyết định của mô hình.

> Hệ thống chỉ mang tính hỗ trợ tham khảo, không thay thế kết luận hoặc chẩn đoán của bác sĩ.

---

# 2. Dữ liệu đầu vào

## 2.1. Đường dẫn dữ liệu hiện có

Dataset đã được tải sẵn vào máy tại:

```text
D:\NCKH_V2\Data
```

Không tải lại dataset nếu thư mục trên đã có dữ liệu.

## 2.2. Nguồn dữ liệu

Dataset Kaggle:

```text
31 classes of skin disease
```

Nguồn tham khảo:

```text
https://www.kaggle.com/datasets/kelixo25/31-classes-of-skin-disease
```

## 2.3. Cấu trúc dữ liệu hiện tại

Dataset đã có sẵn các thư mục:

```text
D:\NCKH_V2\Data
├── train
│   ├── Actinic keratosis
│   ├── Basal Cell Carcinoma
│   ├── Darier_s Disease
│   ├── Dermatofibroma
│   ├── Epidermolysis Bullosa Pruriginosa
│   ├── Hailey-Hailey Disease
│   ├── Herpes Simplex
│   ├── Impetigo
│   ├── Larva Migrans
│   ├── Leprosy Borderline
│   ├── Leprosy Lepromatous
│   ├── Leprosy Tuberculoid
│   ├── Lichen Planus
│   ├── Lupus Erythematosus Chronicus Discoides
│   ├── Melanoma
│   ├── Molluscum Contagiosum
│   ├── Mycosis Fungoides
│   ├── Neurofibromatosis
│   ├── Nevus
│   ├── Papilomatosis Confluentes And Reticulate
│   ├── Pediculosis Capitis
│   ├── Pigmented Benign Keratosis
│   ├── Pityriasis Rosea
│   ├── Porokeratosis Actinic
│   ├── Psoriasis
│   ├── Seborrheic keratosis
│   ├── ...
│   └── Tổng cộng dự kiến: 31 lớp
└── test
    ├── ...
    └── Các thư mục lớp tương ứng
```

Lưu ý:

- Tên thư mục lớp phải được lấy trực tiếp từ dữ liệu thực tế.
- Không tự ý đổi tên lớp trước khi lập bảng ánh xạ.
- Phải kiểm tra chính xác số lượng lớp.
- Phải kiểm tra xem thư mục `test` có cùng đầy đủ lớp với `train` hay không.
- Dataset hiện tại có thể chưa có thư mục `val`. Nếu chưa có, cần tách một phần dữ liệu từ `train` để tạo validation set.

---

# 3. Nguyên tắc bắt buộc khi triển khai

## 3.1. Không làm thay đổi dữ liệu gốc

Thư mục sau là dữ liệu nguyên bản:

```text
D:\NCKH_V2\Data
```

Không xóa, không ghi đè, không đổi tên trực tiếp trong thư mục dữ liệu gốc.

Nếu cần tạo validation set hoặc làm sạch dữ liệu, hãy tạo bản sao hoặc lưu metadata tại thư mục khác.

## 3.2. Không làm rò rỉ dữ liệu

Phải đảm bảo:

- Không sử dụng ảnh trong `test` để train.
- Không dùng ảnh `test` để tinh chỉnh hyperparameter.
- Không để ảnh trùng hoặc ảnh gần trùng xuất hiện ở cả train, validation và test.
- Validation set chỉ được tạo từ tập `train`.
- Nếu phát hiện ảnh trùng, cần ghi nhận vào báo cáo và loại bỏ hợp lý.

## 3.3. So sánh công bằng giữa các mô hình

Các mô hình phải được huấn luyện trong điều kiện thống nhất:

- Cùng train/validation/test split.
- Cùng seed.
- Cùng pipeline augmentation.
- Cùng kích thước ảnh đầu vào trong thí nghiệm chính.
- Cùng số epoch tối đa.
- Cùng điều kiện early stopping.
- Cùng bộ metric.
- Cùng phần cứng hoặc cùng môi trường chạy.
- Cùng nguyên tắc chọn checkpoint tốt nhất.

## 3.4. Tái lập kết quả

Mọi script cần đặt seed cố định, ví dụ:

```python
SEED = 42
```

Cần lưu:

- File cấu hình.
- Phiên bản thư viện.
- Log train.
- Checkpoint tốt nhất.
- Metric theo từng epoch.
- Biểu đồ loss và accuracy.
- Bảng kết quả test.
- Confusion matrix.
- Classification report.

---

# 4. Cấu trúc thư mục project cần tạo

Đặt toàn bộ source code trong:

```text
D:\NCKH_V2
```

Cấu trúc đề xuất:

```text
D:\NCKH_V2
│
├── Data
│   ├── train
│   └── test
│
├── configs
│   ├── base.yaml
│   ├── resnet18.yaml
│   ├── mobilenet_v3.yaml
│   ├── densenet121.yaml
│   └── efficientnet_b0.yaml
│
├── notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_quality_check.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_gradcam_analysis.ipynb
│
├── src
│   ├── __init__.py
│   │
│   ├── data
│   │   ├── __init__.py
│   │   ├── explore_dataset.py
│   │   ├── validate_images.py
│   │   ├── detect_duplicates.py
│   │   ├── create_validation_split.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   │
│   ├── models
│   │   ├── __init__.py
│   │   ├── model_factory.py
│   │   └── classifier.py
│   │
│   ├── training
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── callbacks.py
│   │
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   ├── classification_report.py
│   │   └── benchmark_inference.py
│   │
│   ├── explainability
│   │   ├── __init__.py
│   │   ├── gradcam.py
│   │   ├── gradcam_utils.py
│   │   └── generate_gradcam_samples.py
│   │
│   └── utils
│       ├── __init__.py
│       ├── seed.py
│       ├── logger.py
│       ├── file_utils.py
│       └── config.py
│
├── app
│   ├── streamlit_app.py
│   └── assets
│
├── outputs
│   ├── dataset
│   │   ├── class_distribution.csv
│   │   ├── invalid_images.csv
│   │   ├── duplicate_images.csv
│   │   ├── class_mapping.json
│   │   └── split_metadata.csv
│   │
│   ├── figures
│   │   ├── dataset
│   │   ├── training
│   │   ├── evaluation
│   │   └── gradcam
│   │
│   ├── logs
│   ├── checkpoints
│   │   ├── resnet18
│   │   ├── mobilenet_v3
│   │   ├── densenet121
│   │   └── efficientnet_b0
│   │
│   └── reports
│       ├── model_comparison.csv
│       ├── per_class_metrics.csv
│       ├── inference_benchmark.csv
│       └── final_summary.md
│
├── scripts
│   ├── run_dataset_analysis.ps1
│   ├── run_train_all.ps1
│   ├── run_evaluate_all.ps1
│   └── run_gradcam_all.ps1
│
├── requirements.txt
├── README.md
└── PROJECT_SPEC.md
```

Không bắt buộc phải tạo ngay toàn bộ file, nhưng phải giữ đúng tư duy tách module.

---

# 5. Các giai đoạn triển khai

---

## Giai đoạn 1. Khảo sát dữ liệu

### Mục tiêu

Hiểu chính xác dataset trước khi train.

### Việc cần làm

1. Đọc thư mục:

```text
D:\NCKH_V2\Data\train
D:\NCKH_V2\Data\test
```

2. Liệt kê đầy đủ tên lớp.
3. Đếm số ảnh theo từng lớp trong `train`.
4. Đếm số ảnh theo từng lớp trong `test`.
5. Kiểm tra số lượng lớp giữa `train` và `test`.
6. Kiểm tra ảnh lỗi.
7. Kiểm tra ảnh có phần mở rộng không hỗ trợ.
8. Kiểm tra ảnh trùng.
9. Kiểm tra ảnh gần trùng nếu có thể.
10. Quan sát ảnh mẫu theo từng lớp.
11. Đánh giá mất cân bằng dữ liệu.

### File đầu ra bắt buộc

```text
outputs/dataset/class_distribution.csv
outputs/dataset/invalid_images.csv
outputs/dataset/duplicate_images.csv
outputs/dataset/class_mapping.json
outputs/figures/dataset/class_distribution_train.png
outputs/figures/dataset/class_distribution_test.png
outputs/figures/dataset/sample_images_grid.png
```

### Nội dung `class_distribution.csv`

```text
class_name,train_count,test_count,total_count
```

### Nội dung `class_mapping.json`

Ví dụ:

```json
{
  "Actinic keratosis": 0,
  "Basal Cell Carcinoma": 1,
  "Darier_s Disease": 2
}
```

---

## Giai đoạn 2. Tạo validation set

### Mục tiêu

Dataset hiện có `train` và `test`. Nếu chưa có validation set, cần tách validation từ `train`.

### Yêu cầu

- Không di chuyển dữ liệu gốc.
- Tạo file metadata split thay vì bắt buộc copy ảnh.
- Dùng stratified split theo class.
- Seed cố định.
- Tỷ lệ khuyến nghị:

```text
train mới: 85% dữ liệu từ thư mục train ban đầu
validation: 15% dữ liệu từ thư mục train ban đầu
test: giữ nguyên thư mục test
```

### File đầu ra

```text
outputs/dataset/split_metadata.csv
```

### Cấu trúc file metadata

```text
image_path,class_name,class_index,split
```

Ví dụ:

```text
D:\NCKH_V2\Data\train\Melanoma\image_001.jpg,Melanoma,14,train
D:\NCKH_V2\Data\train\Melanoma\image_002.jpg,Melanoma,14,val
D:\NCKH_V2\Data\test\Melanoma\image_003.jpg,Melanoma,14,test
```

---

## Giai đoạn 3. Tiền xử lý dữ liệu

### Mục tiêu

Tạo pipeline thống nhất cho cả 4 mô hình.

### Cấu hình mặc định

```text
image_size: 224 x 224
batch_size: 16 hoặc 32
num_workers: tùy cấu hình máy
```

### Augmentation cho tập train

Sử dụng mức vừa phải:

- Resize
- Random horizontal flip
- Random rotation nhẹ
- Random crop nhẹ hoặc resized crop
- Color jitter nhẹ
- Normalize theo ImageNet

Không được augmentation quá mạnh làm thay đổi đặc điểm bệnh da liễu.

### Validation và test

Chỉ dùng:

- Resize
- Center crop nếu cần
- Normalize

Không dùng augmentation ngẫu nhiên cho validation và test.

---

# 6. Các mô hình cần triển khai

## 6.1. ResNet18

Vai trò:

- Baseline.
- Kiến trúc residual đơn giản.
- Thời gian train vừa phải.
- Dễ so sánh.

## 6.2. MobileNetV3

Vai trò:

- Mô hình nhẹ.
- Phù hợp triển khai thực tế.
- Dùng để so sánh giữa độ chính xác và tốc độ.

Ưu tiên:

```text
MobileNetV3-Large
```

Nếu tài nguyên hạn chế có thể thử thêm:

```text
MobileNetV3-Small
```

Nhưng thí nghiệm chính cần thống nhất một phiên bản.

## 6.3. DenseNet121

Vai trò:

- Khai thác kết nối dày đặc.
- Tái sử dụng đặc trưng.
- Thường phù hợp với bài toán ảnh y tế.

## 6.4. EfficientNet-B0

Vai trò:

- Cân bằng giữa độ chính xác và chi phí tính toán.
- Là mô hình chính cần so sánh với các kiến trúc còn lại.

## 6.5. Yêu cầu chung

- Dùng pretrained weights từ ImageNet.
- Thay classification head bằng output có số neuron bằng số lớp thực tế.
- Không hard-code số lớp là 31 nếu dữ liệu đọc thực tế khác.
- Số lớp phải được suy ra từ `class_mapping.json`.

---

# 7. Chiến lược huấn luyện

## 7.1. Giai đoạn baseline

- Đóng băng backbone.
- Chỉ train classifier head.
- Epoch khuyến nghị: 5 đến 10.
- Learning rate khuyến nghị:

```text
1e-3
```

## 7.2. Giai đoạn fine-tuning

- Mở một phần block cuối.
- Giảm learning rate.
- Epoch khuyến nghị: 10 đến 30.
- Learning rate khuyến nghị:

```text
1e-4 hoặc 1e-5
```

## 7.3. Optimizer

Ưu tiên:

```text
AdamW
```

## 7.4. Loss function

Mặc định:

```text
CrossEntropyLoss
```

Nếu mất cân bằng lớp đáng kể:

```text
CrossEntropyLoss(weight=class_weights)
```

Có thể thử thêm:

```text
Focal Loss
```

Nhưng phải ghi rõ đây là thí nghiệm bổ sung.

## 7.5. Scheduler

Khuyến nghị:

```text
ReduceLROnPlateau
```

hoặc:

```text
CosineAnnealingLR
```

## 7.6. Early stopping

Theo dõi:

```text
validation loss
```

hoặc:

```text
macro F1-score
```

Phải lưu checkpoint tốt nhất.

---

# 8. Metric đánh giá

## 8.1. Metric phân loại

Bắt buộc:

- Accuracy
- Precision macro
- Recall macro
- F1-score macro
- F1-score weighted
- Top-3 accuracy
- Classification report theo từng lớp
- Confusion matrix

## 8.2. Metric triển khai

Bắt buộc:

- Số lượng tham số
- Kích thước file checkpoint
- Thời gian inference trung bình trên một ảnh
- Throughput nếu có thể
- CPU hoặc GPU được sử dụng

## 8.3. Bảng so sánh cuối cùng

Tạo file:

```text
outputs/reports/model_comparison.csv
```

Cấu trúc:

```text
model_name,accuracy,macro_precision,macro_recall,macro_f1,weighted_f1,top3_accuracy,num_parameters,model_size_mb,inference_time_ms
```

Các mô hình:

```text
resnet18
mobilenet_v3_large
densenet121
efficientnet_b0
```

---

# 9. Grad-CAM

## 9.1. Vai trò

Grad-CAM dùng để trực quan hóa vùng ảnh ảnh hưởng nhiều nhất đến dự đoán của mô hình.

Grad-CAM không phải mô hình phân loại và không tự làm tăng độ chính xác.

## 9.2. Yêu cầu

Tích hợp Grad-CAM cho cả 4 mô hình:

- ResNet18
- MobileNetV3
- DenseNet121
- EfficientNet-B0

## 9.3. Loại ảnh cần phân tích

Mỗi mô hình cần tạo Grad-CAM cho:

1. Ảnh dự đoán đúng, confidence cao.
2. Ảnh dự đoán đúng, confidence thấp.
3. Ảnh dự đoán sai.
4. Một số lớp thường bị nhầm trong confusion matrix.
5. Ảnh có nền phức tạp.
6. Ảnh có dấu hiệu watermark, vật thể ngoài vùng da hoặc nhiễu nền nếu có.

## 9.4. File đầu ra

```text
outputs/figures/gradcam/<model_name>/
```

Ví dụ:

```text
outputs/figures/gradcam/resnet18/
outputs/figures/gradcam/mobilenet_v3_large/
outputs/figures/gradcam/densenet121/
outputs/figures/gradcam/efficientnet_b0/
```

Mỗi mẫu nên lưu:

- Ảnh gốc.
- Heatmap.
- Ảnh overlay.
- Tên lớp thật.
- Tên lớp dự đoán.
- Confidence.

---

# 10. Hệ thống demo

## 10.1. Công nghệ

Ưu tiên:

```text
Streamlit
```

## 10.2. Chức năng

Giao diện cần có:

1. Upload ảnh.
2. Hiển thị ảnh gốc.
3. Chọn mô hình hoặc dùng mô hình tốt nhất mặc định.
4. Hiển thị top-3 dự đoán.
5. Hiển thị xác suất từng lớp.
6. Sinh heatmap Grad-CAM.
7. Hiển thị overlay Grad-CAM.
8. Hiển thị cảnh báo y khoa.

## 10.3. Cảnh báo bắt buộc

```text
Kết quả chỉ mang tính hỗ trợ tham khảo và không thay thế chẩn đoán của bác sĩ.
```

---

# 11. README cần có

File `README.md` phải gồm:

1. Giới thiệu đề tài.
2. Mục tiêu.
3. Dataset.
4. Cấu trúc thư mục.
5. Cách cài môi trường.
6. Cách chạy khảo sát dữ liệu.
7. Cách train từng mô hình.
8. Cách train toàn bộ mô hình.
9. Cách đánh giá.
10. Cách sinh Grad-CAM.
11. Cách chạy Streamlit demo.
12. Kết quả thực nghiệm.
13. Cảnh báo y khoa.
14. Hướng phát triển.

---

# 12. requirements.txt

Tạo file `requirements.txt` tối thiểu gồm:

```text
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
Pillow
opencv-python
PyYAML
tqdm
seaborn
streamlit
grad-cam
imagehash
```

Lưu ý:

- Có thể dùng `pytorch-grad-cam` hoặc package tương đương.
- Nếu không dùng seaborn thì có thể bỏ.
- Cần khóa version sau khi môi trường chạy ổn định:

```bash
pip freeze > requirements-lock.txt
```

---

# 13. Quy ước code

## 13.1. Yêu cầu chung

- Python 3.10 trở lên.
- Dùng type hints.
- Viết docstring.
- Chia nhỏ hàm.
- Không viết toàn bộ logic trong một notebook.
- Notebook chỉ dùng cho EDA và trình bày.
- Logic chính phải nằm trong thư mục `src`.
- Mỗi script phải có:

```python
if __name__ == "__main__":
    main()
```

## 13.2. Không hard-code

Không hard-code:

- Đường dẫn dữ liệu ở nhiều file.
- Số lớp.
- Tên lớp.
- Batch size.
- Epoch.
- Learning rate.

Các giá trị này phải được đọc từ file config.

## 13.3. Logging

Log cần có:

- Tên mô hình.
- Epoch.
- Train loss.
- Validation loss.
- Train accuracy.
- Validation accuracy.
- Macro F1.
- Learning rate.
- Thời gian epoch.
- Checkpoint path.

---

# 14. File config mẫu

Tạo:

```text
configs/base.yaml
```

Nội dung gợi ý:

```yaml
project:
  name: skin_disease_classification
  seed: 42

paths:
  data_root: "D:/NCKH_V2/Data"
  output_root: "D:/NCKH_V2/outputs"

data:
  image_size: 224
  batch_size: 16
  num_workers: 2
  validation_ratio: 0.15
  use_class_weights: true

training:
  pretrained: true
  baseline_epochs: 8
  finetune_epochs: 20
  classifier_lr: 0.001
  finetune_lr: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 5
  scheduler: reduce_on_plateau

evaluation:
  top_k: 3
  benchmark_runs: 100

models:
  - resnet18
  - mobilenet_v3_large
  - densenet121
  - efficientnet_b0
```

Dùng dấu `/` trong đường dẫn Windows để hạn chế lỗi escape.

---

# 15. Thứ tự thực hiện bắt buộc

AI coding assistant phải làm theo đúng thứ tự sau:

## Bước 1

Kiểm tra cấu trúc thư mục dữ liệu thật tại:

```text
D:\NCKH_V2\Data
```

Không giả định dataset đúng hoàn toàn.

## Bước 2

Tạo script:

```text
src/data/explore_dataset.py
```

Script phải:

- Đếm lớp.
- Đếm ảnh từng lớp.
- So sánh train/test.
- Xuất CSV.
- Xuất biểu đồ.

## Bước 3

Tạo script:

```text
src/data/validate_images.py
```

Script phải:

- Kiểm tra ảnh lỗi.
- Kiểm tra ảnh không đọc được.
- Xuất danh sách lỗi.

## Bước 4

Tạo script:

```text
src/data/detect_duplicates.py
```

Script phải:

- Kiểm tra duplicate bằng hash.
- Có thể bổ sung perceptual hash cho near-duplicate.
- Xuất CSV.

## Bước 5

Tạo script:

```text
src/data/create_validation_split.py
```

Script phải:

- Tạo stratified validation split.
- Không thay đổi dữ liệu gốc.
- Xuất `split_metadata.csv`.

## Bước 6

Tạo Dataset class và DataLoader.

## Bước 7

Tạo model factory cho 4 mô hình.

## Bước 8

Tạo training pipeline dùng chung.

## Bước 9

Train baseline.

## Bước 10

Fine-tune.

## Bước 11

Đánh giá test set.

## Bước 12

Sinh bảng so sánh mô hình.

## Bước 13

Tích hợp Grad-CAM.

## Bước 14

Xây Streamlit demo.

---

# 16. Yêu cầu đầu ra của AI coding assistant

Khi bắt đầu thực hiện, AI cần:

1. Không tạo code quá nhiều cùng lúc.
2. Làm từng bước.
3. Sau mỗi bước, giải thích file nào được tạo.
4. Cung cấp lệnh chạy cụ thể trên Windows PowerShell.
5. Nêu kết quả mong đợi.
6. Dừng lại để người dùng chạy thử và gửi lỗi nếu có.
7. Không giả định kết quả khi chưa chạy thật.
8. Không tự bịa số lượng ảnh hoặc metric.
9. Chỉ kết luận sau khi có output thực tế.
10. Ưu tiên code rõ ràng, dễ bảo trì và dễ dùng trong báo cáo nghiên cứu.

---

# 17. Lệnh PowerShell khởi tạo môi trường

Chạy trong thư mục:

```text
D:\NCKH_V2
```

Tạo virtual environment:

```powershell
python -m venv .venv
```

Kích hoạt:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nâng cấp pip:

```powershell
python -m pip install --upgrade pip
```

Cài thư viện:

```powershell
pip install -r requirements.txt
```

---

# 18. Lệnh chạy dự kiến

Khảo sát dữ liệu:

```powershell
python -m src.data.explore_dataset
```

Kiểm tra ảnh lỗi:

```powershell
python -m src.data.validate_images
```

Kiểm tra ảnh trùng:

```powershell
python -m src.data.detect_duplicates
```

Tạo validation split:

```powershell
python -m src.data.create_validation_split
```

Train một mô hình:

```powershell
python -m src.training.train --model resnet18
```

Train toàn bộ:

```powershell
python -m src.training.train --all-models
```

Đánh giá toàn bộ:

```powershell
python -m src.evaluation.evaluate --all-models
```

Sinh Grad-CAM:

```powershell
python -m src.explainability.generate_gradcam_samples --all-models
```

Chạy demo:

```powershell
streamlit run app/streamlit_app.py
```

---

# 19. Nội dung báo cáo nghiên cứu cần bám theo project

## Chương 1. Tổng quan đề tài

- Lý do chọn đề tài.
- Mục tiêu.
- Phạm vi.
- Phương pháp.
- Đóng góp.

## Chương 2. Cơ sở lý thuyết

- Tổng quan phân loại ảnh da liễu.
- CNN.
- Transfer learning.
- ResNet18.
- MobileNetV3.
- DenseNet121.
- EfficientNet-B0.
- Grad-CAM.
- Metric.

## Chương 3. Dữ liệu và phương pháp

- Dataset.
- Thống kê dữ liệu.
- Làm sạch.
- Chia dữ liệu.
- Augmentation.
- Kiến trúc pipeline.
- Cấu hình train.

## Chương 4. Thực nghiệm và đánh giá

- Môi trường.
- Kết quả baseline.
- Kết quả fine-tuning.
- Bảng so sánh.
- Confusion matrix.
- Phân tích lớp dễ nhầm.
- Benchmark inference.

## Chương 5. Grad-CAM và hệ thống demo

- Vai trò Grad-CAM.
- Phân tích ảnh đúng.
- Phân tích ảnh sai.
- Giao diện demo.
- Kết quả top-3.
- Cảnh báo.

## Chương 6. Kết luận và hướng phát triển

- Tổng kết.
- Hạn chế.
- Hướng mở rộng.

---

# 20. Hướng phát triển

Các nội dung sau để ở phần hướng phát triển:

- Thu thập dữ liệu thực tế từ bệnh viện.
- Kiểm định với bác sĩ da liễu.
- Bổ sung thông tin lâm sàng.
- Phân tích fairness theo màu da.
- Dùng segmentation để khoanh vùng tổn thương.
- Triển khai mobile app.
- Tối ưu MobileNetV3 cho thiết bị tài nguyên thấp.
- Mở rộng số lớp bệnh.
- Xây dựng module chuyên biệt hỗ trợ sàng lọc NF1.
- So sánh thêm Vision Transformer hoặc ConvNeXt.

---

# 21. Tiêu chí hoàn thành tối thiểu

Dự án được xem là hoàn thành phiên bản nghiên cứu ban đầu khi có đủ:

- [ ] Đọc được dataset từ `D:\NCKH_V2\Data`.
- [ ] Thống kê chính xác số lớp và số ảnh.
- [ ] Kiểm tra ảnh lỗi.
- [ ] Kiểm tra duplicate.
- [ ] Tạo train/validation/test split hợp lệ.
- [ ] Train được ResNet18.
- [ ] Train được MobileNetV3.
- [ ] Train được DenseNet121.
- [ ] Train được EfficientNet-B0.
- [ ] Đánh giá cùng bộ metric.
- [ ] Có confusion matrix.
- [ ] Có bảng so sánh 4 mô hình.
- [ ] Có Grad-CAM cho cả 4 mô hình.
- [ ] Có Streamlit demo.
- [ ] Có README.
- [ ] Có file cấu hình.
- [ ] Có log và checkpoint.
- [ ] Có cảnh báo y khoa.

---

# 22. Prompt khởi động dành cho AI coding assistant

Sau khi đặt file này vào thư mục gốc project, sử dụng prompt sau:

```text
Hãy đọc kỹ file PROJECT_SPEC.md trong thư mục gốc project D:\NCKH_V2.

Dataset đã có sẵn tại D:\NCKH_V2\Data. Không tải lại dữ liệu và không thay đổi dữ liệu gốc.

Bắt đầu từ Giai đoạn 1. Trước tiên hãy kiểm tra cấu trúc project và tạo các thư mục cần thiết. Sau đó chỉ tạo script src/data/explore_dataset.py cùng các file tối thiểu cần để chạy script này.

Script cần:
- đọc dữ liệu thật từ D:\NCKH_V2\Data;
- liệt kê các lớp;
- thống kê số lượng ảnh train và test theo từng lớp;
- kiểm tra lớp nào chỉ có ở train hoặc test;
- xuất outputs/dataset/class_distribution.csv;
- xuất outputs/dataset/class_mapping.json;
- xuất biểu đồ phân bố train và test;
- in tóm tắt kết quả ra terminal.

Hãy cung cấp lệnh PowerShell để chạy.
Không tiếp tục sang bước tiếp theo cho đến khi tôi gửi output thực tế.
Không tự bịa số lượng ảnh hoặc giả định dataset đã đúng.
```

---

# 23. Ghi chú cuối cùng

Ưu tiên cao nhất trong giai đoạn đầu là:

1. Hiểu đúng dữ liệu.
2. Không làm rò rỉ dữ liệu.
3. So sánh mô hình công bằng.
4. Lưu kết quả đầy đủ.
5. Không kết luận dựa trên accuracy duy nhất.
6. Dùng Grad-CAM để phân tích khả năng giải thích.
7. Giữ phạm vi phù hợp với đề tài nghiên cứu khoa học sinh viên.
