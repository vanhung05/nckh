# Báo cáo tổng hợp kết quả

> Tự động sinh từ các file output thực tế. Kết quả chỉ mang tính hỗ trợ tham khảo, không thay thế chẩn đoán của bác sĩ.

## 1. Dữ liệu

- Số lớp: **31**
- Tổng ảnh train (gốc): **3915**
- Tổng ảnh test: **994**
- Mất cân bằng train: nhiều nhất 382, ít nhất 64, tỉ lệ 5.97

## 2. Làm sạch dữ liệu

- Tổng ảnh train bị loại: **127**
  - conflicting_label: 86
  - leakage_train_test: 36
  - intra_train_duplicate: 5

- Sau làm sạch: train=3221, val=567, test=994

## 3. So sánh mô hình

| model_name | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | top3_accuracy | num_parameters | model_size_mb | inference_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| densenet121 | 0.7928 | 0.8167 | 0.8113 | 0.8066 | 0.7927 | 0.9386 | 6985631 | 27.19 | 15.132 |
| efficientnet_b0 | 0.7948 | 0.8014 | 0.8134 | 0.8022 | 0.7959 | 0.9145 | 4047259 | 15.71 | 8.249 |
| resnet18 | 0.7807 | 0.7951 | 0.8017 | 0.7944 | 0.7832 | 0.9245 | 11192415 | 42.77 | 2.903 |
| mobilenet_v3_large | 0.7575 | 0.7693 | 0.7778 | 0.7694 | 0.7559 | 0.9135 | 4241743 | 16.37 | 6.58 |

- Mô hình tốt nhất theo macro-F1: **densenet121** (macro-F1 = 0.8066, accuracy = 0.7928)
