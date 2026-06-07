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

- (Chưa có model_comparison.csv. Hãy chạy evaluate.)
