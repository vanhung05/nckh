# Hướng dẫn train trên Cloud GPU (Colab / Kaggle)

Máy cá nhân chỉ chạy CPU nên train rất chậm. Tài liệu này hướng dẫn đưa repo
lên GPU miễn phí để train đủ 4 mô hình.

Code đã hỗ trợ 2 biến môi trường để khỏi sửa file config:

- `SKIN_DATA_ROOT`  — thư mục chứa `train/` và `test/`.
- `SKIN_OUTPUT_ROOT` — nơi lưu output (CSV, hình, checkpoint).

---

## Bước 0 (làm một lần): đưa code lên GitHub

Cloud notebook lấy code dễ nhất qua `git clone`. Từ thư mục dự án:

```powershell
git init
git add .
git commit -m "Skin disease CNN + Grad-CAM"
# Tạo repo trống trên GitHub rồi:
git remote add origin https://github.com/<user>/<repo>.git
git branch -M main
git push -u origin main
```

> `.gitignore` đã loại thư mục `Data/` và checkpoint nặng, nên chỉ code được
> đẩy lên. Dataset sẽ nạp riêng trên cloud (xem bên dưới).

---

## Phương án A — Kaggle (khuyên dùng vì dataset có sẵn trên Kaggle)

1. Vào https://www.kaggle.com/code → **New Notebook**.
2. **Settings → Accelerator → GPU T4 x2** (hoặc P100).
3. **Add Input → Datasets** → tìm `31 classes of skin disease` (kelixo25) → Add.
4. **File → Import Notebook** → tải `notebooks/run_on_kaggle.ipynb` lên.
5. Sửa `GITHUB_REPO_URL` và `SKIN_DATA_ROOT` cho khớp (cell 3 in ra đường dẫn
   thật trong `/kaggle/input/...`).
6. Chạy lần lượt các cell. Kết quả nén thành `outputs.zip` để tải về.

Ưu điểm: không cần upload ảnh, dataset gắn trực tiếp. Hạn mức ~30 giờ GPU/tuần.

---

## Phương án B — Google Colab

1. Mở https://colab.research.google.com → **File → Upload notebook** →
   chọn `notebooks/run_on_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Nạp dataset bằng 1 trong 2 cách (ghi trong notebook):
   - **Cách A:** upload `kaggle.json` (Kaggle API token) để tải dataset.
   - **Cách B:** nén `Data` thành `Data.zip`, đưa lên Google Drive, giải nén.
4. Chạy lần lượt các cell. Cell cuối sao kết quả ra Google Drive để không mất
   khi hết phiên.

> Lưu ý: phiên Colab miễn phí có thể bị ngắt sau vài giờ không tương tác. Nếu
> train cả 4 mô hình lâu, nên train từng mô hình
> (`--model resnet18`, ...) và lưu checkpoint ra Drive sau mỗi mô hình.

---

## Sau khi có kết quả

Tải `outputs/` về máy, đặt vào thư mục dự án (ghi đè `outputs/` local). Khi đó:

```powershell
python -m src.evaluation.generate_summary
python -m streamlit run app/streamlit_app.py
```

Demo sẽ tự nạp checkpoint tốt nhất theo `model_comparison.csv`.

---

## Mẹo tăng tốc / tránh hết bộ nhớ GPU

- T4/P100 (16GB) chạy tốt `batch_size: 32`. Có thể nâng lên 64 để nhanh hơn:
  sửa `configs/base.yaml` hoặc tạo config riêng.
- Nếu gặp `CUDA out of memory`, giảm `batch_size` xuống 16.
- `num_workers` trên Kaggle/Colab nên để 2–4.
