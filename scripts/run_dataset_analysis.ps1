# Chạy toàn bộ giai đoạn khảo sát và chuẩn bị dữ liệu.
# Sử dụng: .\scripts\run_dataset_analysis.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== [1/4] Khao sat dataset ===" -ForegroundColor Cyan
python -m src.data.explore_dataset

Write-Host "=== [2/4] Kiem tra anh loi ===" -ForegroundColor Cyan
python -m src.data.validate_images

Write-Host "=== [3/4] Phat hien anh trung ===" -ForegroundColor Cyan
python -m src.data.detect_duplicates

Write-Host "=== [4/4] Tao validation split ===" -ForegroundColor Cyan
python -m src.data.create_validation_split

Write-Host "Hoan tat phan tich du lieu." -ForegroundColor Green
