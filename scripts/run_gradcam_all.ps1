# Sinh mau Grad-CAM cho toan bo 4 mo hinh.
# Su dung: .\scripts\run_gradcam_all.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== Sinh mau Grad-CAM cho toan bo mo hinh ===" -ForegroundColor Cyan
python -m src.explainability.generate_gradcam_samples --all-models

Write-Host "Hoan tat. Xem outputs/figures/gradcam/" -ForegroundColor Green
