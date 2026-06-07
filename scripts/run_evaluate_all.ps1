# Danh gia toan bo 4 mo hinh tren test set.
# Su dung: .\scripts\run_evaluate_all.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== Danh gia toan bo mo hinh ===" -ForegroundColor Cyan
python -m src.evaluation.evaluate --all-models

Write-Host "Hoan tat danh gia. Xem outputs/reports/model_comparison.csv" -ForegroundColor Green
