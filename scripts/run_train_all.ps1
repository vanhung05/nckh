# Huan luyen toan bo 4 mo hinh.
# Su dung: .\scripts\run_train_all.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== Huan luyen toan bo mo hinh ===" -ForegroundColor Cyan
python -m src.training.train --all-models

Write-Host "Hoan tat huan luyen." -ForegroundColor Green
