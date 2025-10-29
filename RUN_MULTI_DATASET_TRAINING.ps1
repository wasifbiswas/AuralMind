# ============================================================================
# Multi-Dataset Training Script
# Helper commands for training on different datasets
# ============================================================================

Write-Host "`n=== MULTI-DATASET TRAINING HELPER ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "AVAILABLE COMMANDS:" -ForegroundColor Yellow
Write-Host ""

Write-Host "1. Train on IEMOCAP (default):" -ForegroundColor Green
Write-Host "   python self_supervised_training.py --dataset IEMOCAP"
Write-Host ""

Write-Host "2. Train on CommonDB:" -ForegroundColor Green
Write-Host "   python self_supervised_training.py --dataset CommonDB"
Write-Host ""

Write-Host "3. Train with custom parameters:" -ForegroundColor Green
Write-Host "   python self_supervised_training.py --dataset CommonDB --batch-size 64 --epochs 15 --clusters 5"
Write-Host ""

Write-Host "4. Evaluate IEMOCAP model:" -ForegroundColor Green
Write-Host "   python evaluate_model.py --dataset IEMOCAP"
Write-Host ""

Write-Host "5. Evaluate CommonDB model:" -ForegroundColor Green
Write-Host "   python evaluate_model.py --dataset CommonDB"
Write-Host ""

Write-Host "=" -NoNewline; 1..79 | ForEach-Object { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host "`nOUTPUT FOLDER STRUCTURE:" -ForegroundColor Yellow
Write-Host ""
Write-Host "IEMOCAP outputs:" -ForegroundColor Cyan
Write-Host "  - Models: d:\MCA Minor1\models_IEMOCAP\"
Write-Host "  - Embeddings: d:\MCA Minor1\embeddings_IEMOCAP\"
Write-Host ""
Write-Host "CommonDB outputs:" -ForegroundColor Cyan
Write-Host "  - Models: d:\MCA Minor1\models_CommonDB\"
Write-Host "  - Embeddings: d:\MCA Minor1\embeddings_CommonDB\"
Write-Host ""

Write-Host "=" -NoNewline; 1..79 | ForEach-Object { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host "`nQUICK START - Train CommonDB now:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Enter to start training on CommonDB, or Ctrl+C to exit..." -ForegroundColor Green
$null = Read-Host

Write-Host "`nStarting CommonDB training..." -ForegroundColor Green
cd "d:\MCA Minor1\src"
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python self_supervised_training.py --dataset CommonDB
