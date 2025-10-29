# ============================================================================
# Mental Health Classification - Complete Training Pipeline
# ============================================================================
# This script trains both CNN+BiLSTM and MLP models for comparison
# Features: GPU optimization, full metrics, plots, and predictions

Write-Host "`n=============================================================================" -ForegroundColor Cyan
Write-Host "🧠 Mental Health Classification Training Pipeline" -ForegroundColor Cyan
Write-Host "   Models: CNN+BiLSTM vs MLP" -ForegroundColor Cyan
Write-Host "=============================================================================" -ForegroundColor Cyan

# Configuration
$CONDA_PATH = "C:\Users\nemesis\anaconda3\Scripts\conda.exe"
$ENV_PATH = "d:\MCA Minor1\.conda"
$SCRIPT_PATH = "d:\MCA Minor1\src\complete_mental_health_classification.py"

# Check if conda exists
if (-not (Test-Path $CONDA_PATH)) {
    Write-Host "❌ Error: Conda not found at $CONDA_PATH" -ForegroundColor Red
    exit 1
}

# Check if script exists
if (-not (Test-Path $SCRIPT_PATH)) {
    Write-Host "❌ Error: Script not found at $SCRIPT_PATH" -ForegroundColor Red
    exit 1
}

Write-Host "`n🚀 Starting training..." -ForegroundColor Green
Write-Host "   This will train BOTH models (CNN+BiLSTM and MLP)" -ForegroundColor Yellow
Write-Host "   Expected time: 6-10 hours for 30 epochs each" -ForegroundColor Yellow
Write-Host "`n=============================================================================" -ForegroundColor Cyan

# Change to src directory
Set-Location "d:\MCA Minor1\src"

# Run with conda environment
& $CONDA_PATH run -p $ENV_PATH --no-capture-output python $SCRIPT_PATH

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=============================================================================" -ForegroundColor Green
    Write-Host "✅ Training Complete!" -ForegroundColor Green
    Write-Host "=============================================================================" -ForegroundColor Green
    Write-Host "`n📊 Results saved in:" -ForegroundColor Cyan
    Write-Host "   - results/cnn_bilstm/        (CNN+BiLSTM results)" -ForegroundColor White
    Write-Host "   - results/mlp/               (MLP results)" -ForegroundColor White
    Write-Host "   - results/training_history.png" -ForegroundColor White
    Write-Host "   - results/confusion_matrices.png" -ForegroundColor White
    Write-Host "   - results/model_comparison.csv" -ForegroundColor White
    Write-Host "`n💾 Models saved in:" -ForegroundColor Cyan
    Write-Host "   - models/cnn_bilstm/best_model.pt" -ForegroundColor White
    Write-Host "   - models/mlp/best_model.pt" -ForegroundColor White
} else {
    Write-Host "`n❌ Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}

Write-Host "`n=============================================================================" -ForegroundColor Cyan
