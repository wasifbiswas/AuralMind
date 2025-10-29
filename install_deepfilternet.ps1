# DeepFilterNet 2 Installation and Setup Script

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DeepFilterNet 2 Migration Script" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Step 1: Uninstall Koala
Write-Host "Step 1: Removing Koala (if installed)..." -ForegroundColor Yellow
try {
    pip uninstall -y pvkoala 2>$null
    Write-Host "✅ Koala removed successfully" -ForegroundColor Green
}
catch {
    Write-Host "⚠️  Koala was not installed" -ForegroundColor Yellow
}

# Step 2: Install DeepFilterNet 2
Write-Host "`nStep 2: Installing DeepFilterNet 2..." -ForegroundColor Yellow
pip install deepfilternet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ DeepFilterNet 2 installed successfully" -ForegroundColor Green
}
else {
    Write-Host "❌ Failed to install DeepFilterNet 2" -ForegroundColor Red
    exit 1
}

# Step 3: Verify Installation
Write-Host "`nStep 3: Verifying installation..." -ForegroundColor Yellow
python -c "from df.enhance import init_df; print('✅ DeepFilterNet 2 imports working')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ DeepFilterNet 2 import failed" -ForegroundColor Red
    exit 1
}

# Step 4: Run GPU Verification
Write-Host "`nStep 4: Running GPU verification..." -ForegroundColor Yellow
python verify_gpu.py

# Step 5: Instructions
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ Installation Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. cd src" -ForegroundColor White
Write-Host "2. python audio_preprocessing.py" -ForegroundColor White
Write-Host "`nNo access keys needed - DeepFilterNet 2 is open source!`n" -ForegroundColor Green
