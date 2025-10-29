# Quick Start Script for Self-Supervised Audio Training Pipeline
# =============================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "🎵 Self-Supervised Audio Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if conda environment exists
$condaEnvPath = "d:\MCA Minor1\.conda"

if (-not (Test-Path $condaEnvPath)) {
    Write-Host "❌ Error: Conda environment not found at $condaEnvPath" -ForegroundColor Red
    Write-Host "Please create the environment first." -ForegroundColor Yellow
    exit 1
}

# Check if feature files exist
$featuresPath = "d:\MCA Minor1\src\audio_data\features\IEMOCAP"

if (-not (Test-Path $featuresPath)) {
    Write-Host "❌ Error: Features directory not found at $featuresPath" -ForegroundColor Red
    Write-Host "Please ensure your wav2vec2 features are extracted first." -ForegroundColor Yellow
    exit 1
}

# Count feature files
$featureCount = (Get-ChildItem -Path $featuresPath -Filter "*.npy" | Measure-Object).Count

Write-Host "✓ Found $featureCount feature files in IEMOCAP directory" -ForegroundColor Green

# Display menu
Write-Host "`nSelect training mode:" -ForegroundColor Yellow
Write-Host "1. Stage 1: Self-supervised pre-training (CNN+BiLSTM encoder)" -ForegroundColor White
Write-Host "2. Stage 2: Downstream MLP classifier training" -ForegroundColor White
Write-Host "3. Run both stages sequentially" -ForegroundColor White
Write-Host "4. Check GPU status" -ForegroundColor White
Write-Host "5. View training logs" -ForegroundColor White
Write-Host "6. Exit" -ForegroundColor White

$choice = Read-Host "`nEnter your choice (1-6)"

switch ($choice) {
    "1" {
        Write-Host "`n🚀 Starting Stage 1: Self-Supervised Pre-training..." -ForegroundColor Green
        Write-Host "This will train the CNN+BiLSTM encoder using temporal prediction." -ForegroundColor Yellow
        Write-Host "Expected time: 2-3 hours on RTX 4060`n" -ForegroundColor Yellow
        
        Set-Location "d:\MCA Minor1\src"
        C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p $condaEnvPath --no-capture-output python self_supervised_training.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✅ Stage 1 completed successfully!" -ForegroundColor Green
            Write-Host "Model saved to: checkpoints/self_supervised_model.pt" -ForegroundColor Cyan
            Write-Host "Embeddings saved to: embeddings/train_embeddings.npy" -ForegroundColor Cyan
        } else {
            Write-Host "`n❌ Stage 1 failed with exit code $LASTEXITCODE" -ForegroundColor Red
        }
    }
    
    "2" {
        # Check if embeddings exist
        $embeddingsPath = "d:\MCA Minor1\src\embeddings\train_embeddings.npy"
        
        if (-not (Test-Path $embeddingsPath)) {
            Write-Host "`n⚠️  Warning: Embeddings not found!" -ForegroundColor Yellow
            Write-Host "You need to run Stage 1 first to extract embeddings." -ForegroundColor Yellow
            Write-Host "Would you like to run Stage 1 now? (Y/N)" -ForegroundColor Yellow
            
            $runStage1 = Read-Host
            
            if ($runStage1 -eq "Y" -or $runStage1 -eq "y") {
                Write-Host "`n🚀 Starting Stage 1 first..." -ForegroundColor Green
                Set-Location "d:\MCA Minor1\src"
                C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p $condaEnvPath --no-capture-output python self_supervised_training.py
                
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "`n❌ Stage 1 failed. Cannot proceed to Stage 2." -ForegroundColor Red
                    exit 1
                }
            } else {
                Write-Host "`nExiting..." -ForegroundColor Yellow
                exit 0
            }
        }
        
        Write-Host "`n🚀 Starting Stage 2: MLP Classifier Training..." -ForegroundColor Green
        Write-Host "This will train the downstream MLP classifier on extracted embeddings." -ForegroundColor Yellow
        Write-Host "Note: Currently uses dummy labels. Replace with actual labels for supervised training.`n" -ForegroundColor Yellow
        
        Set-Location "d:\MCA Minor1\src"
        C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p $condaEnvPath --no-capture-output python downstream_mlp_training.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✅ Stage 2 completed successfully!" -ForegroundColor Green
            Write-Host "Classifier saved to: checkpoints/mlp_classifier.pt" -ForegroundColor Cyan
        } else {
            Write-Host "`n❌ Stage 2 failed with exit code $LASTEXITCODE" -ForegroundColor Red
        }
    }
    
    "3" {
        Write-Host "`n🚀 Starting Complete Pipeline..." -ForegroundColor Green
        Write-Host "This will run both stages sequentially." -ForegroundColor Yellow
        Write-Host "Total expected time: 3-4 hours`n" -ForegroundColor Yellow
        
        # Stage 1
        Write-Host "`n--- Stage 1: Self-Supervised Pre-training ---" -ForegroundColor Cyan
        Set-Location "d:\MCA Minor1\src"
        C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p $condaEnvPath --no-capture-output python self_supervised_training.py
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "`n❌ Stage 1 failed. Aborting pipeline." -ForegroundColor Red
            exit 1
        }
        
        Write-Host "`n✅ Stage 1 completed!" -ForegroundColor Green
        
        # Stage 2
        Write-Host "`n--- Stage 2: MLP Classifier Training ---" -ForegroundColor Cyan
        C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p $condaEnvPath --no-capture-output python downstream_mlp_training.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✅ Complete pipeline finished successfully!" -ForegroundColor Green
        } else {
            Write-Host "`n⚠️  Stage 2 failed, but Stage 1 completed successfully." -ForegroundColor Yellow
        }
    }
    
    "4" {
        Write-Host "`n🔍 Checking GPU status..." -ForegroundColor Green
        Set-Location "d:\MCA Minor1\src"
        C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p $condaEnvPath --no-capture-output python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('CUDA Version:', torch.version.cuda); print('PyTorch Version:', torch.__version__)"
    }
    
    "5" {
        Write-Host "`n📋 Available checkpoints and embeddings:" -ForegroundColor Green
        
        $checkpointsPath = "d:\MCA Minor1\src\checkpoints"
        $embeddingsPath = "d:\MCA Minor1\src\embeddings"
        
        if (Test-Path $checkpointsPath) {
            Write-Host "`nCheckpoints:" -ForegroundColor Cyan
            Get-ChildItem -Path $checkpointsPath -Filter "*.pt" | ForEach-Object {
                $size = [math]::Round($_.Length / 1MB, 2)
                Write-Host "  $($_.Name) - ${size} MB" -ForegroundColor White
            }
        } else {
            Write-Host "`nNo checkpoints found." -ForegroundColor Yellow
        }
        
        if (Test-Path $embeddingsPath) {
            Write-Host "`nEmbeddings:" -ForegroundColor Cyan
            Get-ChildItem -Path $embeddingsPath -Filter "*.npy" | ForEach-Object {
                $size = [math]::Round($_.Length / 1MB, 2)
                Write-Host "  $($_.Name) - ${size} MB" -ForegroundColor White
            }
        } else {
            Write-Host "`nNo embeddings found." -ForegroundColor Yellow
        }
    }
    
    "6" {
        Write-Host "`nExiting..." -ForegroundColor Yellow
        exit 0
    }
    
    default {
        Write-Host "`n❌ Invalid choice. Please select 1-6." -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Script completed." -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
