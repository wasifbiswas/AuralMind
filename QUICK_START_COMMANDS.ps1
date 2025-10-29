# Quick Start Scripts for CNN + BiLSTM Training
# Copy and paste these commands into PowerShell

# ============================================================================
# 1. INSTALL DEPENDENCIES
# ============================================================================

# Install training requirements
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" pip install scikit-learn matplotlib seaborn pandas

# ============================================================================
# 2. TEST INSTALLATION
# ============================================================================

# Run system tests
cd "d:\MCA Minor1\src"
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python test_training_pipeline.py

# ============================================================================
# 3. TRAIN MODEL (BASIC)
# ============================================================================

# Train with default settings (standard model, 50 epochs)
cd "d:\MCA Minor1\src"
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python main_train.py

# ============================================================================
# 4. TRAIN MODEL (CUSTOM SETTINGS)
# ============================================================================

# Example 1: Train lightweight model for quick testing
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python main_train.py --model_type light --num_epochs 20 --batch_size 64

# Example 2: Train deep model for best performance
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python main_train.py --model_type deep --num_epochs 100 --learning_rate 0.0005 --patience 25

# Example 3: Train with custom directories and classes
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python main_train.py --train_dir audio_data/features/IEMOCAP/train --val_dir audio_data/features/IEMOCAP/val --num_classes 4 --model_type standard

# ============================================================================
# 5. RESUME TRAINING
# ============================================================================

# Resume from interrupted or saved checkpoint
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python main_train.py --resume models/checkpoints/checkpoint_epoch10.pt

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================

# Evaluate on test set
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python evaluate_model.py --checkpoint models/checkpoints/best_model.pt --test_dir audio_data/features/test --model_type standard --num_classes 7

# Evaluate with class names
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python evaluate_model.py --checkpoint models/checkpoints/best_model.pt --test_dir audio_data/features/test --model_type standard --num_classes 7 --class_names anger sad happy neutral fear disgust surprise

# ============================================================================
# 7. QUICK MODEL TESTS
# ============================================================================

# Test model architecture
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" python model_cnn_bilstm.py

# Test data loader
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" python data_loader.py

# Test trainer
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" python train_model.py

# ============================================================================
# 8. MONITORING & DEBUGGING
# ============================================================================

# Check GPU status
nvidia-smi

# Check CUDA availability
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Count feature files
Get-ChildItem -Path "d:\MCA Minor1\src\audio_data\features" -Recurse -Filter "*.npy" | Measure-Object | Select-Object -ExpandProperty Count

# Check model size
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" python -c "from model_cnn_bilstm import create_model; print(f'Light: {create_model(\"light\").get_num_params():,}'); print(f'Standard: {create_model(\"standard\").get_num_params():,}'); print(f'Deep: {create_model(\"deep\").get_num_params():,}')"

# ============================================================================
# 9. COMMON TRAINING SCENARIOS
# ============================================================================

# Scenario: Small dataset, want fast results
# Use: light model, small batch, fewer epochs
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python main_train.py --model_type light --batch_size 16 --num_epochs 30 --learning_rate 0.001

# Scenario: Large dataset, want best accuracy
# Use: deep model, large batch, many epochs
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python main_train.py --model_type deep --batch_size 64 --num_epochs 100 --learning_rate 0.0005 --patience 30

# Scenario: Balanced approach (recommended starting point)
# Use: standard model with default settings
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python main_train.py --model_type standard --batch_size 32 --num_epochs 50 --learning_rate 0.001

# Scenario: Low GPU memory (OOM errors)
# Use: smaller batch, lighter model, no mixed precision
C:\Users\nemesis\anaconda3\Scripts\conda.exe run -p "d:\MCA Minor1\.conda" --no-capture-output python main_train.py --model_type light --batch_size 8 --mixed_precision False

# ============================================================================
# 10. TIPS
# ============================================================================

# Tip 1: Monitor GPU usage during training
# Open another terminal and run: nvidia-smi -l 1

# Tip 2: Training logs are saved to logs/training_history.json
# View: Get-Content logs/training_history.json | ConvertFrom-Json

# Tip 3: Best model is always saved as models/checkpoints/best_model.pt

# Tip 4: If training is slow, try:
# - Increase batch_size (more GPU utilization)
# - Enable mixed_precision (if not already)
# - Use lighter model variant

# Tip 5: If accuracy is low, try:
# - Increase num_epochs
# - Decrease learning_rate (0.0001)
# - Use deeper model variant
# - Check class balance in dataset
