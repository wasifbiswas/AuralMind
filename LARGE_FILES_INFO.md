# Large Files Reference - Not Stored in Git

This document lists the large files that are excluded from the Git repository due to GitHub's file size limits.

## 📦 Model Weight Files (Excluded from Git)

These trained model files should be stored in **GitHub Releases** or external storage:

### IEMOCAP Models
- `models_IEMOCAP/encoder_final.pt` (~15 MB)
- `models_IEMOCAP/encoder_iter_*.pt` (~15 MB each)
- `models_IEMOCAP/classifier_best.pt` (~0.5 MB)
- `models_IEMOCAP/classifier_iter_*.pt` (~0.5 MB each)

### CommonDB Models
- `models_CommonDB/encoder_final.pt` (~15 MB)
- `models_CommonDB/encoder_iter_*.pt` (~15 MB each)
- `models_CommonDB/classifier_best.pt` (~0.5 MB)
- `models_CommonDB/classifier_iter_*.pt` (~0.5 MB each)

**Total Size**: ~100 MB

---

## 🗂️ Embedding Files (Excluded from Git)

Large numpy arrays containing embeddings and pseudo-labels:

### IEMOCAP Embeddings
- `embeddings_IEMOCAP/embeddings_iter_*.npy` (~40 MB per file)
- `embeddings_IEMOCAP/pseudo_labels_iter_*.npy` (~0.1 MB per file)

### CommonDB Embeddings
- `embeddings_CommonDB/embeddings_iter_*.npy` (~200 MB per file)
- `embeddings_CommonDB/pseudo_labels_iter_*.npy` (~0.5 MB per file)

**Total Size**: ~500 MB

---

## 🎵 Audio Feature Files (Excluded from Git)

Preprocessed wav2vec2 features:

### IEMOCAP Features
- `src/audio_data/features/IEMOCAP/*.npy` (9,642 files)
- **Total Size**: ~4 GB

### CommonDB Features
- `src/audio_data/features/CommonDB/*.npy` (51,070 files)
- **Total Size**: ~20 GB

---

## 📊 What IS Included in Git

✅ **Included (Small files):**
- All Python source code
- PowerShell helper scripts
- Documentation (README, reports)
- Evaluation results (JSON, TXT)
- Visualization images (PNG)
- Confusion matrices
- Training analysis plots
- Comparative analysis

---

## 🚀 How to Download Model Weights

### Option 1: GitHub Releases (Recommended)
After pushing to GitHub, upload the model files as release assets:

```bash
# Tag and create a release
git tag -a v1.0 -m "Release v1.0: Trained models for IEMOCAP and CommonDB"
git push origin v1.0

# Then upload .pt files to the release via GitHub web interface
```

### Option 2: External Storage
Consider using:
- **Hugging Face Model Hub** - Free hosting for ML models
- **Google Drive / OneDrive** - For team collaboration
- **Git LFS** - For larger repositories (requires setup)

---

## 📥 Reproducing Results

To reproduce the training results:

1. Clone the repository
2. Download the feature files (or extract them from raw audio)
3. Run training: `python self_supervised_training.py --dataset IEMOCAP`
4. Models will be saved automatically in `models_IEMOCAP/`

---

## 📝 Model Architecture

Both IEMOCAP and CommonDB models use identical architecture:

- **Encoder**: CNN + BiLSTM (3.7M parameters)
- **Classifier**: MLP 2-layer (200K parameters)
- **Input**: wav2vec2-large-xlsr-53 features (1024-dim)
- **Output**: 512-dim embeddings → 5 classes

---

## 🔗 Links

- GitHub Repository: https://github.com/wasifbiswas/AuralMind
- Model Release: [Add release link here]
- Feature Extraction: See `src/feature_extraction.py`

---

**Note**: All evaluation results, metrics, and visualizations ARE included in the Git repository, so you can view the model performance without downloading the large weight files.
