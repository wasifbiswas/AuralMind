# Model Evaluation Report

**Generated:** 2025-10-29 22:31:49

## Model Information

- **Model Architecture:** CNN + BiLSTM Encoder
- **Task:** Audio Feature Classification (Unsupervised)
- **Classifier:** MLP (2-layer)

## Performance Metrics

### Training Set

| Metric | Score |
|--------|-------|
| Accuracy | 0.9681 (96.81%) |
| F1-Score (Macro) | 0.9730 |
| F1-Score (Weighted) | 0.9682 |
| Precision (Macro) | 0.9729 |
| Recall (Macro) | 0.9737 |

### Test Set

| Metric | Score |
|--------|-------|
| Accuracy | 0.9705 (97.05%) |
| F1-Score (Macro) | 0.9728 |
| F1-Score (Weighted) | 0.9706 |
| Precision (Macro) | 0.9703 |
| Recall (Macro) | 0.9758 |

## Summary Table

| Dataset | Accuracy | F1-Score (Weighted) |
|---------|----------|---------------------|
| Training | 0.9681 | 0.9682 |
| Test | 0.9705 | 0.9706 |

## Model Architecture Details

### CNN + BiLSTM Encoder
- **Input:** wav2vec2-large-xlsr-53 features (1024-dim)
- **CNN Layers:** [64, 128, 256] channels with BatchNorm & MaxPooling
- **LSTM:** 2-layer Bidirectional LSTM (256 hidden units)
- **Output:** 512-dimensional embeddings

### MLP Classifier
- **Input:** 512-dimensional embeddings
- **Hidden Layers:** [256, 128] with BatchNorm & Dropout(0.4)
- **Output:** 5 classes
- **Total Parameters:** ~3.7M (encoder) + ~200K (classifier)

## Training Configuration

- **Batch Size:** 64
- **Learning Rate:** 0.0001
- **Optimizer:** Adam
- **Mixed Precision:** Enabled (FP16)
- **GPU:** NVIDIA GeForce RTX 4060
- **Training Method:** Self-supervised with iterative pseudo-labeling

## Confusion Matrices

See saved images:
- `confusion_matrix_training.png`
- `confusion_matrix_test.png`

## Notes

This model was trained using a self-supervised learning approach with pseudo-labeling via K-Means clustering. The high accuracy indicates excellent cluster separation and consistent pseudo-label quality across iterations.
