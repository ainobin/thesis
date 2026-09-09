# Full Results & Analysis — Brick NDT Thesis Pipeline

**Project:** A Deep Learning-Based Non-Destructive Testing (NDT) Approach for
Brick Quality Assessment via Mel-Spectrogram Analysis

**Date:** 09 September 2026

**Dataset:** 591 audio files (276 Grade A, 196 Grade B, 119 Grade C) + waveform augmentation

---

## 1. Data Pipeline Results

### 1.1 Preprocessing Configuration

| Parameter | Value |
|---|---|
| Target sample rate | 22050 Hz |
| Fixed window | 0.8 seconds (17640 samples) |
| Mel bands | 128 |
| FFT window | 2048 |
| Hop length | 512 |
| Time steps per sample | 35 |
| Output shape per sample | (128, 35, 1) |
| Output dtype | float32 |
| Dynamic range | [0.0, 1.0] |

### 1.2 Augmentation (Training Files Only)

Four waveform-level augmentations applied **only to training files** after file-level split:

| Augmentation | Parameters | Implementation |
|---|---|---|
| Pitch shift +1 semitone | `n_steps=1` | `librosa.effects.pitch_shift` |
| Pitch shift -1 semitone | `n_steps=-1` | `librosa.effects.pitch_shift` |
| Time stretch (1.1x) | `rate=1.1` | `librosa.effects.time_stretch` |
| Additive Gaussian noise | `noise_factor=0.005` | `np.random.randn` |

Each training file produces 1 original + 4 augmented versions = 5× expansion.
Validation and test files remain as original recordings only.

### 1.3 Data Leakage Prevention

**Critical fix applied:** The pipeline now splits at the **file level** before augmentation.
This prevents augmented versions of the same recording from appearing in different splits.

```
Before (leaky):     All files → augment all → split → train/val/test
After (correct):    All files → split files → augment train only → train/val/test
```

### 1.4 Batch Processing Output

```
Scanning audio files ...
  Found 591 files.

Splitting files into train/val/test ...
  Train files: 413
  Val files:   89
  Test files:  89

Processing training files (with augmentation) ...
  ✓ X_train → (2065, 128, 35, 1)  (files × 5 augmented)

Processing validation files ...
  ✓ X_val → (89, 128, 35, 1)

Processing test files ...
  ✓ X_test → (89, 128, 35, 1)

Class distribution:
  grade_a (0):  train=965  val=41  test=42
  grade_b (1):  train=685  val=30  test=29
  grade_c (2):  train=415  val=18  test=18

Total samples: 2243
  Train: 2065  (augmented 5×)
  Val:   89  (original only)
  Test:  89  (original only)

No data leakage: file-level split applied before augmentation.
```

### 1.5 Data Verification

```
X_train: (2065, 128, 35, 1)  float32  range: [0.0000, 1.0000]
y_train: (2065,)  int32  bins: [965 685 415]
X_val:   (89, 128, 35, 1)  float32  range: [0.0000, 1.0000]
y_val:   (89,)  int32  bins: [41 30 18]
X_test:  (89, 128, 35, 1)  float32  range: [0.0000, 1.0000]
y_test:  (89,)  int32  bins: [42 29 18]
```

**Analysis:** File-level stratified split ensures no data leakage. Training set is
augmented 5× (413 files → 2,065 samples). Validation and test sets contain only
original recordings from held-out files.

---

## 2. Model Architecture

### 2.1 Architecture Details

```
Input:          (None, 128, 35, 1)
├─ Conv2D(32, 3×3, ReLU) + BatchNorm + MaxPool(2×2)   → (64, 17, 32)
├─ Conv2D(64, 3×3, ReLU) + BatchNorm + MaxPool(2×2)   → (32, 8, 64)
├─ Conv2D(128, 3×3, ReLU) + BatchNorm                  → (32, 8, 128)
├─ GlobalAveragePooling2D                               → (128)
├─ Dense(64, ReLU, L2=1e-3) + Dropout(0.3)            → (64)
└─ Dense(3, Softmax)                                    → (3)
```

### 2.2 Parameter Count

| Category | Count | Size |
|---|---|---|
| Total params | ~102,000 | ~399 KB |
| Trainable params | ~101,571 | ~397 KB |
| Non-trainable params (BatchNorm) | 448 | 1.75 KB |

---

## 3. Hyperparameter Search Results

### 3.1 Grid Search Configuration

| Parameter | Values tested |
|---|---|
| Learning rate | 0.001, 0.0001 |
| Dropout rate | 0.3, 0.5 |
| L2 regularization | 0.0001, 0.001 |
| Batch size | 16, 32 |
| **Total combinations** | **16** |

### 3.2 Best Configuration (on corrected data)

```python
{
    'learning_rate': 0.001,
    'dropout': 0.3,
    'l2_reg': 0.001,
    'batch_size': 16,
    'val_accuracy': 0.8989
}
```

**Analysis:** With file-level split (no leakage), the best validation accuracy is
89.89% — lower than the leaked 91.89% but honest and generalizable.

---

## 4. Test-Set Evaluation Results (Final Model)

### 4.1 CNN Summary Metrics

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| **Test accuracy** | **0.8539** | (0.7753, 0.9213) |
| **Precision** | **0.8199** | (0.7253, 0.9063) |
| **Recall** | **0.8014** | (0.7109, 0.8879) |
| **F1-score** | **0.8084** | (0.7124, 0.8909) |

### 4.2 Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Grade A (0) | 0.9333 | 1.0000 | **0.9655** | 42 |
| Grade B (1) | 0.7931 | 0.7931 | **0.7931** | 29 |
| Grade C (2) | 0.7333 | 0.6111 | **0.6667** | 18 |
| **Macro avg** | **0.8199** | **0.8014** | **0.8084** | **89** |

### 4.3 Confusion Matrix

```
              Predicted
              Grade A  Grade B  Grade C
Actual Grade A    42       0        0
Actual Grade B     2      23        4
Actual Grade C     1       6       11
```

**Analysis:**
- **Grade A** is perfectly classified (42/42 correct, 100% recall)
- **Grade B** has 6 samples confused (2→A, 4→C) — 79.31% recall
- **Grade C** has 7 samples confused (1→A, 6→B) — 61.11% recall (weakest)
- Total misclassifications: 13 out of 89 (14.61%)

### 4.4 Bootstrap Confidence Intervals

The 95% CIs are wider than the leaked version (accuracy: 77.53%–92.13%) due to
the smaller test set (89 vs 444 samples), but still provide a reliable estimate.

---

## 5. Baseline Model Comparison

### 5.1 Baseline Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **SVM (RBF)** | **0.8764** | **0.8473** | **0.8525** | **0.8425** |
| **CNN** | 0.8539 | 0.8199 | 0.8014 | 0.8084 |
| Random Forest | 0.8090 | 0.7695 | 0.7660 | 0.7629 |

### 5.2 Baseline Per-Class Breakdown

**SVM (RBF):**
```
              precision    recall  f1-score   support
   Grade A       0.98      1.00      0.99        42
   Grade B       0.91      0.72      0.81        29
   Grade C       0.65      0.83      0.73        18
```

**Random Forest:**
```
              precision    recall  f1-score   support
   Grade A       0.91      0.98      0.94        42
   Grade B       0.83      0.66      0.73        29
   Grade C       0.57      0.67      0.62        18
```

### 5.3 Analysis

With honest evaluation (no data leakage), **SVM outperforms the CNN**:

| Comparison | Accuracy Δ | F1 Δ |
|---|---|---|
| SVM vs CNN | **+2.25%** | **+3.41%** |
| CNN vs RF | **+4.49%** | **+4.55%** |

This is expected for this dataset size: with only 89 test samples and 2,065
training samples, the CNN's 102K parameters can overfit despite regularization.
The SVM's simpler decision boundary generalizes better.

**Grade C note:** SVM achieves 83% recall on Grade C (vs. 61.11% for CNN),
confirming that simpler models are more robust for the minority class.

---

## 6. Data Leakage Impact Analysis

### 6.1 Before vs After Fix

| Metric | Before (Leaked) | After (Honest) | Δ |
|---|---|---|---|
| Test samples | 444 (augmented) | 89 (original only) | -355 |
| CNN accuracy | 94.82% | 85.39% | **-9.43%** |
| CNN F1 | 93.38% | 80.84% | **-12.54%** |
| Best model | CNN | **SVM** | — |

### 6.2 Why the Drop?

The leaked pipeline placed augmented versions of the same file in both training
and test sets. The model could match acoustic signatures of known recordings
rather than learning generalizable features. With file-level split, the model
must generalize to truly unseen recordings.

### 6.3 Key Insight

The previous "CNN outperforms baselines" conclusion was an artifact of data
leakage. With proper evaluation, **SVM achieves the best generalization** on
this dataset size. The CNN would benefit from more training data (1,000+ files
per class) to realize its capacity advantage.

---

## 7. Figure Summary

| Figure | Description |
|---|---|
| `figures/spectrogram_grid.png` | 4×6 grid of random Mel-spectrograms with class labels |
| `figures/training_curves_final.png` | Train/val loss & accuracy over epochs |
| `figures/class_distribution.png` | Bar chart of train/val/test class counts |
| `figures/confusion_matrix.png` | Normalized 3×3 confusion matrix heatmap |
| `figures/roc_curve.png` | ROC curves with AUC per class (one-vs-rest) |
| `figures/pr_curve.png` | Precision-Recall curves with AUC per class |
| `figures/misclassifications.png` | Up to 8 misclassified samples with confidence scores |

---

## 8. Files Produced

### `data/processed/`

| File | Shape | Description |
|---|---|---|
| `X_train.npy` | (2065, 128, 35, 1) | Training features (augmented) |
| `y_train.npy` | (2065,) | Training labels |
| `X_val.npy` | (89, 128, 35, 1) | Validation features (original) |
| `y_val.npy` | (89,) | Validation labels |
| `X_test.npy` | (89, 128, 35, 1) | Test features (original) |
| `y_test.npy` | (89,) | Test labels |

### `models/`

| File | Description |
|---|---|
| `best.keras` | Best model checkpoint (by val_accuracy) |
| `history.npy` | Training history (loss/accuracy per epoch) |
| `best_hparams.npy` | Best hyperparameter config from grid search |
| `results_summary.npy` | All test metrics + bootstrap 95% CIs |

### `figures/`

7 PNG files (see Section 7).

---

## 9. Pipeline Integrity

| Stage | Status | Notes |
|---|---|---|
| Audio loading | ✓ | librosa.load, mono, 22050 Hz |
| Denoising | ✓ | noisereduce, stationary=False, 0.85 |
| Silence trimming | ✓ | top_db=30 |
| Fixed-length padding | ✓ | Symmetric zero-pad to 0.8 s |
| Mel-spectrogram | ✓ | 128 bands, 2048 FFT, 512 hop |
| Normalization | ✓ | Min-max to [0,1] float32 |
| Channel dimension | ✓ | (128, 35, 1) |
| Data augmentation | ✓ | Pitch shift, time stretch, noise (train only) |
| **File-level split** | ✓ | **Split before augmentation — no leakage** |
| Stratified split | ✓ | 70/15/15, class-balanced |
| NumPy save | ✓ | 6 files |
| CNN training | ✓ | Callbacks, early stopping, checkpointing, GPU support |
| Hyperparameter search | ✓ | 16 combos |
| Baseline models | ✓ | SVM (RBF) + Random Forest |
| Bootstrap CI | ✓ | 1000 iterations, 95% CI for all metrics |
| Evaluation | ✓ | Full per-class metrics, confusion matrix, ROC |
| Visualization | ✓ | 7 publication-ready figures |
| GPU acceleration | ✓ | NVIDIA RTX 3050, memory growth enabled |

---

## 10. Conclusion

**Key findings:**

1. **SVM achieves the best generalization** (87.64% accuracy, 84.25% F1) on this
   dataset size, outperforming the CNN by +2.25% accuracy and +3.41% F1.

2. **Data leakage inflated CNN results by ~9.4%** — the previous 94.82% accuracy
   dropped to 85.39% after fixing the augmentation pipeline.

3. **Grade A is perfectly classified** (100% recall) by both CNN and SVM, while
   Grade C remains the weakest class (61.11% CNN recall, 83% SVM recall).

4. **The CNN's capacity is not yet justified** — with 2,065 training samples and
   102K parameters, the model is at the edge of overfitting. More data (1,000+
   files per class) would allow the CNN to realize its advantage over SVM.

5. **The pipeline is production-ready** with correct data handling — adding more
   audio files to `data/raw/grade_{a,b,c}/` and re-running will scale properly.

**Recommendations:**
- **For thesis reporting:** Use SVM as the primary model (best generalization)
- **For future work:** Collect 1,000+ files per class, then retrain CNN
- **For immediate improvement:** Apply ensemble of SVM + CNN for robust predictions
- **For Grade C:** Collect more Grade C recordings to balance the dataset
