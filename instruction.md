# Training Instructions — Brick NDT Model on New Dataset

## Overview

This project classifies brick quality grades (Grade A, B, C) from audio recordings of brick strikes using a 2D CNN trained on Mel-spectrograms. The pipeline converts raw WAV files into spectrogram images, augments the dataset, and trains a convolutional neural network.

**Current Dataset:**

| Class | Files | Label |
|-------|-------|-------|
| Grade A | 276 | 0 |
| Grade B | 196 | 1 |
| Grade C | 119 | 2 |
| Total | 591 | 3 classes |

After 4 augmentations per file: **~2,955 total samples** (70/15/15 split).

---

## Prerequisites

1. Place your `.wav` audio files in the correct directories:

```
data/raw/grade_a/    ← Grade A brick recordings
data/raw/grade_b/    ← Grade B brick recordings
data/raw/grade_c/    ← Grade C brick recordings
```

2. Activate the virtual environment:

```bash
source venv/bin/activate
```

3. Verify dependencies are installed:

```bash
pip install -r requirements.txt
```

4. **GPU Setup (NVIDIA RTX 3050 or similar):**

```bash
# Install CUDA toolkit + cuDNN packages into venv
pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 \
            nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
            nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 nvidia-curand-cu12

# Set library path for CUDA (required every terminal session)
export LD_LIBRARY_PATH=$(python -c "import nvidia; import os; base=os.path.dirname(nvidia.__path__[0]); print(':'.join([os.path.join(base,'nvidia',d,'lib') for d in ['cublas','cudnn','cuda_runtime','cufft','cusolver','cusparse','curand']]))")

# Verify GPU is detected
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

**Expected output:** `GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

**Important:** The `LD_LIBRARY_PATH` must be set in every new terminal session before running any Python scripts. Alternatively, add it to your shell profile (`~/.bashrc`):

```bash
echo 'export LD_LIBRARY_PATH=/path/to/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/path/to/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/path/to/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Step-by-Step Execution Guide

### Step 1: Preprocess, Augment, and Split Data

```bash
python src/batch_process.py
```

**Duration:** ~10-20 minutes

**What it does:**
- Scans `data/raw/grade_{a,b,c}/` for `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg` files
- For each file:
  1. Loads audio at 22,050 Hz (mono)
  2. Applies spectral gating noise reduction
  3. Trims silence (30 dB threshold)
  4. Pads/crops to exactly 0.8 seconds (17,640 samples)
  5. Computes 128-band Mel-spectrogram (FFT=2048, hop=512)
  6. Converts to log scale, normalizes to [0, 1]
- Creates 4 augmented copies per file:
  - Pitch shift +1 semitone
  - Pitch shift -1 semitone
  - Time stretch at 1.1x
  - Additive Gaussian noise (factor=0.005)
- Performs stratified 70/15/15 train/val/test split (random_state=42)
- Saves 6 files to `data/processed/`:
  - `X_train.npy`, `y_train.npy` (70%)
  - `X_val.npy`, `y_val.npy` (15%)
  - `X_test.npy`, `y_test.npy` (15%)

**Output shape per sample:** (128, ~130, 1)

---

### Step 2: Find Optimal Hyperparameters (Optional but Recommended)

```bash
python src/hparam_search.py
```

**Duration:** ~30-60 minutes (16 combinations × 30 epochs each)

**What it does:**
- Tests 16 hyperparameter combinations:

| Parameter | Values Tested |
|-----------|---------------|
| Learning rate | 1e-3, 1e-4 |
| Dropout | 0.3, 0.5 |
| L2 regularization | 1e-4, 1e-3 |
| Batch size | 16, 32 |

- Trains each configuration for up to 30 epochs with EarlyStopping(patience=8)
- Saves best configuration to `models/best_hparams.npy`

**After running:** Read the output to find the best hyperparameters, then update the constants in `src/train.py`:

```python
# In src/train.py, modify these lines:
BATCH_SIZE = <best_batch_size>      # default: 32
LR = <best_learning_rate>           # default: 1e-4

# In src/model.py, modify build_cnn() call or defaults:
dropout = <best_dropout>            # default: 0.5
l2_reg = <best_l2_reg>              # default: 1e-4
```

---

### Step 3: Train the CNN

```bash
python src/train.py
```

**Duration:** ~10-30 minutes

**What it does:**
- Loads `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`
- Builds CNN architecture:
  - 3 Conv2D blocks (32 → 64 → 128 filters) with BatchNorm + MaxPool
  - GlobalAveragePooling
  - Dense(64) + Dropout + Dense(3, softmax)
  - ~110K total parameters
- Compiles with Adam optimizer (lr=1e-4) and sparse categorical crossentropy
- Computes balanced class weights for imbalanced data
- Trains with callbacks:
  - **ModelCheckpoint:** saves best model to `models/best.keras` (monitors val_accuracy)
  - **EarlyStopping:** patience=15, restores best weights
  - **ReduceLROnPlateau:** factor=0.5, patience=8, min_lr=1e-6
- Saves training history to `models/history.npy`

**Output files:**
- `models/best.keras` — best model weights
- `models/history.npy` — training loss/accuracy per epoch

---

### Step 4: Evaluate Model Performance

```bash
python src/evaluate.py
```

**Duration:** ~2 minutes

**What it does:**
- Loads `models/best.keras` and `X_test.npy`, `y_test.npy`
- Generates predictions on the held-out test set
- Computes metrics:
  - Overall accuracy
  - Macro precision, recall, F1-score
  - Per-class precision, recall, F1, support
  - Confusion matrix
- Generates and saves:
  - `figures/confusion_matrix.png`
  - `figures/roc_curve.png` (one-vs-rest ROC with AUC per class)
  - `models/results_summary.npy`

---

### Step 5: Generate Publication Figures

```bash
python src/visualize.py
```

**Duration:** ~5 minutes

**What it does:**
Generates 7 thesis figures and saves them to `figures/`:

1. `spectrogram_grid.png` — 4x6 grid of sample Mel-spectrograms
2. `training_curves_final.png` — loss and accuracy over epochs
3. `class_distribution.png` — bar chart of train/val/test class counts
4. `confusion_matrix.png` — normalized heatmap
5. `roc_curve.png` — ROC curves with AUC per class
6. `pr_curve.png` — Precision-Recall curves
7. `misclassifications.png` — up to 8 misclassified samples with confidence

Also computes:
- Bootstrap 95% confidence intervals (1000 iterations) for accuracy, precision, recall, F1
- Per-class metrics table
- Updates `models/results_summary.npy` with confidence intervals

---

### Step 6: Compare with Traditional ML Baselines (Optional)

```bash
python src/baselines.py
```

**Duration:** ~2 minutes

**What it does:**
- Flattens Mel-spectrograms into 1D feature vectors (~16,640 dimensions)
- Trains two models:
  - SVM (RBF kernel, balanced class weights)
  - Random Forest (200 trees, max_depth=20, balanced)
- Prints comparison table: CNN vs SVM vs Random Forest on test set accuracy, precision, recall, F1

---

## How to Evaluate Performance

### Key Metrics to Check

| Metric | What it Measures | Good Target |
|--------|------------------|-------------|
| **Accuracy** | Overall correct classification rate | >80% |
| **Macro F1** | Balanced F1 across all classes (handles imbalance) | >0.80 |
| **Per-class F1** | Performance on each individual grade | >0.75 each |
| **Confusion matrix** | Which classes get confused with each other | Diagonal dominance |
| **AUC-ROC** | Model's discrimination ability per class | >0.90 |

### Reading the Confusion Matrix

- **Diagonal cells** = correct predictions (should be high)
- **Off-diagonal cells** = misclassifications (should be low)
- Grade C may be harder to classify due to fewer training samples (119 vs 276)

### Reading Bootstrap Confidence Intervals

- **Narrow CI** → reliable performance estimate
- **CI lower bound > 0.75** → acceptable performance
- **Wide CI** → need more data or class is inherently difficult

### Output Files Summary

| File | Contents |
|------|----------|
| `models/results_summary.npy` | All metrics + bootstrap 95% CIs (dict) |
| `figures/confusion_matrix.png` | Heatmap of predictions vs actuals |
| `figures/roc_curve.png` | ROC curves with AUC per class |
| `figures/pr_curve.png` | Precision-Recall curves |
| `figures/training_curves_final.png` | Loss and accuracy over training epochs |
| `figures/misclassifications.png` | Visual examples of errors with confidence scores |

---

## Improvements and Optimizations

### 1. Address Class Imbalance

Grade C has only 119 files (20% of dataset). Current mitigation: balanced class weights.

**Additional steps:**
- Oversample Grade C to match Grade A before augmentation (upsample to 276)
- Apply more aggressive augmentations to minority classes:
  - Pitch shift ±2 semitones for Grade C only
  - Higher noise factor (0.01) for Grade C
- Consider focal loss instead of standard cross-entropy

### 2. Expand Hyperparameter Search

Current grid has 16 combinations. Expand to include:

| Parameter | Current | Add These |
|-----------|---------|-----------|
| Learning rate | 1e-3, 1e-4 | 5e-4, 3e-4 |
| Dropout | 0.3, 0.5 | 0.4, 0.6 |
| Batch size | 16, 32 | 64 |

### 3. Model Architecture Improvements

The current CNN is ~110K parameters. Options to try:
- Add a 4th Conv block (256 filters) with BatchNorm
- Replace GlobalAveragePooling with a small flatten + Dense(128)
- Use depthwise separable convolutions for efficiency
- Add squeeze-and-excitation (SE) blocks for channel attention
- Try a ResNet-style architecture with skip connections

### 4. Training Improvements

- **Mixed precision training:** Use `tf.keras.mixed_precision.set_global_policy('mixed_float16')` for ~2x speedup on GPU
- **Data pipeline:** Convert to `tf.data.Dataset` with `.prefetch(tf.data.AUTOTUNE)` for faster loading
- **K-fold cross-validation:** With only ~2,955 samples, 5-fold CV gives more reliable estimates
- **Learning rate warmup:** Linear warmup for first 5-10% of training epochs
- **Cosine annealing:** Replace ReduceLROnPlateau with cosine annealing schedule

### 5. Preprocessing Tuning

| Parameter | Current | Try These |
|-----------|---------|-----------|
| Window length | 0.8s (17,640 samples) | 1.0s, 1.5s, 2.0s |
| Mel bands | 128 | 64, 96 |
| FFT size | 2048 | 1024, 4096 |
| Noise reduction | prop_decrease=0.85 | 0.7, 0.9 |
| Trim threshold | top_db=30 | 20, 25, 35 |

### 6. Additional Augmentations

Add to `src/augment.py`:
- Time masking (SpecAugment style)
- Frequency masking
- Random gain/volume adjustment
- Reverb simulation
- Mixup between samples (alpha=0.2)
- CutMix for spectrograms

### 7. Ensemble Methods

Train 3-5 models with different random seeds, then average predictions. This typically improves accuracy by 2-5% and reduces variance.

### 8. Transfer Learning

Consider using a pretrained audio model:
- **YAMNet** (Google) — pretrained on AudioSet
- **VGGish** — pretrained on YouTube audio
- **AST (Audio Spectrogram Transformer)** — state-of-the-art
- **ResNet-50** pretrained on ImageNet, fine-tuned on spectrograms

---

## Running the Streamlit Demo App

The project includes an interactive web demo (`show-demo/app.py`) that allows real-time brick quality assessment via audio input.

### Features
- **Live Mic Recording**: Click the microphone button to record a brick strike directly in the browser
- **File Upload**: Upload a pre-recorded `.wav` file as a backup
- **Mel-Spectrogram Visualization**: Displays the time-frequency signature of the audio
- **AI Prediction**: Shows the predicted grade (A/B/C) with confidence scores

### How to Run

```bash
source venv/bin/activate
export LD_LIBRARY_PATH=$(python -c "import nvidia; import os; base=os.path.dirname(nvidia.__path__[0]); print(':'.join([os.path.join(base,'nvidia',d,'lib') for d in ['cublas','cudnn','cuda_runtime','cufft','cusolver','cusparse','curand']]))")

cd show-demo
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Prerequisites
- A trained model at `models/best.keras` (run the training pipeline first)
- `streamlit` installed (`pip install -r requirements.txt`)
- Browser microphone access for live recording

---

## Quick Reference Command Sequence

```bash
# Full pipeline from scratch (with GPU):
source venv/bin/activate
export LD_LIBRARY_PATH=$(python -c "import nvidia; import os; base=os.path.dirname(nvidia.__path__[0]); print(':'.join([os.path.join(base,'nvidia',d,'lib') for d in ['cublas','cudnn','cuda_runtime','cufft','cusolver','cusparse','curand']]))")
python src/batch_process.py          # ~15 min
python src/hparam_search.py          # ~15 min (optional, faster with GPU)
python src/train.py                  # ~5 min (GPU) vs ~30 min (CPU)
python src/evaluate.py               # ~1 min
python src/visualize.py              # ~5 min
python src/baselines.py              # ~2 min (optional)
```

**Total estimated time:** ~25-60 minutes with GPU (vs 60-150 minutes on CPU).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: data/raw/` | Run `python init_workspace.py` to create directory structure |
| CUDA out of memory | Reduce `BATCH_SIZE` in `train.py` from 32 to 16 |
| GPU not detected | Run `export LD_LIBRARY_PATH=...` (see GPU Setup above) |
| `Could not find cuda drivers` | Install NVIDIA packages: `pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 ...` |
| Training stuck at low accuracy | Run `hparam_search.py` to find better hyperparameters |
| Grade C performing poorly | Apply stronger augmentations to minority class |
| Overfitting (train acc >> val acc) | Increase dropout, add L2 regularization, add more augmentations |
| Underfitting (both accuracies low) | Increase model capacity, lower learning rate, train longer |
