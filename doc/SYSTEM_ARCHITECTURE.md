# System Architecture & Dataflow Documentation

> Thesis: Brick Grade Classification using Audio and Deep Learning
> Codebase analysis and complete documentation

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Dataflow Diagram](#2-dataflow-diagram)
3. [End-to-End Working Process](#3-end-to-end-working-process)
4. [Step-by-Step Breakdown](#4-step-by-step-breakdown)
5. [Outputs Summary](#5-outputs-summary)
6. [File Dependency Graph](#6-file-dependency-graph)

---

## 1. System Overview

This system classifies bricks into two grades (Grade A / Grade B) by analyzing the sound of bricks being struck. Raw audio recordings are converted into Mel-spectrogram images, which are then classified by a Convolutional Neural Network (CNN). The project includes a complete ML pipeline: data preprocessing, augmentation, CNN training, evaluation, hyperparameter search, baseline comparisons, and visualization.

**Key components:**
- **Audio Preprocessing** (`preprocess.py`): Raw audio → normalized Mel-spectrograms
- **Augmentation** (`augment.py`): Waveform-level perturbations to expand small datasets
- **Batch Processing** (`batch_process.py`): Orchestrates loading, augmenting all files, splitting, saving
- **CNN Model** (`model.py`): 2D CNN architecture (~110K parameters)
- **Training** (`train.py`): Training loop with callbacks
- **Evaluation** (`evaluate.py`): Test-set metrics and figures
- **Hyperparameter Search** (`hparam_search.py`): Grid search over 16 configurations
- **Baselines** (`baselines.py`): SVM and Random Forest for comparison
- **Visualization** (`visualize.py`): 7 publication-ready thesis figures

---

## 2. Dataflow Diagram

```
                              DATAFLOW DIAGRAM
                              ================

                           +-------------------+
                           |  RAW AUDIO FILES  |
                           | data/raw/         |
                           |  grade_a/ (15)    |
                           |  grade_b/ (15)    |
                           +---------+---------+
                                     |
                                     | librosa.load(sr=22050, mono=True)
                                     v
                           +---------+---------+
                           |  PREROCESSING +   |
                           |  AUGMENTATION     |
                           |  (applied to all  |
                           |   30 files)       |
                           |                   |
                           |  For each file:   |
                           |  1. preprocess    |
                           |     audio→mel     |
                           |  2. pitch_shift+2 |
                           |  3. pitch_shift-2 |
                           |  4. time_stretch  |
                           |  5. add_noise     |
                           +---------+---------+
                                     |
                                     | 30 × 5 = 150 samples
                                     v
                           +---------+---------+
                           |  STRATIFIED SPLIT |
                           |  70/15/15 on 150  |
                           +----+----+----+----+
                                |    |    |
                   +------------+    |    +------------+
                   |                 |                 |
                   v                 v                 v
           +-------+-------+  +-----+-----+  +--------+------+
           |   TRAIN        |  |  VAL       |  |  TEST        |
           |  ~105 samples  |  |  ~22       |  |  ~23         |
           |  (128,130,1)   |  |  samples   |  |  samples     |
           +-------+-------+  +------+------+  +-------+------+
                   |                  |                 |
                   v                  v                 v
           +-------+-------+  +------+------+  +-------+------+
           | X_train.npy   |  | X_val.npy   |  | X_test.npy   |
           | y_train.npy   |  | y_val.npy   |  | y_test.npy   |
           +-------+-------+  +------+------+  +-------+------+
                   |
                   v
           +----------------+      +----------------+
           |  model.py      |      |  train.py      |
           |  build_cnn()   |<-----|  model.fit()   |
           |  ~110K params  |      |  callbacks     |
           +----------------+      +--------+-------+
                                            |
                                   +--------+--------+
                                   | best.keras      |
                                   | history.npy     |
                                   +--------+--------+
                                            |
                      +---------------------+---------------------+
                      |                     |                     |
                      v                     v                     v
              +-------+------+    +---------+------+     +-------+------+
              | evaluate.py  |    | visualize.py   |     | hparam       |
              | test metrics |    | 7 figures      |     | _search.py   |
              +-------+------+    +---------+------+     +-------+------+
                      |                     |                     |
                      v                     v                     v
              +-------+------+    +---------+------+     +-------+------+
              | results       |    | figures/      |     | best_hparams |
              | _summary.npy  |    | *.png (7)     |     | .npy         |
              +-------+------+    +----------------+     +--------------+
                      |
                      v
              +-------+------+
              | baselines.py |
              | SVM + RF     |
              | (console)    |
              +--------------+
```

---

## 3. End-to-End Working Process

### Execution Order

```
Step 0: Environment Setup
  └─ bash setup.sh
  └─ Creates virtualenv + installs dependencies

Step 1: Directory Initialization
  └─ python init_workspace.py
  └─ Creates: data/raw/grade_a/, data/raw/grade_b/,
              data/processed/, src/, models/

Step 2: Raw Data Collection (Manual)
  └─ Place .wav files in data/raw/grade_a/ and data/raw/grade_b/

Step 3: Batch Preprocessing
  └─ python src/batch_process.py
  └─ Converts all audio → Mel-spectrograms, augments ALL files
      (4× each), then splits 70/15/15, saves .npy

Step 4 (Optional): Hyperparameter Search
  └─ python src/hparam_search.py
  └─ Grid search over 16 combos, saves best config

Step 5: Model Training
  └─ python src/train.py
  └─ Trains CNN with callbacks, saves best model + history

Step 6: Model Evaluation
  └─ python src/evaluate.py
  └─ Test-set metrics, confusion matrix, ROC curve

Step 7: Baseline Comparison
  └─ python src/baselines.py
  └─ SVM + Random Forest on flattened features

Step 8: Visualization
  └─ python src/visualize.py
  └─ Generates 7 thesis figures + bootstrap CIs
```

---

## 4. Step-by-Step Breakdown

### Step 0: Environment Setup (`setup.sh`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Creates Python virtual environment, upgrades pip/setuptools/wheel, installs all packages from `requirements.txt`, verifies critical imports |
| **Why needed** | Isolates project dependencies, ensures reproducible environment across machines |
| **Logic** | Checks for existing venv to avoid redundant creation; uses `pip install --upgrade` to get latest compatible versions; import verification catches missing/broken packages early |
| **Outcome** | A ready-to-use Python environment with tensorflow, librosa, noisereduce, scikit-learn, matplotlib, numpy for all downstream steps |

### Step 1: Directory Initialization (`init_workspace.py`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Creates the directory tree: `data/raw/grade_a/`, `data/raw/grade_b/`, `data/processed/`, `src/`, `models/` |
| **Why needed** | Standardized directory structure ensures all scripts can find data using consistent relative paths |
| **Logic** | Uses `os.makedirs(exist_ok=True)` for idempotency — safe to rerun without errors |
| **Outcome** | Well-organized project skeleton ready for data placement and processing |

### Step 2: Raw Data Collection (Manual)

| Aspect | Detail |
|--------|--------|
| **What it does** | User places audio recordings (.wav) of brick strikes into class-labeled directories |
| **Why needed** | Raw data is the system input; separation by directory encodes ground-truth labels |
| **Logic** | Directory name (`grade_a`/`grade_b`) maps to numeric label (0/1) in `batch_process.py:LABEL_MAP` |
| **Outcome** | 30 raw audio files (15 per class) organized by grade, ready for feature extraction |

### Step 3: Batch Preprocessing (`src/batch_process.py`)

This is the most complex step. It consists of several sub-steps:

#### 3a. Audio File Walking

| Aspect | Detail |
|--------|--------|
| **What it does** | `_walk_audio_files()` scans `data/raw/grade_a/` and `data/raw/grade_b/` for supported audio files (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`) |
| **Why needed** | Dynamically discovers all available audio files without hardcoding file lists |
| **Logic** | Uses `glob.glob()` with extension patterns; yields `(filepath, label)` tuples; supports multiple audio formats |
| **Outcome** | Complete list of all data files with their class labels |

#### 3b. Audio Loading (`librosa.load`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Reads audio file into a 1D numpy array (waveform) at 22050 Hz sample rate, mono channel |
| **Why needed** | Converts compressed audio formats into raw numerical representation suitable for signal processing |
| **Logic** | `sr=SR` resamples to exactly 22050 Hz (standard for speech/audio ML); `mono=True` collapses stereo to single channel reducing data dimensionality |
| **Outcome** | Raw waveform `y` of shape `(N,)` where N = samples at 22050 Hz |

#### 3c. Noise Reduction (`noisereduce.reduce_noise`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Applies spectral gating to suppress background noise |
| **Why needed** | Brick strike recordings may contain environmental noise (handling sounds, room echo) that corrupts the acoustic signature |
| **Logic** | `stationary=False` uses non-stationary noise reduction (better for varying noise); `prop_decrease=0.85` reduces noise by 85% (aggressive but preserves signal) |
| **Outcome** | Cleaned waveform with reduced background noise |

#### 3d. Silence Trimming (`librosa.effects.trim`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Removes leading/trailing silence below 30 dB threshold |
| **Why needed** | Brick strike sounds are short (<1s); silence before/after strike is uninformative and wastes model capacity on padding |
| **Logic** | `top_db=30` — frames below 30 dB relative to max are considered silence and removed |
| **Outcome** | Trimmed waveform containing only the active brick strike sound |

#### 3e. Length Normalization (Pad/Crop to 3 seconds)

| Aspect | Detail |
|--------|--------|
| **What it does** | Forces every audio sample to exactly 66150 samples (3s × 22050 Hz) via symmetric padding or center-cropping |
| **Why needed** | Neural networks require fixed-size inputs; batch processing needs uniform tensor dimensions |
| **Logic** | If shorter: pad equally on both sides (preserves temporal alignment); if longer: center-crop (assumes strike is in the middle). 3 seconds is generous to capture full strike + resonance |
| **Outcome** | Uniform-length waveform: exactly 66150 samples |

#### 3f. Mel-Spectrogram Conversion

| Aspect | Detail |
|--------|--------|
| **What it does** | Computes the Mel-scaled spectrogram: 128 Mel frequency bands, 2048-point FFT, 512-sample hop length, resulting in a (128, 130) time-frequency representation |
| **Why needed** | Mel-spectrograms mimic human auditory perception (better frequency resolution at low frequencies); provide a 2D "image-like" representation suitable for CNNs; more compact than raw audio or linear spectrograms |
| **Logic** | `n_mels=128` provides good frequency resolution; `hop_length=512` gives ~46 frames/second; 130 time frames from 66150/512 ≈ 129.2 → 130 |
| **Outcome** | 2D array (128 × 130) representing time-frequency energy distribution |

#### 3g. Log Amplitude Conversion (`power_to_db`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Converts power spectrogram to decibel (log) scale |
| **Why needed** | Human hearing perceives sound logarithmically; log compression makes quiet features more visible and prevents large values from dominating |
| **Logic** | `ref=np.max` normalizes so the loudest point is 0 dB; all other values become negative dB |
| **Outcome** | Log-scaled spectrogram where values represent dB relative to peak |

#### 3h. Min-Max Normalization

| Aspect | Detail |
|--------|--------|
| **What it does** | Scales all values to range [0.0, 1.0] |
| **Why needed** | Neural networks train more stably when inputs are in a consistent, bounded range; prevents gradient saturation |
| **Logic** | `(x - min) / (max - min)` — preserves relative differences; safe division handles edge case of constant input |
| **Outcome** | Normalized spectrogram with values in [0, 1], dtype float32 |

#### 3i. Channel Dimension Addition

| Aspect | Detail |
|--------|--------|
| **What it does** | Adds a trailing dimension: (128, 130) → (128, 130, 1) |
| **Why needed** | Keras/TensorFlow Conv2D layers expect 4D tensors (batch, height, width, channels) |
| **Logic** | `np.newaxis` at the last axis creates a single-channel "image" |
| **Outcome** | 3D tensor (128, 130, 1) ready for CNN input |

#### 3j. Data Augmentation (Applied to Every File)

| Aspect | Detail |
|--------|--------|
| **What it does** | For every file, creates 4 additional versions via pitch shift (±2 semitones), time stretch (1.1×), and additive Gaussian noise |
| **Why needed** | Only 30 original samples — too few for deep learning. Augmentation creates plausible variations, expanding the total sample pool before splitting |
| **Logic** | Applied at waveform level (before spectrogram) for realism; each augmentation preserves the class label |
| **Outcome** | 5× expansion: 30 files → 150 total samples (original + 4 augmented each) |

#### 3k. Stratified Train/Val/Test Split

| Aspect | Detail |
|--------|--------|
| **What it does** | Splits the 150 augmented samples into 70% train, 15% validation, 15% test while preserving class proportions |
| **Why needed** | Ensures each split has representative class distribution |
| **Logic** | Two-stage `train_test_split` with `stratify` parameter: first splits off 15% as test, then splits 15/85 of remaining as val; `random_state=42` ensures reproducibility |
| **Outcome** | Train (~105), val (~22), test (~23) samples with balanced classes |

#### 3l. Save NumPy Arrays

| Aspect | Detail |
|--------|--------|
| **What it does** | Saves stacked arrays to `data/processed/` as 6 `.npy` files |
| **Why needed** | Decouples expensive preprocessing from training; allows multiple training runs without reprocessing |
| **Logic** | `np.stack` creates uniform 4D batches; `np.save` uses efficient binary format |
| **Outcome** | `X_train (~105,128,130,1)`, `y_train (~105,)`, `X_val (~22,...)`, `y_val (~22,)`, `X_test (~23,...)`, `y_test (~23,)` |

### Step 4: Hyperparameter Search (`src/hparam_search.py`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Exhaustive grid search over 4 parameters × 2 values = 16 combinations: learning rate (1e-3, 1e-4), dropout (0.3, 0.5), L2 regularization (1e-4, 1e-3), batch size (16, 32) |
| **Why needed** | CNN performance is sensitive to hyperparameters; manual tuning is unreliable; best config may differ from defaults |
| **Logic** | `itertools.product` generates Cartesian product; each combo trained for 30 epochs with early stopping (patience=8); tracks best val_accuracy; `restore_best_weights=True` prevents overfitting |
| **Outcome** | `models/best_hparams.npy` — dict with best config + achieved val_accuracy |

### Step 5: Model Architecture (`src/model.py`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Builds a 3-block 2D CNN with Batch Normalization, Global Average Pooling, Dense head |
| **Why needed** | CNNs are the standard for spectrogram classification; they exploit local frequency-time patterns (textures, edges) |
| **Logic** | |
| | - **Conv2D(32, 3×3)** — 32 filters detect basic spectrotemporal patterns |
| | - **BatchNorm** — stabilizes training, reduces internal covariate shift |
| | - **MaxPool(2×2)** — downsamples 2×, reduces parameters, adds translation invariance |
| | - **Conv2D(64, 3×3)** — 64 filters detect more complex patterns |
| | - **Conv2D(128, 3×3)** — 128 filters, no pooling (preserve resolution before GAP) |
| | - **GlobalAveragePooling2D** — replaces Flatten + Dense; drastically reduces parameters, prevents overfitting, provides translation invariance |
| | - **Dense(128, ReLU, L2)** — final feature projection with L2 regularization |
| | - **Dropout(0.5)** — randomly drops 50% of neurons during training, prevents co-adaptation |
| | - **Dense(2, Softmax)** — outputs class probabilities |
| | - Total: ~110K parameters (very compact) |
| **Outcome** | A compiled Keras Model ready for training |

### Step 6: Model Training (`src/train.py`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Loads preprocessed data, builds CNN, trains with Adam optimizer, 3 callbacks |
| **Why needed** | Core model fitting step — learns to map spectrograms to brick grades |
| **Logic** | |
| | - **Adam(lr=1e-3)** — adaptive optimizer with per-parameter learning rates |
| | - **sparse_categorical_crossentropy** — appropriate for integer labels (0, 1) |
| | - **ModelCheckpoint(val_accuracy, mode='max')** — saves only the epoch with highest validation accuracy |
| | - **EarlyStopping(patience=10)** — stops if val_accuracy doesn't improve for 10 epochs; restores best weights |
| | - **ReduceLROnPlateau(factor=0.5, patience=5)** — halves LR when val_loss plateaus, enabling finer convergence |
| | - **batch_size=32** — good balance for 100 training samples |
| | - **epochs=100** — upper bound; early stopping typically stops sooner |
| **Outcome** | `models/best.keras` (best model weights), `models/history.npy` (per-epoch loss/accuracy) |

### Step 7: Model Evaluation (`src/evaluate.py`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Loads best model, predicts on held-out test set, computes comprehensive metrics |
| **Why needed** | Test set provides unbiased estimate of real-world performance; metrics quantify different aspects of model quality |
| **Logic** | |
| | - **Accuracy** — overall correctness (but misleading for imbalanced data) |
| | - **Precision** — of predicted Grade B, how many are correct? (false positive cost) |
| | - **Recall** — of actual Grade B, how many caught? (false negative cost) |
| | - **F1-score** — harmonic mean of precision & recall |
| | - **Confusion Matrix** — shows exactly which errors the model makes |
| | - **ROC AUC** — measures separation ability across all thresholds |
| **Outcome** | Console metrics output + `figures/confusion_matrix.png` + `figures/roc_curve.png` + `models/results_summary.npy` |

### Step 8: Baseline Comparison (`src/baselines.py`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Trains SVM (RBF kernel) and Random Forest (200 trees, max_depth=20) on flattened 16640-dimensional Mel-spectrogram features |
| **Why needed** | Validates that the task is learnable (if baselines fail, the problem is too hard). Provides a reference point: the CNN must outperform simple models to justify its complexity |
| **Logic** | |
| | - **SVM (RBF)** — non-linear kernel captures complex decision boundaries; `class_weight="balanced"` handles class imbalance |
| | - **Random Forest** — ensemble of 200 trees robust to overfitting; feature importance interpretable |
| | - Features flattened from (128, 130, 1) → 16640 — no spatial structure preserved |
| **Outcome** | Console comparison table: SVM + RF accuracy, precision, recall, F1 |

### Step 9: Visualization (`src/visualize.py`)

| Aspect | Detail |
|--------|--------|
| **What it does** | Generates 7 publication-ready figures from trained model + data |
| **Why needed** | Visual communication is essential for thesis; figures illustrate data characteristics, model behavior, and results |
| **Logic** | |
| | **Figure 1 — Spectrogram Grid (4×6):** 24 random training samples with labels; shows what the model "sees" |
| | **Figure 2 — Training Curves:** loss & accuracy over epochs for train/val; diagnoses overfitting |
| | **Figure 3 — Class Distribution:** bar chart across splits; validates stratification |
| | **Figure 4 — Confusion Matrix:** normalized heatmap; visual classification errors |
| | **Figure 5 — ROC Curve:** TPR vs FPR with AUC; measures discriminability |
| | **Figure 6 — PR Curve:** precision vs recall with AUC; better for imbalanced data |
| | **Figure 7 — Misclassifications:** spectrograms of errors with predicted/true labels; reveals failure modes |
| | **Bootstrap CI (1000 iterations):** resamples test set with replacement to compute 95% confidence intervals for all metrics |
| **Outcome** | 7 `.png` files in `figures/` + updated `models/results_summary.npy` with bootstrap CIs |

---

## 5. Outputs Summary

### Data Files (`data/processed/`)
| File | Approx. Shape | Description |
|------|---------------|-------------|
| `X_train.npy` | (~105, 128, 130, 1) | Training features (augment-all before split) |
| `y_train.npy` | (~105,) | Training labels |
| `X_val.npy` | (~22, 128, 130, 1) | Validation features |
| `y_val.npy` | (~22,) | Validation labels |
| `X_test.npy` | (~23, 128, 130, 1) | Test features |
| `y_test.npy` | (~23,) | Test labels |

### Model Files (`models/`)
| File | Description |
|------|-------------|
| `best.keras` | Best model checkpoint by val_accuracy |
| `final_best.keras` | Alternative best model (e.g., after HPO retrain) |
| `history.npy` | Per-epoch training metrics dict |
| `history_final.npy` | Alternative training history |
| `best_hparams.npy` | Best hyperparameter config from grid search |
| `results_summary.npy` | All test metrics + bootstrap CIs |

### Figures (`figures/`)
| File | Description |
|------|-------------|
| `spectrogram_grid.png` | 4×6 grid of random sample spectrograms |
| `training_curves_final.png` | Loss & accuracy over epochs |
| `class_distribution.png` | Class counts across train/val/test |
| `confusion_matrix.png` | Normalized confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC |
| `pr_curve.png` | Precision-Recall curve with AUC |
| `misclassifications.png` | Up to 8 misclassified samples |

---

## 6. File Dependency Graph

```
                     +------------------+
                     |  requirements.txt |
                     +--------+---------+
                              |
                              v
                     +--------+---------+
                     |    setup.sh      |
                     |  (environment)   |
                     +------------------+

                     +------------------+
                     | init_workspace.py|
                     |  (directories)   |
                     +------------------+

                     +------------------+
                     |  preprocess.py   |
                     |  (mel conversion)|
                     +--------+---------+
                              |
                              v
+------------------+  +-------+---------+  +------------------+
|   augment.py     |  | batch_process   |  |   (librosa,      |
| (pitch, stretch, |->| .py             |  |    noisereduce)  |
|  noise)          |  | (orchestrator)  |  +------------------+
+------------------+  +-------+---------+
                              |
                     +--------+---------+
                     | data/processed/  |
                     | *.npy (6 files)  |
                     +--------+---------+
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
  +---------+-------+  +------+-------+  +-----+--------+
  |   model.py      |  |  train.py    |  | baselines.py |
  | (CNN arch)      |  | (training)   |  | (SVM + RF)   |
  +---------+-------+  +------+-------+  +--------------+
            |                 |
            +-----------------+
                              |
                     +--------+---------+
                     |  evaluate.py     |
                     |  (test metrics)  |
                     +--------+---------+
                              |
                     +--------+---------+
                     |  visualize.py    |
                     |  (7 figures)     |
                     +------------------+
                              |
                     +--------+---------+
                     |  hparam_search   |
                     |  .py             |
                     | (grid search)    |
                     +------------------+
```
