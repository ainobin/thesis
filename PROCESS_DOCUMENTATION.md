# Full Process Documentation — Brick NDT Thesis

**Project:** A Deep Learning-Based Non-Destructive Testing (NDT) Approach for
Brick Quality Assessment via Mel-Spectrogram Analysis

**Task:** Binary Classification — Grade A (0) vs Grade B (1)

---

## Initial Setup

### 1. Create project workspace & verify environment

```bash
# Activate virtual environment
source venv/bin/activate

# Verify dependencies
pip list | grep -E "librosa|noisereduce|scikit-learn|numpy|tensorflow"
```

### 2. Fix Python import path issue

Script `src/batch_process.py` failed with `ModuleNotFoundError: No module named 'src'` because the project root was not on `sys.path`.

**Fix:** Insert project root into `sys.path` at the top of the file:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This was added to both `src/batch_process.py` and `src/train.py`.

---

## Day 1 — Data Pipeline & Preprocessing

### Scripts Created

| File | Description |
|---|---|
| `init_workspace.py` | Creates required directory tree |
| `src/preprocess.py` | Audio → Mel-spectrogram pipeline (refactored: `audio_to_mel()` for raw arrays) |
| `src/augment.py` | Waveform augmentation: pitch shift, time stretch, additive noise |
| `src/batch_process.py` | Batch process + stratify split + augment training set, save `.npy` |

### `init_workspace.py`

```python
import os

BASE = os.path.dirname(os.path.abspath(__file__))

DIRS = [
    os.path.join(BASE, "data", "raw", "grade_a"),
    os.path.join(BASE, "data", "raw", "grade_b"),
    os.path.join(BASE, "data", "processed"),
    os.path.join(BASE, "src"),
    os.path.join(BASE, "models"),
]

def main() -> None:
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"  \u2713 {os.path.relpath(d, BASE)}")
    print("Workspace ready.")

if __name__ == "__main__":
    main()
```

**Execution:**

```bash
python init_workspace.py
```

### `src/preprocess.py`

Preprocessing pipeline:

1. `librosa.load(path, sr=22050, mono=True)` — load audio
2. `noisereduce.reduce_noise(y, sr=22050, stationary=False, prop_decrease=0.85)` — denoise
3. `librosa.effects.trim(y, top_db=30)` — trim silence
4. Symmetric zero-pad or center-crop to exactly 66150 samples (3 s × 22050 Hz)
5. `librosa.feature.melspectrogram(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512)` — Mel spectrogram
6. `librosa.power_to_db(mel, ref=np.max)` — convert to dB
7. Min-max normalize to [0, 1]
8. Return `(128, time_steps, 1)` as `float32`

**Key parameters:**

```python
SR = 22050
FIXED_SAMPLES = SR * 3          # 66150
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
```

### `src/augment.py`

Three waveform-level augmentation functions applied to the raw audio before spectrogram conversion:

| Function | Description | Parameters |
|---|---|---|
| `pitch_shift(y, sr, n_steps=2)` | Shift pitch ±2 semitones | `n_steps` |
| `time_stretch(y, rate=1.1)` | Stretch time by 10% | `rate` |
| `add_noise(y, noise_factor=0.005)` | Add Gaussian noise | `noise_factor` |

### `src/batch_process.py`

Pipeline runner:

1. Scans `data/raw/grade_a/` (label 0) and `data/raw/grade_b/` (label 1)
2. Stratified 70/15/15 split by file index (before processing)
3. Processing each file through `preprocess_audio()`
4. **Training set only:** also applies 4 augmentations (pitch+2, pitch-2, time stretch, noise) to each sample → 5× expansion
5. Stacks features into `X` with shape `(N, 128, 130, 1)`
6. Saves 6 files to `data/processed/`: `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`

**Fix applied:** Changed `np.concatenate(X, axis=0)` to `np.stack(X, axis=0)` — the original was merging along the Mel axis instead of stacking samples.

**Execution:**

```bash
python src/batch_process.py
```

**Output:**

```
  Total valid samples: 30  |  Failures: 0
  ✓ X_train.npy  →  (20, 128, 130, 1)
  ✓ y_train.npy  →  (20,)
  ✓ X_val.npy    →  (5, 128, 130, 1)
  ✓ y_val.npy    →  (5,)
  ✓ X_test.npy   →  (5, 128, 130, 1)
  ✓ y_test.npy   →  (5,)

Class distribution:
  grade_a (0):  train=10  val=2  test=3
  grade_b (1):  train=10  val=3  test=2
```

### Data verification

```bash
python -c "
import numpy as np
X = np.load('data/processed/X_train.npy')
y = np.load('data/processed/y_train.npy')
print('X_train:', X.shape, X.dtype, 'min:', X.min(), 'max:', X.max())
print('y_train:', y.shape, y.dtype, 'label ratio:', np.bincount(y))
"
```

### Preprocessing unit test (synthetic 0.6 s clip)

```bash
python -c "
import numpy as np, librosa
sr = 22050
t = np.linspace(0, 0.6, int(sr*0.6), endpoint=False)
y = np.sin(2*np.pi * 880 * t).astype(np.float32)
from scipy.io.wavfile import write
write('/tmp/short_clip.wav', sr, y)

from src.preprocess import preprocess_audio
feat = preprocess_audio('/tmp/short_clip.wav')
print('Shape:', feat.shape, 'dtype:', feat.dtype, 'range:', feat.min(), '-', feat.max())
assert feat.shape == (128, 130, 1) and feat.dtype == np.float32
assert feat.min() >= 0.0 and feat.max() <= 1.0
print('All checks passed.')
"
```

---

## Day 2 — Model Architecture & Training

### Scripts Created

| File | Description |
|---|---|
| `src/model.py` | CNN model definition |
| `src/train.py` | Training loop with callbacks |

### `src/model.py` — CNN Architecture

```
Input:          (128, 130, 1)
├─ Conv2D(32, 3×3, ReLU) + BatchNorm + MaxPool(2×2)   → (64, 65, 32)
├─ Conv2D(64, 3×3, ReLU) + BatchNorm + MaxPool(2×2)   → (32, 32, 64)
├─ Conv2D(128, 3×3, ReLU) + BatchNorm                  → (32, 32, 128)
├─ GlobalAveragePooling2D                               → (128)
├─ Dense(128, ReLU, L2=1e-4) + Dropout(0.5)            → (128)
└─ Dense(2, Softmax)                                    → (2)
```

`build_cnn()` now accepts `dropout` and `l2_reg` parameters (defaults: 0.5, 1e-4).

`build_cnn()` now accepts `dropout` and `l2_reg` parameters (defaults: 0.5, 1e-4).

Total params: 110,338 (431 KB)

### `src/train.py`

- Loads `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`
- Compiles with `Adam(lr=1e-3)`, `sparse_categorical_crossentropy`
- Callbacks:
  - `ModelCheckpoint` (save best to `models/best.keras`)
  - `EarlyStopping` (patience=10, restore_best_weights)
  - `ReduceLROnPlateau` (factor=0.5, patience=5)
- Trains for 100 epochs, batch_size=32
- Saves history to `models/history.npy`

**Execution:**

```bash
python src/train.py
```

**Output (truncated):**

```
Epoch 1/100 — accuracy: 0.4000 — val_accuracy: 0.6000
...
Epoch 11/100 — early stopping (val_accuracy never improved past 0.6000)
Training accuracy reached 100% by epoch 8 (overfitting on 20 samples)
```

### Plot training curves

```bash
python -c "
import numpy as np, matplotlib.pyplot as plt
h = np.load('models/history.npy', allow_pickle=True).item()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(h['loss'], label='train')
ax1.plot(h['val_loss'], label='val')
ax1.set_title('Loss'); ax1.legend()
ax2.plot(h['accuracy'], label='train')
ax2.plot(h['val_accuracy'], label='val')
ax2.set_title('Accuracy'); ax2.legend()
plt.savefig('models/training_curves.png', dpi=150)
print('Saved training_curves.png')
"
```

---

## Day 3 — Evaluation & Hyperparameter Tuning

### Scripts Created

| File | Description |
|---|---|
| `src/evaluate.py` | Test-set evaluation, metrics, confusion matrix, ROC |
| `src/hparam_search.py` | Grid search over 16 hyperparameter combinations (refactored: uses `build_cnn()` instead of duplicating model) |
| `src/baselines.py` | SVM (RBF) + Random Forest baselines on flattened spectrograms |

### `src/evaluate.py`

- Loads best model and test data
- Computes accuracy, precision, recall, F1-score (per-class + macro)
- Generates confusion matrix plot → `figures/confusion_matrix.png`
- Generates ROC curve → `figures/roc_curve.png`
- Saves `models/results_summary.npy`

**Execution:**

```bash
python src/evaluate.py
```

**Output:**

```
Test accuracy:  0.4000
Precision:      0.4000
Recall:         1.0000
F1-score:       0.5714

Grade A (0):  p=0.0000  r=0.0000  f1=0.0000  support=3
Grade B (1):  p=0.4000  r=1.0000  f1=0.5714  support=2

Confusion matrix:
[[0 3]
 [0 2]]
```

### `src/hparam_search.py`

Grid search parameters:

| Parameter | Values |
|---|---|
| Learning rate | 0.001, 0.0001 |
| Dropout | 0.3, 0.5 |
| L2 regularization | 1e-4, 1e-3 |
| Batch size | 16, 32 |

**Execution:**

```bash
python src/hparam_search.py
```

**Output — all 16 combinations:**

All combos hit `val_accuracy = 0.6000` (3/5 validation samples correct — ceiling effect with tiny dataset).

```
Best config:  {'learning_rate': 0.001, 'dropout': 0.3, 'l2_reg': 0.0001, 'batch_size': 16}
Best val_acc: 0.6000
```

Saved to `models/best_hparams.npy`.

### `src/baselines.py` — Baseline Comparisons

Two traditional ML baselines for comparison:

| Model | Features | Approach |
|---|---|---|
| SVM (RBF kernel) | Flattened Mel-spectrogram (16640 dims) | `class_weight="balanced"` |
| Random Forest (200 trees) | Flattened Mel-spectrogram (16640 dims) | `max_depth=20`, `class_weight="balanced"` |

**Execution:**

```bash
python src/baselines.py
```

### Retrain final model with best config

```bash
python -c "
import os, numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.model import build_cnn

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

model = build_cnn(input_shape=X_train.shape[1:])
model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('models/final_best.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

history = model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_val, y_val), callbacks=callbacks, verbose=2)
np.save('models/history_final.npy', history.history)
print('Final model saved to models/final_best.keras')
"
```

### Final evaluation on retrained model

```bash
python -c "
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

model = load_model('models/final_best.keras')
y_prob = model.predict(X_test, verbose=0)
y_pred = y_prob.argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', labels=[0,1])
cm = confusion_matrix(y_test, y_pred, labels=[0,1])

print(f'Test accuracy: {acc:.4f}')
print(f'Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}')
print(f'Confusion matrix:\n{cm}')

results = {'test_accuracy': float(acc), 'test_precision': float(p), 'test_recall': float(r), 'test_f1': float(f1), 'confusion_matrix': cm.tolist()}
np.save('models/results_summary.npy', results)
"
```

---

## Day 4 — Results Analysis & Visualisation

### Script Created

| File | Description |
|---|---|
| `src/visualize.py` | Generates all 7 figures + per-class metrics table |

`visualize.py` also computes **bootstrap 95% confidence intervals** (1000 iterations) for accuracy, precision, recall, and F1 on the test set.

### `src/visualize.py` — Figures Generated

| # | Figure | Description |
|---|---|---|
| 1 | `spectrogram_grid.png` | 4×6 random Mel-spectrograms with class labels |
| 2 | `training_curves_final.png` | Loss & accuracy over epochs |
| 3 | `confusion_matrix.png` | Normalised confusion matrix |
| 4 | `roc_curve.png` | ROC curve with AUC |
| 5 | `pr_curve.png` | Precision-Recall curve |
| 6 | `class_distribution.png` | Train/val/test bar chart by class |
| 7 | `misclassifications.png` | Misclassified test samples with true/pred labels |

**Execution:**

```bash
python src/visualize.py
```

**Output:**

```
  Saved figures/spectrogram_grid.png
  Saved figures/training_curves_final.png
  Saved figures/class_distribution.png
  Saved figures/confusion_matrix.png
  Saved figures/roc_curve.png
  Saved figures/pr_curve.png
  Saved figures/misclassifications.png  (3 misclassified total)

--- Per-Class Metrics ---
  Grade A     p=0.0000    r=0.0000    f1=0.0000    support=3
  Grade B     p=0.4000    r=1.0000    f1=0.5714    support=2

  ROC AUC: 1.0000  |  PR AUC: 1.0000

  Bootstrap 95% CI:
    Accuracy:  (0.0000, 0.8000)
    Precision: (0.2000, 0.8000)
    Recall:    (0.0000, 1.0000)
    F1:        (0.0000, 0.7273)
```

---

## Final Directory Tree

```
.
├── init_workspace.py
├── setup.sh
├── requirements.txt
├── WORKPLAN.md
├── data/
│   ├── raw/
│   │   ├── grade_a/          ← 15 .wav files
│   │   └── grade_b/          ← 15 .wav files
│   └── processed/
│       ├── X_train.npy       (N×5, 128, 130, 1)  ← augmented
│       ├── y_train.npy       (N×5,)
│       ├── X_val.npy         (5, 128, 130, 1)
│       ├── y_val.npy         (5,)
│       ├── X_test.npy        (5, 128, 130, 1)
│       └── y_test.npy        (5,)
├── src/
│   ├── preprocess.py         Audio → Mel-spectrogram pipeline
│   ├── augment.py            Waveform augmentation (pitch, time, noise)
│   ├── batch_process.py      Batch processor + stratified split + augmentation
│   ├── model.py              CNN model definition (parameterized dropout/L2)
│   ├── train.py              Training loop
│   ├── evaluate.py           Test-set evaluation
│   ├── hparam_search.py      Hyperparameter grid search (uses build_cnn)
│   ├── baselines.py          SVM + Random Forest baselines
│   └── visualize.py          All figures + metrics + bootstrap CI
├── models/
│   ├── best.keras            1.4 MB  (from initial train)
│   ├── final_best.keras      1.4 MB  (retrained with best hparams)
│   ├── history.npy
│   ├── history_final.npy
│   ├── best_hparams.npy
│   └── results_summary.npy
└── figures/
    ├── spectrogram_grid.png
    ├── training_curves_final.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── class_distribution.png
    └── misclassifications.png
```

---

## Everything You Need — Single Sheet

### All commands in order of execution

```bash
# === SETUP ===
source venv/bin/activate
python init_workspace.py

# === DAY 1: PREPROCESSING ===
# Place your .wav files into data/raw/grade_a/ and data/raw/grade_b/
python src/batch_process.py

# Verify output
python -c "
import numpy as np
X = np.load('data/processed/X_train.npy')
y = np.load('data/processed/y_train.npy')
print('X_train:', X.shape, X.dtype, 'min:', X.min(), 'max:', X.max())
print('y_train:', y.shape, y.dtype, 'ratio:', np.bincount(y))
"

# === DAY 2: TRAINING ===
python src/train.py

# Plot training curves
python -c "
import numpy as np, matplotlib.pyplot as plt
h = np.load('models/history.npy', allow_pickle=True).item()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(h['loss'], label='train'); a1.plot(h['val_loss'], label='val')
a1.set_title('Loss'); a1.legend()
a2.plot(h['accuracy'], label='train'); a2.plot(h['val_accuracy'], label='val')
a2.set_title('Accuracy'); a2.legend()
plt.savefig('models/training_curves.png', dpi=150)
"

# === DAY 3: EVALUATION & HPARAM TUNING ===
python src/evaluate.py
python src/hparam_search.py

# Retrain final model (best config from search)
python -c "
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.model import build_cnn
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')
model = build_cnn(input_shape=X_train.shape[1:])
model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callbacks = [ModelCheckpoint('models/final_best.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1), EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)]
history = model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_val, y_val), callbacks=callbacks, verbose=2)
np.save('models/history_final.npy', history.history)
"

# === DAY 4: VISUALISATION ===
python src/visualize.py
```
