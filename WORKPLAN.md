# Thesis Workplan: Brick Quality Assessment via Mel-Spectrogram Analysis

**Title:** A Deep Learning-Based Non-Destructive Testing (NDT) Approach for
Brick Quality Assessment via Mel-Spectrogram Analysis

**Task:** Binary Classification — Grade A (0) vs Grade B (1)

**Total Duration:** 5 Days

---

## Day 1 — Data Pipeline & Preprocessing

### Objectives
- Place raw audio files into the correct directories
- Run the preprocessing pipeline end-to-end
- Verify output `.npy` arrays and class balance

### Tasks

- [ ] **1.1 Place raw audio data**
  - Copy all Grade A `.wav` files into `data/raw/grade_a/`
  - Copy all Grade B `.wav` files into `data/raw/grade_b/`
  - Verify files are present:
    ```bash
    ls data/raw/grade_a/*.wav | wc -l
    ls data/raw/grade_b/*.wav | wc -l
    ```

- [ ] **1.2 Run workspace initializer**
  ```bash
  python init_workspace.py
  ```

- [ ] **1.3 Activate virtual environment & check dependencies**
  ```bash
  source venv/bin/activate
  pip list | grep -E "librosa|noisereduce|scikit-learn|numpy|tensorflow"
  ```

- [ ] **1.4 Execute batch processing pipeline**
  ```bash
  python src/batch_process.py
  ```
  Expected output:
  - Prints total files found and any failures
  - Saves 6 files to `data/processed/`:
    - `X_train.npy`, `y_train.npy`
    - `X_val.npy`, `y_val.npy`
    - `X_test.npy`, `y_test.npy`
  - Prints shapes and class distribution table

- [ ] **1.5 Sanity-check the saved arrays**
  ```bash
  python -c "
  import numpy as np
  X = np.load('data/processed/X_train.npy')
  y = np.load('data/processed/y_train.npy')
  print('X_train:', X.shape, X.dtype)
  print('y_train:', y.shape, y.dtype)
  print('min:', X.min(), 'max:', X.max())
  print('Label ratio:', np.bincount(y))
  "
  ```

- [ ] **1.6 Visualise a few Mel-spectrograms for confirmation**
  ```bash
  python -c "
  import numpy as np, matplotlib.pyplot as plt
  X = np.load('data/processed/X_train.npy')
  y = np.load('data/processed/y_train.npy')
  fig, axes = plt.subplots(2, 3, figsize=(12, 6))
  for i, ax in enumerate(axes.flat):
      idx = np.random.randint(len(X))
      ax.imshow(X[idx, :, :, 0], aspect='auto', origin='lower', cmap='magma')
      ax.set_title(f'Grade {\"A\" if y[idx]==0 else \"B\"}')
      ax.axis('off')
  plt.tight_layout()
  plt.savefig('data/processed/sample_spectrograms.png', dpi=150)
  print('Saved sample_spectrograms.png')
  "
  ```

### Deliverables
- [ ] `data/processed/X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`
- [ ] `data/processed/sample_spectrograms.png`
- [ ] Console output with matrix shapes & class distribution

---

## Day 2 — Model Architecture & Training

### Objectives
- Design a CNN or CNN+LSTM architecture that consumes `(128, T, 1)` input
- Train the model on the preprocessed data
- Log metrics (accuracy, loss, precision, recall, F1)

### Tasks

- [ ] **2.1 Create `src/model.py` with model definition**

  ```python
  # src/model.py
  from tensorflow.keras import layers, Model, regularizers

  def build_cnn(input_shape=(128, None, 1), num_classes=2):
      inp = layers.Input(shape=input_shape)

      x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
      x = layers.BatchNormalization()(x)
      x = layers.MaxPooling2D((2, 2))(x)

      x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
      x = layers.BatchNormalization()(x)
      x = layers.MaxPooling2D((2, 2))(x)

      x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
      x = layers.BatchNormalization()(x)
      x = layers.GlobalAveragePooling2D()(x)

      x = layers.Dense(128, activation='relu',
                       kernel_regularizer=regularizers.l2(1e-4))(x)
      x = layers.Dropout(0.5)(x)
      out = layers.Dense(num_classes, activation='softmax')(x)

      return Model(inputs=inp, outputs=out)
  ```

- [ ] **2.2 Create `src/train.py`**

  - Load `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`
  - Convert labels to one-hot encoding
  - Compile model with `Adam(learning_rate=1e-3)`, `sparse_categorical_crossentropy`
  - Callbacks:
    - `ModelCheckpoint` (save best to `models/best.keras`)
    - `EarlyStopping` (patience=10, restore_best_weights=True)
    - `ReduceLROnPlateau` (factor=0.5, patience=5)
  - `model.fit()` with batch_size=32, epochs=100
  - Save training history to `models/history.npy`

- [ ] **2.3 Run training**
  ```bash
  python src/train.py
  ```

- [ ] **2.4 Plot loss & accuracy curves**
  ```bash
  python -c "
  import numpy as np, matplotlib.pyplot as plt
  h = np.load('models/history.npy', allow_pickle=True).item()
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
  ax1.plot(h['loss'], label='train'); ax1.plot(h['val_loss'], label='val')
  ax1.set_title('Loss'); ax1.legend()
  ax2.plot(h['accuracy'], label='train'); ax2.plot(h['val_accuracy'], label='val')
  ax2.set_title('Accuracy'); ax2.legend()
  plt.savefig('models/training_curves.png', dpi=150)
  print('Saved training_curves.png')
  "
  ```

### Deliverables
- [ ] `src/model.py`
- [ ] `src/train.py`
- [ ] `models/best.keras` (best checkpoint)
- [ ] `models/history.npy`
- [ ] `models/training_curves.png`

---

## Day 3 — Evaluation & Hyperparameter Tuning

### Objectives
- Evaluate the saved model on the held-out test set
- Compute full classification metrics
- Run a small hyperparameter sweep (learning rate, dropout, L2)
- Document best configuration

### Tasks

- [ ] **3.1 Create `src/evaluate.py`**

  - Load `X_test.npy`, `y_test.npy` and the best model
  - Predict probabilities and class labels
  - Compute and print:
    - Test loss & accuracy
    - Precision, Recall, F1-score (per class and macro)
    - Confusion matrix
  - Save confusion matrix as `models/confusion_matrix.png`

- [ ] **3.2 Run evaluation**
  ```bash
  python src/evaluate.py
  ```

- [ ] **3.3 Hyperparameter grid search (lightweight)**

  Test combinations in `src/hparam_search.py`:

  | Parameter         | Values                 |
  |-------------------|------------------------|
  | Learning rate     | 1e-3, 1e-4            |
  | Dropout           | 0.3, 0.5              |
  | L2 regularizer    | 1e-4, 1e-3            |
  | Batch size        | 16, 32                |

  Only 2×2×2×2 = 16 runs. Use a `for` loop, train for 30 epochs each,
  record validation accuracy, pick the best combo.

- [ ] **3.4 Train final model with best hyperparameters**

  - Update `src/train.py` with the winning config
  - Retrain for full 100 epochs
  - Save as `models/final_best.keras`
  - Regenerate curves & confusion matrix

- [ ] **3.5 Save a concise results summary**

  ```bash
  python -c "
  import numpy as np
  results = {
      'best_lr': 1e-3, 'best_dropout': 0.5, 'best_l2': 1e-4,
      'best_batch_size': 32,
      'test_accuracy': 0.XX,
      'test_f1': 0.XX,
      'test_precision': 0.XX,
      'test_recall': 0.XX
  }
  np.save('models/results_summary.npy', results)
  "
  ```

### Deliverables
- [ ] `src/evaluate.py`
- [ ] `src/hparam_search.py`
- [ ] `models/confusion_matrix.png`
- [ ] `models/final_best.keras`
- [ ] `models/results_summary.npy`

---

## Day 4 — Results Analysis & Visualisation

### Objectives
- Build a comprehensive evaluation notebook / script
- Generate all figures for the thesis
- Extract key talking points (accuracy, failure analysis)

### Tasks

- [ ] **4.1 Create `src/visualize.py`**

  Produce the following figures and save as high-res PNGs:

  | Figure | Description |
  |--------|-------------|
  | `figures/spectrogram_grid.png` | 6×4 grid of sample Mel-spectrograms per class |
  | `figures/training_curves_final.png` | Final loss & accuracy over epochs |
  | `figures/confusion_matrix.png` | Normalised confusion matrix (already from Day 3) |
  | `figures/roc_curve.png` | ROC curve with AUC for each class |
  | `figures/pr_curve.png` | Precision-Recall curve |
  | `figures/class_distribution.png` | Bar chart of train/val/test counts |

- [ ] **4.2 Compute per-class metrics**

  ```
  | Class   | Precision | Recall | F1-Score | Support |
  |---------|-----------|--------|----------|---------|
  | Grade A | 0.XX      | 0.XX   | 0.XX     |  XX     |
  | Grade B | 0.XX      | 0.XX   | 0.XX     |  XX     |
  ```

- [ ] **4.3 Failure analysis**

  - Identify misclassified test samples
  - Plot a subset (4–6) of false positives & false negatives
  - Save as `figures/misclassifications.png`
  - Write 2–3 sentences on possible causes (signal similarity, noise, etc.)

- [ ] **4.4 Model summary statistics**

  - Total parameters
  - Inference time per sample (average over 100 runs)
  - Model size on disk (MB)

- [ ] **4.5 Create `figures/` directory and export all plots**

  ```bash
  mkdir -p figures
  python src/visualize.py
  ```

### Deliverables
- [ ] `src/visualize.py`
- [ ] `figures/` directory with all 7+ figures
- [ ] `figures/misclassifications.png`
- [ ] Per-class metrics table (printed or saved as CSV)

---

## Day 5 — Thesis Writing & Final Polish

### Objectives
- Write/complete the thesis sections using the generated figures and metrics
- Finalise code repository
- Prepare any presentation materials

### Tasks

- [ ] **5.1 Thesis sections to complete**

  - **Abstract** — Summarise problem, method, dataset, key results
  - **Introduction** — Motivation for NDT in brick quality, DL approach
  - **Related Work** — Cite 5–10 papers on acoustic NDT, mel-spectrogram DL
  - **Methodology** — 2–3 pages covering:
    - Data acquisition & preprocessing pipeline
    - Mel-spectrogram extraction parameters
    - CNN architecture diagram
    - Training details (split, hyperparams, callbacks)
  - **Experiments & Results** — Present all figures + metrics table
  - **Discussion** — Interpret results, limitations, failure cases
  - **Conclusion & Future Work** — Summary + next steps

- [ ] **5.2 Generate final numbers for the thesis**

  ```bash
  python -c "
  import numpy as np
  r = np.load('models/results_summary.npy', allow_pickle=True).item()
  print(f'Test Accuracy:  {r[\"test_accuracy\"]:.4f}')
  print(f'Test F1-score:  {r[\"test_f1\"]:.4f}')
  print(f'Model size:     {r[\"model_size_mb\"]:.2f} MB')
  print(f'Inference/sample: {r[\"inference_ms\"]*1000:.2f} ms')
  "
  ```

- [ ] **5.3 Clean up the repository**

  - Add `.gitignore` entries for `*.npy`, `*.keras`, `figures/`, `__pycache__/`, `venv/`
  - Remove any debug/temp files
  - Final commit with message `"Thesis final submission"`
  - Verify everything runs from a clean clone:
    ```bash
    git clone ... && cd thesis && bash setup.sh && python src/batch_process.py
    ```

- [ ] **5.4 Optional: Build a simple inference demo**

  ```bash
  python -c "
  from src.preprocess import preprocess_audio
  from tensorflow.keras.models import load_model
  model = load_model('models/final_best.keras')
  feat = preprocess_audio('path/to/test.wav')
  pred = model.predict(feat[np.newaxis, ...])[0]
  grade = 'Grade A' if pred.argmin() == 0 else 'Grade B'
  print(f'Prediction: {grade} (confidence: {pred.max():.2%})')
  "
  ```

- [ ] **5.5 [Optional] Create presentation slides**

  - Title slide
  - Problem statement & motivation (1 slide)
  - Proposed method pipeline diagram (1 slide)
  - Dataset & preprocessing (1 slide)
  - Model architecture (1 slide)
  - Results table & key figures (2–3 slides)
  - Conclusion (1 slide)

### Deliverables
- [ ] Completed thesis (PDF or DOCX)
- [ ] Clean, tagged GitHub repository
- [ ] `models/results_summary.npy` with final numbers
- [ ] [Optional] Presentation slides

---

## Quick Reference — All Commands

```bash
# Day 1
source venv/bin/activate
python init_workspace.py
python src/batch_process.py

# Day 2
python src/train.py

# Day 3
python src/evaluate.py
python src/hparam_search.py

# Day 4
python src/visualize.py

# Day 5
# (thesis writing — no scripts)
```

---

## File Tree (final)

```
.
├── init_workspace.py
├── setup.sh
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── grade_a/          ← place your .wav files here
│   │   └── grade_b/          ← place your .wav files here
│   └── processed/
│       ├── X_train.npy
│       ├── y_train.npy
│       ├── X_val.npy
│       ├── y_val.npy
│       ├── X_test.npy
│       └── y_test.npy
├── src/
│   ├── preprocess.py
│   ├── batch_process.py
│   ├── model.py              ← Day 2
│   ├── train.py              ← Day 2
│   ├── evaluate.py           ← Day 3
│   ├── hparam_search.py      ← Day 3
│   └── visualize.py          ← Day 4
├── models/
│   ├── best.keras
│   ├── final_best.keras
│   ├── history.npy
│   ├── results_summary.npy
│   └── training_curves.png
└── figures/                  ← Day 4
    ├── spectrogram_grid.png
    ├── training_curves_final.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── class_distribution.png
    └── misclassifications.png
```
