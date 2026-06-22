# EXECUTION PLAN (5-Day Sprint)

## Brick Quality Assessment via Mel-Spectrogram CNN

**Goal:** Raw `.wav` → Grade (A/B/C) in 5 days, 24/5 crunch mode.

---

## Day 1 — Environment + Audio Ingestion

- [ ] `python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip`
- [ ] `pip install tensorflow librosa numpy soundfile noisereduce matplotlib scikit-learn tqdm`
- [ ] Create folder structure:
  ```
  data/raw/grade_a/  data/raw/grade_b/  data/raw/grade_c/
  data/processed/spectrograms/train/grade_a/
  data/processed/spectrograms/train/grade_b/
  data/processed/spectrograms/train/grade_c/
  data/processed/spectrograms/val/grade_a/
  data/processed/spectrograms/val/grade_b/
  data/processed/spectrograms/val/grade_c/
  data/processed/spectrograms/test/grade_a/
  data/processed/spectrograms/test/grade_b/
  data/processed/spectrograms/test/grade_c/
  src/
  models/
  ```
- [ ] Paste your `.wav` files into `data/raw/grade_*/`
- [ ] Write `src/audio_loader.py` — `load_wav(path)` returns `np.ndarray`

---

## Day 2 — DSP Pipeline (Noise Reduction → Mel-Spectrogram)

- [ ] Write `src/preprocess.py`:
  ```python
  # 1. load_wav
  # 2. noisereduce.reduce_noise(stationary=False, prop_decrease=0.85)
  # 3. librosa.effects.trim(top_db=20)
  # 4. pad/center-crop to 3 seconds (132300 samples)
  # 5. librosa.stft(n_fft=2048, hop_length=512)
  # 6. librosa.feature.melspectrogram(n_mels=128)
  # 7. librosa.power_to_db()
  # 8. tf.image.resize to (128, 128)
  # 9. min-max normalise to [0, 1]
  # Output: numpy array (128, 128, 1)
  ```
- [ ] Test on 3 samples (one per grade), visually confirm specs look different
- [ ] Write `src/batch_process.py` — walks `data/raw/`, calls preprocess on each file, saves `.npy` to `data/processed/`

---

## Day 3 — Batch Processing + Train/Val/Test Split

- [ ] Finish `batch_process.py` — process ALL raw files into `.npy` spectrograms
- [ ] Write `src/split_data.py` — split into train (70%) / val (15%) / test (15%), move `.npy` files into respective folders
- [ ] Print class distribution — ensure balanced
- [ ] Build `src/data_loader.py`:
  ```python
  def load_dataset(split):
      # Walk data/processed/spectrograms/{split}/
      # Load .npy files, pair with labels
      # Return (X, y) numpy arrays
  ```

---

## Day 4 — CNN Training

- [ ] Write `src/model.py`:
  ```python
  # Input: (128, 128, 1)
  # Conv2D 32 (3x3) → BN → MaxPool(2x2)
  # Conv2D 64 (3x3) → BN → MaxPool(2x2)
  # Conv2D 128 (3x3) → BN → MaxPool(2x2)
  # Conv2D 256 (3x3) → BN → MaxPool(2x2)
  # GlobalAvgPool → Dense(256, ReLU) → Dropout(0.5) → Dense(3, Softmax)
  ```
- [ ] Write `src/train.py`:
  ```python
  # compile with Adam(1e-3), categorical_crossentropy
  # callbacks: EarlyStopping(patience=10), ModelCheckpoint(best.weights.h5)
  # fit for 100 epochs (will likely stop earlier)
  ```
- [ ] Run training — monitor val_accuracy
- [ ] Save best model to `models/best.keras`

---

## Day 5 — Evaluation + Inference Script

- [ ] Write `src/evaluate.py`:
  ```python
  # Load test set
  # model.predict → confusion matrix, precision, recall, f1
  # Print classification report
  # Save confusion matrix plot
  ```
- [ ] Write `src/predict.py`:
  ```python
  # python src/predict.py --wav path/to/file.wav
  # Output: "Grade A (94.2%)"
  ```
- [ ] Run inference on 5 test files per grade — verify correctness
- [ ] `git init && git add . && git commit -m "brick-ndt v1"`
- [ ] **DONE.**

---

## File Tree (Final State)

```
.
├── src/
│   ├── audio_loader.py
│   ├── preprocess.py
│   ├── batch_process.py
│   ├── split_data.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── best.keras
├── data/
│   ├── raw/grade_{a,b,c}/
│   └── processed/spectrograms/{train,val,test}/grade_{a,b,c}/
└── requirements.txt
```

No notebooks. No visualisations. No augmentation. No transfer learning. No Grad-CAM. No ablation. One pipeline, one model, one inference script. Done in 5 days.
