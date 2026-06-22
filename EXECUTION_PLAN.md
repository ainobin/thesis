# EXECUTION PLAN & PROJECT ROADMAP

## A Deep Learning-Based NDT Approach for Brick Quality Assessment via Mel-Spectrogram Analysis

**Timeline:** 4 Months (Days 1–120)  
**Core Stack:** Python 3.10+ · Librosa · TensorFlow/Keras · NumPy · Noisereduce · Soundfile · Matplotlib  
**Target Architecture:** 2D-CNN (Transfer Learning: MobileNetV2 / ResNet50 or Custom Optimized CNN)

---

## Table of Contents

1. [System Architecture & Data Pipeline Overview](#1-system-architecture--data-pipeline-overview)
2. [Phase 1: Environment & Workspace Blueprint (Days 1–5)](#2-phase-1-environment--workspace-blueprint-days-15)
3. [Phase 2: Audio Preprocessing & DSP Engine (Days 6–15)](#3-phase-2-audio-preprocessing--dsp-engine-days-615)
4. [Phase 3: Automation & Batch Processing Script (Days 16–25)](#4-phase-3-automation--batch-processing-script-days-1625)
5. [Phase 4: Model Architecture & Training Optimization (Days 26–45)](#5-phase-4-model-architecture--training-optimization-days-2645)
6. [Phase 5: Evaluation, Metrics & Validation (Days 46–60)](#6-phase-5-evaluation-metrics--validation-days-4660)
7. [Phase 6: Thesis Writing & Defense Preparation (Days 61–120)](#7-phase-6-thesis-writing--defense-preparation-days-61120)
8. [Grand Master Checkbox Matrix](#8-grand-master-checkbox-matrix)

---

## 1. System Architecture & Data Pipeline Overview

### Textual Description

The system ingests raw `.wav` recordings of struck bricks, applies digital signal preprocessing to isolate the acoustically salient portion of the impact, converts the time-domain signal into a 2D time–frequency representation (Mel-Spectrogram), and feeds that image-like tensor into a 2D Convolutional Neural Network. The classifier outputs one of three grades: **A** (metallic ring — high quality), **B** (intermediate), or **C** (dull thud — low quality / "picker").

### ASCII Flow Diagram

```
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────────────────┐
│  Raw Audio (.wav) │────>│ Noise Reduction     │────>│ Dynamic Silence Trimming     │
│  fs=44.1kHz mono  │     │ (noisereduceNR)     │     │ (librosa.effects.trim,       │
│  bit-depth=16bit  │     │ stationary=False    │     │  top_db=20, frame_length=256)│
└──────────────────┘     └─────────────────────┘     └──────────┬───────────────────┘
                                                                 │
                                                                 ▼
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────────────────┐
│  Save to Disk    │<────│  Mel-Spectrogram     │<────│  Short-Time Fourier Transform│
│  /data/processed │     │  (librosa.power_to_db│     │  (librosa.stft,              │
│  {grade}_{id}.npy│     │   → librosa.display  │     │   n_fft=2048, hop=512,       │
│  + .png preview  │     │   n_mels=128)        │     │   win_length=2048)           │
└──────────────────┘     └─────────────────────┘     └──────────────────────────────┘
                                                                 │
                                                                 ▼
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────────────────┐
│  Grade Output    │<────│  2D-CNN Classifier   │<────│  Train/Val/Test Split        │
│  (A / B / C)     │     │  Transfer Learning   │     │  (70/15/15) via              │
│  + confidence %  │     │  (MobileNetV2 /      │     │  image_dataset_from_directory│
│                  │     │   ResNet50) + FC head │     │  or custom generator         │
└──────────────────┘     └─────────────────────┘     └──────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Sample rate | 44.1 kHz | Captures full audible range of brick resonance (fundamental ~800 Hz – 4 kHz) |
| Mono channel | Yes | Single impact source; stereo provides no benefit |
| n_fft | 2048 | ~46 Hz frequency resolution at 44.1 kHz — sufficient to resolve formants |
| hop_length | 512 | 75% overlap; ~11.6 ms temporal resolution |
| n_mels | 128 | Balances frequency detail and tensor size for CNN input |
| Spectrogram size | 128×128 (resized) | Standard input for MobileNetV2 / ResNet50; lower resolution for custom nets |
| Normalisation | Min-max per sample | Unit interval [0,1] per spectrogram; preserves relative dynamics |
| Augmentation | Time-stretch + pitch-shift (online) | Small variations (±5%) improve generalisation without destroying physics |

---

## 2. Phase 1: Environment & Workspace Blueprint (Days 1–5)

### Objective

Establish a reproducible, isolated Python environment and a clean directory skeleton. Verify core DSP libraries can read `.wav` files safely.

### Developer TODO List

#### Day 1 — Project Initialisation

- [ ] Create root project directory: `mkdir -p /home/nobin/ProjectWork/Thesis`
- [ ] Initialise git repo: `git init .`
- [ ] Create `.gitignore` (add `__pycache__/`, `*.pyc`, `.env`, `venv/`, `data/`, `models/`, `*.egg-info/`, `*.so`)
- [ ] Create Python virtual environment: `python3.10 -m venv venv`
- [ ] Activate venv: `source venv/bin/activate`
- [ ] Upgrade pip: `pip install --upgrade pip setuptools wheel`

#### Day 2 — Directory Structure

- [ ] Create scaffold:
  ```
  mkdir -p data/raw/grade_a data/raw/grade_b data/raw/grade_c
  mkdir -p data/processed/spectrograms/train/grade_a
  mkdir -p data/processed/spectrograms/train/grade_b
  mkdir -p data/processed/spectrograms/train/grade_c
  mkdir -p data/processed/spectrograms/val/grade_a
  mkdir -p data/processed/spectrograms/val/grade_b
  mkdir -p data/processed/spectrograms/val/grade_c
  mkdir -p data/processed/spectrograms/test/grade_a
  mkdir -p data/processed/spectrograms/test/grade_b
  mkdir -p data/processed/spectrograms/test/grade_c
  mkdir -p notebooks
  mkdir -p src/utils src/preprocessing src/models src/evaluation
  mkdir -p models/checkpoints models/exported logs
  mkdir -p config
  ```
- [ ] Verify structure with `tree -d -L 4` (install `tree` if needed)

#### Day 3 — Dependencies & Setup Script

- [ ] Create `requirements.txt` with pinned versions:
  ```
  tensorflow==2.13.0
  librosa==0.10.1
  numpy==1.24.3
  scipy==1.11.2
  soundfile==0.12.1
  noisereduce==3.0.0
  matplotlib==3.7.2
  scikit-learn==1.3.0
  pandas==2.0.3
  seaborn==0.12.2
  pydub==0.25.1
  tqdm==4.66.1
  ```
- [ ] Create `setup.sh`:
  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  echo "Environment ready."
  ```
- [ ] Make executable: `chmod +x setup.sh`
- [ ] Run `./setup.sh` and confirm no errors

#### Day 4 — Boilerplate Ingestion Verifier

- [ ] Create `src/utils/audio_loader.py`:
  ```python
  import librosa
  import soundfile as sf
  import numpy as np
  from pathlib import Path

  EXPECTED_SR = 44100
  EXPECTED_CHANNELS = 1
  MAX_DURATION_SEC = 5.0

  def verify_wav(filepath: str) -> dict:
      """Validate a .wav file for thesis pipeline compatibility.
      Returns a dict with status, metadata, or error message."""
      fp = Path(filepath)
      if not fp.exists():
          return {"status": "FAIL", "error": "File not found"}
      if fp.suffix.lower() not in (".wav", ".wave"):
          return {"status": "FAIL", "error": "Not a .wav file"}
      try:
          y, sr = librosa.load(str(fp), sr=None, mono=False)
      except Exception as e:
          return {"status": "FAIL", "error": f"librosa.load failed: {e}"}

      if y.ndim == 2:
          return {"status": "FAIL", "error": f"Stereo file ({y.shape[0]} channels); mono required"}
      duration = librosa.get_duration(y=y, sr=sr)
      result = {
          "status": "PASS",
          "file": fp.name,
          "sample_rate": sr,
          "channels": 1,
          "samples": int(y.shape[0]),
          "duration_sec": round(duration, 3),
          "duration_ok": duration <= MAX_DURATION_SEC,
          "sr_ok": sr == EXPECTED_SR,
      }
      return result

  def load_safe(filepath: str, sr: int = EXPECTED_SR) -> np.ndarray:
      """Load audio, convert to mono, resample to target sr. Returns float32 array."""
      y, _ = librosa.load(filepath, sr=sr, mono=True)
      return y.astype(np.float32)
  ```
- [ ] Write `src/utils/__init__.py` (empty or re-export)
- [ ] Create `config/paths.py`:
  ```python
  from pathlib import Path

  BASE = Path(__file__).resolve().parent.parent
  DATA_RAW = BASE / "data" / "raw"
  DATA_PROCESSED = BASE / "data" / "processed"
  SPECTROGRAM_DIR = DATA_PROCESSED / "spectrograms"
  MODEL_DIR = BASE / "models"
  CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
  EXPORT_DIR = MODEL_DIR / "exported"
  LOG_DIR = BASE / "logs"
  SRC_DIR = BASE / "src"
  CONFIG_DIR = BASE / "config"

  GRADES = ["grade_a", "grade_b", "grade_c"]
  CLASS_MAP = {"grade_a": 0, "grade_b": 1, "grade_c": 2}
  CLASS_MAP_INV = {v: k for k, v in CLASS_MAP.items()}
  ```

#### Day 5 — Sanity Check & First Notebook

- [ ] Create `notebooks/00_environment_sanity.ipynb`:
  ```python
  # Cell 1 — imports
  import sys, librosa, tensorflow as tf, numpy as np, soundfile as sf
  print(f"Python {sys.version}")
  print(f"Librosa {librosa.__version__}")
  print(f"TensorFlow {tf.__version__}")
  print(f"NumPy {np.__version__}")

  # Cell 2 — load and play a test .wav
  from src.utils.audio_loader import verify_wav, load_safe
  # (copy a file into data/raw/test/ or use one provided)
  result = verify_wav("data/raw/grade_a/sample_a_001.wav")
  print(result)

  # Cell 3 — compute STFT
  y = load_safe("data/raw/grade_a/sample_a_001.wav")
  D = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048)
  mag = np.abs(D)
  print(f"STFT shape: {mag.shape}")

  # Cell 4 — mel spectrogram
  mel = librosa.feature.melspectrogram(S=mag**2, sr=44100, n_mels=128)
  log_mel = librosa.power_to_db(mel, ref=np.max)
  print(f"Mel-spec shape: {log_mel.shape}")
  ```
- [ ] Place one sample `.wav` per grade in `data/raw/grade_*/`
- [ ] Commit everything: `git add . && git commit -m "Phase 1: environment, scaffold, audio verifier"`

---

## 3. Phase 2: Audio Preprocessing & DSP Engine (Days 6–15)

### Objective

Build a robust DSP pipeline that converts raw `.wav` files into clean, standardised Mel-Spectrogram tensors ready for CNN ingestion. Every step must be deterministic and logged.

### Algorithm Pipeline (Pseudocode)

```
Input:  raw_wav (float32[N], sr=44100)
Output: log_mel (float32[128, T])

1. NOISE_REDUCTION:
   - Sample first 0.5 s (assumed silence / ambient) as noise clip
   - Apply noisereduce.reduce_noise(y=y, sr=sr, y_noise=noise_clip,
       stationary=False, prop_decrease=0.85, n_fft=2048, hop_length=512)
   - Output: y_clean

2. SILENCE_TRIMMING:
   - librosa.effects.trim(y_clean, top_db=20, frame_length=256, hop_length=64)
   - Output: y_trimmed, [start, end] indices
   - If trimmed duration < 0.1 s: discard sample (corrupt recording)

3. PADDING / TRUNCATION:
   - Target length = sr * 3.0  (3-second fixed window)
   - If len(y_trimmed) < target: pad symmetrically with 0s
   - If len(y_trimmed) > target: centre-crop
   - Output: y_fixed (float32[132300]) — 44 100 × 3

4. MEL-SPECTROGRAM:
   - D = librosa.stft(y_fixed, n_fft=2048, hop_length=512, win_length=2048)
   - S = np.abs(D) ** 2
   - mel_S = librosa.feature.melspectrogram(S=S, sr=44100, n_mels=128,
       fmin=0, fmax=22050)
   - log_mel = librosa.power_to_db(mel_S, ref=np.max)
   - Output: log_mel (float32[128, T])  — T ≈ 259 for 3 s @ hop=512

5. RESIZE (if required by model):
   - tf.image.resize(..., (128, 128)) — bilinear, antialias
   - Output: log_mel_resized (float32[128, 128])

6. NORMALISE:
   - log_mel_resized = (log_mel_resized - log_mel_resized.min()) /
     (log_mel_resized.max() - log_mel_resized.min() + 1e-8)
   - Output: float32[128, 128] in [0, 1]
```

### Developer TODO List

#### Day 6 — Noise Reduction Module

- [ ] Create `src/preprocessing/noise_reduction.py`:
  ```python
  import numpy as np
  import noisereduce as nr

  def reduce_noise(y: np.ndarray, sr: int = 44100, noise_duration: float = 0.5,
                   prop_decrease: float = 0.85) -> np.ndarray:
      """Reduce stationary + non-stationary noise using spectral gating.
      Uses the first `noise_duration` seconds as the noise profile."""
      noise_len = int(sr * noise_duration)
      noise_clip = y[:noise_len] if len(y) > noise_len else y
      y_clean = nr.reduce_noise(
          y=y, sr=sr, y_noise=noise_clip,
          stationary=False, prop_decrease=prop_decrease,
          n_fft=2048, hop_length=512, win_length=2048,
      )
      return y_clean.astype(np.float32)
  ```

#### Day 7 — Silence Trimming & Length Normalisation

- [ ] Create `src/preprocessing/trim_pad.py`:
  ```python
  import librosa
  import numpy as np

  TARGET_DURATION_SEC = 3.0
  TARGET_SR = 44100

  def trim_silence(y: np.ndarray, top_db: int = 20) -> np.ndarray:
      y_trimmed, _ = librosa.effects.trim(y, top_db=top_db,
                                           frame_length=256, hop_length=64)
      if len(y_trimmed) < 0.1 * TARGET_SR:
          raise ValueError(f"Trimmed audio too short ({len(y_trimmed)/TARGET_SR:.3f}s)")
      return y_trimmed

  def fix_length(y: np.ndarray, target_len: int = TARGET_DURATION_SEC * TARGET_SR) -> np.ndarray:
      if len(y) < target_len:
          pad = target_len - len(y)
          y = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
      elif len(y) > target_len:
          start = (len(y) - target_len) // 2
          y = y[start:start + target_len]
      return y.astype(np.float32)
  ```

#### Day 8 — Mel-Spectrogram Computation

- [ ] Create `src/preprocessing/spectrogram.py`:
  ```python
  import librosa
  import numpy as np
  import tensorflow as tf

  N_FFT = 2048
  HOP_LENGTH = 512
  WIN_LENGTH = 2048
  N_MELS = 128
  TARGET_SIZE = (128, 128)

  def compute_mel_spec(y: np.ndarray, sr: int = 44100) -> np.ndarray:
      D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       win_length=WIN_LENGTH)
      S = np.abs(D) ** 2
      mel_S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=N_MELS,
                                             fmin=0, fmax=sr // 2)
      log_mel = librosa.power_to_db(mel_S, ref=np.max)
      return log_mel.astype(np.float32)

  def resize_spec(spec: np.ndarray, size: tuple = TARGET_SIZE) -> np.ndarray:
      spec_tensor = tf.convert_to_tensor(spec[..., np.newaxis], dtype=tf.float32)
      resized = tf.image.resize(spec_tensor, size, method="bilinear",
                                antialias=True)
      return resized.numpy().squeeze()

  def normalise_spec(spec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
      s_min, s_max = spec.min(), spec.max()
      return (spec - s_min) / (s_max - s_min + eps)
  ```

#### Day 9 — Full Pipeline Orchestrator

- [ ] Create `src/preprocessing/pipeline.py`:
  ```python
  import numpy as np
  from src.utils.audio_loader import load_safe
  from src.preprocessing.noise_reduction import reduce_noise
  from src.preprocessing.trim_pad import trim_silence, fix_length
  from src.preprocessing.spectrogram import compute_mel_spec, resize_spec, normalise_spec

  def audio_to_spectrogram(filepath: str, sr: int = 44100) -> np.ndarray:
      """End-to-end: .wav → normalised 128×128 Mel-Spectrogram."""
      y = load_safe(filepath, sr=sr)                    # 1. Load
      y = reduce_noise(y, sr=sr)                         # 2. Denoise
      y = trim_silence(y)                                # 3. Trim
      y = fix_length(y)                                  # 4. Fix length
      spec = compute_mel_spec(y, sr=sr)                  # 5. STFT → Mel
      spec = resize_spec(spec)                            # 6. Resize
      spec = normalise_spec(spec)                         # 7. Normalise
      return spec  # shape (128, 128), dtype float32, range [0, 1]
  ```

#### Day 10 — Visualisation & Quality Check Utilities

- [ ] Create `src/utils/visualise.py`:
  ```python
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import librosa.display
  import numpy as np
  from pathlib import Path

  def save_spec_figure(spec: np.ndarray, save_path: str, title: str = "") -> None:
      """Save a borderless, axis-less spectrogram figure for CNN input preview."""
      fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
      librosa.display.specshow(spec, cmap="magma", ax=ax)
      ax.axis("off")
      fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
      fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
      plt.close(fig)

  def plot_waveform(y: np.ndarray, sr: int, save_path: str) -> None:
      fig, ax = plt.subplots(figsize=(10, 3))
      librosa.display.waveshow(y, sr=sr, ax=ax)
      ax.set_xlabel("Time (s)")
      ax.set_ylabel("Amplitude")
      fig.savefig(save_path, dpi=150)
      plt.close(fig)
  ```

#### Day 11 — Unit Tests for DSP Modules

- [ ] Create `tests/test_preprocessing.py`:
  ```python
  import pytest
  import numpy as np
  import soundfile as sf
  import tempfile, os
  from src.preprocessing.noise_reduction import reduce_noise
  from src.preprocessing.trim_pad import trim_silence, fix_length
  from src.preprocessing.spectrogram import compute_mel_spec, resize_spec, normalise_spec

  def _sine_wave(freq=800, dur=1.0, sr=44100):
      t = np.linspace(0, dur, int(sr * dur), endpoint=False)
      return np.sin(2 * np.pi * freq * t).astype(np.float32)

  def test_fix_length_pad():
      y = _sine_wave(dur=0.5)
      y_fixed = fix_length(y, target_len=44100 * 3)
      assert len(y_fixed) == 44100 * 3

  def test_fix_length_crop():
      y = _sine_wave(dur=5.0)
      y_fixed = fix_length(y, target_len=44100 * 3)
      assert len(y_fixed) == 44100 * 3

  def test_mel_spec_shape():
      y = _sine_wave(dur=3.0)
      spec = compute_mel_spec(y)
      assert spec.shape[0] == 128  # n_mels

  def test_resize_spec():
      y = _sine_wave(dur=3.0)
      spec = compute_mel_spec(y)
      resized = resize_spec(spec, size=(128, 128))
      assert resized.shape == (128, 128)

  def test_normalise_range():
      y = _sine_wave(dur=3.0)
      spec = compute_mel_spec(y)
      normed = normalise_spec(spec)
      assert normed.min() >= 0.0 and normed.max() <= 1.0
  ```
- [ ] Run `pytest tests/test_preprocessing.py -v` and fix failures
- [ ] Create `tests/conftest.py` with shared fixtures

#### Day 12 — End-to-End Pipeline Validation

- [ ] Process 3 samples per grade through `audio_to_spectrogram`
- [ ] Save preprocessed outputs to `data/processed/spectrograms/train/grade_*/*.npy`
- [ ] Generate `.png` previews with `save_spec_figure` and visually inspect for:
  - **Grade A:** Distinct harmonic rows across multiple frequency bands; long decay tail
  - **Grade C:** Energy concentrated in low frequencies; short decay; no harmonic structure
- [ ] Log any failures in `logs/pipeline_validation.log`

#### Day 13 — Edge Case Handling

- [ ] Add try/except guards in `pipeline.py` for:
  ```python
  except sf.LibsndfileError:
      # corrupt file; log and skip
  except ValueError as e:
      # too-short audio after trim; log and skip
  except Exception as e:
      # unexpected; log full trace
  ```
- [ ] Create `src/preprocessing/__init__.py` re-exporting `audio_to_spectrogram`
- [ ] Write corrupt-wav handler in `audio_loader.py`:
  ```python
  def is_corrupt(filepath: str) -> bool:
      try:
          with sf.SoundFile(filepath) as f:
              _ = f.read(1)
          return False
      except Exception:
          return True
  ```

#### Day 14 — Param Sweep Documentation

- [ ] Create `notebooks/01_param_sweep.ipynb`:
  - Sweep `n_mels` ∈ {64, 128, 256}
  - Sweep `hop_length` ∈ {256, 512, 1024}
  - Sweep `target_duration` ∈ {1.5, 3.0, 5.0}
  - For each combo, compute mean spectrogram and note:
    - Frequency resolution vs. temporal resolution trade-off
    - Tensor size impact on GPU memory
  - [ ] Select winning params and hard-code them in `spectrogram.py`

#### Day 15 — Phase 2 Code Freeze

- [ ] Run `pytest tests/ -v --tb=short` — all green
- [ ] `cd src && python -c "from preprocessing.pipeline import audio_to_spectrogram; print('Phase 2 OK')"`
- [ ] `git add . && git commit -m "Phase 2: DSP pipeline — denoise, trim, mel-spec, resize, normalise"`
- [ ] Tag: `git tag phase2-complete`

---

## 4. Phase 3: Automation & Batch Processing Script (Days 16–25)

### Objective

Build a production-grade batch processor that walks the raw data directory tree, applies the DSP pipeline to every file, logs all operations, and automatically splits the results into train / val / test sets.

### Manifest CSV Schema

| Column | Type | Description |
|---|---|---|
| `uid` | str | Unique ID, e.g. `grade_a_00042` |
| `source_file` | str | Absolute path to original `.wav` |
| `grade` | str | `grade_a`, `grade_b`, or `grade_c` |
| `duration_sec` | float | Duration of the trimmed, fixed-length audio |
| `spec_path` | str | Relative path to saved `.npy` |
| `split` | str | `train` / `val` / `test` |
| `status` | str | `ok`, `corrupt`, `too_short`, `error` |
| `error_msg` | str | Empty or error description |

### Developer TODO List

#### Day 16 — Inventory & Manifest Writer

- [ ] Create `src/preprocessing/manifest.py`:
  ```python
  import csv, uuid, time, os
  from pathlib import Path
  from typing import Optional
  from config.paths import GRADES, DATA_RAW

  MANIFEST_HEADER = [
      "uid", "source_file", "grade", "duration_sec",
      "spec_path", "split", "status", "error_msg"
  ]
  MANIFEST_FILENAME = "manifest.csv"

  def generate_uid(prefix: str = "brick") -> str:
      ts = int(time.time() * 1_000_000)
      return f"{prefix}_{ts}"

  def init_manifest(overwrite: bool = False) -> str:
      from config.paths import DATA_PROCESSED
      path = str(DATA_PROCESSED / MANIFEST_FILENAME)
      if not overwrite and os.path.exists(path):
          return path
      with open(path, "w", newline="") as f:
          writer = csv.writer(f)
          writer.writerow(MANIFEST_HEADER)
      return path

  def append_manifest_row(manifest_path: str, row: dict) -> None:
      with open(manifest_path, "a", newline="") as f:
          writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
          writer.writerow(row)
  ```

#### Day 17 — Directory Scanner

- [ ] Create `src/preprocessing/scanner.py`:
  ```python
  from pathlib import Path
  from typing import List, Tuple
  from config.paths import DATA_RAW, GRADES

  def scan_raw_directory() -> List[Tuple[str, str]]:
      """Walk data/raw/ and return list of (filepath, grade) tuples.
      Only returns .wav files in recognized grade directories."""
      files = []
      for grade in GRADES:
          grade_dir = DATA_RAW / grade
          if not grade_dir.exists():
              continue
          for fpath in sorted(grade_dir.glob("*.wav")):
              files.append((str(fpath.resolve()), grade))
          for fpath in sorted(grade_dir.glob("*.WAV")):
              files.append((str(fpath.resolve()), grade))
      return files
  ```

#### Day 18 — Batch Processor Core

- [ ] Create `src/preprocessing/batch_processor.py`:
  ```python
  import os, sys, time, traceback
  import numpy as np
  from pathlib import Path
  from tqdm import tqdm
  from concurrent.futures import ProcessPoolExecutor, as_completed

  from src.utils.audio_loader import load_safe, is_corrupt
  from src.preprocessing.pipeline import audio_to_spectrogram
  from src.preprocessing.manifest import init_manifest, append_manifest_row, generate_uid
  from src.preprocessing.scanner import scan_raw_directory
  from config.paths import DATA_PROCESSED, SPECTROGRAM_DIR, CLASS_MAP

  NUM_WORKERS = 4  # adjust based on CPU core count

  def _process_one(args):
      source_path, grade = args
      result = {
          "uid": generate_uid(),
          "source_file": source_path,
          "grade": grade,
          "duration_sec": "",
          "spec_path": "",
          "split": "",
          "status": "error",
          "error_msg": "",
      }
      try:
          if is_corrupt(source_path):
              raise ValueError("Corrupt WAV file")

          y = load_safe(source_path)
          result["duration_sec"] = round(len(y) / 44100, 3)

          spec = audio_to_spectrogram(source_path)  # (128, 128)

          # Save .npy
          grade_short = grade.replace("grade_", "")
          out_dir = SPECTROGRAM_DIR / "unassigned" / grade
          out_dir.mkdir(parents=True, exist_ok=True)
          npy_path = out_dir / f"{result['uid']}.npy"
          np.save(str(npy_path), spec)

          result["spec_path"] = str(npy_path.relative_to(DATA_PROCESSED))
          result["status"] = "ok"

      except Exception as e:
          result["error_msg"] = f"{type(e).__name__}: {e}"
      return result

  def run_batch():
      print("[Batch Processor] Scanning raw directory...")
      files = scan_raw_directory()
      print(f"[Batch Processor] Found {len(files)} files.")

      manifest_path = init_manifest(overwrite=True)
      print(f"[Batch Processor] Manifest: {manifest_path}")

      results = []
      with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
          futures = {pool.submit(_process_one, f): f for f in files}
          for future in tqdm(as_completed(futures), total=len(files),
                             desc="Processing", unit="file"):
              results.append(future.result())

      for row in results:
          append_manifest_row(manifest_path, row)

      ok_count = sum(1 for r in results if r["status"] == "ok")
      err_count = sum(1 for r in results if r["status"] == "error")
      print(f"[Batch Processor] Done. OK={ok_count}  Error={err_count}")
      return manifest_path
  ```

#### Day 19 — CLI Entry Point

- [ ] Create `src/preprocessing/cli.py`:
  ```python
  import argparse
  from src.preprocessing.batch_processor import run_batch

  def main():
      parser = argparse.ArgumentParser(description="BrickN DT Batch Audio Processor")
      parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
      parser.add_argument("--overwrite-manifest", action="store_true", default=True)
      args = parser.parse_args()

      import src.preprocessing.batch_processor as bp
      bp.NUM_WORKERS = args.workers
      run_batch()

  if __name__ == "__main__":
      main()
  ```
- [ ] Make it runnable: `python -m src.preprocessing.cli --workers 4`

#### Day 20 — Train/Val/Test Splitter

- [ ] Create `src/preprocessing/splitter.py`:
  ```python
  import csv, random, shutil
  from pathlib import Path
  from config.paths import SPECTROGRAM_DIR, GRADES

  SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
  RANDOM_SEED = 42

  def load_manifest_ok_rows(manifest_path: str):
      rows = []
      with open(manifest_path, newline="") as f:
          reader = csv.DictReader(f)
          for row in reader:
              if row["status"] == "ok" and row.get("split", "") == "":
                  rows.append(row)
      return rows

  def assign_splits(rows):
      random.seed(RANDOM_SEED)
      by_grade = {}
      for r in rows:
          by_grade.setdefault(r["grade"], []).append(r)

      for grade, grade_rows in by_grade.items():
          random.shuffle(grade_rows)
          n = len(grade_rows)
          n_train = int(n * SPLIT_RATIOS["train"])
          n_val = int(n * SPLIT_RATIOS["val"])
          for i, r in enumerate(grade_rows):
              if i < n_train:
                  r["split"] = "train"
              elif i < n_train + n_val:
                  r["split"] = "val"
              else:
                  r["split"] = "test"
      return rows

  def reorganise_files(rows, manifest_path: str):
      for r in rows:
          src = SPECTROGRAM_DIR / "unassigned" / r["grade"] / f"{r['uid']}.npy"
          if not src.exists():
              continue
          dst_dir = SPECTROGRAM_DIR / r["split"] / r["grade"]
          dst_dir.mkdir(parents=True, exist_ok=True)
          dst = dst_dir / f"{r['uid']}.npy"
          shutil.move(str(src), str(dst))
          r["spec_path"] = str(dst.relative_to(SPECTROGRAM_DIR.parent))

      # Rewrite manifest with split assigned
      with open(manifest_path, "w", newline="") as f:
          writer = csv.DictWriter(f, fieldnames=[
              "uid", "source_file", "grade", "duration_sec",
              "spec_path", "split", "status", "error_msg",
          ])
          writer.writeheader()
          writer.writerows(rows)

      # Clean up unassigned dirs if empty
      for grade in GRADES:
          d = SPECTROGRAM_DIR / "unassigned" / grade
          if d.exists() and not any(d.iterdir()):
              d.rmdir()

  def run_split(manifest_path: str):
      rows = load_manifest_ok_rows(manifest_path)
      rows = assign_splits(rows)
      reorganise_files(rows, manifest_path)
      counts = {"train": 0, "val": 0, "test": 0}
      for r in rows:
          counts[r["split"]] += 1
      print(f"[Splitter] train={counts['train']}  val={counts['val']}  test={counts['test']}")
  ```

#### Day 21 — Dry Run on Real Data

- [ ] Place minimum 30 `.wav` files (10 per grade) in `data/raw/grade_*/`
- [ ] Run: `python -m src.preprocessing.cli --workers 2`
- [ ] Run: `python -c "from src.preprocessing.splitter import run_split; run_split('data/processed/manifest.csv')"`
- [ ] Verify directory tree under `data/processed/spectrograms/`

#### Day 22 — Progress Bar & Logging

- [ ] Add `rich` progress bar to `batch_processor.py` (or keep `tqdm`)
- [ ] Add `logging` config to `config/logging_config.py`:
  ```python
  import logging

  def setup_logger(name: str = "brick_ndt", level=logging.INFO):
      logger = logging.getLogger(name)
      logger.setLevel(level)
      if not logger.handlers:
          fh = logging.FileHandler("logs/pipeline.log")
          fh.setFormatter(logging.Formatter(
              "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
          logger.addHandler(fh)
          sh = logging.StreamHandler()
          sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
          logger.addHandler(sh)
      return logger
  ```
- [ ] Integrate logger into `batch_processor.py` (log each file + aggregate summary)

#### Day 23 — Error Recovery & Resumability

- [ ] Modify `init_manifest` to append (not overwrite) by default
- [ ] Add `--resume` flag to `cli.py` that skips files already in manifest
  ```python
  def get_processed_uids(manifest_path: str) -> set:
      uids = set()
      with open(manifest_path, newline="") as f:
          for row in csv.DictReader(f):
              uids.add(row["uid"])
      return uids
  ```
- [ ] Filter `scan_raw_directory` results against already-processed source paths
- [ ] Test resumability by killing mid-run, re-running with `--resume`

#### Day 24 — Data Augmentation (Online, in Dataset Loader)

- [ ] Create `src/models/data_augmentation.py`:
  ```python
  import tensorflow as tf

  @tf.function
  def augment_spectrogram(spec: tf.Tensor) -> tf.Tensor:
      """Online augmentation: time-stretch, freq masking, time masking."""
      spec = tf.expand_dims(spec, axis=-1)  # (H, W, 1)
      # Frequency masking (SpecAugment style)
      spec = tf.image.random_crop(spec, size=(128, 128, 1))
      # Random frequency shift via roll
      shift = tf.random.uniform([], minval=-8, maxval=8, dtype=tf.int32)
      spec = tf.roll(spec, shift=shift, axis=1)
      # Gaussian noise (very small)
      noise = tf.random.normal(tf.shape(spec), mean=0.0, stddev=0.005)
      spec = tf.clip_by_value(spec + noise, 0.0, 1.0)
      return tf.squeeze(spec, axis=-1)
  ```

#### Day 25 — Phase 3 Code Freeze & Stress Test

- [ ] Populate `data/raw/` with ≥ 100 `.wav` files across all grades
- [ ] Run full batch: `python -m src.preprocessing.cli --workers 8`
- [ ] Run splitter: `python -c "from src.preprocessing.splitter import run_split; run_split('data/processed/manifest.csv')"`
- [ ] Count files per split: `find data/processed/spectrograms -name '*.npy' | wc -l`
- [ ] `git add . && git commit -m "Phase 3: batch processor, manifest, splitter, resume, augmentation"`
- [ ] `git tag phase3-complete`

---

## 5. Phase 4: Model Architecture & Training Optimization (Days 26–45)

### Objective

Design, train, and optimise a 2D-CNN classifier. Experiment with transfer learning (MobileNetV2 / ResNet50) vs. a custom lightweight CNN. Implement early stopping, checkpointing, LR scheduling, and TensorBoard logging.

### Architecture Blueprint — Custom CNN (Fallback)

```
Input:        (128, 128, 1) — float32 [0, 1]

Conv2D 1:     32 filters, 3×3, ReLU, padding="same"
BatchNorm + MaxPool2D(2×2)           →  (64, 64, 32)

Conv2D 2:     64 filters, 3×3, ReLU, padding="same"
BatchNorm + MaxPool2D(2×2)           →  (32, 32, 64)

Conv2D 3:     128 filters, 3×3, ReLU, padding="same"
BatchNorm + MaxPool2D(2×2)           →  (16, 16, 128)

Conv2D 4:     256 filters, 3×3, ReLU, padding="same"
BatchNorm + MaxPool2D(2×2)           →  (8, 8, 256)

GlobalAveragePooling2D                →  (256)

Dense(256, ReLU) + Dropout(0.5)
Dense(128, ReLU) + Dropout(0.3)
Dense(3, Softmax)                     →  (3)
```

### Transfer Learning Blueprint

```
Input:        (128, 128, 3) — convert grayscale → 3-channel via tf.tile
Backbone:     MobileNetV2 (weights="imagenet", include_top=False)
              → Freeze all layers
GAP:          GlobalAveragePooling2D
Dense(256) + BatchNorm + Dropout(0.5)
Dense(3, Softmax)
```

### Developer TODO List

#### Day 26 — Data Loader (NumPy + tf.data)

- [ ] Create `src/models/data_loader.py`:
  ```python
  import numpy as np
  import tensorflow as tf
  from pathlib import Path
  from typing import Tuple, Optional
  from config.paths import SPECTROGRAM_DIR, CLASS_MAP, CLASS_MAP_INV

  def _load_npy(path: str) -> np.ndarray:
      spec = np.load(path.decode()) if isinstance(path, bytes) else np.load(path)
      return spec.astype(np.float32)

  def _parse_example(spec_path: str, label: int) -> Tuple[tf.Tensor, tf.Tensor]:
      spec = tf.numpy_function(_load_npy, [spec_path], tf.float32)
      spec.set_shape((128, 128))
      spec = tf.expand_dims(spec, axis=-1)  # (128, 128, 1)
      label = tf.one_hot(label, depth=len(CLASS_MAP))
      return spec, label

  def create_dataset(split: str, batch_size: int = 32, shuffle: bool = True,
                     augment: bool = False) -> tf.data.Dataset:
      split_dir = SPECTROGRAM_DIR / split
      specs, labels = [], []
      for grade, label in CLASS_MAP.items():
          grade_dir = split_dir / grade
          if not grade_dir.exists():
              continue
          for fpath in sorted(grade_dir.glob("*.npy")):
              specs.append(str(fpath.resolve()))
              labels.append(label)

      ds = tf.data.Dataset.from_tensor_slices((specs, labels))
      ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)

      if augment:
          from src.models.data_augmentation import augment_spectrogram
          ds = ds.map(lambda x, y: (augment_spectrogram(x), y),
                      num_parallel_calls=tf.data.AUTOTUNE)

      if shuffle:
          ds = ds.shuffle(buffer_size=len(specs), seed=42)
      ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
      return ds
  ```

#### Day 27 — Custom CNN Model Builder

- [ ] Create `src/models/custom_cnn.py`:
  ```python
  import tensorflow as tf
  from tensorflow.keras import layers, Model

  def build_custom_cnn(input_shape=(128, 128, 1), num_classes: int = 3) -> Model:
      inputs = layers.Input(shape=input_shape, dtype=tf.float32)

      x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
      x = layers.BatchNormalization()(x)
      x = layers.MaxPool2D(2)(x)               # (64, 64, 32)

      x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
      x = layers.BatchNormalization()(x)
      x = layers.MaxPool2D(2)(x)               # (32, 32, 64)

      x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
      x = layers.BatchNormalization()(x)
      x = layers.MaxPool2D(2)(x)               # (16, 16, 128)

      x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
      x = layers.BatchNormalization()(x)
      x = layers.MaxPool2D(2)(x)               # (8, 8, 256)

      x = layers.GlobalAveragePooling2D()(x)    # (256)
      x = layers.Dense(256, activation="relu")(x)
      x = layers.Dropout(0.5)(x)
      x = layers.Dense(128, activation="relu")(x)
      x = layers.Dropout(0.3)(x)
      outputs = layers.Dense(num_classes, activation="softmax")(x)

      return Model(inputs, outputs, name="custom_cnn")
  ```

#### Day 28 — Transfer Learning Model Builder

- [ ] Create `src/models/transfer_model.py`:
  ```python
  import tensorflow as tf
  from tensorflow.keras import layers, Model

  def build_transfer_model(
      backbone_name: str = "MobileNetV2",
      input_shape=(128, 128, 3),
      num_classes: int = 3,
      freeze_backbone: bool = True,
  ) -> Model:
      # Build pretrained backbone
      if backbone_name == "MobileNetV2":
          base = tf.keras.applications.MobileNetV2(
              weights="imagenet", include_top=False, input_shape=input_shape
          )
      elif backbone_name == "ResNet50":
          base = tf.keras.applications.ResNet50(
              weights="imagenet", include_top=False, input_shape=input_shape
          )
      else:
          raise ValueError(f"Unknown backbone: {backbone_name}")

      if freeze_backbone:
          base.trainable = False

      # Grayscale → 3-channel by tiling
      inputs = layers.Input(shape=(128, 128, 1), dtype=tf.float32)
      x = layers.Concatenate(axis=-1)([inputs, inputs, inputs])  # (128, 128, 3)

      # Normalise for imagenet stats
      x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

      x = base(x, training=False)
      x = layers.GlobalAveragePooling2D()(x)
      x = layers.Dense(256, activation="relu")(x)
      x = layers.BatchNormalization()(x)
      x = layers.Dropout(0.5)(x)
      outputs = layers.Dense(num_classes, activation="softmax")(x)

      return Model(inputs, outputs, name=f"{backbone_name}_transfer")
  ```

#### Day 29 — Training Script with Callbacks

- [ ] Create `src/models/train.py`:
  ```python
  import os, argparse, json
  import tensorflow as tf
  from datetime import datetime
  from config.paths import CHECKPOINT_DIR, LOG_DIR
  from src.models.data_loader import create_dataset
  from src.models.custom_cnn import build_custom_cnn
  from src.models.transfer_model import build_transfer_model

  def get_callbacks(model_name: str):
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      ckpt_path = CHECKPOINT_DIR / f"{model_name}_{timestamp}" / "best.weights.h5"
      log_path = LOG_DIR / f"{model_name}_{timestamp}"

      return [
          tf.keras.callbacks.EarlyStopping(
              monitor="val_loss", patience=15, restore_best_weights=True,
              verbose=1,
          ),
          tf.keras.callbacks.ModelCheckpoint(
              filepath=str(ckpt_path),
              monitor="val_accuracy", save_best_only=True,
              save_weights_only=True, verbose=1,
          ),
          tf.keras.callbacks.ReduceLROnPlateau(
              monitor="val_loss", factor=0.5, patience=5,
              min_lr=1e-7, verbose=1,
          ),
          tf.keras.callbacks.CSVLogger(
              filename=str(log_path / "training_log.csv"),
          ),
          tf.keras.callbacks.TensorBoard(
              log_dir=str(log_path), histogram_freq=1, write_graph=True,
          ),
      ]

  def compile_model(model, lr: float = 1e-3):
      model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
          loss="categorical_crossentropy",
          metrics=["accuracy", tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
      )
      return model

  def train(model_name: str = "custom_cnn", backbone: str = "",
            batch_size: int = 32, epochs: int = 100, lr: float = 1e-3,
            freeze: bool = True):
      # Model selection
      if model_name == "custom_cnn":
          model = build_custom_cnn()
      elif model_name == "transfer":
          model = build_transfer_model(backbone_name=backbone or "MobileNetV2",
                                       freeze_backbone=freeze)
      else:
          raise ValueError(f"Unknown model: {model_name}")

      model.summary()
      compile_model(model, lr=lr)

      # Data
      train_ds = create_dataset("train", batch_size=batch_size, shuffle=True, augment=True)
      val_ds = create_dataset("val", batch_size=batch_size, shuffle=False, augment=False)

      callbacks = get_callbacks(model_name)
      history = model.fit(
          train_ds, validation_data=val_ds,
          epochs=epochs, callbacks=callbacks, verbose=1,
      )

      # Save training history
      model_name_ = model.name
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      hist_path = LOG_DIR / f"{model_name_}_{timestamp}" / "history.json"
      hist_path.parent.mkdir(parents=True, exist_ok=True)
      with open(hist_path, "w") as f:
          json.dump(history.history, f, indent=2)

      return model, history
  ```

#### Day 30 — Experiment Tracking Config

- [ ] Create `config/experiment_config.yaml` (optional, or keep as Python dict):
  ```python
  # config/training_config.py
  EXPERIMENTS = {
      "custom_cnn_baseline": {
          "model_name": "custom_cnn",
          "batch_size": 32,
          "epochs": 100,
          "lr": 1e-3,
          "augment": True,
      },
      "mobilenetv2_frozen": {
          "model_name": "transfer",
          "backbone": "MobileNetV2",
          "freeze": True,
          "batch_size": 32,
          "epochs": 80,
          "lr": 1e-3,
      },
      "mobilenetv2_finetune": {
          "model_name": "transfer",
          "backbone": "MobileNetV2",
          "freeze": False,
          "batch_size": 16,
          "epochs": 120,
          "lr": 5e-5,  # low LR for fine-tuning
      },
      "resnet50_frozen": {
          "model_name": "transfer",
          "backbone": "ResNet50",
          "freeze": True,
          "batch_size": 32,
          "epochs": 80,
          "lr": 1e-3,
      },
  }
  ```

#### Day 31 — GPU / MPS Setup & Mixed Precision

- [ ] Configure GPU memory growth in `src/utils/tf_config.py`:
  ```python
  import tensorflow as tf

  def configure_gpu():
      gpus = tf.config.list_physical_devices("GPU")
      if gpus:
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
          print(f"[TF Config] {len(gpus)} GPU(s) configured with memory growth")
      else:
          print("[TF Config] No GPU found; running on CPU")

  def enable_mixed_precision():
      tf.keras.mixed_precision.set_global_policy("mixed_float16")
      print("[TF Config] Mixed precision (float16) enabled")
  ```
- [ ] Call `configure_gpu()` at the top of `train.py`

#### Day 32 — First Training Run

- [ ] Run custom CNN baseline:
  ```bash
  python -m src.models.train \
      --model custom_cnn \
      --batch-size 32 \
      --epochs 100 \
      --lr 1e-3
  ```
- [ ] Monitor with TensorBoard: `tensorboard --logdir logs`

#### Day 33 — Transfer Learning Training: MobileNetV2 (Frozen)

- [ ] Run:
  ```bash
  python -m src.models.train \
      --model transfer \
      --backbone MobileNetV2 \
      --freeze \
      --batch-size 32 \
      --epochs 80 \
      --lr 1e-3
  ```

#### Day 34 — Transfer Learning Fine-Tuning

- [ ] Load frozen model weights, unfreeze backbone, re-compile with low LR:
  ```python
  # In src/models/train.py, add a --finetune-from flag
  # 1. Load best weights from frozen run
  # 2. Unfreeze all layers
  # 3. Re-compile with lr=5e-5
  # 4. Train for another 40 epochs
  ```
- [ ] Create `src/models/finetune.py`:
  ```python
  import tensorflow as tf
  from src.models.transfer_model import build_transfer_model
  from src.models.train import compile_model, get_callbacks, create_dataset

  def finetune(checkpoint_path: str, backbone: str = "MobileNetV2",
               initial_lr: float = 5e-5, epochs: int = 40, batch_size: int = 16):
      model = build_transfer_model(backbone_name=backbone, freeze_backbone=True)
      model.load_weights(checkpoint_path)

      # Unfreeze backbone
      for layer in model.layers[3].layers:  # layer[3] = backbone
          layer.trainable = True

      compile_model(model, lr=initial_lr)

      train_ds = create_dataset("train", batch_size=batch_size, shuffle=True, augment=True)
      val_ds = create_dataset("val", batch_size=batch_size, shuffle=False, augment=False)

      callbacks = get_callbacks(f"{backbone}_finetune")
      history = model.fit(train_ds, validation_data=val_ds,
                          epochs=epochs, callbacks=callbacks, verbose=1)
      return model, history
  ```

#### Day 35 — ResNet50 (Frozen) Training

- [ ] Run:
  ```bash
  python -m src.models.train \
      --model transfer \
      --backbone ResNet50 \
      --freeze \
      --batch-size 32 \
      --epochs 80 \
      --lr 1e-3
  ```

#### Day 36 — ResNet50 Fine-Tuning

- [ ] Repeat fine-tuning procedure for best ResNet50 frozen checkpoint

#### Day 37 — Hyperparameter Grid Search (Limited)

- [ ] Create `notebooks/02_hparam_search.ipynb`:
  - Sweep over:
    - Learning rate: {1e-2, 1e-3, 1e-4}
    - Dropout rate: {0.3, 0.5, 0.7}
    - Batch size: {16, 32, 64}
    - Optimizer: {Adam, AdamW}
  - For each combo: train 30 epochs, record val_accuracy
  - [ ] Select optimal set
  - [ ] Re-run full training with optimal hp

#### Day 38 — Class Imbalance Handling (if needed)

- [ ] Inspect class distribution in `data/processed/manifest.csv`
- [ ] If imbalance > 2:1, implement in `data_loader.py`:
  ```python
  # Compute class weights
  from sklearn.utils.class_weight import compute_class_weight
  classes = np.array(list(CLASS_MAP.values()))
  class_weights = compute_class_weight("balanced", classes=classes, y=train_labels)
  class_weight_dict = dict(enumerate(class_weights))

  # Pass to model.fit(class_weight=class_weight_dict)
  ```
- [ ] Alternatively, use `tf.data.Dataset.rejection_resample`

#### Day 39 — Class Activation Map (CAM) Visualisation

- [ ] Implement Grad-CAM in `src/utils/gradcam.py`:
  ```python
  import tensorflow as tf
  import numpy as np
  import matplotlib.pyplot as plt

  def make_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
      grad_model = tf.keras.models.Model(
          inputs=model.input,
          outputs=[model.get_layer(last_conv_layer_name).output, model.output]
      )
      with tf.GradientTape() as tape:
          conv_outputs, predictions = grad_model(img_array)
          if pred_index is None:
              pred_index = tf.argmax(predictions[0])
          loss = predictions[:, pred_index]
      grads = tape.gradient(loss, conv_outputs)
      pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
      conv_outputs = conv_outputs[0]
      heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
      heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
      return heatmap.numpy()
  ```
- [ ] Generate CAM overlays for 5 samples per class, archive to `logs/cam/`

#### Day 40 — Overfitting Diagnosis

- [ ] Plot training vs. validation loss from CSVLogger output
- [ ] If val_loss diverges > 0.1 from train_loss after convergence:
  - [ ] Increase Dropout (0.5 → 0.6)
  - [ ] Add L2 regularisation to Conv2D layers:
    ```python
    layers.Conv2D(..., kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    ```
  - [ ] Reduce model capacity (fewer filters, fewer dense units)
- [ ] Re-run and compare

#### Day 41 — Model Ensembling Experiment

- [ ] Implement soft-voting ensemble in `src/models/ensemble.py`:
  ```python
  import numpy as np

  def ensemble_predict(models, X):
      probs = np.mean([m.predict(X, verbose=0) for m in models], axis=0)
      return probs  # (N, 3)
  ```
- [ ] Evaluate: custom CNN + MobileNetV2 + ResNet50
- [ ] Compare ensemble accuracy vs. best single model

#### Day 42 — Model Export (SavedModel + TFLite)

- [ ] Export best model to `models/exported/`:
  ```python
  model.save("models/exported/best_model.keras")
  model.save("models/exported/best_model_savedmodel/")

  # Convert to TFLite
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()
  with open("models/exported/best_model.tflite", "wb") as f:
      f.write(tflite_model)
  ```

#### Day 43 — Model Quantisation (Post-Training)

- [ ] Apply int8 quantisation for edge deployment:
  ```python
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset  # ~100 calib samples
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8
  quant_model = converter.convert()
  ```
- [ ] Compare accuracy drop (should be < 2%)

#### Day 44 — Final Model Selection

- [ ] Tabulate results:
  | Model | Val Acc | Precision | Recall | F1 | Size (MB) | Latency (ms) |
  |---|---|---|---|---|---|---|
  | Custom CNN | | | | | | |
  | MobileNetV2 (frozen) | | | | | | |
  | MobileNetV2 (finetune) | | | | | | |
  | ResNet50 (frozen) | | | | | | |
  | ResNet50 (finetune) | | | | | | |
  | Ensemble (soft vote) | | | | | | |
- [ ] Select best candidate based on accuracy / size / speed trade-off
- [ ] Export final model: `models/exported/final_model.keras`

#### Day 45 — Phase 4 Code Freeze

- [ ] `pytest tests/ -v --tb=short`
- [ ] `git add . && git commit -m "Phase 4: models, training, augmentation, export, ensemble, gradcam"`
- [ ] `git tag phase4-complete`

---

## 6. Phase 5: Evaluation, Metrics & Validation (Days 46–60)

### Objective

Rigorously evaluate the trained model on the held-out test set. Build a CLI inference script that accepts a `.wav` path and returns a predicted grade with confidence. Generate all figures for the thesis.

### Developer TODO List

#### Day 46 — Test Set Evaluation Script

- [ ] Create `src/evaluation/evaluate.py`:
  ```python
  import numpy as np
  import tensorflow as tf
  from sklearn.metrics import (
      confusion_matrix, classification_report,
      precision_score, recall_score, f1_score,
      roc_curve, auc, roc_auc_score,
  )
  from src.models.data_loader import create_dataset
  from config.paths import CLASS_MAP, CLASS_MAP_INV, EXPORT_DIR

  def evaluate_model(model_path: str, split: str = "test"):
      model = tf.keras.models.load_model(model_path)
      ds = create_dataset(split, batch_size=32, shuffle=False, augment=False)

      y_true, y_pred_probs = [], []
      for specs, labels in ds:
          probs = model.predict(specs, verbose=0)
          y_pred_probs.append(probs)
          y_true.append(labels.numpy())

      y_true = np.concatenate(y_true, axis=0)
      y_pred_probs = np.concatenate(y_pred_probs, axis=0)
      y_pred = np.argmax(y_pred_probs, axis=1)
      y_true_idx = np.argmax(y_true, axis=1)

      cm = confusion_matrix(y_true_idx, y_pred)
      report = classification_report(
          y_true_idx, y_pred,
          target_names=[CLASS_MAP_INV[i] for i in range(len(CLASS_MAP))],
          output_dict=True,
      )
      precision = precision_score(y_true_idx, y_pred, average="weighted")
      recall = recall_score(y_true_idx, y_pred, average="weighted")
      f1 = f1_score(y_true_idx, y_pred, average="weighted")

      # ROC-AUC (one-vs-rest)
      roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr")

      return {
          "confusion_matrix": cm,
          "classification_report": report,
          "precision": precision,
          "recall": recall,
          "f1_score": f1,
          "roc_auc": roc_auc,
          "y_true": y_true_idx,
          "y_pred_probs": y_pred_probs,
      }
  ```

#### Day 47 — Confusion Matrix Plotter

- [ ] Create `src/evaluation/plot_confusion.py`:
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  import numpy as np
  from config.paths import CLASS_MAP

  def plot_confusion_matrix(cm: np.ndarray, save_path: str,
                            normalize: bool = True):
      if normalize:
          cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

      labels = [k.replace("grade_", "").upper() for k in CLASS_MAP.keys()]
      fig, ax = plt.subplots(figsize=(6, 5))
      sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                  cmap="Blues", xticklabels=labels, yticklabels=labels,
                  ax=ax)
      ax.set_xlabel("Predicted")
      ax.set_ylabel("True")
      ax.set_title("Confusion Matrix")
      fig.tight_layout()
      fig.savefig(save_path, dpi=200)
      plt.close(fig)
  ```

#### Day 48 — ROC-AUC Curve Plotter

- [ ] Create `src/evaluation/plot_roc.py`:
  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  from sklearn.metrics import roc_curve, auc
  from config.paths import CLASS_MAP

  def plot_roc_curves(y_true_bin: np.ndarray, y_pred_probs: np.ndarray,
                      save_path: str):
      labels = [k.replace("grade_", "").upper() for k in CLASS_MAP.keys()]
      fig, ax = plt.subplots(figsize=(8, 6))
      for i, label in enumerate(labels):
          fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
          roc_auc = auc(fpr, tpr)
          ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
      ax.plot([0, 1], [0, 1], "k--", lw=1)
      ax.set_xlim([0.0, 1.0])
      ax.set_ylim([0.0, 1.05])
      ax.set_xlabel("False Positive Rate")
      ax.set_ylabel("True Positive Rate")
      ax.set_title("ROC-AUC Curves (One-vs-Rest)")
      ax.legend(loc="lower right")
      fig.tight_layout()
      fig.savefig(save_path, dpi=200)
      plt.close(fig)
  ```

#### Day 49 — Per-Class Metrics Table Generator

- [ ] Create `src/evaluation/metrics_table.py`:
  ```python
  import pandas as pd
  from sklearn.metrics import precision_recall_fscore_support
  from config.paths import CLASS_MAP

  def generate_metrics_table(y_true: np.ndarray, y_pred: np.ndarray,
                              save_path: str):
      labels = [k.replace("grade_", "").upper() for k in CLASS_MAP.keys()]
      precision, recall, f1, support = precision_recall_fscore_support(
          y_true, y_pred, average=None
      )
      df = pd.DataFrame({
          "Grade": labels,
          "Precision": precision,
          "Recall": recall,
          "F1-Score": f1,
          "Support": support,
      })
      df.to_csv(save_path, index=False)
      print(f"[Metrics] Table saved to {save_path}")
      return df
  ```

#### Day 50 — Inference Script (Single .wav → Grade)

- [ ] Create `src/inference.py` (top-level script):
  ```python
  #!/usr/bin/env python
  """
  Brick NDT Inference
  Usage:
      python src/inference.py --wav path/to/brick.wav
      python src/inference.py --wav path/to/brick.wav --model models/exported/final_model.keras
  """
  import argparse, sys, os
  import numpy as np
  import tensorflow as tf

  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from src.preprocessing.pipeline import audio_to_spectrogram
  from config.paths import CLASS_MAP_INV, EXPORT_DIR, CLASS_MAP

  def predict(filepath: str, model_path: str = None) -> dict:
      if model_path is None:
          model_path = str(EXPORT_DIR / "final_model.keras")
          if not os.path.exists(model_path):
              model_path = str(EXPORT_DIR / "best_model.keras")

      model = tf.keras.models.load_model(model_path)
      spec = audio_to_spectrogram(filepath)  # (128, 128)
      spec = np.expand_dims(spec, axis=(0, -1))  # (1, 128, 128, 1)

      probs = model.predict(spec, verbose=0)[0]  # (3,)
      pred_idx = int(np.argmax(probs))
      confidence = float(probs[pred_idx])
      grade = CLASS_MAP_INV[pred_idx].replace("grade_", "").upper()

      all_probs = {
          CLASS_MAP_INV[i].replace("grade_", "").upper(): float(probs[i])
          for i in range(len(CLASS_MAP))
      }
      return {
          "file": filepath,
          "predicted_grade": grade,
          "confidence": round(confidence, 4),
          "all_probabilities": all_probs,
      }

  def main():
      parser = argparse.ArgumentParser(description="Brick NDT Inference")
      parser.add_argument("--wav", required=True, help="Path to .wav file")
      parser.add_argument("--model", default=None, help="Path to .keras model")
      args = parser.parse_args()
      result = predict(args.wav, args.model)
      print(f"File:      {result['file']}")
      print(f"Predicted: {result['predicted_grade']}")
      print(f"Confidence: {result['confidence']:.2%}")
      for grade, prob in result["all_probabilities"].items():
          print(f"  P({grade}) = {prob:.2%}")

  if __name__ == "__main__":
      main()
  ```

#### Day 51 — Inference Script Testing

- [ ] Run on 10 test files per grade:
  ```bash
  python src/inference.py --wav data/raw/grade_a/sample_a_001.wav
  python src/inference.py --wav data/raw/grade_b/sample_b_001.wav
  python src/inference.py --wav data/raw/grade_c/sample_c_001.wav
  ```
- [ ] Verify predictions are correct (compare against ground-truth labels)

#### Day 52 — Batch Inference for Test Set

- [ ] Run evaluation:
  ```python
  from src.evaluation.evaluate import evaluate_model
  from src.evaluation.plot_confusion import plot_confusion_matrix
  from src.evaluation.plot_roc import plot_roc_curves
  from src.evaluation.metrics_table import generate_metrics_table

  results = evaluate_model("models/exported/final_model.keras", split="test")
  plot_confusion_matrix(results["confusion_matrix"], "logs/confusion_matrix.png")
  plot_roc_curves(results["y_true"], results["y_pred_probs"], "logs/roc_curves.png")
  generate_metrics_table(results["y_true"], results["y_pred"], "logs/metrics_table.csv")
  ```

#### Day 53 — Error Analysis

- [ ] Identify misclassified samples from confusion matrix
- [ ] Create `logs/error_analysis/` directory
- [ ] For each misclassified sample:
  - [ ] Plot its spectrogram
  - [ ] Plot Grad-CAM heatmap over the spectrogram
  - [ ] Write a 1-line hypothesis on why it failed
- [ ] Summarise failure patterns in `logs/error_analysis/summary.md`:
  - e.g. "Grade B often confused with C when impact energy is low"
  - e.g. "Grade A misclassified when brick has surface cracks causing dampened harmonics"

#### Day 54 — Ablation Study

- [ ] Evaluate model variants to justify preprocessing choices:
  | Variant | Test Accuracy |
  |---|---|
  | Full pipeline | baseline |
  | No noise reduction | |
  | No silence trimming | |
  | No normalisation | |
  | 64 mel bands (vs 128) | |
  | 1.5 s window (vs 3.0 s) | |
- [ ] Save results to `logs/ablation_study.csv`
- [ ] Draw conclusions for thesis methodology chapter

#### Day 55 — Robustness Test (Noisy Environment Simulation)

- [ ] Generate synthetic noisy versions of 50 test samples:
  ```python
  import noisereduce as nr
  # Add real-world noise at SNR {10, 5, 0} dB
  ```
- [ ] Evaluate model on noisy samples
- [ ] Plot accuracy vs. SNR curve → `logs/robustness_snr.csv`
- [ ] Document degradation pattern

#### Day 56 — Inference Latency Benchmark

- [ ] Create `src/evaluation/benchmark.py`:
  ```python
  import time, numpy as np
  import tensorflow as tf

  def benchmark(model_path: str, num_runs: int = 100):
      model = tf.keras.models.load_model(model_path)
      dummy = np.random.randn(1, 128, 128, 1).astype(np.float32)
      # Warm-up
      model.predict(dummy, verbose=0)
      times = []
      for _ in range(num_runs):
          t0 = time.perf_counter()
          model.predict(dummy, verbose=0)
          times.append(time.perf_counter() - t0)
      return {
          "mean_ms": np.mean(times) * 1000,
          "std_ms": np.std(times) * 1000,
          "p95_ms": np.percentile(times, 95) * 1000,
          "p99_ms": np.percentile(times, 99) * 1000,
      }
  ```
- [ ] Benchmark both `.keras` and `.tflite` variants
- [ ] Report in thesis results section

#### Day 57 — Final Metrics Compilation

- [ ] Create `notebooks/03_final_results.ipynb`:
  - Load all evaluation metrics
  - Display confusion matrix, ROC curves, metrics table
  - Show 5 correctly classified + 5 misclassified spectrograms side-by-side
  - Generate LaTeX-ready tables for thesis
- [ ] Export all figures to `logs/thesis_figures/`

#### Day 58 — Cross-Validation (Optional / Time Permitting)

- [ ] Implement 5-fold cross-validation in `src/evaluation/crossval.py`:
  ```python
  from sklearn.model_selection import StratifiedKFold
  # Split dataset into 5 folds
  # For each fold: train on 4, validate on 1
  # Report mean ± std accuracy
  ```
- [ ] Compare k-fold results with single train/val/test split
- [ ] Update results section if variance is high

#### Day 59 — Documentation & Code Cleanup

- [ ] Add docstrings to every public function across `src/`
- [ ] Remove dead code / commented-out blocks
- [ ] Verify no hardcoded paths (all paths via `config/paths.py`)
- [ ] Add `README.md` with:
  - Project title and abstract
  - Setup instructions
  - CLI usage examples
  - Reproduce training: `python -m src.models.train ...`
  - Run inference: `python src/inference.py --wav ...`
  - Evaluation results summary

#### Day 60 — Phase 5 Code Freeze

- [ ] Full pipeline end-to-end test:
  ```bash
  # From scratch
  ./setup.sh
  python -m src.preprocessing.cli --workers 4
  python -c "from src.preprocessing.splitter import run_split; run_split('data/processed/manifest.csv')"
  python -m src.models.train --model custom_cnn --epochs 50
  python src/evaluation/run_full_eval.py  # (consolidate all eval steps)
  python src/inference.py --wav data/raw/grade_a/sample_a_001.wav
  ```
- [ ] `pytest tests/ -v --tb=short`
- [ ] `git add . && git commit -m "Phase 5: evaluation, inference, robustness, benchmark, crossval"`
- [ ] `git tag phase5-complete`

---

## 7. Phase 6: Thesis Writing & Defense Preparation (Days 61–120)

### Objective

Write the full thesis manuscript, compile all figures, prepare the defense presentation, and submit.

### Developer TODO List

#### Days 61–70 — Literature Review & Chapter 1

- [ ] Write Chapter 1: Introduction
  - [ ] Problem statement (traditional brick testing subjectivity)
  - [ ] Proposed solution (deep learning + acoustic NDT)
  - [ ] Research objectives and contributions
  - [ ] Thesis outline
- [ ] Compile annotated bibliography (≥ 30 references)
  - [ ] NDT in construction materials
  - [ ] Audio classification using CNNs
  - [ ] Mel-spectrogram analysis for impact acoustics
  - [ ] Transfer learning for small datasets

#### Days 71–80 — Methodology Chapters

- [ ] Write Chapter 2: Literature Review
  - [ ] Traditional brick quality testing methods
  - [ ] NDT techniques (ultrasonic, acoustic emission, impact acoustics)
  - [ ] Deep learning for audio classification
  - [ ] Mel-spectrogram theory and applications
  - [ ] Transfer learning in acoustic classification
- [ ] Write Chapter 3: Methodology
  - [ ] Data collection procedure
  - [ ] Preprocessing pipeline (noise reduction, trimming, STFT, Mel-filterbank)
  - [ ] Model architectures (custom CNN, MobileNetV2, ResNet50)
  - [ ] Training protocol (optimizer, loss, callbacks, augmentation)
  - [ ] Evaluation metrics

#### Days 81–95 — Results & Discussion

- [ ] Write Chapter 4: Results
  - [ ] Dataset statistics table
  - [ ] Training curves (loss & accuracy across epochs)
  - [ ] Confusion matrix (test set)
  - [ ] ROC-AUC curves
  - [ ] Precision / Recall / F1 table per class
  - [ ] Ablation study results table
  - [ ] Robustness (noise) test results
  - [ ] Inference latency benchmark
  - [ ] Grad-CAM visualisations with interpretation
- [ ] Write Chapter 5: Discussion
  - [ ] Interpretation of results
  - [ ] Comparison with traditional testing
  - [ ] Limitations and failure case analysis
  - [ ] Practical deployment considerations

#### Days 96–105 — Writing & Revision

- [ ] Write Chapter 6: Conclusion & Future Work
- [ ] Write Abstract (250–300 words)
- [ ] Write Acknowledgments
- [ ] Compile List of Figures, List of Tables
- [ ] First full draft review (self-edit)
- [ ] Advisor review submission
- [ ] Incorporate advisor feedback

#### Days 106–115 — Defense Preparation

- [ ] Create defense presentation (15–20 slides):
  - [ ] Title slide
  - [ ] Problem statement (2 slides)
  - [ ] Related work (1–2 slides)
  - [ ] Methodology (4–5 slides)
  - [ ] Results (4–5 slides)
  - [ ] Conclusion & contributions (2 slides)
  - [ ] Q&A backup slides (detailed model diagrams, ablation tables)
- [ ] Practice defense talk (≥ 3 times, timed)
- [ ] Prepare answers for likely questions:
  - Why Mel-spectrogram over raw waveform / MFCC?
  - Why MobileNetV2 over ResNet50?
  - How was the dataset collected? What are its limitations?
  - How would you deploy this in a real brick factory?
  - What is the minimum audio quality needed?
- [ ] Create demo video (screen recording of inference script)

#### Days 116–120 — Final Submission

- [ ] Final thesis PDF generation (LaTeX or Word)
- [ ] Verify all figures are high-resolution (≥ 300 DPI)
- [ ] Verify all references are correctly formatted
- [ ] Submit thesis to department
- [ ] Upload supplementary materials (code, dataset link, trained models)
- [ ] Archive final code: `git tag v1.0.0`

---

## 8. Grand Master Checkbox Matrix

### Coding Priority Key

| Priority | Label | Meaning |
|---|---|---|
| 🔴 High | Must-do for core thesis | Code or experiment without which the thesis cannot proceed |
| 🟡 Medium | Important but optional | Substantially improves results or usability |
| 🟢 Low | Nice-to-have | Polishing, edge deployment, advanced experiments |

### Consolidated Chronological Checklist

#### Days 1–5: Environment & Workspace

- [ ] 🔴 Project scaffold (`mkdir -p`, `.gitignore`)
- [ ] 🔴 Python venv + `requirements.txt`
- [ ] 🔴 `setup.sh` automated install script
- [ ] 🔴 `src/utils/audio_loader.py` — `.wav` verifier
- [ ] 🔴 `config/paths.py` — centralised path config
- [ ] 🟡 `notebooks/00_environment_sanity.ipynb`
- [ ] 🔴 First git commit

#### Days 6–15: Audio Preprocessing & DSP Engine

- [ ] 🔴 `src/preprocessing/noise_reduction.py`
- [ ] 🔴 `src/preprocessing/trim_pad.py` — silence trim + length fix
- [ ] 🔴 `src/preprocessing/spectrogram.py` — STFT → Mel → resize → normalise
- [ ] 🔴 `src/preprocessing/pipeline.py` — `audio_to_spectrogram()` orchestrator
- [ ] 🔴 `src/utils/visualise.py` — borderless spec figure saver
- [ ] 🔴 `tests/test_preprocessing.py` — unit tests for all DSP functions
- [ ] 🔴 End-to-end pipeline validation on 3 samples per grade
- [ ] 🟡 `notebooks/01_param_sweep.ipynb` — n_mels, hop_length, duration sweep
- [ ] 🟡 Edge-case handling: corrupt `.wav`, too-short audio
- [ ] 🔴 Git commit + tag `phase2-complete`

#### Days 16–25: Automation & Batch Processing

- [ ] 🔴 `src/preprocessing/manifest.py` — CSV manifest writer
- [ ] 🔴 `src/preprocessing/scanner.py` — directory tree scanner
- [ ] 🔴 `src/preprocessing/batch_processor.py` — multiprocess pipeline + tqdm
- [ ] 🔴 `src/preprocessing/cli.py` — argparse entry point
- [ ] 🔴 `src/preprocessing/splitter.py` — 70/15/15 stratified splitter + file move
- [ ] 🔴 Run batch processor on ≥ 100 real `.wav` files
- [ ] 🟡 Resume functionality (`--resume` flag, skip processed)
- [ ] 🟢 Rich logging (`config/logging_config.py`)
- [ ] 🟢 `src/models/data_augmentation.py` — online SpecAugment-style augmentation
- [ ] 🔴 Git commit + tag `phase3-complete`

#### Days 26–45: Model Architecture & Training

- [ ] 🔴 `src/models/data_loader.py` — `tf.data` pipeline from `.npy` files
- [ ] 🔴 `src/models/custom_cnn.py` — 4-block ConvNet
- [ ] 🔴 `src/models/transfer_model.py` — MobileNetV2 + ResNet50 wrappers
- [ ] 🔴 `src/models/train.py` — full training loop with callbacks
- [ ] 🔴 Run custom CNN baseline (100 epochs)
- [ ] 🔴 Run MobileNetV2 frozen (80 epochs)
- [ ] 🔴 Run MobileNetV2 fine-tune (40 more epochs)
- [ ] 🟡 Run ResNet50 frozen + fine-tune
- [ ] 🟡 `src/utils/tf_config.py` — GPU memory growth + mixed precision
- [ ] 🟡 `config/training_config.py` — experiment registry
- [ ] 🟡 `notebooks/02_hparam_search.ipynb` — LR, dropout, batch size sweep
- [ ] 🟡 Class imbalance handling (class weights / resampling)
- [ ] 🟢 `src/utils/gradcam.py` — Grad-CAM visualisations
- [ ] 🟢 `src/models/ensemble.py` — soft-voting ensemble
- [ ] 🟡 Model export: `.keras`, SavedModel, TFLite
- [ ] 🟢 Post-training quantisation (int8)
- [ ] 🔴 Final model selection table
- [ ] 🔴 Git commit + tag `phase4-complete`

#### Days 46–60: Evaluation, Metrics & Validation

- [ ] 🔴 `src/evaluation/evaluate.py` — full test set evaluation
- [ ] 🔴 `src/evaluation/plot_confusion.py` — normalised confusion matrix
- [ ] 🔴 `src/evaluation/plot_roc.py` — multi-class ROC-AUC curves
- [ ] 🔴 `src/evaluation/metrics_table.py` — per-class precision/recall/F1
- [ ] 🔴 `src/inference.py` — single `.wav` → grade + confidence CLI
- [ ] 🔴 Run evaluation on test set, save figures
- [ ] 🟡 Error analysis (misclassified samples + Grad-CAM)
- [ ] 🟡 Ablation study (no denoise, no trim, 64 mels, 1.5 s window)
- [ ] 🟡 Robustness test (noisy audio at SNR 10/5/0 dB)
- [ ] 🟢 `src/evaluation/benchmark.py` — latency benchmark (keras vs tflite)
- [ ] 🟢 `notebooks/03_final_results.ipynb` — compiles all figures
- [ ] 🟡 5-fold cross-validation (optional)
- [ ] 🔴 Code cleanup, docstrings, `README.md`
- [ ] 🔴 End-to-end pipeline integration test
- [ ] 🔴 Git commit + tag `phase5-complete`

#### Days 61–120: Thesis Writing & Defense

- [ ] 🔴 Chapter 1: Introduction
- [ ] 🔴 Chapter 2: Literature Review (≥ 30 refs)
- [ ] 🔴 Chapter 3: Methodology (pipeline + model architecture)
- [ ] 🔴 Chapter 4: Results (all tables, figures, metrics)
- [ ] 🔴 Chapter 5: Discussion (interpretation + limitations)
- [ ] 🔴 Chapter 6: Conclusion & Future Work
- [ ] 🔴 Abstract + Acknowledgments + ToC
- [ ] 🔴 Advisor review + revisions
- [ ] 🔴 Defense presentation (15–20 slides)
- [ ] 🟡 Practice talk (≥ 3 times)
- [ ] 🟡 Demo video
- [ ] 🔴 Final submission
- [ ] 🔴 `git tag v1.0.0`

---

*End of Execution Plan. Generate this file at project root as `EXECUTION_PLAN.md` and refer to it daily for task prioritisation.*
