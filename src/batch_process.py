"""
batch_process.py — Batch Preprocessing & Stratified Split Runner with Augmentation

Purpose:
  Scans all raw audio files from data/raw/grade_{a,b,c}, splits them into
  train/val/test at the FILE level first, then applies augmentation ONLY to
  training files. This prevents data leakage where augmented versions of the
  same file end up in different splits.

Augmentation (applied to training files only):
  - Pitch shift (+1 semitone)
  - Pitch shift (-1 semitone)
  - Time stretch (1.1x)
  - Additive Gaussian noise

Output (6 files in data/processed/):
  Training:   N_train files × 5 (1 original + 4 augmented)
  Validation: N_val files × 1 (original only)
  Test:       N_test files × 1 (original only)
  Stratified 70/15/15 split at file level

Class labels:
  grade_a -> 0,  grade_b -> 1,  grade_c -> 2

Usage:
  python src/batch_process.py
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.preprocess import preprocess_audio, audio_to_mel_raw, SR
from src.augment import pitch_shift, time_stretch, add_noise

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data", "processed")

LABEL_MAP = {"grade_a": 0, "grade_b": 1, "grade_c": 2}
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

AUGMENTATIONS = [
    ("pitch_shift+1", lambda y, sr: pitch_shift(y, sr, n_steps=1)),
    ("pitch_shift-1", lambda y, sr: pitch_shift(y, sr, n_steps=-1)),
    ("time_stretch",  lambda y, sr: time_stretch(y, rate=1.1)),
    ("add_noise",     lambda y, sr: add_noise(y, noise_factor=0.005)),
]


def _walk_audio_files(root: str):
    for grade_name, label in LABEL_MAP.items():
        grade_dir = os.path.join(root, grade_name)
        if not os.path.isdir(grade_dir):
            continue
        for ext in SUPPORTED_EXTENSIONS:
            pattern = os.path.join(grade_dir, f"*{ext}")
            for path in sorted(glob.glob(pattern)):
                yield path, label


def _process_file(path: str, label: int, augment: bool) -> tuple:
    """Process a single file. Returns list of (feature, label) tuples."""
    samples = []

    # Original
    feat = preprocess_audio(path)
    samples.append((feat, label))

    # Augment only for training
    if augment:
        y_raw, sr = librosa.load(path, sr=SR, mono=True)
        for aug_name, aug_fn in AUGMENTATIONS:
            try:
                y_aug = aug_fn(y_raw.copy(), sr)
                feat_aug = audio_to_mel_raw(y_aug, sr)
                samples.append((feat_aug, label))
            except Exception as exc:
                print(f"\n  ! {aug_name} failed for {os.path.relpath(path, BASE)}: {exc}")

    return samples


def run_pipeline() -> None:
    print("Scanning audio files ...")
    items = list(_walk_audio_files(RAW))

    if not items:
        print("ERROR: No audio files found under data/raw/.", file=sys.stderr)
        sys.exit(1)

    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    print(f"  Found {len(paths)} files.\n")

    # ── Step 1: Split at FILE level (before any processing) ──
    print("Splitting files into train/val/test ...")
    trainval_idx, test_idx = train_test_split(
        np.arange(len(paths)), test_size=0.15, stratify=labels, random_state=42
    )
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.15 / 0.85,
        stratify=np.array(labels)[trainval_idx], random_state=42
    )

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_paths = [paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"  Train files: {len(train_paths)}")
    print(f"  Val files:   {len(val_paths)}")
    print(f"  Test files:  {len(test_paths)}\n")

    os.makedirs(OUT, exist_ok=True)

    # ── Step 2: Process training files (with augmentation) ──
    print("Processing training files (with augmentation) ...")
    X_train, y_train = [], []
    for i in tqdm(range(len(train_paths)), desc="Train"):
        samples = _process_file(train_paths[i], train_labels[i], augment=True)
        for feat, lbl in samples:
            X_train.append(feat)
            y_train.append(lbl)

    X_train = np.stack(X_train, axis=0)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"  ✓ X_train → {X_train.shape}  (files × 5 augmented)\n")

    # ── Step 3: Process validation files (no augmentation) ──
    print("Processing validation files ...")
    X_val, y_val = [], []
    for i in tqdm(range(len(val_paths)), desc="Val"):
        samples = _process_file(val_paths[i], val_labels[i], augment=False)
        for feat, lbl in samples:
            X_val.append(feat)
            y_val.append(lbl)

    X_val = np.stack(X_val, axis=0)
    y_val = np.array(y_val, dtype=np.int32)
    print(f"  ✓ X_val → {X_val.shape}\n")

    # ── Step 4: Process test files (no augmentation) ──
    print("Processing test files ...")
    X_test, y_test = [], []
    for i in tqdm(range(len(test_paths)), desc="Test"):
        samples = _process_file(test_paths[i], test_labels[i], augment=False)
        for feat, lbl in samples:
            X_test.append(feat)
            y_test.append(lbl)

    X_test = np.stack(X_test, axis=0)
    y_test = np.array(y_test, dtype=np.int32)
    print(f"  ✓ X_test → {X_test.shape}\n")

    # ── Step 5: Save ──
    print("Saving ...")
    np.save(os.path.join(OUT, "X_train.npy"), X_train)
    np.save(os.path.join(OUT, "y_train.npy"), y_train)
    np.save(os.path.join(OUT, "X_val.npy"), X_val)
    np.save(os.path.join(OUT, "y_val.npy"), y_val)
    np.save(os.path.join(OUT, "X_test.npy"), X_test)
    np.save(os.path.join(OUT, "y_test.npy"), y_test)
    print(f"  ✓ Saved 6 files to {OUT}/\n")

    # ── Summary ──
    print("Class distribution:")
    for grade_name, label in LABEL_MAP.items():
        train_c = int((y_train == label).sum())
        val_c   = int((y_val   == label).sum())
        test_c  = int((y_test  == label).sum())
        print(f"  {grade_name} ({label}):  train={train_c}  val={val_c}  test={test_c}")

    total = len(y_train) + len(y_val) + len(y_test)
    print(f"\nTotal samples: {total}")
    print(f"  Train: {len(y_train)}  (augmented 5×)")
    print(f"  Val:   {len(y_val)}  (original only)")
    print(f"  Test:  {len(y_test)}  (original only)")
    print(f"\nNo data leakage: file-level split applied before augmentation.")


if __name__ == "__main__":
    run_pipeline()
