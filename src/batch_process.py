"""
batch_process.py — Batch Preprocessing & Stratified Split Runner with Augmentation

Purpose:
  Scans all raw audio files from data/raw/grade_{a,b}, runs them through
  preprocess_audio(), stacks into unified NumPy matrices, performs a
  stratified 70/15/15 train/val/test split, and augments the training set.

Augmentation (training set only):
  - Pitch shift (±2 semitones)
  - Time stretch (1.1x)
  - Additive Gaussian noise

Output (6 files in data/processed/):
  X_train.npy, y_train.npy    (70%, augmented)
  X_val.npy,   y_val.npy      (15%)
  X_test.npy,  y_test.npy     (15%)

Class labels:
  grade_a → 0,  grade_b → 1

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

from src.preprocess import preprocess_audio, audio_to_mel, SR
from src.augment import pitch_shift, time_stretch, add_noise

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data", "processed")

LABEL_MAP = {"grade_a": 0, "grade_b": 1}
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

AUGMENTATIONS = [
    ("pitch_shift+2", lambda y, sr: pitch_shift(y, sr, n_steps=2)),
    ("pitch_shift-2", lambda y, sr: pitch_shift(y, sr, n_steps=-2)),
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


def run_pipeline() -> None:
    print("Scanning audio files ...")
    items = list(_walk_audio_files(RAW))

    if not items:
        print("ERROR: No audio files found under data/raw/.", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(items)} files.\n")

    paths = [p for p, l in items]
    labels = [l for p, l in items]

    # Stratified split by index (before processing)
    train_idx, test_idx = train_test_split(
        np.arange(len(paths)), test_size=0.15, stratify=labels, random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15 / 0.85,
        stratify=[labels[i] for i in train_idx], random_state=42
    )

    splits = {
        "train": (train_idx, True),
        "val":   (val_idx,   False),
        "test":  (test_idx,  False),
    }

    os.makedirs(OUT, exist_ok=True)

    for split_name, (indices, do_augment) in splits.items():
        X_split, y_split = [], []
        desc = f"Processing {split_name}"
        for i in tqdm(indices, desc=desc):
            path = paths[i]
            label = labels[i]
            try:
                feat = preprocess_audio(path)
                X_split.append(feat)
                y_split.append(label)

                if do_augment:
                    y_raw, sr = librosa.load(path, sr=SR, mono=True)
                    for aug_name, aug_fn in AUGMENTATIONS:
                        try:
                            y_aug = aug_fn(y_raw.copy(), sr)
                            feat_aug = audio_to_mel(y_aug, sr)
                            X_split.append(feat_aug)
                            y_split.append(label)
                        except Exception as exc:
                            print(f"\n  ! {aug_name} failed for {os.path.relpath(path, BASE)}: {exc}")
            except Exception as exc:
                print(f"\n  \u2717 {os.path.relpath(path, BASE)} \u2014 {exc}")

        X_arr = np.stack(X_split, axis=0)
        y_arr = np.array(y_split, dtype=np.int32)

        name_x = f"X_{split_name}.npy"
        name_y = f"y_{split_name}.npy"
        np.save(os.path.join(OUT, name_x), X_arr)
        np.save(os.path.join(OUT, name_y), y_arr)
        print(f"  \u2713 {name_x}  \u2192  {X_arr.shape}")
        print(f"  \u2713 {name_y}  \u2192  {y_arr.shape}")

    # Summary
    print("\nClass distribution:")
    for grade_name, label in LABEL_MAP.items():
        y_train = np.load(os.path.join(OUT, "y_train.npy"))
        y_val   = np.load(os.path.join(OUT, "y_val.npy"))
        y_test  = np.load(os.path.join(OUT, "y_test.npy"))
        train_c = int((y_train == label).sum())
        val_c   = int((y_val   == label).sum())
        test_c  = int((y_test  == label).sum())
        print(f"  {grade_name} ({label}):  train={train_c}  val={val_c}  test={test_c}")

    print(f"\nAugmentation: training set expanded 5x (1 original + 4 augmented)")


if __name__ == "__main__":
    run_pipeline()
