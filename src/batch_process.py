"""
batch_process.py — Batch Preprocessing & Stratified Split Runner

Purpose:
  Scans all raw audio files from data/raw/grade_{a,b}, runs them through
  preprocess_audio(), stacks into unified NumPy matrices, and performs a
  stratified 70/15/15 train/val/test split.

Output (6 files in data/processed/):
  X_train.npy, y_train.npy    (70%)
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

# ensure project root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.preprocess import preprocess_audio

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data", "processed")

LABEL_MAP = {"grade_a": 0, "grade_b": 1}

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def _walk_audio_files(root: str):
    for grade_name, label in LABEL_MAP.items():
        grade_dir = os.path.join(root, grade_name)
        if not os.path.isdir(grade_dir):
            continue
        for ext in SUPPORTED_EXTENSIONS:
            pattern = os.path.join(grade_dir, f"*{ext}")
            for path in sorted(glob.glob(pattern)):
                yield path, label, grade_name


def run_pipeline() -> None:
    print("Scanning audio files ...")
    items = list(_walk_audio_files(RAW))

    if not items:
        print("ERROR: No audio files found under data/raw/.", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(items)} files.\n")

    X, y = [], []
    failed = 0

    for path, label, grade_name in tqdm(items, desc="Preprocessing"):
        try:
            feat = preprocess_audio(path)
            X.append(feat)
            y.append(label)
        except Exception as exc:
            print(f"\n  \u2717 {os.path.relpath(path, BASE)} \u2014 {exc}")
            failed += 1

    if not X:
        print("ERROR: No features could be extracted.", file=sys.stderr)
        sys.exit(1)

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int32)

    print(f"\n  Total valid samples: {len(X)}  |  Failures: {failed}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=42
    )

    os.makedirs(OUT, exist_ok=True)

    for name, arr in [("X_train", X_train), ("y_train", y_train),
                      ("X_val",   X_val),   ("y_val",   y_val),
                      ("X_test",  X_test),  ("y_test",  y_test)]:
        path = os.path.join(OUT, f"{name}.npy")
        np.save(path, arr)
        print(f"  \u2713 {name}.npy  \u2192  {arr.shape}")

    print("\nClass distribution:")
    for grade_name, label in LABEL_MAP.items():
        train_c = int((y_train == label).sum())
        val_c   = int((y_val   == label).sum())
        test_c  = int((y_test  == label).sum())
        print(f"  {grade_name} ({label}):  train={train_c}  val={val_c}  test={test_c}")


if __name__ == "__main__":
    run_pipeline()
