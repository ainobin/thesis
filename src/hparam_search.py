"""
hparam_search.py — Hyperparameter Grid Search

Purpose:
  Runs a lightweight grid search over 4 hyperparameters to find the
  best configuration for the CNN.

Grid (16 combinations):
  learning_rate  → [1e-3, 1e-4]
  dropout        → [0.3, 0.5]
  l2_reg         → [1e-4, 1e-3]
  batch_size     → [16, 32]

Each combination trains for up to 30 epochs with EarlyStopping.
The best config (highest val_accuracy) is saved.

Output:
  models/best_hparams.npy  — dict of best config + val_accuracy

Usage:
  python src/hparam_search.py
"""

import os
import sys
import itertools
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.model import build_cnn  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HPARAM_GRID = {
    "learning_rate": [1e-3, 1e-4],
    "dropout": [0.3, 0.5],
    "l2_reg": [1e-4, 1e-3],
    "batch_size": [16, 32],
}

HPO_EPOCHS = 30
HPO_PATIENCE = 8


def main():
    print("Loading data ...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}  y_val:   {y_val.shape}")

    keys = list(HPARAM_GRID.keys())
    values = list(HPARAM_GRID.values())
    total = len(list(itertools.product(*values)))
    print(f"\nHyperparameter search: {total} combinations\n")

    best_val_acc = 0.0
    best_config = None

    for i, combo in enumerate(itertools.product(*values)):
        config = dict(zip(keys, combo))
        lr, dropout, l2_reg, bs = config["learning_rate"], config["dropout"], config["l2_reg"], config["batch_size"]

        print(f"[{i+1}/{total}]  lr={lr}  dropout={dropout}  l2={l2_reg}  batch={bs}")

        model = build_cnn(X_train.shape[1:], num_classes=2, dropout=dropout, l2_reg=l2_reg)
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            X_train,
            y_train,
            batch_size=bs,
            epochs=HPO_EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(monitor="val_accuracy", patience=HPO_PATIENCE, restore_best_weights=True)],
            verbose=0,
        )

        val_acc = max(history.history["val_accuracy"])
        train_acc = max(history.history["accuracy"])
        print(f"    train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  epochs={len(history.history['loss'])}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config

    print(f"\n{'='*50}")
    print(f"Best config:  {best_config}")
    print(f"Best val_acc: {best_val_acc:.4f}")

    best_path = os.path.join(MODELS_DIR, "best_hparams.npy")
    np.save(best_path, {**best_config, "val_accuracy": float(best_val_acc)})
    print(f"Saved to {best_path}")


if __name__ == "__main__":
    main()
