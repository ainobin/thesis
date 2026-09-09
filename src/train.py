"""
train.py — Model Training Loop

Purpose:
  Loads preprocessed .npy arrays, builds the CNN, and trains with
  callbacks for checkpointing, early stopping, and LR reduction.

Callbacks:
  - ModelCheckpoint   saves best model to models/best.keras
  - EarlyStopping     patience=10, restore_best_weights
  - ReduceLROnPlateau factor=0.5, patience=5, min_lr=1e-6

Output:
  models/best.keras      — best checkpoint by val_accuracy
  models/history.npy    — per-epoch metrics dict

Usage:
  python src/train.py
"""

import os
import sys
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

# ensure project root is on sys.path for data/model imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model import build_cnn  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configured: {len(gpus)} device(s) detected with memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("WARNING: No GPU detected. Training will run on CPU (slow).")


def main() -> None:
    configure_gpu()
    print("Loading data ...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}  y_val:   {y_val.shape}")

    num_classes = len(np.unique(y_train))
    model = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes,  dropout=0.3, l2_reg=1e-3)
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    print("Computed Class Weights:", class_weight_dict)

    callbacks = [
        ModelCheckpoint(
            os.path.join(MODELS_DIR, "best.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nTraining ...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    hist_path = os.path.join(MODELS_DIR, "history.npy")
    np.save(hist_path, history.history)
    print(f"\nHistory saved to {hist_path}")
    print("Done.")


if __name__ == "__main__":
    main()
