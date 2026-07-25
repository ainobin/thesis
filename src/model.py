"""
model.py — CNN Architecture Definition

Purpose:
  Defines the 2D CNN used for brick-grade classification from
  Mel-spectrograms.

Architecture:
  Input (128, T, 1)
  ├─ Conv2D(32, 3×3, ReLU) + BatchNorm + MaxPool(2×2)   → (64, T/2, 32)
  ├─ Conv2D(64, 3×3, ReLU) + BatchNorm + MaxPool(2×2)   → (32, T/4, 64)
  ├─ Conv2D(128, 3×3, ReLU) + BatchNorm                  → (32, T/4, 128)
  ├─ GlobalAveragePooling2D                               → (128)
  ├─ Dense(128, ReLU, L2=1e-4) + Dropout(0.5)            → (128)
  └─ Dense(2, Softmax)                                    → (2)

Parameters: ~110K (431 KB)

Usage:
  from src.model import build_cnn
  model = build_cnn(input_shape=(128, 130, 1), num_classes=2)
"""

from tensorflow.keras import layers, Model, regularizers


def build_cnn(input_shape=(128, None, 1), num_classes=2):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inp, outputs=out, name="brick_ndt_cnn")
