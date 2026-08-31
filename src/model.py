"""
model.py — CNN Architecture Definition

Purpose:
  Defines the 2D CNN used for brick-grade classification from
  Mel-spectrograms.

Architecture (with T=130 time steps):
  Input (128, 130, 1)
  ├─ Conv2D(32, 3x3, ReLU) + BatchNorm + MaxPool(2x2)   → (64, 65, 32)
  ├─ Conv2D(64, 3x3, ReLU) + BatchNorm + MaxPool(2x2)   → (32, 32, 64)
  ├─ Conv2D(128, 3x3, ReLU) + BatchNorm                  → (32, 32, 128)
  ├─ GlobalAveragePooling2D                               → (128)
  ├─ Dense(128, ReLU, L2=1e-4) + Dropout(0.5)            → (128)
  └─ Dense(num_classes, Softmax)                          → (num_classes)

Parameters: ~110K (431 KB)

Usage:
  from src.model import build_cnn
  model = build_cnn(input_shape=(128, 130, 1), num_classes=3)
  model = build_cnn(input_shape=(128, 130, 1), num_classes=3, dropout=0.3, l2_reg=1e-3)
"""

from tensorflow.keras import layers, Model, regularizers


def build_cnn(input_shape=(128, None, 1), num_classes=3, dropout=0.5, l2_reg=1e-4):
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

    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inp, outputs=out, name="brick_ndt_cnn")
