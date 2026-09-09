"""
evaluate.py — Test-Set Evaluation

Purpose:
  Loads the best saved model and runs it on the held-out test set.
  Computes accuracy, precision, recall, F1-score (per-class + macro),
  confusion matrix, and ROC curves (one-vs-rest).

Output:
  figures/confusion_matrix.png   — heatmap
  figures/roc_curve.png          — ROC curves with AUC per class
  models/results_summary.npy     — dict of all test metrics

Usage:
  python src/evaluate.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

LABELS = ["Grade A", "Grade B", "Grade C"]


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configured: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")


def main():
    configure_gpu()
    print("Loading test data ...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"  X_test: {X_test.shape}  y_test: {y_test.shape}")

    model_path = os.path.join(MODELS_DIR, "best.keras")
    print(f"\nLoading model from {model_path} ...")
    model = load_model(model_path)

    print("\nPredicting ...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(LABELS)))
    )

    print(f"\n  Test accuracy:  {acc:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-score:       {f1:.4f}")
    print()
    for i, name in enumerate(LABELS):
        print(f"  {name} ({i}):  precision={precision_per[i]:.4f}  recall={recall_per[i]:.4f}  f1={f1_per[i]:.4f}  support={support_per[i]}")

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(LABELS))))
    print(f"\nConfusion matrix:\n{cm}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"  Saved {cm_path}")

    n_classes = y_prob.shape[1]
    colors = ["steelblue", "coral", "seagreen"]
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for i in range(n_classes):
        binary_y = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_y, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f"{LABELS[i]} (AUC = {roc_auc:.3f})", lw=2, color=colors[i % len(colors)])
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve (One-vs-Rest)")
    ax2.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    print(f"  Saved {roc_path}")

    results = {
        "test_accuracy": float(acc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }
    res_path = os.path.join(MODELS_DIR, "results_summary.npy")
    np.save(res_path, results)
    print(f"\nResults summary saved to {res_path}")
    print("Done.")


if __name__ == "__main__":
    main()
