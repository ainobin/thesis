"""
visualize.py — Results Visualisation & Figure Generation

Purpose:
  Generates all publication-ready figures for the thesis using the
  trained model and preprocessed data.

Figures (7 total, saved to figures/):
  1. spectrogram_grid.png       — 4×6 random sample Mel-spectrograms
  2. training_curves_final.png  — loss & accuracy over epochs
  3. class_distribution.png     — train/val/test bar chart
  4. confusion_matrix.png       — normalised heatmap
  5. roc_curve.png              — ROC with AUC
  6. pr_curve.png               — Precision-Recall with AUC
  7. misclassifications.png     — false positives & false negatives

Also prints a per-class metrics table and updates results_summary.npy.

Usage:
  python src/visualize.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

LABELS = ["Grade A", "Grade B"]


def _load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def _load_model():
    path = os.path.join(MODELS_DIR, "final_best.keras")
    if not os.path.exists(path):
        path = os.path.join(MODELS_DIR, "best.keras")
    return load_model(path), path


def plot_spectrogram_grid(X_train, y_train):
    fig, axes = plt.subplots(4, 6, figsize=(16, 10))
    for i, ax in enumerate(axes.flat):
        idx = np.random.randint(len(X_train))
        ax.imshow(X_train[idx, :, :, 0], aspect="auto", origin="lower", cmap="magma")
        ax.set_title(LABELS[y_train[idx]], fontsize=9)
        ax.axis("off")
    plt.suptitle("Sample Mel-Spectrograms per Class", fontsize=14, y=0.98)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "spectrogram_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_pr_curve(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pr_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return pr_auc


def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return roc_auc


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return cm


def plot_class_distribution(y_train, y_val, y_test):
    counts = {
        "Train": [int((y_train == 0).sum()), int((y_train == 1).sum())],
        "Val":   [int((y_val   == 0).sum()), int((y_val   == 1).sum())],
        "Test":  [int((y_test  == 0).sum()), int((y_test  == 1).sum())],
    }
    x = np.arange(len(counts))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, [c[0] for c in counts.values()], width, label="Grade A", color="steelblue")
    ax.bar(x + width / 2, [c[1] for c in counts.values()], width, label="Grade B", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(counts.keys())
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution Across Splits")
    ax.legend()
    for i, k in enumerate(counts):
        for j, v in enumerate(counts[k]):
            ax.text(i + (-0.12 if j == 0 else 0.12), v + 0.1, str(v), fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return counts


def plot_misclassifications(X_test, y_test, y_pred, y_prob):
    errors = np.where(y_test != y_pred)[0]
    if len(errors) == 0:
        print("  No misclassifications found.")
        return

    n_plot = min(len(errors), 8)
    cols = 4
    rows = (n_plot + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if n_plot > 1 else [axes]
    for i, ax in zip(range(n_plot), axes):
        idx = errors[i]
        ax.imshow(X_test[idx, :, :, 0], aspect="auto", origin="lower", cmap="magma")
        true_lbl = LABELS[y_test[idx]]
        pred_lbl = LABELS[y_pred[idx]]
        conf = y_prob[idx].max()
        ax.set_title(f"True: {true_lbl} | Pred: {pred_lbl}\n(conf: {conf:.2%})", fontsize=9)
        ax.axis("off")
    for ax in axes[n_plot:]:
        ax.axis("off")
    plt.suptitle("Misclassified Samples", fontsize=14, y=0.98)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "misclassifications.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}  ({len(errors)} misclassified total)")


def plot_training_curves():
    path = os.path.join(MODELS_DIR, "history_final.npy")
    if not os.path.exists(path):
        path = os.path.join(MODELS_DIR, "history.npy")
    h = np.load(path, allow_pickle=True).item()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(h["loss"], label="train")
    ax1.plot(h["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(h["accuracy"], label="train")
    ax2.plot(h["val_accuracy"], label="val")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "training_curves_final.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def main():
    print("Loading data ...")
    X_train, y_train, X_val, y_val, X_test, y_test = _load_data()
    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    print("\nLoading model ...")
    model, model_path = _load_model()
    print(f"  Loaded {model_path}")

    print("\nPredicting ...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    print("\nGenerating figures ...\n")
    plot_spectrogram_grid(X_train, y_train)
    plot_training_curves()
    plot_class_distribution(y_train, y_val, y_test)
    cm = plot_confusion_matrix(y_test, y_pred)
    roc_auc = plot_roc_curve(y_test, y_prob)
    pr_auc = plot_pr_curve(y_test, y_prob)
    plot_misclassifications(X_test, y_test, y_pred, y_prob)

    print("\n--- Per-Class Metrics ---")
    tn, fp, fn, tp = cm.ravel()
    print(f"  {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"  {'-'*58}")
    for i, label in enumerate(LABELS):
        support = int((y_test == i).sum())
        p = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0.0
        r = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"  {label:<12} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {support:<10}")
    print(f"\n  ROC AUC: {roc_auc:.4f}  |  PR AUC: {pr_auc:.4f}")

    results = {
        "test_accuracy": float((y_test == y_pred).mean()),
        "test_f1": float(2 * (cm[1, 1] / (cm[1, 1] + (cm[0, 1] + cm[1, 0]) / 2)) if (cm[1, 1] + (cm[0, 1] + cm[1, 0]) / 2) > 0 else 0.0),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
    }
    np.save(os.path.join(MODELS_DIR, "results_summary.npy"), results)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
