"""
visualize.py — Results Visualisation & Figure Generation

Purpose:
  Generates all publication-ready figures for the thesis using the
  trained model and preprocessed data.

Figures (7 total, saved to figures/):
  1. spectrogram_grid.png       — 4x6 random sample Mel-spectrograms
  2. training_curves_final.png  — loss & accuracy over epochs
  3. class_distribution.png     — train/val/test bar chart
  4. confusion_matrix.png       — normalised heatmap
  5. roc_curve.png              — ROC with AUC (one-vs-rest per class)
  6. pr_curve.png               — Precision-Recall with AUC (one-vs-rest per class)
  7. misclassifications.png     — false positives & false negatives

Also prints a per-class metrics table and updates results_summary.npy.

Usage:
  python src/visualize.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    accuracy_score,
    precision_recall_fscore_support,
)
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

LABELS = ["Grade A", "Grade B", "Grade C"]


def _load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def _load_model():
    path = os.path.join(MODELS_DIR, "best.keras")
    if not os.path.exists(path):
        path = os.path.join(MODELS_DIR, "final_best.keras")
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
    n_classes = y_prob.shape[1]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["steelblue", "coral", "seagreen"]
    for i in range(n_classes):
        binary_y = (y_test == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_y, y_prob[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=2, color=colors[i % len(colors)],
                label=f"{LABELS[i]} (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (One-vs-Rest)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pr_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_roc_curve(y_test, y_prob):
    n_classes = y_prob.shape[1]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["steelblue", "coral", "seagreen"]
    for i in range(n_classes):
        binary_y = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_y, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                label=f"{LABELS[i]} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (One-vs-Rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_confusion_matrix(y_test, y_pred):
    labels = list(range(len(LABELS)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return cm


def plot_class_distribution(y_train, y_val, y_test):
    n_classes = len(LABELS)
    colors = ["steelblue", "coral", "seagreen"]
    counts = {}
    for split_name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts[split_name] = [int((y == i).sum()) for i in range(n_classes)]
    x = np.arange(len(counts))
    width = 0.8 / n_classes
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(n_classes):
        offset = (i - (n_classes - 1) / 2) * width
        vals = [c[i] for c in counts.values()]
        ax.bar(x + offset, vals, width, label=LABELS[i], color=colors[i % len(colors)])
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.1, str(v), fontsize=8, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(counts.keys())
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution Across Splits")
    ax.legend()
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


def bootstrap_ci(y_true, y_pred_proba, n_iterations=1000, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for _ in range(n_iterations):
        idx = rng.choice(n, n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred_proba[idx].argmax(axis=1)

        metrics["accuracy"].append(accuracy_score(y_true_boot, y_pred_boot))
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_boot, y_pred_boot, average="macro", zero_division=0
        )
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["f1"].append(f1)

    ci = {}
    for metric, values in metrics.items():
        values_sorted = sorted(values)
        lower = values_sorted[int(n_iterations * alpha / 2)]
        upper = values_sorted[int(n_iterations * (1 - alpha / 2))]
        ci[metric] = (lower, upper)
    return ci


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
    plot_roc_curve(y_test, y_prob)
    plot_pr_curve(y_test, y_prob)
    plot_misclassifications(X_test, y_test, y_pred, y_prob)

    print("\n--- Per-Class Metrics ---")
    print(f"  {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"  {'-'*58}")
    for i, label in enumerate(LABELS):
        support = int((y_test == i).sum())
        p = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0.0
        r = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"  {label:<12} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {support:<10}")

    print("\n--- Bootstrap 95% Confidence Intervals (1000 iterations) ---")
    ci = bootstrap_ci(y_test, y_prob)
    print(f"  {'Metric':<12} {'Point Est.':<12} {'95% CI':<24}")
    print(f"  {'-'*48}")
    point_acc = accuracy_score(y_test, y_pred)
    point_p, point_r, point_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    print(f"  {'Accuracy':<12} {point_acc:<12.4f} ({ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f})")
    print(f"  {'Precision':<12} {point_p:<12.4f} ({ci['precision'][0]:.4f}, {ci['precision'][1]:.4f})")
    print(f"  {'Recall':<12} {point_r:<12.4f} ({ci['recall'][0]:.4f}, {ci['recall'][1]:.4f})")
    print(f"  {'F1':<12} {point_f1:<12.4f} ({ci['f1'][0]:.4f}, {ci['f1'][1]:.4f})")

    results = {
        "test_accuracy": float(point_acc),
        "test_f1": float(point_f1),
        "test_precision": float(point_p),
        "test_recall": float(point_r),
        "confusion_matrix": cm.tolist(),
        "bootstrap_ci_95": {k: [float(v[0]), float(v[1])] for k, v in ci.items()},
    }
    np.save(os.path.join(MODELS_DIR, "results_summary.npy"), results)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
