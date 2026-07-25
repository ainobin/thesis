"""
baselines.py — Traditional ML Baselines (SVM & Random Forest)

Purpose:
  Trains SVM (RBF kernel) and Random Forest on flattened Mel-spectrograms
  for comparison against the CNN. Flattens each (128, 130, 1) sample into
  a 16640-dimensional feature vector.

Output:
  Prints accuracy, precision, recall, F1 for both baselines.

Usage:
  python src/baselines.py
"""

import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def main():
    print("Loading data ...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    n_train, h, w, c = X_train.shape
    n_test = X_test.shape[0]
    print(f"  Train: {X_train.shape}  → flatten to ({n_train}, {h*w*c})")
    print(f"  Test:  {X_test.shape}  → flatten to ({n_test}, {h*w*c})")

    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat  = X_test.reshape(n_test, -1)

    models = {
        "SVM (RBF)": SVC(kernel="rbf", random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    print()
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", labels=[0, 1]
        )
        results[name] = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
        print(f"  Per-class:\n{classification_report(y_test, y_pred, target_names=['Grade A', 'Grade B'], zero_division=0)}")

    print("\n--- Baseline Comparison ---")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    print(f"{'CNN (reference)':<20} {'—':<10} {'—':<10} {'—':<10} {'—':<10}")
    print("  (run src/evaluate.py for CNN metrics)")


if __name__ == "__main__":
    main()
