"""
barchart.py — Class Distribution Bar Chart

Generates a publication-ready bar chart showing the aggregate class
distribution of the acoustic brick-impact dataset.

Usage:
  python figures/barchart.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data ───────────────────────────────────────────────────────────────────
grades = ["Grade A", "Grade B", "Grade C"]
counts = [465, 509, 452]
percentages = [32.6, 35.7, 31.7]

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.5,
})

# ── Colors ─────────────────────────────────────────────────────────────────
colors = ["#3B7DD8", "#2CA02C", "#D62728"]  # blue, green, red
edge_colors = ["#2A5A9E", "#1E7A1E", "#A31E1E"]

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(
    grades, counts,
    color=colors, edgecolor=edge_colors, linewidth=1.0,
    width=0.55, zorder=3,
)

# Gradient-like effect: slight inner glow via a lighter inner bar
for bar, c in zip(bars, colors):
    inner = plt.Rectangle(
        (bar.get_x() + bar.get_width() * 0.15, 0),
        bar.get_width() * 0.7,
        bar.get_height(),
        facecolor=c, alpha=0.15, edgecolor="none", zorder=4,
    )
    ax.add_patch(inner)

# ── Annotations ────────────────────────────────────────────────────────────
for bar, count, pct in zip(bars, counts, percentages):
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    ax.annotate(
        f"{count}",
        xy=(x, y), xytext=(0, 6),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold", color="#222222",
    )
    ax.annotate(
        f"({pct}%)",
        xy=(x, y), xytext=(0, 22),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=10, color="#555555", fontstyle="italic",
    )

# ── Mean line ──────────────────────────────────────────────────────────────
mean_val = np.mean(counts)
ax.axhline(mean_val, color="#888888", linestyle="--", linewidth=0.9, zorder=2, alpha=0.7)
ax.annotate(
    f"Mean: {mean_val:.0f}",
    xy=(2.35, mean_val + 5),
    fontsize=9, color="#666666", fontstyle="italic",
)

# ── Axes ───────────────────────────────────────────────────────────────────
ax.set_ylabel("Number of Audio Samples", labelpad=10)
ax.set_xlabel("Brick Quality Grades", labelpad=10)
ax.set_title(
    "Aggregate Class Distribution of Acoustic Dataset",
    fontsize=14, fontweight="bold", pad=14,
)

ax.set_ylim(0, 620)
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc", zorder=0)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Save ───────────────────────────────────────────────────────────────────
plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "barchart.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
