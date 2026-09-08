"""
fig_mel_spectrogram_grid.py — Standalone Figure Generator

Generates a 3x3 grid of Mel-spectrograms: 3 rows × (Grade A, Grade B, Grade C),
each row showing a different sample to illustrate the distinct time-frequency
energy decay patterns across brick quality grades.

Usage:
  python figures/fig_mel_spectrogram_grid.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

SR = 22050
FIXED_SAMPLES = int(SR * 0.8)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

GRADES = [
    ("Grade A", os.path.join(RAW_DIR, "grade_a")),
    ("Grade B", os.path.join(RAW_DIR, "grade_b")),
    ("Grade C", os.path.join(RAW_DIR, "grade_c")),
]


def load_and_mel(file_path):
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    n = len(y)
    if n < FIXED_SAMPLES:
        pad_left = (FIXED_SAMPLES - n) // 2
        pad_right = FIXED_SAMPLES - n - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")
    elif n > FIXED_SAMPLES:
        start = (n - FIXED_SAMPLES) // 2
        y = y[start : start + FIXED_SAMPLES]
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max), sr


def pick_sample(folder, seed=42):
    wavs = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))
    rng = np.random.default_rng(seed)
    return os.path.join(folder, wavs[rng.integers(len(wavs))])


def main():
    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 13))

    for col_idx, (label, folder) in enumerate(GRADES):
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            wav_path = pick_sample(folder, seed=42 + row_idx)
            mel_db, sr = load_and_mel(wav_path)

            n_frames = mel_db.shape[1]
            duration = n_frames * HOP_LENGTH / sr

            img = ax.imshow(
                mel_db, aspect="auto", origin="lower", cmap="magma",
                extent=[0, duration, 0, sr / 2], interpolation="nearest",
            )

            ax.set_xlim(0, 0.8)
            ax.set_ylim(0, sr / 2)
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
            ax.set_yticks([0, 2048, 4096, 6144, 8192])
            ax.tick_params(axis="both", labelsize=8)

            if col_idx == 0:
                ax.set_ylabel("Frequency (Hz)", fontsize=9)
            else:
                ax.set_ylabel("")

            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            else:
                ax.set_xlabel("")

            if row_idx == 0:
                ax.set_title(label, fontsize=14, fontweight="bold", pad=10)

    fig.suptitle(
        "Sample Mel-spectrogram representations showing the distinct time-\n"
        "frequency energy decay of Grade A, Grade B, and Grade C brick impacts",
        fontsize=14, y=0.98,
    )

    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.07, right=0.88,
                        hspace=0.30, wspace=0.30)

    cbar = fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.6, pad=0.02)
    cbar.set_label("Amplitude (dB)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    out_path = os.path.join(FIGURES_DIR, "fig_mel_spectrogram_grid.png")
    fig.savefig(out_path, dpi=300, pad_inches=0.3, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
