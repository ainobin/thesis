"""
fig_mel_spectrogram_samples.py — Standalone Figure Generator

Generates Fig. 2: Sample Mel-spectrogram representations showing the
distinct time-frequency energy decay of Grade A, Grade B, and Grade C
brick impacts.

This script is independent from the main thesis pipeline.
It reads raw WAV files directly and produces a publication-ready figure.

Usage:
  python figures/fig_mel_spectrogram_samples.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Audio parameters (matches preprocessing.py) ────────────────────────────
SR = 22050
FIXED_SAMPLES = int(SR * 0.8)  # 0.8 s
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

GRADES = [
    ("Grade A", os.path.join(RAW_DIR, "grade_a")),
    ("Grade B", os.path.join(RAW_DIR, "grade_b")),
    ("Grade C", os.path.join(RAW_DIR, "grade_c")),
]


def load_and_mel(file_path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file and return its Mel-spectrogram (dB) and SR."""
    y, sr = librosa.load(file_path, sr=SR, mono=True)

    # trim silence
    y, _ = librosa.effects.trim(y, top_db=30)

    # pad / crop to fixed length
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
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db, sr


def pick_sample(folder: str, seed: int = 42) -> str:
    """Pick a deterministic sample WAV from a folder."""
    wavs = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(wavs))
    return os.path.join(folder, wavs[idx])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.07, right=0.88, wspace=0.35)

    for ax, (label, folder) in zip(axes, GRADES):
        wav_path = pick_sample(folder)
        mel_db, sr = load_and_mel(wav_path)

        n_frames = mel_db.shape[1]
        duration = n_frames * HOP_LENGTH / sr

        img = ax.imshow(
            mel_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=mel_db.min(),
            vmax=mel_db.max(),
            extent=[0, duration, 0, sr / 2],
            interpolation="nearest",
        )

        ax.set_title(label, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Frequency (Hz)" if ax == axes[0] else "", fontsize=11)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, sr / 2)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.set_yticks([0, 2048, 4096, 6144, 8192])
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        "Sample Mel-spectrogram representations showing the distinct time-\n"
        "frequency energy decay of Grade A, Grade B, and Grade C brick impacts",
        fontsize=13,
        y=1.0,
    )

    cbar = fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.7, pad=0.02)
    cbar.set_label("Amplitude (dB)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    out_path = os.path.join(FIGURES_DIR, "fig_mel_spectrogram_samples.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
