"""
preprocess.py — Audio-to-Mel-Spectrogram Converter

Purpose:
  Converts a raw audio file into a normalized Mel-spectrogram tensor
  ready for CNN consumption.

Pipeline:
  1. librosa.load  (mono, 22050 Hz)
  2. noisereduce   (stationary=False, prop_decrease=0.85)
  3. librosa.effects.trim  (top_db=30)
  4. Symmetric padding or center-crop to 3 s (66150 samples)
  5. Mel-spectrogram  (128 bands, n_fft=2048, hop=512)
  6. power_to_db      (log scale, ref=max)
  7. Min-max normalize  → [0, 1] float32
  8. Add channel dim   → (128, time_steps, 1)

Usage:
  from src.preprocess import preprocess_audio
  feat = preprocess_audio("path/to/file.wav")     # → ndarray (128, 130, 1)
"""

import librosa
import numpy as np
import noisereduce as nr

SR = 22050
FIXED_SAMPLES = SR * 3
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def preprocess_audio(file_path: str) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=SR, mono=True)

    y = nr.reduce_noise(y=y, sr=SR, stationary=False, prop_decrease=0.85)

    y, _ = librosa.effects.trim(y, top_db=30)

    n = len(y)
    if n < FIXED_SAMPLES:
        pad_left = (FIXED_SAMPLES - n) // 2
        pad_right = FIXED_SAMPLES - n - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant", constant_values=0)
    elif n > FIXED_SAMPLES:
        start = (n - FIXED_SAMPLES) // 2
        y = y[start : start + FIXED_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mmin, mmax = mel_db.min(), mel_db.max()
    if mmax - mmin > 1e-8:
        mel_norm = (mel_db - mmin) / (mmax - mmin)
    else:
        mel_norm = np.zeros_like(mel_db)

    return mel_norm.astype(np.float32)[..., np.newaxis]
