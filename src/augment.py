import numpy as np
import librosa


def pitch_shift(y: np.ndarray, sr: int, n_steps: int = 2) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_stretch(y: np.ndarray, rate: float = 1.1) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate)


def add_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(y)).astype(np.float32) * noise_factor
    return y + noise
