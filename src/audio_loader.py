import librosa
import numpy as np

def load_wav(path: str, sr: int = 44100) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)
