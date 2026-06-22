import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

EXPECTED_SR = 44100
EXPECTED_CHANNELS = 1
MAX_DURATION_SEC = 5.0

def verify_wav(filepath: str) -> dict:
    fp = Path(filepath)
    if not fp.exists():
        return {"status": "FAIL", "error": "File not found"}
    if fp.suffix.lower() not in (".wav", ".wave"):
        return {"status": "FAIL", "error": "Not a .wav file"}
    try:
        y, sr = librosa.load(str(fp), sr=None, mono=False)
    except Exception as e:
        return {"status": "FAIL", "error": f"librosa.load failed: {e}"}

    if y.ndim == 2:
        return {"status": "FAIL", "error": f"Stereo file ({y.shape[0]} channels); mono required"}
    duration = librosa.get_duration(y=y, sr=sr)
    result = {
        "status": "PASS",
        "file": fp.name,
        "sample_rate": sr,
        "channels": 1,
        "samples": int(y.shape[0]),
        "duration_sec": round(duration, 3),
        "duration_ok": duration <= MAX_DURATION_SEC,
        "sr_ok": sr == EXPECTED_SR,
    }
    return result

def load_safe(filepath: str, sr: int = EXPECTED_SR) -> np.ndarray:
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    return y.astype(np.float32)

def is_corrupt(filepath: str) -> bool:
    try:
        with sf.SoundFile(filepath) as f:
            _ = f.read(1)
        return False
    except Exception:
        return True
