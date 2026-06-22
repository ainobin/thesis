from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_RAW = BASE / "data" / "raw"
DATA_PROCESSED = BASE / "data" / "processed"
SPECTROGRAM_DIR = DATA_PROCESSED / "spectrograms"
MODEL_DIR = BASE / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
EXPORT_DIR = MODEL_DIR / "exported"
LOG_DIR = BASE / "logs"
SRC_DIR = BASE / "src"
CONFIG_DIR = BASE / "config"

GRADES = ["grade_a", "grade_b", "grade_c"]
CLASS_MAP = {"grade_a": 0, "grade_b": 1, "grade_c": 2}
CLASS_MAP_INV = {v: k for k, v in CLASS_MAP.items()}
