#!/usr/bin/env bash
set -euo pipefail

echo "--- Brick NDT Setup ---"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Verifying installation..."
python -c "
import librosa
import tensorflow as tf
import numpy as np
import soundfile as sf
import noisereduce as nr
import sklearn
import matplotlib
print(f'librosa {librosa.__version__}')
print(f'TensorFlow {tf.__version__}')
print(f'NumPy {np.__version__}')
print('All imports OK')
"
echo "Setup complete. Run: source venv/bin/activate"
