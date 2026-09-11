# Brick Quality Assessment Demo

AI-powered acoustic Non-Destructive Testing (NDT) for brick structural grading.

## Prerequisites

- Python 3.9+
- A trained model at `models/best.keras` (relative to project root)
- `src/preprocess.py` module (located in project root)

## Setup

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r show-demo/requirements.txt
```

## Running the App

```bash
# From the project root (not from show-demo/)
streamlit run show-demo/app.py
```

The app opens in your browser at `http://localhost:8501`.

## Usage

1. **Live Test** — Click "Record" and strike two bricks together near the mic.
2. **File Upload** — Upload a pre-recorded `.wav` file.

The AI analyzes the Mel-spectrogram and returns one of:

| Grade | Meaning |
|-------|---------|
| Grade A | High Density / Intact (Metallic Clang) |
| Grade B | Medium Density / Minor Flaws |
| Grade C | Low Density / Micro-cracks (Dull Thud) |

## Project Structure

```
Thesis/
├── models/
│   └── best.keras          # Trained CNN model
├── src/
│   └── preprocess.py       # Audio-to-Mel pipeline
└── show-demo/
    ├── app.py              # Streamlit demo
    └── requirements.txt
```
