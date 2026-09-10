import sys
import os

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.preprocess import audio_to_mel, SR, N_MELS, N_FFT, HOP_LENGTH

# --- Page Configuration ---
st.set_page_config(page_title="Sirius Lab | Acoustic NDT", layout="wide")

# --- Load trained model (cached) ---
@st.cache_resource
def load_model():
    model_path = os.path.join(PROJECT_ROOT, "models", "best.keras")
    return tf.keras.models.load_model(model_path)

MODEL = load_model()
CLASS_NAMES = ["Grade A", "Grade B", "Grade C"]
CLASS_DESC = {
    "Grade A": "High Density / Intact (Metallic Clang)",
    "Grade B": "Medium Density / Minor Flaws",
    "Grade C": "Low Density / Micro-cracks (Dull Thud)",
}
CLASS_COLORS = {"Grade A": "green", "Grade B": "orange", "Grade C": "red"}

# --- UI Header ---
st.title("AI-Powered Brick Quality Assessment")
st.markdown("**Sirius Lab Ltd.** | Automated Acoustic Non-Destructive Testing (NDT)")
st.write("Strike two bricks together and let the AI analyze the Mel-spectrogram to determine its structural grade.")

st.divider()

# --- Input Methods (Live Mic or File Upload) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Live Strike Test")
    recorded_audio = st.audio_input("Click to Record (Strike the bricks now!)")

with col2:
    st.subheader("Upload Backup Audio")
    uploaded_audio = st.file_uploader("Upload pre-recorded .wav file", type=["wav"])

# Determine which audio to process
audio_file = recorded_audio if recorded_audio else uploaded_audio

# --- Processing & Display ---
if audio_file is not None:
    st.success("Audio captured successfully! Processing...")

    st.audio(audio_file)

    # Load audio using librosa
    y, sr = librosa.load(audio_file, sr=SR)

    # 1. Generate Mel-Spectrogram for visualization (raw, no noise reduction)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

    # Create the visualization layout
    st.subheader("AI Analysis & Prediction")
    viz_col, pred_col = st.columns([2, 1])

    with viz_col:
        fig, ax = plt.subplots(figsize=(8, 4))
        img = librosa.display.specshow(mel_spect_db, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='viridis')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel-Spectrogram (Time-Frequency Signature)')
        st.pyplot(fig)

    with pred_col:
        st.write("### Model Prediction")
        with st.spinner("Extracting features and classifying..."):

            # Preprocess with the full pipeline (noise reduction, trim, pad, normalize)
            features = audio_to_mel(y, sr)  # shape: (128, ~130, 1)
            features = np.expand_dims(features, axis=0)  # add batch dim → (1, 128, ~130, 1)

            # Run CNN inference
            predictions = MODEL.predict(features, verbose=0)[0]  # softmax probabilities
            pred_idx = int(np.argmax(predictions))
            confidence = float(predictions[pred_idx])

            grade = CLASS_NAMES[pred_idx]
            desc = CLASS_DESC[grade]
            color = CLASS_COLORS[grade]

            # Display result
            st.markdown(f"<h1 style='text-align: center; color: {color};'>{grade}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'><i>{desc}</i></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 1.2em;'><b>{confidence*100:.1f}%</b> confidence</p>", unsafe_allow_html=True)

            # Confidence bars for all classes
            st.write("**Confidence Scores:**")
            for i, name in enumerate(CLASS_NAMES):
                pct = float(predictions[i]) * 100
                st.progress(int(pct), text=f"{name}: {pct:.1f}%")
