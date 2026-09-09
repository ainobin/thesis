import os
import re
import glob
import librosa
import soundfile as sf
import numpy as np

def get_next_start_index(output_dir: str, prefix: str) -> int:
    """Finds the highest existing file index to prevent overwriting."""
    if not os.path.exists(output_dir):
        return 1
    existing_files = glob.glob(os.path.join(output_dir, f"{prefix}_*.wav"))
    indices = []
    for f in existing_files:
        match = re.search(rf"{re.escape(prefix)}_(\d+)\.wav$", os.path.basename(f))
        if match:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 1

def auto_segment_audio(
    input_wav: str, 
    output_dir: str, 
    prefix: str = "sample", 
    clip_duration: float = 0.8, 
    pre_attack_buffer: float = 0.1,
    sr: int = 22050
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Determine starting file index
    start_idx_num = get_next_start_index(output_dir, prefix)
    print(f"Starting extraction for '{prefix}' from index: {start_idx_num:03d}")
    
    # 2. Load full audio
    y, sr = librosa.load(input_wav, sr=sr, mono=True)
    
    # 3. Detect onset events (sharp energy spikes)
    onset_frames = librosa.onset.onset_detect(
        y=y, 
        sr=sr, 
        hop_length=512, 
        backtrack=True, 
        pre_max=20, 
        post_max=20, 
        pre_avg=100, 
        post_avg=100, 
        delta=0.2, 
        wait=int(sr * 0.4 / 512)
    )
    
    onset_samples = librosa.frames_to_samples(onset_frames)
    clip_len = int(clip_duration * sr)
    buffer_len = int(pre_attack_buffer * sr)
    
    saved_count = 0
    for strike_sample in onset_samples:
        start_idx = max(0, strike_sample - buffer_len)
        end_idx = start_idx + clip_len
        
        if end_idx > len(y):
            continue
            
        clip = y[start_idx:end_idx]
        
        # Save without overwriting previous extractions
        current_file_num = start_idx_num + saved_count
        out_path = os.path.join(output_dir, f"{prefix}_{current_file_num:03d}.wav")
        sf.write(out_path, clip, sr)
        saved_count += 1
        
    end_idx_num = start_idx_num + saved_count - 1
    print(f"✓ Extracted {saved_count} new strikes ({prefix}_{start_idx_num:03d}.wav to {prefix}_{end_idx_num:03d}.wav) into '{output_dir}'")

if __name__ == "__main__":
    # Clean the second MP3 via ffmpeg first if needed, then run:
    auto_segment_audio("q3s1_clean.wav", "data/raw/grade_c", prefix="grade_c")