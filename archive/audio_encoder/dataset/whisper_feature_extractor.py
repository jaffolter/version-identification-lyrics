import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from loguru import logger
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

# === CONFIGURATION ===
AUDIO_DIR = "src/data/audio/audio"
METADATA_PATH = "src/data/to_process.csv"
OUTPUT_DIR = "src/data/new_features"
SAMPLE_RATE = 16_000
SEGMENT_DURATION = 30
BATCH_SIZE = 32
NUM_WORKERS = 4
ID_BATCH = [5, 6]  # List of batch IDs to process (from 1 to 17)

WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"

os.makedirs(OUTPUT_DIR, exist_ok=True)
processor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_NAME)


def extract_whisper_features_batch(waveforms: List[np.ndarray]) -> np.ndarray:
    """
    Extract Whisper features from a batch of waveforms.
    
    Args:
        waveforms (List[np.ndarray]): List of audio waveforms, each as a numpy array.
        
    Returns:
        np.ndarray: Extracted features for each waveform in the batch.
    """
    with torch.no_grad():
        inputs = processor(waveforms, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt", device="cuda")
        return inputs.input_features.cpu().numpy()


def process(md5_encoded: str, df: pd.DataFrame):
    """ 
    Process a batch of audio files, to extract Whisper features.
    
    Args:
        md5_encoded (str): The MD5 encoded of the audio file to process.
        df (pd.DataFrame): DataFrame containing metadata for the audio files.
    """
    waveforms = []
    filenames = []
    valid_examples = []

    # Create a batch of waveforms to process later on -----------------------------------
    try:
        # Retrieve all 30s segments for the given md5_encoded
        rows = df[df["md5_encoded"] == md5_encoded]

        # Load the associated audio (full track)
        waveform, sr = torchaudio.load(f"{AUDIO_DIR}/{md5_encoded}.mp3")

        # Convert to mono and resample
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)  
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Loop over segments and load the corresponding audio [start, end] + pad or truncate to get 30s segments
        for row in rows.itertuples():
            start = float(row.start)
            end = float(row.end)

            waveform_segment = waveform[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]

            # Pad if < 30s
            if waveform_segment.shape[-1] < SAMPLE_RATE * SEGMENT_DURATION:
                pad_len = SAMPLE_RATE * SEGMENT_DURATION - waveform_segment.shape[-1]
                waveform_segment = F.pad(waveform_segment, (0, pad_len))
                
            # Truncate if > 30s
            if waveform_segment.shape[-1] > SAMPLE_RATE * SEGMENT_DURATION:
                waveform_segment = waveform_segment[: SAMPLE_RATE * SEGMENT_DURATION]

            waveforms.append(waveform_segment)
            filenames.append(f"{OUTPUT_DIR}/{row.filename}")

    except Exception as e:
        logger.info(f"[ERROR] Failed example at idx {md5_encoded}: {e}")
        return
    
    # Extract features for the batch. -----------------------------------------------------------------
    try:
        waveforms = np.array(waveforms)
        features_batch = extract_whisper_features_batch(waveforms)
    except Exception as e:
        logger.info(f"[ERROR] Feature extraction failed: {e}")
        return

    # Save the features to disk. ----------------------------------------------------------------------
    for filename, features in zip(filenames, features_batch):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, features)


def main():
    # only select items based on ID_BATCH
    df = pd.read_csv(METADATA_PATH)
    df = df[df["batch_id"].isin(ID_BATCH)]
    
    # create filename based on id and chunk_id
    df["filename"] = df.apply(
        lambda x: f"{x['id'].replace('V-', '').replace('_', '')[:3]}/{x['id'].replace('V-', '').replace('_', '')}{x['chunk_id']}",
        axis=1,
    )

    # process by batches
    for md5 in tqdm(df["md5_encoded"].unique(), desc="Processing batches"):
        process(md5, df)


if __name__ == "__main__":
    main()
