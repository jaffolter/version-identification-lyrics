"""
Whisper feature extraction pipeline.

------------------------------------------------------------
Purpose:
    Extract log-Mel features from raw audio segments using the
    Whisper feature extractor. Each 30-second chunk is saved as
    a .npy file, ready to be used in downstream datasets.

Expected inputs:
    - Audio files in AUDIO_DIR (e.g. <md5_encoded>.mp3).
    - Metadata CSV with at least:
        md5_encoded, start, end, id, chunk_id
        -> It corresponds to the file created after running the vocal extraction
        pipeline (see `extract-vocals-dataset` command).

Expected outputs:
    - .npy files containing Whisper features
      saved under OUTPUT_DIR / <filename>.npy
------------------------------------------------------------
"""

import ast
import os
from typing import List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from loguru import logger
from tqdm import tqdm
from transformers import WhisperFeatureExtractor
from functools import lru_cache

from livi.core.data.utils.audio_toolbox import load_audio


class WhisperExtractor:
    """
    Wrapper around HuggingFace WhisperFeatureExtractor for log-Mel feature extraction.

    Usage
    -----
    - Input: audio waveforms already chunked into ~30s segments.
    These can be:
        • Raw audio chunks (e.g., sliding windows over a track).
        • Segments produced by the `VocalDetector.extract_vocals()` pipeline.
    - Output: log-Mel spectrogram features (float32 tensors), ready for
    downstream training or inference.

    Notes
    -----
    - This class does *not* perform segmentation itself; you must provide
        fixed-length waveforms externally.
    - All waveforms must be at `self.sample_rate` (e.g., 16 kHz for Whisper).
    """

    def __init__(self, sample_rate: int, model_name: str = "openai/whisper-large-v3-turbo"):
        self.sample_rate = sample_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.processor = WhisperFeatureExtractor.from_pretrained(model_name)

    def pipeline(self, waveforms: List[np.ndarray]) -> torch.Tensor:
        """
        Extract log-Mel features from a list of audio waveforms.

        Args:
            waveforms: List of audio waveforms as numpy arrays.

        Returns:
            np.ndarray: Extracted log-Mel features.
        """
        with torch.no_grad():
            audio_features = self.processor(
                waveforms, sampling_rate=self.sample_rate, padding=True, return_tensors="pt", device="cuda"
            )
            mel = audio_features.input_features.to(self.device)

        return mel


# ------------------------------- RUNNERS -------------------------------


def create_uid(version_id: str, chunk_id: int) -> str:
    """
    Create a unique identifier for each audio chunk.
    Format: <version_id><chunk_id>, with 'V-' and '_' removed from version
    """
    if "V-" in version_id:
        uid = f"{version_id.replace('V-', '').replace('_', '')}{chunk_id}"
    else:
        uid = f"{version_id}{chunk_id}"
    return uid


@lru_cache(maxsize=8)
def get_cached_feature_extractor(
    sample_rate: int = 16_000,
    model_name: str = "openai/whisper-large-v3-turbo",
) -> WhisperExtractor:
    """Cache detectors keyed by config to avoid reloading the model repeatedly."""
    return WhisperExtractor(
        sample_rate=sample_rate,
        model_name=model_name,
    )


def extract_mel(
    waveforms: List[np.ndarray],
    feature_extractor: Optional[WhisperExtractor],
    sample_rate: Optional[int] = 16_000,
    model_name: Optional[str] = "openai/whisper-large-v3-turbo",
) -> torch.Tensor:
    """
    Extract vocal components from a waveform.

    Args:
        waveforms: List of audio waveforms as numpy arrays.
        feature_extractor: Pre-initialized Whisper feature extractor.
        sample_rate: Sampling rate of Whisper (16000).
        model_name: Model name for feature extraction.

    Returns:
        torch.Tensor: Extracted log-Mel features.
    """
    feature_extractor = feature_extractor or get_cached_feature_extractor(
        sample_rate=sample_rate,
        model_name=model_name,
    )
    return feature_extractor.pipeline(waveforms=waveforms)


def extract_whisper_features_batch(
    waveforms: List[np.ndarray],
    processor: WhisperFeatureExtractor,
    sample_rate: int,
) -> np.ndarray:
    """
    Extract Whisper log-Mel features from a batch of waveforms.
    Batch is composed of 30s extracted segments from a single track.

    Args:
        waveforms (List[np.ndarray]): List of audio waveforms as numpy arrays.
        processor (WhisperFeatureExtractor): HuggingFace Whisper feature extractor.
        sample_rate (int): Sampling rate of Whisper (16000).

    Returns:
        np.ndarray: Extracted features for each waveform in the batch.
    """
    with torch.no_grad():
        inputs = processor(
            waveforms,
            sampling_rate=sample_rate,
            padding=True,
            return_tensors="pt",
            device="cuda",
        )
        return inputs.input_features.cpu().numpy()


def process_single_track(
    row: pd.Series,
    audio_dir: str,
    output_dir: str,
    sample_rate: int,
    segment_duration: int,
    processor: WhisperFeatureExtractor,
) -> None:
    """
    Process all segments of a single audio track.

    Args:
        row (pd.Series): Metadata for the audio file.
        audio_dir (str): Directory containing raw audio (.mp3).
        output_dir (str): Directory where .npy features will be stored.
        sample_rate (int): Target sampling rate.
        segment_duration (int): Duration (s) of each segment.
        processor (WhisperFeatureExtractor): Whisper feature extractor.
    """
    waveforms, filenames = [], []

    # Load the audio file, and extract audio segments based on metadata (start, end)
    try:
        # Convert to mono and resample
        waveform = load_audio(f"{audio_dir}/{row['md5_encoded']}.mp3", target_sample_rate=sample_rate)

        # Retrieve all segments for this track
        # chunks_sec = [(start, end), (start, end), ...]
        chunks_sec = ast.literal_eval(row["chunks_sec"])

        for chunk_id, (start, end) in enumerate(chunks_sec):
            start, end = float(start), float(end)
            segment = waveform[int(start * sample_rate) : int(end * sample_rate)]

            # Pad or truncate to fixed length
            if segment.shape[-1] < sample_rate * segment_duration:
                pad_len = sample_rate * segment_duration - segment.shape[-1]
                segment = F.pad(segment, (0, pad_len))
            elif segment.shape[-1] > sample_rate * segment_duration:
                segment = segment[: sample_rate * segment_duration]

            waveforms.append(segment.numpy())

            # Define path where the extracted mel-spectrogram will be saved
            # Format: <output_dir>/<uid[:3]>/<uid>.npy
            uid = create_uid(row["version_id"], chunk_id)
            filenames.append(Path(output_dir) / f"{uid[:3]}/{uid}.npy")

    except Exception as e:
        logger.error(f"❌ Failed track {row['md5_encoded']}: {e}")
        return

    # Extract features for all extracted segments of a track
    try:
        features_batch = extract_whisper_features_batch(waveforms, processor, sample_rate)
    except Exception as e:
        logger.error(f"❌ Feature extraction failed for {row['md5_encoded']}: {e}")
        return

    # Save features
    for filename, features in zip(filenames, features_batch):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, features)


def run_extract_whisper_features(
    audio_dir: str,
    metadata_path: str,
    output_dir: str,
    sample_rate: int = 16000,
    segment_duration: int = 30,
    model_name: Optional[str] = "openai/whisper-large-v3-turbo",
) -> None:
    """
    Extract Whisper features for all tracks in metadata.

    Args:
        audio_dir: Directory containing raw audio files (.mp3).
        metadata_path: Path to CSV metadata file.
        output_dir: Directory to save extracted features.
        sample_rate: Target sampling rate (Hz).
        segment_duration: Segment duration in seconds.
        model_name: Optional model name for feature extraction.
    """
    os.makedirs(output_dir, exist_ok=True)
    processor = WhisperFeatureExtractor.from_pretrained(model_name)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load the dataset and filter to tracks within a pre-defined batch
    df = pd.read_csv(
        metadata_path,
        dtype={"version_id": str},
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting mel-spectrograms"):
        if row["is_vocal"]:
            process_single_track(
                row,
                audio_dir=audio_dir,
                output_dir=output_dir,
                sample_rate=sample_rate,
                segment_duration=segment_duration,
                processor=processor,
            )
