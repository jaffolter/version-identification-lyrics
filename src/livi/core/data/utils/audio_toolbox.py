"""
Audio toolbox utilities.

Reusable helpers for loading, resampling, and preprocessing audio...
"""

import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Optional, List, Dict, Sequence, Tuple
import numpy as np
import torch.nn.functional as F


def load_audio(
    path: str,
    target_sample_rate: int = 16_000,
    mono: bool = True,
) -> torch.Tensor:
    """
    Load an audio file, convert to mono, and resample.

    Args:
        path (str): Path to the audio file (e.g. .mp3, .wav).
        target_sample_rate (int): Desired sample rate (Hz). Default = 16k.
        mono (bool): If True, convert multi-channel audio to mono.

    Returns:
        torch.Tensor: Waveform tensor of shape (T,), resampled and optionally mono.
    """
    # Load waveform
    waveform, sr = torchaudio.load_with_torchcodec(path)  # shape (C, T)

    # Convert to mono if multiple channels
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    # Resample if needed
    if sr != target_sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform


def split_audio_30s(audio_path: Path, sample_rate: Optional[int] = 16000, chunk_duration: Optional[float] = 30.0):
    """
    Split a single audio file into chunks of 30s

    Args:
        audio_path (Path): Path to the input audio file
        sample_rate (int): Target sample rate for the output chunks. Default=16,000
        chunk_duration (float): Duration of each chunk in seconds (default is 30s)
    """
    waveform = load_audio(audio_path, target_sample_rate=sample_rate)

    T_total = waveform.shape[1]
    chunk_size = chunk_duration * sample_rate

    chunks: List[np.ndarray] = []
    for start in range(0, T_total, chunk_size):
        end = min(start + chunk_size, T_total)
        chunk = waveform[:, start:end]

        # pad the last chunk if it's too short
        if chunk.shape[1] < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - chunk.shape[1]))

        chunks.append(chunk.squeeze(0).numpy())

    return chunks


def split_audio_predefined(
    audio_path: Path,
    windows_sec: Sequence[Tuple[float, float]],
    chunk_duration: Optional[float] = 30.0,
    sample_rate: Optional[int] = 16000,
) -> torch.Tensor:
    """
    Convert time windows into fixed-length audio chunks.

    Parameters
    ----------
    waveform : torch.Tensor, shape (T,) or (1, T)
        Mono waveform already at `self.sample_rate`.
    windows_sec : list of (start_sec, end_sec)
        Must all be exactly `chunk_duration` long (by construction).
    chunk_duration : float
        Duration of each chunk in seconds (default is 30s).
    sample_rate : int
        Sample rate for the audio chunks (default is 16k).

    Returns
    -------
    chunks_audio : torch.Tensor, shape (N, T_chunk)
        N exact-length chunks. If no windows, returns an empty (0, T_chunk) tensor.
    """
    target_len = int(round(chunk_duration * sample_rate))
    segs: List[torch.Tensor] = []

    waveform = load_audio(audio_path, target_sample_rate=sample_rate)

    for start, end in windows_sec:
        s = int(round(start * sample_rate))
        e = int(round(end * sample_rate))
        chunk = waveform[s:e]
        # Pad/Truncate
        if chunk.shape[-1] < target_len:
            chunk = F.pad(chunk, (0, target_len - chunk.shape[-1]))
        elif chunk.shape[-1] > target_len:
            chunk = chunk[:target_len]
        segs.append(chunk)

    return np.array(segs)
