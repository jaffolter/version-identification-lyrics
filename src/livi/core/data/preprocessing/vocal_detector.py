from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Sequence, Optional
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from loguru import logger
from tqdm import tqdm
import os
from loguru import logger

from livi.core.data.utils.audio_toolbox import load_audio

# --- Fix for torch loading numpy scalars (as in your original) ---
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# External model providing segment-level vocalness
from instrumentalvocal import Session


class VocalDetector:
    """
    Lightweight wrapper around `instrumentalvocal.Session` with fixed, reproducible behavior.

    Pipeline:
    1) Run the model on an audio file -> segments with (start, end, vocalness).
    2) Keep segments where `vocalness > vocal_threshold`.
    3) Merge consecutive vocal segments that touch exactly (segment.start == previous.end).
    4) Compute mean vocalness across all segments (before filtering).
    5) If mean vocalness < mean_vocalness_threshold -> reject track.
    6) For accepted tracks:
        a) Symmetrically pad each kept vocal segment to approach `chunk_sec`,
            with total padding capped by `max_total_pad_sec`.
        b) Merge overlapping/touching padded windows.
        c) Slice into exact `chunk_sec` chunks; shorter leftovers.

    Parameters
    ----------
    vocal_threshold : float
        Per-segment threshold to classify a segment as "vocal".
    mean_vocalness_threshold : float
        Track-level acceptance threshold on mean vocalness across all segments.
        If below, the track is considered instrumental / too low vocal content.
    sample_rate : int
        Expected sampling rate for downstream slicing (used by `extract_fixed_chunks`).
    chunk_sec : float
        Exact chunk duration in seconds kept for downstream use (e.g., 30.0).
    max_total_pad_sec : float
        Max total symmetric padding (left + right) applied to each segment
        before merging and slicing. (e.g., 10.0 means up to 5s left + 5s right)
    """

    def __init__(
        self,
        vocal_threshold: float = 0.5,
        mean_vocalness_threshold: float = 0.5,
        sample_rate: int = 16_000,
        chunk_sec: float = 30.0,
        max_total_pad_sec: float = 10.0,
    ) -> None:
        self.model = Session()
        self.vocal_threshold = float(vocal_threshold)
        self.mean_vocalness_threshold = float(mean_vocalness_threshold)

        self.sample_rate = int(sample_rate)
        self.chunk_sec = float(chunk_sec)
        self.max_total_pad_sec = float(max_total_pad_sec)

    # ---------------------------------------------------------------------
    # Stage 1: Detection
    # ---------------------------------------------------------------------
    def detect(self, audio_path: str | Path) -> Tuple[bool, float, List[Dict[str, float]]]:
        """
        Run vocal detection on an audio file.

        Returns
        -------
        is_vocal_track : bool
            True if mean vocalness >= mean_vocalness_threshold, else False.
        mean_vocalness : float
            Mean vocalness across all segments provided by the model.
        vocal_segments : list of {"start": float, "end": float}
            Merged "vocal" segments (vocalness > vocal_threshold and touching segments merged).
        """
        audio_path = str(audio_path)
        try:
            result = self.model.analyze(audio_path)
        except Exception as e:
            print(f"[VocalDetector] Error processing {audio_path}: {e}")
            return False, 0.0, []

        vocal_segments: List[Dict[str, float]] = []
        vocalness_values: List[float] = []
        last_end: float = 0.0  # for merging touching segments

        for seg in result.segments:
            v = float(seg.vocalness)
            s = float(seg.start)
            e = float(seg.end)

            vocalness_values.append(v)

            if v > self.vocal_threshold:
                if vocal_segments and s == last_end:
                    # extend previous if touching
                    vocal_segments[-1]["end"] = e
                    vocal_segments[-1]["vocalness"] = (vocal_segments[-1]["vocalness"] + v) / 2

                else:
                    vocal_segments.append({"start": s, "end": e, "vocalness": v})
                last_end = e

        mean_vocalness = float(np.mean(vocalness_values)) if vocalness_values else 0.0
        accepted = True
        if mean_vocalness < self.mean_vocalness_threshold:
            accepted = False

        return accepted, mean_vocalness, vocal_segments

    # ---------------------------------------------------------------------
    # Stage 2: Pad → Merge → Slice
    # ---------------------------------------------------------------------
    def pad_merge_and_slice(
        self,
        segments: Sequence[Dict[str, float]],
    ) -> List[Tuple[float, float]]:
        """
        1) Symmetric pad each (start,end) to approach `chunk_sec` with total padding
            capped by `max_total_pad_sec`.
        2) Merge overlapping/touching padded windows.
        3) Slice into exact `chunk_sec` chunks when duration > `chunk_sec`.

        Returns
        -------
        chunks_sec : list[(start_sec, end_sec)]
            Time windows in seconds, each of length `chunk_sec`.
        """
        chunk_sec = self.chunk_sec
        max_total_pad = self.max_total_pad_sec

        # 1) pad per segment
        padded: List[Tuple[float, float]] = []
        for s in segments:
            start, end = float(s["start"]), float(s["end"])
            dur = end - start
            if dur > chunk_sec:
                # keep as-is (> chunk_sec). We'll slice later.
                padded.append((start, end))
                continue
            half_pad = min(max_total_pad, (chunk_sec - dur) / 2.0)
            padded.append((max(0.0, start - half_pad), end + half_pad))

        # 2) merge overlapping/touching
        padded.sort()
        merged: List[List[float]] = []
        for s, e in padded:
            if not merged or merged[-1][1] < s:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)

        # 3) slice into exact chunk_sec
        chunks: List[Tuple[float, float]] = []
        for s, e in merged:
            d = e - s
            while d >= chunk_sec:
                chunks.append((s, s + chunk_sec))
                s += chunk_sec
                d = e - s
            if e > s:
                chunks.append((s, e))
        return chunks

    # ---------------------------------------------------------------------
    # Stage 3: Time windows → Audio chunks
    # ---------------------------------------------------------------------
    def extract_fixed_chunks(
        self,
        waveform: torch.Tensor,
        windows_sec: Sequence[Tuple[float, float]],
    ) -> torch.Tensor:
        """
        Convert time windows into fixed-length audio chunks.

        Parameters
        ----------
        waveform : torch.Tensor, shape (T,) or (1, T)
            Mono waveform already at `self.sample_rate`.
        windows_sec : list of (start_sec, end_sec)
            Must all be exactly `self.chunk_sec` long (by construction).

        Returns
        -------
        chunks_audio : torch.Tensor, shape (N, T_chunk)
            N exact-length chunks. If no windows, returns an empty (0, T_chunk) tensor.
        """
        target_len = int(round(self.chunk_sec * self.sample_rate))
        segs: List[torch.Tensor] = []
        for start, end in windows_sec:
            s = int(round(start * self.sample_rate))
            e = int(round(end * self.sample_rate))
            chunk = waveform[s:e]
            # Pad/Truncate
            if chunk.shape[-1] < target_len:
                chunk = F.pad(chunk, (0, target_len - chunk.shape[-1]))
            elif chunk.shape[-1] > target_len:
                chunk = chunk[:target_len]
            segs.append(chunk)

        return np.array(segs)

    # --------------------- High-level single-call helper ---------------------
    def pipeline_no_audio(
        self,
        audio_path: str | Path,
    ) -> Tuple[
        bool,  # is_vocal_track
        float,  # mean_vocalness
        List[Dict[str, float]],  # raw_vocal_segments (results from vocal detection)
        List[Tuple[float, float]],  # chunks_sec (start-end of outputs chunks)
    ]:
        """
        Pipeline (without extracting audio chunks, just extracting start-end times):

        1) Detect + merge touching vocal segments.
        2) Pad → merge → slice into exact `chunk_sec` windows.

        Returns
        -------
        (is_vocal_track, mean_vocalness, raw_vocal_segments, chunks_sec)
        """
        # 1) detection
        is_vocal, mean_voc, raw_segments = self.detect(audio_path)
        if not is_vocal:
            return False, mean_voc, raw_segments, []

        # 2) slicing windows
        chunks_sec = self.pad_merge_and_slice(raw_segments)

        return True, mean_voc, raw_segments, chunks_sec

    def pipeline(
        self,
        audio_path: str | Path,
        waveform: Optional[torch.Tensor] = None,
    ) -> Tuple[
        bool,  # is_vocal_track
        float,  # mean_vocalness
        List[Dict[str, float]],  # raw_vocal_segments (results from vocal detection)
        List[Tuple[float, float]],  # chunks_sec (start-end of outputs chunks)
        torch.Tensor,  # chunks_audio (N, T_chunk)
    ]:
        """
        Full pipeline (with audio chunks extraction):

        1) Detect + merge touching vocal segments.
        2) Pad → merge → slice into exact `chunk_sec` windows.
        3) If `waveform` is provided, convert windows to audio chunks.

        Notes
        -----
        - `waveform` must be mono and already at `sample_rate`.

        Returns
        -------
        (is_vocal_track, mean_vocalness, raw_vocal_segments, chunks_sec, chunks_audio)
        """
        # 1) detection
        is_vocal, mean_voc, raw_segments = self.detect(audio_path)
        if not is_vocal:
            return False, mean_voc, raw_segments, [], torch.empty(0)

        # 2) slicing windows
        chunks_sec = self.pad_merge_and_slice(raw_segments)

        # 3) audio extraction
        chunks_audio = self.extract_fixed_chunks(waveform, chunks_sec)
        return True, mean_voc, raw_segments, chunks_sec, chunks_audio


# ------------------------------- Runners -------------------------------
@lru_cache(maxsize=8)
def get_cached_vocal_detector(
    vocal_threshold: float = 0.5,
    mean_vocalness_threshold: float = 0.5,
    sample_rate: int = 16_000,
    chunk_sec: float = 30.0,
    max_total_pad_sec: float = 10.0,
) -> VocalDetector:
    """
    Cache detectors keyed by config to avoid reloading the model repeatedly.
    """
    return VocalDetector(
        vocal_threshold=vocal_threshold,
        mean_vocalness_threshold=mean_vocalness_threshold,
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
        max_total_pad_sec=max_total_pad_sec,
    )


def extract_vocals(
    audio_path: str | Path,
    waveform: torch.Tensor,
    vocal_detector: Optional[VocalDetector],
    vocal_threshold: Optional[float] = 0.5,
    mean_vocalness_threshold: Optional[float] = 0.5,
    sample_rate: Optional[int] = 16_000,
    chunk_sec: Optional[float] = 30.0,
    max_total_pad_sec: Optional[float] = 10.0,
) -> Tuple[
    bool,  # is_vocal_track
    float,  # mean_vocalness
    List[Dict[str, float]],  # raw_vocal_segments (results from vocal detection)
    List[Tuple[float, float]],  # chunks_sec (start-end of outputs chunks)
    torch.Tensor,  # chunks_audio (N, T_chunk)
]:
    """
    Extract vocal components from a waveform, on a single audio file.
    """

    vocal_detector = vocal_detector or get_cached_vocal_detector(
        vocal_threshold=vocal_threshold,
        mean_vocalness_threshold=mean_vocalness_threshold,
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
        max_total_pad_sec=max_total_pad_sec,
    )

    return vocal_detector.pipeline(audio_path=audio_path, waveform=waveform)


def extract_vocals_no_audio(
    audio_path: str | Path,
    vocal_detector: Optional[VocalDetector],
    vocal_threshold: Optional[float] = 0.5,
    mean_vocalness_threshold: Optional[float] = 0.5,
    sample_rate: Optional[int] = 16_000,
    chunk_sec: Optional[float] = 30.0,
    max_total_pad_sec: Optional[float] = 10.0,
) -> Tuple[
    bool,  # is_vocal_track
    float,  # mean_vocalness
    List[Dict[str, float]],  # raw_vocal_segments (results from vocal detection)
    List[Tuple[float, float]],  # chunks_sec (start-end of outputs chunks)
]:
    """
    Extract vocal components from a waveform (only metadata), on a single audio file.
    """

    vocal_detector = vocal_detector or get_cached_vocal_detector(
        vocal_threshold=vocal_threshold,
        mean_vocalness_threshold=mean_vocalness_threshold,
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
        max_total_pad_sec=max_total_pad_sec,
    )

    return vocal_detector.pipeline_no_audio(audio_path=audio_path)


def run_detection(audio_path: Path, detector: Optional[VocalDetector]) -> Dict[str, float]:
    """
    Run vocal detection only on a single file, returning a summary dictionary.

    Args:
        audio_path: Path to the audio file.
        detector: Optional pre-loaded VocalDetector instance. If None, a cached instance will be used.
    Returns:
        A dictionary with keys:
            - "vocal_detected": bool, whether vocals were detected.
            - "vocalness_score": float, mean vocalness score of the track.
            - "vocal_segments": list of dicts
    """
    detector = detector or get_cached_vocal_detector()
    is_vocal, mean_voc, segments = detector.detect(audio_path)
    return {
        "vocal_detected": is_vocal,
        "vocalness_score": mean_voc,
        "vocal_segments": segments,
    }


def run_detection_dataset(
    metadata_path: Path, audio_dir: Path, out_path: Path, id_col: Optional[str] = "md5_encoded"
) -> Dict[str, float]:
    """
    Run vocal detection on a dataset (no extraction of audio chunks) and save datasets in a csv.

    Args:
        metadata_path: Path to the metadata CSV file containing at least an id column
        audio_dir: Directory containing the audio files.
        out_path: Path to save the output CSV with detection results.
        id_col: Column name in the metadata CSV containing the unique track identifier (default:
            "md5_encoded").

    Returns:
        A dictionary with keys:
            - "vocal_detected": bool, whether vocals were detected.
            - "vocalness_score": float, mean vocalness score of the track.
            - "vocal_segments": list of dicts
            - "chunks_sec": list of (start_sec, end_sec) tuples for vocal chunks.
    """

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    # Load dataset
    df = pd.read_csv(metadata_path)

    logger.info(f"Preparing {len(df)} tracks for vocal detection → {out_path}")

    # Load detector
    detector = get_cached_vocal_detector()

    # Apply detection to each audio file
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running Vocal Detector"):
        path_audio = Path(audio_dir) / f"{row[id_col]}.mp3"
        if not path_audio.exists():
            logger.warning(f"Audio file {path_audio} does not exist. Skipping.")
            continue
        result = run_detection(path_audio, detector=detector)
        results.append(result)

    # Save results to CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)


def run_extract_vocals_dataset(
    metadata_path: Path,
    audio_dir: Path,
    out_path: Path,
    id_col: Optional[str] = "md5_encoded",
    vocal_threshold: Optional[float] = 0.5,
    mean_vocalness_threshold: Optional[float] = 0.5,
    sample_rate: Optional[int] = 16_000,
    chunk_sec: Optional[float] = 30.0,
    max_total_pad_sec: Optional[float] = 10.0,
) -> Dict[str, float]:
    """
    Run vocal detection on a dataset and save datasets in a csv.

    Args:
        metadata_path: Path to the metadata CSV file containing at least an id column
        audio_dir: Directory containing the audio files.
        out_path: Path to save the output CSV with detection results.
        id_col: Column name in the metadata CSV containing the unique track identifier (default:
            "md5_encoded").
        vocal_threshold: Per-segment threshold to classify a segment as "vocal" (default: 0.5).
        mean_vocalness_threshold: Mean vocalness threshold to classify a track as "vocal" (default: 0.5).
        sample_rate: Sample rate for audio processing (default: 16,000).
        chunk_sec: Duration of audio chunks in seconds (default: 30.0).
        max_total_pad_sec: Maximum padding duration in seconds (default: 10.0).

    Returns:
        A dictionary with keys:
            - "vocal_detected": bool, whether vocals were detected.
            - "vocalness_score": float, mean vocalness score of the track.
            - "vocal_segments": list of dicts
            - "chunks_sec": list of (start_sec, end_sec) tuples for vocal chunks
            - "chunks_audio": list of audio chunks as numpy arrays
    """

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    # Load dataset
    df = pd.read_csv(metadata_path)

    logger.info(f"Preparing {len(df)} tracks for vocal detection → {out_path}")

    # Load detector
    detector = get_cached_vocal_detector()

    # Apply detection to each audio file
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running Vocal Detector"):
        audio_path = Path(audio_dir) / f"{row[id_col]}.mp3"
        if not audio_path.exists():
            logger.warning(f"Audio file {audio_path} does not exist. Skipping.")
            continue

        is_vocal, mean_voc, raw_segments, chunks_sec = extract_vocals_no_audio(
            audio_path=audio_path,
            vocal_detector=detector,
            vocal_threshold=vocal_threshold,
            mean_vocalness_threshold=mean_vocalness_threshold,
            sample_rate=sample_rate,
            chunk_sec=chunk_sec,
            max_total_pad_sec=max_total_pad_sec,
        )

        result = {
            "version_id": row["version_id"] if "version_id" in row else "",
            "md5_encoded": row[id_col],
            "is_vocal": is_vocal,
            "vocalness_score": mean_voc,
            "res_detection": raw_segments,
            "chunks_sec": chunks_sec,
        }

        results.append(result)

    # Save results to CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
