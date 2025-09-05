from __future__ import annotations

import datasets
import os
import subprocess
from functools import partial
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from loguru import logger
import numpy as np
import torch
import torchaudio
from nnAudio.features.cqt import CQT2010v2


# -----------------------------------------------------------------------------
# Low-level: extract CQT for a batch (HF datasets .map(batched=True) style)
# -----------------------------------------------------------------------------
def extract_cqt_batch(
    batch: Mapping[str, List[str]],
    cqt_dir: str | os.PathLike[str],
    fmin: float,
    n_bins: int,
    bins_per_octave: int,
    device: str | torch.device,
    *,
    target_sr: int = 16000,
    hop_length: int = 640,
    overwrite: bool = False,
) -> Dict[str, List[Optional[str]]]:
    """
    Compute CQT features for a batch of audio files and save each as .npy.

    Parameters
    ----------
    batch : mapping with keys "version_id" and "audio_path"
        `version_id`: list[str], unique IDs for each audio file
        `audio_path`: list[str], paths to source audio files (e.g., .mp3)
    cqt_dir : str | Path
        Output directory where <version_id>.cqt.npy files are written.
    fmin : float
        Minimum frequency (Hz) for the CQT.
    n_bins : int
        Total number of CQT bins.
    bins_per_octave : int
        Bins per octave for the CQT.
    device : str | torch.device
        "cuda" / "cpu" / "mps", etc. The transform runs on this device.
    target_sr : int, default 16000
        Target sampling rate. Inputs are resampled if needed.
    hop_length : int, default 640
        Hop size (in samples) for the CQT.
    overwrite : bool, default False
        If False and output exists, skip recomputation.

    Returns
    -------
    dict: {"cqt_path": list[Optional[str]]}
        For each input item, the path to the generated CQT numpy file, or None on failure.
    """
    device = torch.device(device)
    out_dir = Path(cqt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the CQT transform ONCE for the whole batch
    # (nnAudio modules are PyTorch Modules and can run on GPU if available)
    cqt_transform = CQT2010v2(
        sr=target_sr,
        hop_length=hop_length,
        n_bins=n_bins,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        verbose=False,
    ).to(device)

    # helper: convert any audio to a mono tensor at target_sr
    def _load_mono_resampled(path: Path) -> torch.Tensor:
        """
        Load audio (mp3/wav/etc.) into a mono waveform at `target_sr`.
        If torchaudio fails on MP3 due to backend issues, fall back to ffmpeg -> wav.
        """
        try:
            wav, sr = torchaudio.load(str(path))
        except Exception:
            # Fallback: transcode to temp WAV via ffmpeg, then load
            tmp_wav = path.with_suffix("")  # remove extension
            tmp_wav = tmp_wav.parent / f"{tmp_wav.name}_temp.wav"
            logger.info("ffmpeg transcode -> %s", tmp_wav)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-i", str(path), "-ar", str(target_sr), "-ac", "1", str(tmp_wav)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wav, sr = torchaudio.load(str(tmp_wav))
            # Cleanup temp wav
            try:
                tmp_wav.unlink(missing_ok=True)
            except Exception:
                logger.warning("Could not remove temp WAV: %s", tmp_wav)

        # Stereo → mono
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr).to(device)
            wav = resampler(wav.to(device)).cpu()  # keep result on CPU for consistency
            sr = target_sr

        return wav  # shape: [1, num_samples]

    cqt_paths: List[Optional[str]] = []

    # Iterate current batch items
    for version_id, audio_path in zip(batch["version_id"], batch["audio_path"]):
        vid = str(version_id)
        src = Path(audio_path)
        out_path = out_dir / f"{vid}.cqt.npy"

        try:
            if out_path.exists() and not overwrite:
                # Already computed
                cqt_paths.append(str(out_path))
                continue

            logger.info("Processing version_id=%s | audio=%s", vid, src)
            # 1) Load → mono → resample
            wav = _load_mono_resampled(src)  # [1, N]
            wav = wav.to(device)

            # 2) Normalize amplitude gently to [-0.999, 0.999]
            # avoid division by zero with a small epsilon
            max_abs = torch.max(torch.abs(wav))
            denom = torch.clamp(max_abs, min=1e-3)
            wav = (wav / denom) * 0.999

            # 3) Run CQT (output ~ [1, n_bins, time])
            cqt = cqt_transform(wav) + 1e-9  # add epsilon before log
            cqt = cqt.squeeze(0)  # [n_bins, time]

            # 4) Convert to dB scale: 20*log10(x) - 20*log10(max)
            ref = torch.log10(torch.max(cqt))
            cqt_db = 20.0 * torch.log10(cqt) - 20.0 * ref

            # 5) Time-major (T, F) for downstream code
            cqt_db = torch.swapaxes(cqt_db, 0, 1)  # [time, n_bins]

            # 6) Save as float32 numpy
            np.save(str(out_path), cqt_db.cpu().to(torch.float32).numpy())
            cqt_paths.append(str(out_path))

        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg failed for %s: %s", src, e)
            cqt_paths.append(None)
        except Exception as e:
            logger.exception("Failed to process %s (%s): %s", vid, src, e)
            cqt_paths.append(None)

    return {"cqt_path": cqt_paths}


# -----------------------------------------------------------------------------
# High-level: wrap for an entire HF dataset
# -----------------------------------------------------------------------------
def extract_cqt_dataset(
    ds,  # HuggingFace datasets.Dataset or IterableDataset with map()
    dataset_dir: str | os.PathLike[str],
    *,
    fmin: float = 32.0,
    n_bins: int = 96,
    bins_per_octave: int = 12,
    target_sr: int = 16000,
    hop_length: int = 640,
    device: Optional[str | torch.device] = None,
    batch_size: int = 1,
    num_proc: int = 1,
    overwrite: bool = False,
) -> datasets.Dataset:
    """
    Apply CQT extraction to an entire HF dataset and add a `cqt_path` column.

    Parameters
    ----------
    ds : datasets.Dataset
        Must have columns: "version_id", "audio_path".
    dataset_dir : str | Path
        Base directory where `cqt_feat/` will be created.
    fmin, n_bins, bins_per_octave, target_sr, hop_length : CQT parameters
    device : str | torch.device, optional
        Defaults to CUDA if available else CPU.
    batch_size : int
        Batch size for `datasets.Dataset.map(batched=True)`.
    num_proc : int
        Parallel processes for `map`. (Be careful with GPU transforms; keep 1 if using GPU.)
    overwrite : bool
        Recompute even if the .npy already exists.

    Returns
    -------
    datasets.Dataset
        Same dataset with an extra column `cqt_path`.
    """
    # Pick device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("CQT extraction device: %s", device)

    # Ensure output dir exists
    cqt_dir = Path(dataset_dir) / "cqt_feat"
    cqt_dir.mkdir(parents=True, exist_ok=True)

    # Partially apply parameters so map() only provides the batch
    wrapped = partial(
        extract_cqt_batch,
        cqt_dir=str(cqt_dir),
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        device=device,
        target_sr=target_sr,
        hop_length=hop_length,
        overwrite=overwrite,
    )

    # NOTE:
    #   - If device is GPU, keep num_proc=1 (GPU transforms are not fork-safe).
    #   - If CPU-only, you may increase num_proc for speed.
    ds = ds.map(
        wrapped,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Extracting CQT features",
    )

    return ds
