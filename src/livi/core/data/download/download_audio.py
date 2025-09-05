"""
Download audio files from GCS via Deezer's TrackStorage.
https://github.deezerdev.com/Research/deezer-datasource

---------------------------------------------------------------------------
Purpose
    Given a track identifier (md5_encoded) and encoding, download the
    audio file to an output directory using deezer_datasource.gcs.TrackStorage.

Security
    DO NOT hardcode secrets. Put `VAULT_TOKEN=...` in a `.env` file or pass it
    in the shell environment. This script will load `.env` automatically.
    Tips: 
         - vault-login prod 
         - enter: username -> only name, not full email / password -> your actual password
         - echo $VAULT_TOKEN

Requirements 
    - Environment variable VAULT_TOKEN set (e.g. via .env)
    - `deezer_datasource` package configured in your environment

Typical usage
    poetry run livi-data download-audio \
        --csv-path data/metadata.csv \
        --out-dir data/audio \
        --id-col md5_encoded \
        --encoding-col encoding \
        --num-workers 8 

Notes
    - Files are written to <out-dir>/<id>.mp3, e.g., data/audio/083123456.mp3
--------------------------------------------------------------------------- 
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

try:
    from deezer_datasource.gcs import TrackStorage
except Exception as e:
    raise ImportError(
        "Failed to import deezer_datasource.gcs.TrackStorage. Make sure the package is installed and accessible."
    ) from e


# -----------------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------------
def ensure_vault_token() -> str:
    """
    Load .env (if present) and return VAULT_TOKEN from environment.
    Raises if missing.
    """
    load_dotenv()
    token = os.environ.get("VAULT_TOKEN")
    if not token:
        raise EnvironmentError(
            "Missing VAULT_TOKEN in environment. Put it in a .env file "
            "or export it before running.\n\nExample .env:\n  VAULT_TOKEN=xxxxxx\n"
        )
    logger.info(f"VAULT_TOKEN found in environment: {'*' * (len(token) - 4) + token[-4:]}")
    return token


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def _download_one(args) -> Tuple[str, bool, Optional[str]]:
    """
    Worker for Pool.imap_unordered.

    Parameters
    ----------
    args : Tuple[str, str, Path]
        (md5_encoded, encoding, out_dir)
    """
    md5_encoded, encoding, out_dir = args
    try:
        ts = TrackStorage()  # instantiate in the worker process
        out_path = out_dir / f"{md5_encoded}.mp3"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            logger.info(f"Track already downloaded: {md5_encoded}")
            return md5_encoded, True, None

        ts.download_track(md5_encoded, encoding, str(out_dir))
        return md5_encoded, True, None

    except Exception as e:
        return md5_encoded, False, str(e)


def _download_multiple(md5_encoded: List[str], encoding: List[str], out_dir: Path, num_workers: int):
    """
    Downloads multiple audio tracks in parallel.

    Args:
        md5_encoded (List[str]): List of track IDs (md5_encoded).
        encoding (List[str]): List of encodings corresponding to each track ID.
        out_dir (Path): Output directory for downloaded audio files.
        num_workers (int): Number of parallel worker processes.

    """
    work_items = [(i, e, out_dir) for i, e in zip(md5_encoded, encoding)]
    logger.info(f"Preparing to download {len(work_items)} tracks → {out_dir}")

    # Parallel download
    n_workers = max(1, min(num_workers, cpu_count()))
    successes, failures = 0, 0
    errors: List[Tuple[str, str]] = []

    with Pool(processes=n_workers) as pool:
        for track_id, ok, err in tqdm(
            pool.imap_unordered(_download_one, work_items),
            total=len(work_items),
            desc="Downloading audio files from GCS",
        ):
            if ok:
                successes += 1
            else:
                logger.error(f"Failed to download {track_id}: {err}")
                failures += 1
                errors.append((track_id, err or "Unknown error"))

    logger.info(f"✅ Downloads complete: {successes} ok, {failures} failed.")
    if failures:
        logger.warning("Some downloads failed. First 10 errors:")
        for tid, err in errors[:10]:
            logger.warning(f"- {tid}: {err}")


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def run_download_audio(
    csv_path: Path,
    out_dir: Path,
    id_col: str = "md5_encoded",
    encoding_col: str = "encoding",
    num_workers: int = 4,
) -> None:
    """
    Orchestrates downloads from a CSV list.

    Parameters
    ----------
    csv_path : Path
        CSV containing at least columns [id_col, encoding_col].
    out_dir : Path
        Destination directory for downloaded files.
    id_col : str, default "md5_encoded"
        Column name containing track identifiers.
    encoding_col : str, default "encoding"
        Column name containing encoding for each track.
    num_workers : int, default 4
        Number of parallel worker processes.
    """
    # Validate env early
    ensure_vault_token()

    # FS checks
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV and validate columns
    df = pd.read_csv(csv_path)
    for col in (id_col, encoding_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_path}")

    ids: List[str] = df[id_col].astype(str).tolist()
    encs: List[str] = df[encoding_col].astype(str).tolist()

    # Download audio based from ids
    _download_multiple(ids, encs, out_dir, num_workers)
