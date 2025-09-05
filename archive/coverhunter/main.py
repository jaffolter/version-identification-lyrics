from __future__ import annotations

from loguru import logger
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset

from coverhunter.extract_cqt import extract_cqt_dataset
from coverhunter.inference import get_coverhunter_embeddings


# Default dataset name; can be overridden by env or CLI later if you wish
DATASET = "covers80"


def main() -> None:
    """
    End-to-end pipeline:
      1) Load dataset metadata (CSV with at least: version_id, md5_encoded or equivalent)
      2) Build paths: audio_path (input MP3/WAV) + cqt_path (output .npy per track)
      3) Extract CQT features to disk
      4) Generate embeddings with the CoverHunter model
    """

    # ------------------------------------------------------------------
    # 0) Configurable paths
    # ------------------------------------------------------------------
    dataset_name = DATASET

    # Directory that contains: config/hparams.yaml and checkpoints/
    model_path = Path("model")

    # Base folder to write features + embeddings
    base_out = Path("/nfs/interns/jaffolter/data/embeddings") / "audio_baselines" / "coverhunter" / dataset_name
    cqt_dir = base_out / "cqt_feat"

    # Input metadata CSV (one row per track)
    csv_path = Path("/nfs/interns/jaffolter/data/benchmarks") / f"{dataset_name}.csv"

    # Root folder containing the raw audio files for this dataset
    # (e.g., /.../audio/covers80/<md5>.mp3)
    audio_root = Path("/nfs/interns/jaffolter/data/audio") / dataset_name

    # ------------------------------------------------------------------
    # 1) Sanity checks & setup
    # ------------------------------------------------------------------
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {csv_path}")

    cqt_dir.mkdir(parents=True, exist_ok=True)
    base_out.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset: {}", dataset_name)
    logger.info("Model path: {}", model_path)
    logger.info("CSV path: {}", csv_path)
    logger.info("Audio root: {}", audio_root)
    logger.info("Output dir: {}", base_out)

    # ------------------------------------------------------------------
    # 2) Load CSV → HuggingFace Dataset
    #    Expect at least a 'version_id' column; for mp3 naming, we try 'md5_encoded'
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)

    required_cols = {"version_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    ds = Dataset.from_pandas(df)

    # For audio file names: if you don't have 'md5_encoded', adapt here to your scheme
    has_md5 = "md5_encoded" in ds.column_names

    def _row_to_paths(x: dict) -> dict:
        """
        Compute file paths for each row.
        - audio_path: where to read the raw audio (mp3/wav)
        - cqt_path  : where to write the CQT numpy file
        """
        # Choose audio filename; adjust this if your dataset uses another field
        if has_md5 and pd.notna(x.get("md5_encoded", None)):
            audio_file = f"{x['md5_encoded']}.mp3"
        else:
            # Fallback: you may encode 'version_id' or another field here
            # e.g., f"{x['version_id']}.mp3" if your audio is named by version_id
            audio_file = f"{x['version_id']}.mp3"  # <-- change if needed

        return {
            "audio_path": str(audio_root / audio_file),
            "cqt_path": str(cqt_dir / f"{x['version_id']}.cqt.npy"),
        }

    ds = ds.map(_row_to_paths)
    logger.info("Dataset loaded with {} rows", len(ds))

    # Optional: quick peek to confirm paths look right
    sample = ds[0]
    logger.debug("Sample paths: audio_path={}, cqt_path={}", sample["audio_path"], sample["cqt_path"])

    # ------------------------------------------------------------------
    # 3) Extract CQT features (writes <version_id>.cqt.npy files)
    # ------------------------------------------------------------------
    logger.info("Extracting CQT features → {}", cqt_dir)
    ds = extract_cqt_dataset(ds, base_out)

    # ------------------------------------------------------------------
    # 4) Compute embeddings (saves <dataset_name>_embeddings.npz in base_out)
    # ------------------------------------------------------------------
    logger.info("Computing embeddings…")
    embeddings = get_coverhunter_embeddings(
        ds=ds,
        model_path=model_path,
        dataset_dir=base_out,
        dataset_name=dataset_name,
    )
    logger.info("Done. Total embeddings: {}", len(embeddings))


if __name__ == "__main__":
    main()
