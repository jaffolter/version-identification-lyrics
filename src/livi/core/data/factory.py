"""
Dataset factory: build datasets used for training and evaluation.
"""

import sys
from pathlib import Path
from typing import Literal, Optional, Tuple


import pandas as pd
import webdataset as wds
from tqdm import tqdm
import typer
from loguru import logger
from livi.apps.retrieval_eval.ranker import DenseRanker
from livi.core.data.utils.io_toolbox import get_embeddings
import numpy as np


def remove_tracks_errors_editorial(
    path_embeddings_transcription: Path,
    path_embeddings_editorial: Path,
    path_to_remove: Optional[Path] = None,
    threshold: Optional[float] = 0.7,
):
    """
    Remove tracks with errors in the editorial embeddings. These are retrieved
    using the similarity between transcription and editorial embeddings.
    If their similarity is below a certain threshold, they are considered erroneous
    and removed.

    Args:
        path_embeddings_transcription (Path): Path to the transcription embeddings.
        path_embeddings_editorial (Path): Path to the editorial embeddings.
    """
    embeddings_transcription = get_embeddings(path_embeddings_transcription)
    embeddings_editorial = get_embeddings(path_embeddings_editorial)

    ranker = DenseRanker(embeddings=embeddings_transcription, query_embeddings=embeddings_editorial)

    ids = list(embeddings_transcription.keys())

    # Get similarity matrix
    similarity_matrix = ranker.compute_similarity_matrix()

    # Get diagonal elements and retrieve ids where similarity is below threshold
    diagonal_elements = np.diag(similarity_matrix)

    to_remove = [ids[i] for i in range(len(diagonal_elements)) if diagonal_elements[i] < threshold]

    if path_to_remove:
        with open(path_to_remove, "w") as f:
            for item in to_remove:
                f.write(f"{item}\n")

    logger.info(f"Removed {len(to_remove)} tracks with errors (Total: {len(ids)}).")


def split_metadata(
    csv_path: Path,
    out_dir: Path,
    duration: float = 30.0,
    train_frac: float = 0.8,
    val_frac: float = 0.5,
    random_state: int = 42,
) -> None:
    """
    Load metadata from a CSV, filter for rows of a specific duration,
    create unique IDs, and split into train/val/test sets.

    Args:
        csv_path (Path): Path to the input metadata CSV (must contain 'version_id', 'chunk_id', 'start', 'end').
        out_dir (Path): Directory to save output splits.
        duration (float): Required segment duration (in seconds). Default = 30.0.
        train_frac (float): Fraction of rows for the training set. Default = 0.8.
        val_frac (float): Fraction of the remaining rows for validation. Default = 0.5.
        random_state (int): Random seed for reproducibility.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Filter rows with exact duration
    df = df[df["end"] - df["start"] == duration]
    print(f"Filtered metadata: {len(df)} rows with {duration:.1f} seconds duration.")

    # Create UID = version_id + chunk_id
    def create_uid(version_id: str, chunk_id: int) -> str:
        return f"{version_id.replace('V-', '').replace('_', '')}{chunk_id}"

    df["uid"] = df.apply(lambda x: create_uid(x["version_id"], x["chunk_id"]), axis=1)

    # Train/val/test split
    train_df = df.sample(frac=train_frac, random_state=random_state)
    remaining_df = df.drop(train_df.index)
    val_df = remaining_df.sample(frac=val_frac, random_state=random_state)
    test_df = remaining_df.drop(val_df.index)

    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")

    # Save to disk
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(f"Saved splits to {out_dir}")

    return train_df, val_df, test_df


# ----------------------------------------------------------------------
# Create of dataset split for audio encoder (train, val, test)
# ----------------------------------------------------------------------
def run_create_audio_encoder_dataset(
    split: Literal["train", "val", "test"],
    path_out: Optional[str],
    metadata_path: Optional[str],
    mel_path: Optional[str],
    lyrics_embeddings_path: Optional[str],
    shard_size: int = 1000,
) -> None:
    """
    Create a WebDataset shard archive for a given split.

    Args:
        split: Which split to process ("train", "val", "test").

        path_out: Output path pattern for shards, e.g. "data/train/shard-%06d.tar".
            If None, set by default to <DATA_DIR>/audio_encoder_dataset/<SPLIT>/shard-%06d.tar

        metadata_path: Path of the .csv file containing metadata for the split
            Should include the "version_id" and "chunk_id" columns, to create the uid <VERSION_ID><CHUNK_ID>
            If None, set by default to <DATA_DIR>/audio_encoder_dataset/metadata

        mel_path: Folder containing precomputed audio features (.npy).
            If None, set by default to <DATA_DIR>/audio_encoder_dataset/mel

        lyrics_embeddings_path: Folder containing precomputed lyrics-informed embeddings (.npy).
            If None, set by default to <DATA_DIR>/audio_encoder_dataset/lyrics_embeddings

        shard_size: Maximum number of samples per shard.
    """
    # Load metadata
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    # Create uid column
    df = pd.read_csv(metadata_path)

    df["uid"] = df.apply(
        lambda x: f"{x['version_id'].replace('V-', '').replace('_', '')}{x['chunk_id']}",
        axis=1,
    )
    df = df.sort_values("uid")
    uids = df["uid"].tolist()

    # Ensure output directory exists
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)

    # Write shards
    missing, kept = 0, 0
    with wds.ShardWriter(path_out, maxcount=shard_size) as sink:
        for uid in tqdm(uids, total=len(uids), desc=f"Writing {split} shards"):
            feat_path = Path(mel_path) / uid[:3] / f"{uid}.npy"
            emb_path = Path(lyrics_embeddings_path) / uid[:3] / f"{uid}.npy"

            if not (feat_path.exists() and emb_path.exists()):
                logger.warning(f"Missing files for UID {uid}: {feat_path}, {emb_path}")
                missing += 1
                continue

            sample = {
                "__key__": uid,
                "features.npy": feat_path.read_bytes(),
                "text.npy": emb_path.read_bytes(),
            }
            sink.write(sample)
            kept += 1

    print(f"✅ {split}: {kept} samples written, {missing} skipped.", file=sys.stderr)


def split_q1_median_q3(
    path_metadata: Path, output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into quartile-based subsets using `vocalness_score`.

    This function computes the 25th (Q1), 50th (median), and 75th (Q3)
    percentiles of the `vocalness_score` column, and creates three subsets:

        - Q1: samples with `vocalness_score >= Q1`
        - Q2: samples with `vocalness_score >= median`
        - Q3: samples with `vocalness_score >= Q3`

    Optionally, the resulting subsets are saved as CSV files in
    `output_dir` named `q1.csv`, `q2.csv`, and `q3.csv`.

    Parameters
    ----------
    path_metadata : Path
        Path to the input CSV file. Must contain a column `vocalness_score`.
    output_dir : Path, optional
        Directory to save the resulting CSVs. If None, results are not written.

    Returns
    -------
    (df_q1, df_q2, df_q3) : tuple of pd.DataFrame
        DataFrames corresponding to the three subsets.

    Logs
    ----
    Logs the number of samples in each subset.
    """
    df = pd.read_csv(path_metadata)

    q1, q2, q3 = df["vocalness_score"].quantile([0.25, 0.5, 0.75])
    df_q1 = df[df["vocalness_score"] >= q1]
    df_q2 = df[df["vocalness_score"] >= q2]
    df_q3 = df[df["vocalness_score"] >= q3]

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        df_q1.to_csv(output_dir / "q1.csv", index=False)
        df_q2.to_csv(output_dir / "q2.csv", index=False)
        df_q3.to_csv(output_dir / "q3.csv", index=False)

    logger.info(f"[Q1] {len(df_q1)} samples")
    logger.info(f"[Q2] {len(df_q2)} samples")
    logger.info(f"[Q3] {len(df_q3)} samples")

    return df_q1, df_q2, df_q3
