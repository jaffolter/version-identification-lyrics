# src/livi/core/data/cli.py

import typer
from livi.core.data.factory import (
    run_create_audio_encoder_dataset,
    split_metadata,
    remove_tracks_errors_editorial,
    split_q1_median_q3,
)
from livi.core.data.preprocessing.whisper_feature_extractor import run_extract_whisper_features
from livi.core.data.download.download_audio import run_download_audio
from livi.core.data.preprocessing.vocal_detector import run_extract_vocals_dataset, run_detection_dataset
from typing import Optional, List
from livi.config import settings
from pathlib import Path
from loguru import logger


app = typer.Typer(help="Commands for data processing")


# --------------------------------------------------------------------
# Simple smoke test command
# Run: poetry run livi-data hello
# --------------------------------------------------------------------
@app.command()
def hello():
    """Print a hello message (used for quick smoke test)."""
    typer.echo("[data] Hello, World!")


# --------------------------------------------------------------------
# Command to download audio files from a CSV
#
# Purpose:
#   Download audio tracks from Deezer's TrackStorage given a CSV file
#   containing track identifiers and encoding information. Each track
#   is saved locally under <out-dir>/<id>.mp3
#
# Requirements:
#   - See https://github.deezerdev.com/Research/deezer-datasource for details
#   - CSV file must exist and contain at least the following columns:
#       <id_col>        : identifier of the track (e.g., md5_encoded)
#       <encoding_col>  : encoding format
#
# Typical usage:
#   poetry run livi-data download-audio \
#       --csv-path src/livi/test_data/covers80.csv \
#       --out-dir src/livi/test_data/audio/ \
#       --dataset-name covers80 \
#       --id-col md5_encoded \
#       --encoding-col encoding \
#       --num-workers 4
#
# Arguments:
#   csv_path    : Path to CSV file listing tracks to download.
#   out_dir     : Output directory for downloaded audio.
#   dataset_name : Name of the dataset (used for default paths).
#   id_col      : CSV column containing track IDs. Default = "md5_encoded".
#   encoding_col: CSV column containing encodings. Default = "encoding".
#   num_workers : Number of parallel worker processes. Default = 1 (sequential).
# --------------------------------------------------------------------
@app.command("download-audio")
def cli_download_audio(
    csv_path: Optional[Path] = typer.Option(None, help="CSV file listing tracks to download."),
    out_dir: Optional[Path] = typer.Option(None, help="Output directory for downloaded audio."),
    dataset_name: str = typer.Option("covers80", help="Name of the dataset (used for default paths)."),
    id_col: str = typer.Option("md5_encoded", help="CSV column name of the track identifier."),
    encoding_col: str = typer.Option("encoding", help="CSV column name of the encoding."),
    num_workers: int = typer.Option(8, help="Number of parallel worker processes."),
):
    """
    Download audio files listed in a CSV using Deezer's TrackStorage.

    For each row, writes <out-dir>/<id>.mp3
    """
    # Default path fallbacks

    csv_path = csv_path or settings.DATA_DIR / "benchmarks" / f"{dataset_name}.csv"
    out_dir = out_dir or settings.AUDIO_DIR / dataset_name

    logger.info(f"Downloading audio files listed in {csv_path} to {out_dir}")

    run_download_audio(
        csv_path=csv_path,
        out_dir=out_dir,
        id_col=id_col,
        encoding_col=encoding_col,
        num_workers=num_workers,
    )


# --------------------------------------------------------------------
# Command to run Vocal Detector + Vocal Segments extraction over a dataset
#
# Purpose:
#   Detect vocal segments per track using the Vocal model,
#   then pad/merge/slice them into chunk windows. Results
#   are saved to a CSV so you can reuse them downstream (transcription,
#   feature extraction, etc.).
#
# Requirements:
#   - A metadata CSV containing at least a track identifier. You can pass:
#       • --id-col     : column with md5 and we’ll infer "<audio_dir>/<md5>.mp3"
#
# Typical usage:
#   poetry run livi-data extract-vocals-dataset \
#       --metadata-path src/livi/test_data/covers80.csv \
#       --audio-dir src/livi/test_data/audio/ \
#       --out-path src/livi/test_data/covers80_vocals.csv
#
# Arguments:
#   metadata_path           : (Optional) Path to metadata CSV.
#   audio_dir               : (Optional) Directory containing audio files.
#   out_path                : (Optional) Output CSV path for detection results.
#   dataset_name            : (Optional) Name of the dataset (used for default paths).
#   id_col                  : (Optional) Column with track id (e.g. md5).
#   vocal_threshold         : (Optional) Per-segment vocalness threshold. Default = 0.5.
#   mean_vocalness_threshold: (Optional) Track-level mean vocalness threshold. Default = 0.5.
#   sample_rate             : (Optional) Expected sampling rate for waveform. Default = 16,000.
#   chunk_sec               : (Optional) Length (sec) of target chunks. Default = 30.0.
#   max_total_pad_sec       : (Optional) Max total padding per segment (L+R). Default = 10.0.
# --------------------------------------------------------------------
@app.command("extract-vocals-dataset")
def cli_extract_vocals_dataset(
    metadata_path: Path = typer.Option(None, help="Path to metadata CSV."),
    audio_dir: Path = typer.Option(None, help="Directory containing audio files."),
    dataset_name: str = typer.Option("covers80", help="Name of the dataset (used for default paths)."),
    out_path: Path = typer.Option(None, help="Where to write the results CSV."),
    id_col: Optional[str] = typer.Option(
        "md5_encoded",
        help="Column holding the track id (used if --audio-col is not provided).",
    ),
    vocal_threshold: float = typer.Option(0.5, help="Per-segment vocalness threshold."),
    mean_vocalness_threshold: float = typer.Option(0.5, help="Track-level mean vocalness threshold."),
    sample_rate: int = typer.Option(16_000, help="Expected sample rate of audio on disk."),
    chunk_sec: float = typer.Option(30.0, help="Target chunk length in seconds."),
    max_total_pad_sec: float = typer.Option(10.0, help="Max total padding per segment (L+R)."),
):
    """
    Detect vocal segments for each row in a metadata CSV and save results to a CSV.
    """
    # Fallback to default value
    metadata_path = metadata_path or settings.DATA_DIR / "benchmarks" / f"{dataset_name}.csv"
    audio_dir = audio_dir or settings.AUDIO_DIR / dataset_name
    out_path = out_path or settings.DATA_DIR / "benchmarks" / dataset_name / f"{dataset_name}_res_vocal_detection.csv"

    run_extract_vocals_dataset(
        metadata_path=metadata_path,
        audio_dir=audio_dir,
        out_path=out_path,
        id_col=id_col,
        vocal_threshold=vocal_threshold,
        mean_vocalness_threshold=mean_vocalness_threshold,
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
        max_total_pad_sec=max_total_pad_sec,
    )


# --------------------------------------------------------------------
# Command to run simple Vocal Detection (no post-processing) over a dataset
#
# Purpose:
#   Apply the VocalDetector to each track in a metadata CSV and save
#   results (is_vocal, mean_vocalness, etc.) into a new CSV.
#
# Requirements:
#   - A metadata CSV containing at least a track identifier. You can pass:
#       • --id-col     : column with md5 and we’ll infer "<audio_dir>/<md5>.mp3"
#
# Typical usage:
#   poetry run livi-data run-detection-dataset \
#       --metadata-path src/livi/test_data/covers80.csv \
#       --audio-dir src/livi/test_data/audio/ \
#       --out-path src/livi/test_data/covers80_vocal_detection.csv \
#
# Arguments:
#   metadata_path : Path to metadata CSV file (with "audio_path" col).
#   audio_dir     : Directory containing audio files.
#   out_path      : Output CSV path for detection results.
#   dataset_name  : (Optional) Name of the dataset (used for default paths).
# --------------------------------------------------------------------
@app.command("vocal-detection-dataset")
def cli_run_detection_dataset(
    metadata_path: Path = typer.Option(None, help="Path to metadata CSV (must include 'audio_path')."),
    audio_dir: Path = typer.Option(None, help="Directory containing audio files."),
    out_path: Path = typer.Option(None, help="Output CSV path for detection results."),
    dataset_name: str = typer.Option("covers80", help="Name of the dataset (used for default paths)."),
):
    """
    Run vocal detection on each row in metadata and save results to CSV.
    """
    # Fallback to default value
    metadata_path = metadata_path or settings.DATA_DIR / "benchmarks" / f"{dataset_name}.csv"
    audio_dir = audio_dir or settings.AUDIO_DIR / dataset_name
    out_path = (
        out_path
        or settings.DATA_DIR / "benchmarks" / dataset_name / f"{dataset_name}_res_vocal_detection_no_processing.csv"
    )

    run_detection_dataset(
        metadata_path=metadata_path,
        audio_dir=audio_dir,
        out_path=out_path,
    )


# --------------------------------------------------------------------
# Command to extract Whisper features
#
# Purpose:
#   Extract log-Mel features from raw audio segments using the Whisper
#   feature extractor. Each 30-second chunk is converted into a .npy
#   file, ready for use in training/evaluation of the audio encoder.
#
# Requirements:
#   - Raw audio files must be available in AUDIO_DIR:
#       <AUDIO_DIR>/<MD5_ENCODED>.mp3
#   - Metadata CSV must exist and include the following columns:
#       md5_encoded, start, end, id, chunk_id, batch_id
#       -> It corresponds to the file created after running the vocal extraction
#       pipeline (see `extract-vocals-dataset` command).
#   - Each segment is padded or truncated to SEGMENT_DURATION seconds.
#
# Typical usage:
#   poetry run livi-data extract-whisper-features \
#       --audio-dir src/livi/test_data/audio/ \
#       --metadata-path src/livi/test_data/covers80_vocals.csv \
#       --output-dir src/livi/test_data/mel
#
# Arguments:
#   audio_dir        : Directory containing raw audio files (.mp3).
#                      Defaults to <DATA_DIR>/audio/discogs_vi
#   metadata_path    : Path to metadata CSV file.
#                      Defaults to <DATA_DIR>/audio_encoder_data/metadata/full.csv
#   output_dir       : Directory to save extracted Whisper features (.npy).
#                      Defaults to <DATA_DIR>/audio_encoder_data/mel
#   dataset_name     : (Optional) Name of the dataset (used for default paths).
#   sample_rate      : (Optional) Target sampling rate (Hz). Default = 16,000.
#   segment_duration : (Optional) Segment duration in seconds. Default = 30.
# --------------------------------------------------------------------
@app.command("extract-whisper-features")
def extract_whisper_features(
    audio_dir: Optional[str] = typer.Option(
        None,
        help=("Directory containing raw audio files (.mp3). If omitted, defaults to <DATA_DIR>/audio/discogs_vi"),
    ),
    metadata_path: Optional[str] = typer.Option(
        None,
        help=(
            "Path to metadata CSV file (must include md5_encoded,start,end,id,chunk_id,batch_id). "
            "If omitted, defaults to <DATA_DIR>/audio_encoder_data/metadata/full.csv"
        ),
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        help=(
            "Directory to save extracted Whisper features (.npy). "
            "If omitted, defaults to <DATA_DIR>/audio_encoder_data/mel"
        ),
    ),
    dataset_name: Optional[str] = typer.Option(
        "covers80", "--dataset-name", help="Name of the dataset (used for default paths)."
    ),
    sample_rate: int = typer.Option(16000, help="Target sampling rate (Hz)."),
    segment_duration: int = typer.Option(30, help="Segment duration in seconds."),
    model_name: Optional[str] = typer.Option(
        "openai/whisper-large-v3-turbo", help="Model name for feature extraction."
    ),
):
    """
    Create Whisper log-Mel features from raw audio segments and save them as .npy files.
    """

    # Default paths fallback
    audio_dir = audio_dir or settings.DATA_DIR / f"audio/{dataset_name}"
    metadata_path = metadata_path or settings.DATA_DIR / "audio_encoder_data/metadata/full.csv"
    output_dir = output_dir or settings.DATA_DIR / "audio_encoder_data/mel"

    run_extract_whisper_features(
        audio_dir=audio_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        model_name=model_name,
    )


# --------------------------------------------------------------------
# Command to split audio encoder dataset into train/val/test sets
#
# Purpose:
#   Filter metadata rows to only those with a fixed duration (default 30s),
#   create unique IDs, and split into train/val/test sets.
#
# Requirements:
#   - Input CSV must contain at least:
#       version_id, chunk_id, start, end
#
# Typical usage:
#   poetry run livi-data split-metadata \
#       --csv-path data/audio_encoder_dataset/metadata/full.csv \
#       --out-dir data/audio_encoder_dataset/metadata/splits \
#       --duration 30
#
# Arguments:
#   csv_path     : Path to metadata CSV. Defaults to <DATA_DIR>/audio_encoder_dataset/metadata/full.csv
#   out_dir      : Output directory for split CSVs. Defaults to <DATA_DIR>/audio_encoder_dataset/metadata
#   duration     : Required segment duration (seconds). Default = 30.0
#   train_frac   : Fraction of rows for train set. Default = 0.8
#   val_frac     : Fraction of remaining rows for validation. Default = 0.5
#   random_state : Random seed for reproducibility. Default = 42
# --------------------------------------------------------------------
@app.command("split-metadata")
def cli_split_metadata(
    csv_path: Optional[Path] = typer.Option(None, help="Path to input metadata CSV file."),
    out_dir: Optional[Path] = typer.Option(None, help="Output directory for split CSV files."),
    duration: float = typer.Option(30.0, help="Required segment duration in seconds."),
    train_frac: float = typer.Option(0.8, help="Fraction of rows for train set."),
    val_frac: float = typer.Option(0.5, help="Fraction of remaining rows for validation set."),
    random_state: int = typer.Option(42, help="Random seed for reproducibility."),
):
    # Defaults
    if csv_path is None:
        csv_path = settings.DATA_DIR / "audio_encoder_dataset/metadata/full.csv"
    if out_dir is None:
        out_dir = settings.DATA_DIR / "audio_encoder_dataset/metadata"

    split_metadata(
        csv_path=csv_path,
        out_dir=out_dir,
        duration=duration,
        train_frac=train_frac,
        val_frac=val_frac,
        random_state=random_state,
    )


# --------------------------------------------------------------------
# Command to create the audio encoder dataset
#
# Purpose:
#   Generate the WebDataset shards for training the audio encoder model for a
#   given split (train, val, test).
#
# Requirements:
#   - Lyrics-informed embeddings and Mel-spectrograms must be pre-computed.
#   - Files should be organized with fan-out directories:
#       <MEL_DIR>/<UID[:3]>/<UID>.npy
#       <EMBED_DIR>/<UID[:3]>/<UID>.npy
#   - Metadata CSV (<split>.csv) must exist with at least:
#       VERSION_ID and CHUNK_ID to create the UID  # e.g. <VERSION_ID><CHUNK_ID>
#
# Typical usage:
#   poetry run livi-data create-audio-encoder-dataset \
#       --split train \
#       --path-out "data/audio_encoder_dataset/train/shard-%06d.tar" \
#       --metadata-path "data/audio_encoder_dataset/metadata/train.csv" \
#       --mel-path "data/audio_encoder_dataset/mel" \
#       --lyrics-embeddings-path "data/audio_encoder_dataset/lyrics_embeddings" \
#       --shard-size 1000
#
# Arguments:
#   split                : Dataset split (train/val/test).
#   path_out             : Output path pattern for shards.
#                          Defaults to <DATA_DIR>/audio_encoder_dataset/<split>/shard-%06d.tar
#   metadata_path        : Path to metadata CSV.
#                          Defaults to <DATA_DIR>/audio_encoder_dataset/metadata/<split>.csv
#   mel_path             : Root folder for Mel-spectrogram .npy files.
#                          Defaults to <DATA_DIR>/audio_encoder_dataset/mel
#   lyrics_embeddings_path : Root folder for lyrics embeddings .npy files.
#                          Defaults to <DATA_DIR>/audio_encoder_dataset/lyrics_embeddings
#   shard_size           : Maximum number of samples per shard.
# --------------------------------------------------------------------


@app.command("create-audio-encoder-dataset")
def create_audio_encoder_dataset(
    split: str = typer.Option(..., help="Dataset split to process: train/val/test"),
    path_out: Optional[str] = typer.Option(
        None,
        help="Output shard pattern (e.g., data/train/shard-%06d.tar). "
        "If omitted, defaults to <DATA_DIR>/audio_encoder_dataset/<split>/shard-%06d.tar",
    ),
    metadata_path: Optional[str] = typer.Option(
        None,
        help="Path to split metadata CSV with a 'uid' column. "
        "If omitted, defaults to <DATA_DIR>/audio_encoder_dataset/metadata/<split>.csv",
    ),
    mel_path: Optional[str] = typer.Option(
        None,
        help="Root folder for Mel .npy files, organized as <uid[:3]>/<uid>.npy. "
        "If omitted, defaults to <DATA_DIR>/audio_encoder_dataset/mel",
    ),
    lyrics_embeddings_path: Optional[str] = typer.Option(
        None,
        help="Root folder for lyrics embedding .npy files, organized as <uid[:3]>/<uid>.npy. "
        "If omitted, defaults to <DATA_DIR>/audio_encoder_dataset/lyrics_embeddings",
    ),
    shard_size: int = typer.Option(1000, help="Maximum number of samples per shard."),
):
    """Create sharded WebDataset archives from precomputed Mel features + lyrics embeddings."""

    # Default paths fallback
    metadata_path = metadata_path or settings.DATA_DIR / "audio_encoder_dataset" / "metadata" / f"{split}.csv"
    mel_path = mel_path or settings.DATA_DIR / "audio_encoder_dataset" / "mel"
    lyrics_embeddings_path = lyrics_embeddings_path or settings.DATA_DIR / "audio_encoder_dataset" / "lyrics_embeddings"

    run_create_audio_encoder_dataset(
        split=split,
        path_out=path_out,
        metadata_path=metadata_path,
        mel_path=mel_path,
        lyrics_embeddings_path=lyrics_embeddings_path,
        shard_size=shard_size,
    )


# --------------------------------------------------------------------
# CLI: split-q1-median-q3
#
# Purpose
#   Split a dataset into subsets based on the quartiles (Q1, Median, Q3)
#   of the `vocalness_score` column. This helps in stratifying the data
#   by vocalness intensity for training or evaluation.
#
# Requirements
#   - Metadata CSV file must exist and contain a `vocalness_score` column.
#
# Outputs
#   - Optionally saves three CSV files in the specified output directory:
#       q1.csv : All rows with `vocalness_score >= Q1`
#       q2.csv : All rows with `vocalness_score >= Median`
#       q3.csv : All rows with `vocalness_score >= Q3`
#
# Typical usage
#   poetry run livi-data split-q1-median-q3 \
#       --metadata-path data/vocals/vocal_segments.csv \
#       --output-dir data/vocals/splits
#       --dataset-name covers80
#
# Arguments
#   metadata_path : Path to the metadata CSV file.
#   output_dir    : Directory where the resulting CSVs are saved.
#                   If omitted, results are not written to disk.
#   dataset_name   : Name of the dataset (used for default paths).
# --------------------------------------------------------------------
@app.command("split-q1-median-q3")
def cli_split_q1_median_q3(
    path_metadata: Path = typer.Option(
        None, "--metadata-path", help="Path to metadata CSV with 'vocalness_score' column."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Directory to save split CSVs (q1.csv, q2.csv, q3.csv)."
    ),
    dataset_name: Optional[Path] = typer.Option(
        "covers80", "--dataset-name", help="Name of the dataset (used for default paths)."
    ),
):
    """
    Split dataset into Q1/Median/Q3 subsets by `vocalness_score`.

    Example usage:
        poetry run livi-data split-q1-median-q3 \\
            --metadata-path data/vocals/vocal_segments.csv \\
            --output-dir data/vocals/splits
    """
    # Default fallback
    path_metadata = path_metadata or settings.DATA_DIR / "benchmarks" / f"{dataset_name}" / f"{dataset_name}.csv"
    output_dir = output_dir or settings.DATA_DIR / "benchmarks" / f"{dataset_name}"

    split_q1_median_q3(path_metadata=path_metadata, output_dir=output_dir)


if __name__ == "__main__":
    app()
