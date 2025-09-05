# src/livi/apps/training/cli.py
import typer
from pathlib import Path
from livi.utils.paths import p

from typing import Optional, List
from livi.apps.frozen_encoder.models.transcriber import transcribe_dataset
from livi.apps.frozen_encoder.models.text_encoder import encode_text_dataset
from livi.config import settings
from livi.apps.frozen_encoder.infer.session import run_inference_single, run_estimate_time, run_inference

app = typer.Typer(help="Frozen Encoder")


# --------------------------------------------------------------------
# Simple smoke test command
# Run: poetry run livi-frozen-encoder hello
# --------------------------------------------------------------------
@app.command()
def hello():
    """Print a hello message (used for quick smoke test)."""
    typer.echo("[frozen-encoder] Hello, World!")


# --------------------------------------------------------------------
# Command to transcribe a dataset with Whisper
#
# Purpose:
#   Run batched transcription over a metadata CSV + audio dir,
#   optionally using precomputed vocal segments (chunks_sec in CSV).
#
# Requirements:
#   - Metadata CSV must exist. If --vocal is True, it should contain
#     a "chunks_sec" column (list of [start, end] in seconds).
#     Run the `extract-vocals-dataset` command first to create this.
#   - Audio files must exist at <AUDIO_DIR>/<md5_encoded>.mp3
#
# Typical usage:
#   poetry run livi-frozen-encoder transcribe-dataset \
#       --metadata-path src/livi/test_data/covers80_vocals.csv \
#       --audio-dir src/livi/test_data/audio/ \
#       --output-path src/livi/test_data/covers80_transcriptions.csv \
#       --vocal
#
# Arguments:
#   metadata_path     : Path to metadata CSV.
#   audio_dir         : Directory containing audio files (.mp3).
#   output_path       : Where to save the transcriptions CSV.
#   vocal             : If True, use precomputed vocal segments from CSV.
#   dataset_name      : Name of the dataset.
#   translate         : If True, force English output.
#   batch_size        : Generation batch size per forward pass.
#   nb_tracks_max     : Accumulate this many tracks before a batch call.
#   chunk_duration    : Chunk length in seconds when not using vocal segments.
#   model_name        : HF model id (e.g., openai/whisper-large-v3-turbo).
#   device            : "cuda" or "cpu" (auto if omitted).
#   dtype-fp16-on-cuda: Use float16 on CUDA (default: True).
#   sampling_rate     : Expected sampling rate of chunks (default: 16000).
#   num_beams         : Beam search beams (default: 1).
#   condition-on-prev-tokens : Whisper generation knob (default: False).
#   compression-ratio-threshold : Filter for gzip ratio (default: 1.35).
#   temperature       : Repeatable flag; e.g., --temperature 0.0 --temperature 0.2 ...
#   logprob-threshold : Whisper generation knob (default: -1.0).
#   return-timestamps : Whether to return timestamps (default: True).
#   remove-phrase     : Repeatable flag to strip phrases; default: "Thank you.", "music".
#   repeat-threshold  : Collapse repeated words beyond this count (default: 3).
#   min-words-per-chunk: Keep chunks with >= this many words (default: 4).
# --------------------------------------------------------------------
@app.command("transcribe-dataset")
def cli_transcribe_dataset(
    metadata_path: Path = typer.Option(None, "--metadata-path", help="Path to metadata CSV."),
    audio_dir: Path = typer.Option(None, "--audio-dir", help="Directory containing audio files (.mp3)."),
    output_path: Path = typer.Option(None, "--output-path", help="CSV to write transcriptions."),
    dataset_name: str = typer.Option("covers80", "--dataset-name", help="Name of the dataset."),
    vocal: bool = typer.Option(True, help="Use precomputed vocal segments from CSV (chunks_sec)."),
    translate: bool = typer.Option(False, help="Force English output."),
    batch_size: int = typer.Option(64, help="Batch size for transcription."),
    nb_tracks_max: int = typer.Option(32, help="Accumulate this many tracks per batch."),
    chunk_duration: float = typer.Option(30.0, help="Chunk duration (secs) when not using vocal segments."),
    # model / runtime
    model_name: str = typer.Option("openai/whisper-large-v3-turbo", help="HF model id to use."),
    device: Optional[str] = typer.Option(None, help='Device override: "cuda" or "cpu".'),
    dtype_fp16_on_cuda: bool = typer.Option(True, "--dtype-fp16-on-cuda", help="Use float16 on CUDA."),
    sampling_rate: int = typer.Option(16000, help="Expected sampling rate of chunks."),
    num_beams: int = typer.Option(1, help="Beam search width."),
    condition_on_prev_tokens: bool = typer.Option(False, help="Whisper gen knob."),
    compression_ratio_threshold: float = typer.Option(1.35, help="Compression ratio threshold."),
    temperature: List[float] = typer.Option(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help="Repeatable: --temperature 0.0 --temperature 0.2 ..."
    ),
    logprob_threshold: float = typer.Option(-1.0, help="Logprob threshold."),
    return_timestamps: bool = typer.Option(True, help="Return timestamps."),
    remove_phrase: List[str] = typer.Option(["Thank you.", "music"], help="Repeatable: --remove-phrase 'foo'"),
    repeat_threshold: int = typer.Option(3, help="Collapse repeated words beyond this count."),
    min_words_per_chunk: int = typer.Option(4, help="Minimum words to keep a chunk."),
):
    """
    Create (or update) a CSV of transcriptions for a dataset.
    """
    # Fallback to default value
    metadata_path = metadata_path or settings.DATA_DIR / "benchmarks" / f"{dataset_name}_res_vocal_detection.csv"
    audio_dir = audio_dir or settings.AUDIO_DIR / dataset_name
    output_path = (
        output_path or settings.DATA_DIR / "benchmarks" / dataset_name / f"{dataset_name}_res_transcription.csv"
    )

    # Convert lists to tuples where the function expects tuples
    temperature_tuple = tuple(temperature)
    remove_phrases_tuple = tuple(remove_phrase)

    transcribe_dataset(
        path_metadata=metadata_path,
        dir_audio=audio_dir,
        path_output=output_path,
        vocal=vocal,
        translate=translate,
        batch_size=batch_size,
        nb_tracks_max=nb_tracks_max,
        chunk_duration=chunk_duration,
        model_name=model_name,
        device=device,
        dtype_fp16_on_cuda=dtype_fp16_on_cuda,
        sampling_rate=sampling_rate,
        num_beams=num_beams,
        condition_on_prev_tokens=condition_on_prev_tokens,
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=temperature_tuple,
        logprob_threshold=logprob_threshold,
        return_timestamps=return_timestamps,
        remove_phrases=remove_phrases_tuple,
        repeat_threshold=repeat_threshold,
        min_words_per_chunk=min_words_per_chunk,
    )


# ------------------------------------------------------------
# Command: encode-text-dataset
#
# Purpose:
#   Read a CSV, encode a specified text column with the chosen
#   text encoder, and save (id -> embedding) mapping to disk.
#
# Typical usage:
#   poetry run livi-frozen-encoder encode-text-dataset \
#       --metadata-path src/livi/test_data/covers80_transcriptions.csv \
#       --output-path   src/livi/test_data/covers80_transcription.npz \
#       --col-text      joined \
#       --model-name    Alibaba-NLP/gte-multilingual-base \
#       --no-chunking \
#
# Requirements:
#   - Metadata CSV must exist and contain the specified --col-text and --col-id columns.
#
# Arguments:
#   metadata_path       : Path to input CSV.
#   output_path         : Destination file for embeddings (e.g. .npz or .pkl).
#   col_text            : Column containing text to encode.
#   col_id              : Column containing unique row identifiers.
#   model_name          : (Optional) Encoder model identifier if no encoder instance is provided.
#   chunking            : If True, save per-chunk embeddings (id -> (num_chunks, dim) matrix).
#   batch_size          : Number of texts to process per forward pass.
#   get_single_embedding: If True, mean-pool chunk embeddings into a single vector.
# ------------------------------------------------------------
@app.command("encode-text-dataset")
def cli_encode_text_dataset(
    metadata_path: Path = typer.Option(
        ...,
        "--metadata-path",
        exists=True,
        readable=True,
        help="Path to input CSV (must contain --col-text and --col-id).",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output-path",
        help="Destination embeddings file (.npz/.pkl). Parent dirs are created if needed.",
    ),
    col_text: str = typer.Option(
        "transcription",
        "--col-text",
        help="Name of the text column to encode.",
    ),
    col_id: str = typer.Option(
        "md5_encoded",
        "--col-id",
        help="Name of the unique identifier column.",
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model-name",
        help="Text encoder model identifier (used when no encoder instance is provided).",
    ),
    chunking: bool = typer.Option(
        False,
        "--chunking/--no-chunking",
        help="If true, save per-chunk embeddings (id -> (num_chunks, dim)).",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        min=1,
        help="Batch size for text encoding.",
    ),
    get_single_embedding: bool = typer.Option(
        False,
        "--get-single-embedding",
        help="If true, mean-pool chunk embeddings into a single vector.",
    ),
):
    """
    Encode a column of text from a CSV and save embeddings.
        - If --chunking=false: saves id -> single vector
        - If --chunking=true : saves id -> (num_chunks, dim) matrix
        - If --get-single-embedding is set, mean-pools chunk embeddings into a single vector.
    """
    encode_text_dataset(
        metadata_path=metadata_path,
        output_path=output_path,
        col_text=col_text,
        col_id=col_id,
        model_name=model_name,
        chunking=chunking,
        batch_size=batch_size,
        get_single_embedding=get_single_embedding,
    )


# ------------------------------------------------------------
# Command: infer-one
#
# Purpose:
#   Run inference on a single audio file using the frozen encoder:
#   Audio -> vocal segments extraction -> Whisper transcription -> text encoding.
#
# Typical usage:
#   poetry run livi-frozen-encoder infer-one \
#       --audio-path   src/livi/test_data/test.mp3 \
#
# Requirements:
# - Audio file must exist and be readable.
# - Config YAML must define preprocessing and model setup (default parameters are provided in this file)
#
# Arguments:
#   config_path : Path
#       Path to YAML config file controlling preprocessing and encoder setup.
#   audio_path : Path
#       Path to input audio file (.mp3, .wav, etc.).
#
# Output:
#   - Prints the shape of the resulting embedding (tuple).
# ------------------------------------------------------------
@app.command("infer-one")
def cli_infer_one(
    config_path: Path = typer.Option(None, help="Path to model/config YAML (used for data/preproc)."),
    audio_path: Path = typer.Option(..., help="Path to an audio file (.mp3, .wav, ...)"),
):
    """
    Encode a single audio file with the frozen encoder pipeline.
    """
    config_path = config_path or Path("src/livi/apps/frozen_encoder/config/infer.yaml")
    emb = run_inference_single(
        config_path=config_path,
        audio_path=audio_path,
    )
    typer.echo(f"Shape: {tuple(emb.shape)}")


# ------------------------------------------------------------
# Command: inference
#
# Purpose:
#   Run frozen-encoder inference on all audio files in a directory.
#   Extracts vocal segments, transcribes with Whisper, encodes with
#   a multilingual text encoder, and saves embeddings to disk.
#
# Typical usage:
#   poetry run livi-frozen-encoder inference \
#       --audio-dir src/livi/test_data/audio \
#       --out-path src/livi/test_data/covers80_lyrics_embeddings.npz
#
# Requirements:
# - `config_path` must point to a valid infer.yaml (defines transcriber,
#   text encoder, and vocal detection settings).
# - `audio_dir` must contain .mp3 files.
# - `out_path` should finish with .npz
#
# Arguments:
#   config_path : Path to inference config (YAML).
#   audio_dir   : Directory containing audio files.
#   out_path    : Optional destination for embeddings (.npz).
#                 Defaults to <audio_dir>/<audio_dir.name>_embeddings.npz.
# ------------------------------------------------------------
@app.command("inference")
def cli_infer_dir(
    config_path: Path = typer.Option(None, exists=True, readable=True, help="Path to infer.yaml."),
    audio_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Directory containing audio files."),
    out_path: Optional[Path] = typer.Option(..., help="Output .npz path"),
):
    """
    Batch inference over a directory of audio files and save embeddings as a .npz.
    """
    config_path = config_path or Path("src/livi/apps/frozen_encoder/config/infer.yaml")
    embeddings = run_inference(
        config_path=config_path,
        audio_dir=audio_dir,
        path_out=out_path,
    )
    typer.echo(f"Done. Wrote {len(embeddings)} embeddings.")


# ------------------------------------------------------------
# Command: estimate-time
#
# Purpose:
#   Benchmark the frozen-encoder inference pipeline by estimating
#   average runtime for preprocessing, transcription, text encoding,
#   and overall end-to-end inference on a sample of audio files.
#
# Typical usage:
#   poetry run livi-frozen-encoder estimate-time \
#       --audio-dir src/livi/test_data/audio \
#
# Requirements:
# - `config_path` must point to a valid infer.yaml (defines all
#   transcriber, text encoder, and vocal detection settings).
# - `audio_dir` must contain audio files (.mp3).
#
# Arguments:
#   config_path : Path to inference config (YAML).
#   audio_dir   : Directory containing audio files (recursively scanned).
#   sample_size : Number of audio files to randomly sample (default: 200).
#   start_after : Number of warm-up iterations to skip (default: 5).
#   seed        : RNG seed for reproducibility (default: 42).
#
# Output:
#   - Logs mean and standard deviation for each stage:
#       * Preprocessing (load + vocal detection)
#       * Transcription (Whisper)
#       * Text encoding
#       * Total time
#
# Notes:
#   - Skips warm-up iterations before timing to avoid torch.compile overhead.
#   - Results are printed in seconds (mean ± std).
# ------------------------------------------------------------
@app.command("estimate-time")
def cli_estimate_time(
    config_path: Path = typer.Option(None, help="Path to model/config YAML."),
    audio_dir: Path = typer.Option(..., help="Directory to search recursively for *.mp3"),
    sample_size: int = typer.Option(200, help="Number of files to sample."),
    start_after: int = typer.Option(5, help="Warm-up iterations to skip."),
    seed: int = typer.Option(42, help="Sampling seed."),
):
    """
    Estimate average preprocessing, transcription, text-encoding, and total time.
    """
    config_path = config_path or Path("src/livi/apps/frozen_encoder/config/infer.yaml")

    run_estimate_time(
        config_path=config_path,
        audio_dir=audio_dir,
        sample_size=sample_size,
        start_after=start_after,
        seed=seed,
    )
