import typer
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from livi.apps.audio_encoder.train.trainer import Trainer
from pathlib import Path
from livi.apps.audio_encoder.infer.session import run_inference_single, run_estimate_time, run_inference

from typing import Optional

from livi.config import settings

app = typer.Typer(help="Audio Encoder")


# --------------------------------------------------------------------
# Simple smoke test command
# Run: poetry run livi-audio-encoder hello
# --------------------------------------------------------------------
@app.command()
def hello():
    """Print a hello message (used for quick smoke test)."""
    typer.echo("[audio-encoder] Hello, World!")


# ------------------------------------------------------------
# Command: infer-one
#
# Purpose:
#   Encode a single audio file with the audio encoder. Optionally
#   mean-pool over chunk embeddings to return a single vector.
#
# Typical usage:
#   poetry run livi-audio-encoder infer-one
#       --audio-path src/livi/test_data/test.mp3 \
#       --get-global-embedding
#
# Arguments:
#   checkpoint_path     : Path to the trained model checkpoint (.pth).
#       Default to livi/apps/audio_encoder/checkpoints/livi.pth
#   config_path         : Path to the model/config YAML used at train/infer time.
#       Default to livi/apps/audio_encoder/config/infer.yaml
#   audio_path          : Path to the input audio file (.wav/.mp3/...).
#   get_global_embedding: If true, mean-pools chunk embeddings into one vector.
#
# Output:
#   Prints the embedding shape to stdout. You can pipe/save as needed.
# ------------------------------------------------------------
@app.command("infer-one")
def cli_infer_one(
    checkpoint_path: Path = typer.Option(None, help="Path to model checkpoint (.pth)"),
    config_path: Path = typer.Option(None, help="Path to model/config YAML"),
    audio_path: Path = typer.Option(..., help="Path to an audio file"),
    get_global_embedding: bool = typer.Option(True, help="Mean-pool chunk embeddings into a single vector."),
):
    """
    Encode a single audio file and (optionally) save the embedding to disk.
    """
    # Default path fallback
    checkpoint_path = checkpoint_path or Path("src/livi/apps/audio_encoder/checkpoints/livi.pth")
    config_path = config_path or Path("src/livi/apps/audio_encoder/config/infer.yaml")

    emb = run_inference_single(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        audio_path=audio_path,
        get_global_embedding=get_global_embedding,
    )

    typer.echo(f"Shape: {tuple(emb.shape)}")


# ------------------------------------------------------------
# Command: inference
#
# Purpose:
#   Run audio encoder inference on all audio files in a directory.
#   Extracts vocal segments, and forward pass through the audio encoder.
#
# Typical usage:
#   poetry run livi-audio-encoder inference \
#       --audio-dir src/livi/test_data/audio \
#       --out-path src/livi/test_data/covers80_livi_embeddings.npz
#
# Requirements:
# - `config_path` must point to a valid infer.yaml
# - `audio_dir` must contain .mp3 files.
# - `out_path` should finish with .npz
#
# Arguments:
#   checkpoint_path : Path to model checkpoint (.pth).
#   config_path : Path to inference config (YAML).
#   audio_dir   : Directory containing audio files.
#   out_path    : Optional destination for embeddings (.npz).
#   get_global_embedding : If True, mean-pools chunk embeddings into one vector.
# ------------------------------------------------------------
@app.command("inference")
def cli_infer_dir(
    checkpoint_path: Path = typer.Option(None, help=" Path to model checkpoint (.pth)"),
    config_path: Path = typer.Option(None, exists=True, readable=True, help="Path to infer.yaml."),
    audio_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Directory containing audio files."),
    out_path: Optional[Path] = typer.Option(..., help="Output .npz path"),
    get_global_embedding: bool = typer.Option(True, help="Mean-pool chunk embeddings into a single vector."),
):
    """
    Batch inference over a directory of audio files and save embeddings as a .npz.
    """
    checkpoint_path = checkpoint_path or Path("src/livi/apps/audio_encoder/checkpoints/livi.pth")
    config_path = config_path or Path("src/livi/apps/audio_encoder/config/infer.yaml")
    embeddings = run_inference(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        audio_dir=audio_dir,
        path_out=out_path,
        get_global_embedding=get_global_embedding,
    )
    typer.echo(f"Done. Wrote {len(embeddings)} embeddings.")


# ------------------------------------------------------------
# Command: estimate-time
#
# Purpose:
#   Benchmark average preprocessing time, inference time, and total time
#   across a random sample of audio files. Skips the first few iterations
#   as warm-up (to smooth out JIT/IO effects).
#
# Typical usage:
#   poetry run livi-audio-encoder estimate-time
#       --audio-dir src/livi/test_data/audio
#
# Arguments:
#   checkpoint_path     : Path to the trained model checkpoint (.pth).
#   config_path         : Path to the model/config YAML used at train/infer time.
#   audio_dir           : Directory containing audio files to sample from.
#   sample_size         : Number of files to sample for timing (without replacement).
#       Default to 200
#   start_after         : Number of initial files to skip from stats (warm-up).
#       Default to 5
#   seed                : RNG seed used for deterministic sampling.
#       Default to 42
#
# Output:
#   Logs mean ± std for preprocessing, inference, and total time to stdout.
# ------------------------------------------------------------
@app.command("estimate-time")
def cli_estimate_time(
    checkpoint_path: Path = typer.Option(None, help="Path to model checkpoint (.pth)"),
    config_path: Path = typer.Option(None, help="Path to model/config YAML"),
    audio_dir: str = typer.Option(..., help="Directory to select audio files, e.g. 'data/'"),
    sample_size: int = typer.Option(200, help="How many files to sample for timing."),
    start_after: int = typer.Option(5, help="Warm-up iterations to skip."),
    seed: int = typer.Option(42, help="Sampling seed."),
):
    """
    Estimate average preprocessing, inference, and total time over a sampled set of files.
    """
    # Set default values if None
    checkpoint_path = checkpoint_path or Path("src/livi/apps/audio_encoder/checkpoints/livi.pth")
    config_path = config_path or Path("src/livi/apps/audio_encoder/config/infer.yaml")

    run_estimate_time(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        audio_dir=Path(audio_dir),
        sample_size=sample_size,
        start_after=start_after,
        seed=seed,
    )


# --------------------------------------------------------------------
# Training command for the Audio Encoder
#
# Typical usage:
#   poetry run livi-audio-encoder launch-training
#
# Arguments:
#   config_path : Path to Hydra config.
# --------------------------------------------------------------------
@app.command("launch-training")
def cli_train(
    config_path: Path = typer.Option(None, help="Path to Hydra config directory."),
):
    """
    Launch training using Hydra configuration.
    """
    config_path = config_path or Path("src/livi/apps/audio_encoder/config/livi.yaml")
    cfg = OmegaConf.load(config_path)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    app()
