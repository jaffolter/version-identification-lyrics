import typer
from livi.apps.audio_baselines.inference import run_inference

app = typer.Typer(help="Audio Baselines")


# --------------------------------------------------------------------
# Simple smoke test command
# Run: poetry run livi-audio-baselines hello
# --------------------------------------------------------------------
@app.command()
def hello():
    """Print a hello message (used for quick smoke test)."""
    typer.echo("[audio-baselines] Hello, World!")


# --------------------------------------------------------------------
# Inference command for audio baseline models
#
# Purpose:
#   Generate audio embeddings for a given dataset using one of the
#   baseline encoder models (dvinet, bytecover, cqtnet, clews).
#
# Typical usage:
#   poetry run livi-audio-baselines infer --dataset covers80 --model-name dvinet
#
# Arguments:
#   dataset       : Name of dataset (e.g. covers80, discogs-vi, shs100k, ...).
#   model_name    : Model to run (dvinet, bytecover, cqtnet, clews).
#   path_in       : Optional path to input audio directory.
#                   Defaults to <AUDIO_DIR>/<dataset>/
#   path_out      : Optional path to save embeddings.
#                   Defaults to <EMBEDDINGS_DIR>/audio_baselines/<model>/<dataset>/
#   config        : Optional path to model config file.
#                   Defaults to ./config/<model_name>.yaml
#   checkpoint    : Optional path to model checkpoint file.
#                   Defaults to ./checkpoints/<model_name>.ckpt
#   device        : "cuda" (GPU) or "cpu".
#   hop_size      : Hop size in seconds for feature extraction.
#   win_len       : Window length (set -1 to use model default).
#   inference_time: If True, measure and report inference time.
#   truncate      : Maximum number of audio files to process
#                   (1 = just the first file, -1 = all).
# --------------------------------------------------------------------
@app.command()
def infer(
    dataset: str = "covers80",
    model_name: str = "dvinet",
    path_in: str = None,
    path_out: str = None,
    config: str = None,
    checkpoint: str = None,
    device: str = "cuda",
    hop_size: float = 5.0,
    win_len: float = -1,
    inference_time: bool = False,
    truncate: int = 1,
):
    """Run audio embedding inference."""
    typer.echo("[audio-baselines] Running inference...")
    run_inference(
        dataset=dataset,
        model_name=model_name,
        path_in=path_in,
        path_out=path_out,
        config=config,
        checkpoint=checkpoint,
        device=device,
        hop_size=hop_size,
        win_len=win_len,
        inference_time=inference_time,
        truncate=truncate,
    )
