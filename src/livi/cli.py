# ------------------------------------------------------------
# Command-line interface (CLI) for the repository livi.
# This file defines the commands accessible via the terminal
# using Typer.

# To use a function, you can just type in the terminal:
# poetry run python -m livi.cli hello --name YourName
# where hello is the function name and name is the parameter of the function
# ------------------------------------------------------------

import typer
from livi.config import settings
from livi.utils.paths import p
from livi.apps.audio_encoder.cli import app as audio_encoder_app
from livi.apps.frozen_encoder.cli import app as frozen_encoder_app
from livi.apps.retrieval_eval.cli import app as retrieval_eval_app
from livi.apps.audio_baselines.cli import app as audio_baselines_app
from livi.core.data.cli import app as data_app

app = typer.Typer(help="Command-line interface for livi")
app.add_typer(audio_encoder_app, name="audio-encoder", help="Commands for the audio encoder app")
app.add_typer(frozen_encoder_app, name="frozen-encoder", help="Commands for the frozen encoder app")
app.add_typer(retrieval_eval_app, name="retrieval-eval", help="Commands for the retrieval eval app")
app.add_typer(audio_baselines_app, name="audio-baselines", help="Commands for the audio baselines app")
app.add_typer(data_app, name="data", help="Commands for the data app")


# Usage : poetry run python -m livi.cli hello --name Joanne
@app.command()
def hello(name: str = "world"):
    """Example command."""
    typer.echo(f"Hello, {name}! DATA_DIR is {p.data}")


@app.command()
def hello2(name: str = "world"):
    """Example command."""
    typer.echo(f"Hello, {name}! DATA_DIR is {p.data}")


if __name__ == "__main__":
    app()
