# src/livi/apps/training/cli.py
import typer
from pathlib import Path
from typing import Optional
import json

from livi.apps.retrieval_eval.ranker import run_evaluation


app = typer.Typer(help="Retrieval Evaluation")


# --------------------------------------------------------------------
# Simple smoke test command
# Run: poetry run livi-retrieval-eval hello
# --------------------------------------------------------------------
@app.command()
def hello():
    """Print a hello message (used for quick smoke test)."""
    typer.echo("[retrieval-eval] Hello, World!")


# --------------------------------------------------------------------
# Command: Evaluate retrieval models
#
# Purpose:
#   Run retrieval evaluation for various rankers (dense, sparse, BGE,
#   BM25, chunked, or fusion). Computes similarity, ranks results,
#   collects first-relevant items and false positives, and outputs
#   evaluation metrics.
#
# Requirements:
#   - Metadata CSV with at least ID and text columns (e.g., version_id, lyrics)
#   - Embeddings or BM25 index depending on eval_type
#   - Optional second embeddings for fusion (dense+sparse or dense+bm25)
#
# Typical usage:
#   poetry run livi-retrieval-eval evaluate \
#       --path-metadata src/livi/test_data/covers80.csv \
#       --path-embeddings src/livi/test_data/covers80_dense.npz  \
#       --col-id version_id \
#       --text-id lyrics \
#       --k 100 \
#       --path-metrics src/livi/test_data/covers80_dense_metrics.csv
#
# Arguments:
#   path_metadata    : Path to CSV with metadata containing IDs and texts.
#   path_embeddings  : Path to embeddings file (.npz or .pkl).
#   path_embeddings_2: Path to second embeddings file (for fusion).
#   get_single_embedding : Whether to average multiple vectors per ID. Default=True.
#   col_id           : Column name in metadata for IDs. Default="version_id".
#   text_id          : Column name in metadata for text (lyrics). Default="lyrics".
#   k                : Top-K results to keep. Default=100.
#   path_metrics : Where to save metrics CSV.
# --------------------------------------------------------------------
@app.command("evaluate")
def cli_evaluate(
    path_metadata: Path = typer.Option(..., help="CSV with at least the ID and text columns."),
    path_embeddings: Optional[Path] = typer.Option(None, help="Embeddings for the chosen eval (npz)."),
    path_embeddings_2: Optional[Path] = typer.Option(None, help="Second embeddings (for e5-large-instruct -> query)."),
    get_single_embedding: bool = typer.Option(True, help="Average multiple vectors per id when loading embeddings."),
    col_id: str = typer.Option("version_id", help="ID column name in metadata."),
    text_id: str = typer.Option("lyrics", help="Text column name in metadata."),
    k: int = typer.Option(100, help="Top-K to keep."),
    path_metrics: Optional[Path] = typer.Option(None, help="Where to save metrics CSV."),
):
    """
    CLI wrapper around `run_evaluation`. Saves artifacts if paths are provided and prints metrics JSON.
    """
    metrics = run_evaluation(
        path_metadata=path_metadata,
        path_embeddings=path_embeddings,
        path_embeddings_2=path_embeddings_2,
        get_single_embedding=get_single_embedding,
        col_id=col_id,
        text_id=text_id,
        k=k,
        path_metrics=path_metrics,
    )

    for metric, res in metrics.items():
        typer.echo(f"{metric}\t: {res:.3f}")


if __name__ == "__main__":
    app()
