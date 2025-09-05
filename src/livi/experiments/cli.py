# src/livi/apps/training/cli.py
import typer
from pathlib import Path
from typing import Optional
import json

from livi.apps.retrieval_eval.ranker import run_evaluation
from livi.apps.retrieval_eval.reranker import run_reranker


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
#   poetry run livi-data evaluate \
#       --path-metadata data/metadata.csv \
#       --path-embeddings data/embeddings.npz \
#       --col-id version_id \
#       --text-id lyrics \
#       --k 100 \
#       --path-metrics results/metrics.csv
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
    eval_type: str = typer.Option(..., help="dense | sparse | bge | bm25 | chunked | fusion"),
    path_metadata: Path = typer.Option(..., help="CSV with at least the ID and text columns."),
    path_embeddings: Optional[Path] = typer.Option(None, help="Embeddings for the chosen eval (npz/pkl)."),
    path_embeddings_2: Optional[Path] = typer.Option(None, help="Second embeddings (for fusion)."),
    get_single_embedding: bool = typer.Option(True, help="Average multiple vectors per id when loading embeddings."),
    col_id: str = typer.Option("version_id", help="ID column name in metadata."),
    text_id: str = typer.Option("lyrics", help="Text column name in metadata."),
    k: int = typer.Option(100, help="Top-K to keep."),
    hub_repo: Optional[str] = typer.Option(None, help="HF Hub repo for BM25 index (bm25 or fusion with bm25)."),
    fusion_sparse_type: Optional[str] = typer.Option(None, help="For fusion: 'sparse' or 'bm25'."),
    path_ranked_csv: Optional[Path] = typer.Option(None, help="Where to save ranked results CSV."),
    path_fp_tp_csv: Optional[Path] = typer.Option(None, help="Where to save FP/first-relevant join CSV."),
    path_metrics_csv: Optional[Path] = typer.Option(None, help="Where to save metrics CSV."),
):
    """
    CLI wrapper around `run_evaluation`. Saves artifacts if paths are provided and prints metrics JSON.
    """
    metrics = run_evaluation(
        eval_type=eval_type,
        path_metadata=path_metadata,
        path_embeddings=path_embeddings,
        path_embeddings_2=path_embeddings_2,
        get_single_embedding=get_single_embedding,
        col_id=col_id,
        text_id=text_id,
        k=k,
        hub_repo=hub_repo,
        fusion_sparse_type=fusion_sparse_type,
        path_ranked_csv=path_ranked_csv,
        path_fp_tp_csv=path_fp_tp_csv,
        path_metrics_csv=path_metrics_csv,
    )
    typer.echo(json.dumps(metrics, indent=2))


# --------------------------------------------------------------------
# Command: Rerank retrieval results with a cross-encoder
#
# Purpose:
#   Take an initial retrieval ranking (CSV of query_id, corpus_id, rank, relevant),
#   re-score candidates per query with a cross-encoder-style reranker, and output:
#     - a reranked results CSV (query_id, corpus_id, rank, relevant, score)
#     - a metrics JSON/CSV (MR1, HR1, HR10, HR100, MAP10)
#
# Requirements:
#   - Ranked input CSV must contain at least: query_id, corpus_id, rank, relevant
#   - Metadata CSV must contain text to score (e.g., lyrics) and IDs
#   - A reranker model that supports `.compute_score([query, doc], normalize=True)`
#     (e.g. FlagEmbedding's reranker) or SBERT-style `.predict(...)` if you adapt
#     the call in `Reranker.rerank`.
#
# Typical usage:
#   poetry run livi-data rerank \
#       --path-metadata data/metadata.csv \
#       --path-ranked-input results/ranking.csv \
#       --id-col version_id \
#       --text-col lyrics \
#       --out-reranked results/reranked.csv \
#       --out-metrics results/rerank_metrics.json
#
# Arguments:
#   --path-metadata        : Path to metadata CSV with ID and text columns.
#   --path-ranked-input    : Path to CSV of initial ranking.
#   --id-col          : ID column name in metadata. Default="version_id".
#   --text-col        : Text column name in metadata. Default="lyrics".
#   --out-metrics     : Output JSON for metrics.
# --------------------------------------------------------------------


@app.command("rerank")
def cli_rerank(
    path_metadata: Path = typer.Option(..., help="Metadata CSV with ID and text columns."),
    path_ranked_input: Path = typer.Option(
        ..., help="Initial ranking CSV with columns: query_id, corpus_id, rank, relevant."
    ),
    out_metrics: Optional[Path] = typer.Option(None, help="Path to save metrics JSON."),
    id_col: str = typer.Option("version_id", help="ID column in metadata."),
    text_col: str = typer.Option("lyrics", help="Text column in metadata."),
):
    """
    CLI wrapper around `run_reranker`. Loads an optional FlagEmbedding reranker if
    --reranker-name is provided.
    """
    metrics = run_reranker(
        path_metadata=path_metadata,
        path_ranked_input=path_ranked_input,
        out_metrics=out_metrics,
        id_col=id_col,
        text_col=text_col,
    )
    typer.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
