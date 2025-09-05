# -*- coding: utf-8 -*-
"""
Test-set embedding loaders + PCA/similarity runners + CLI

Provides:
    1) Loading helpers
        - load_segment_pairs(text_dir, audio_dir)
        - get_ids_to_keep(lyrics_dir, test_dir)             (>=2 chunks policy)
        - load_and_aggregate_test_set(df_ids, text_dir, audio_dir)

    2) Runners
        - run_pca_aggregated(text_dir, audio_dir, ...)
        - run_similarity_aggregated(text_dir, audio_dir, ...)
        - run_similarity_segments(text_dir, audio_dir, ...)

    3) Typer CLI with commented commands:
        - pca-agg
        - cosine-agg
        - cosine-segments
        - load-segments (quick sanity check)
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------------------


def get_ids_to_keep(dir_lyrics_embed: Path, dir_test_set: Path) -> pd.DataFrame:
    """
    Retrieve version_ids that have at least two chunks (based on filenames).

    Parameters
    ----------
    dir_lyrics_embed : Path
        Directory containing .npy files with lyrics embeddings. Filenames must
        be "<version_id><chunk_id>.npy" (last char is the chunk id).
    dir_test_set : Path
        Unused here (kept for API symmetry).

    Returns
    -------
    pd.DataFrame with columns:
        - ids         : "<version_id><chunk_id>"
        - version_id  : base id (all but last char)
        - chunk_id    : last char
      Only version_ids with count > 1 are retained.
    """
    lyrics_embed_files = glob.glob(str(dir_lyrics_embed / "*.npy"))
    ids = [os.path.basename(f)[:-4] for f in lyrics_embed_files]  # strip ".npy"

    df = pd.DataFrame({"ids": ids})
    df["version_id"] = df["ids"].str[:-1]
    df["chunk_id"] = df["ids"].str[-1]

    counts = df.groupby("version_id").size().reset_index(name="count")
    valid = counts[counts["count"] > 1]["version_id"]
    df = df[df["version_id"].isin(valid)].reset_index(drop=True)
    return df


def load_and_aggregate_test_set(
    df_version_ids_to_keep: pd.DataFrame,
    text_dir: Path,
    audio_dir: Path,
    *,
    id_col: str = "version_id",
    file_ext: str = ".npy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strict aggregate loader: for each version_id, load ALL its chunk files from
    `text_dir` and `audio_dir`, average, and return aligned arrays in the same
    order as the DataFrame.

    Assumptions
    -----------
    - Filenames are "<version_id><chunk_id>.npy".
    - Every version_id in df_version_ids_to_keep exists in BOTH modalities (>=1 chunk).

    Returns
    -------
    text_embeddings : (N, D_t)
    audio_embeddings: (N, D_a)
    """
    if id_col not in df_version_ids_to_keep.columns:
        raise ValueError(f"`df_version_ids_to_keep` must contain '{id_col}' column.")

    order: List[str] = df_version_ids_to_keep[id_col].astype(str).tolist()

    def file_to_vid(path: str) -> str:
        stem = os.path.basename(path)[: -len(file_ext)]
        return stem[:-1]  # strip chunk id

    v2text: Dict[str, List[str]] = {}
    for f in glob.glob(str(text_dir / f"*{file_ext}")):
        v2text.setdefault(file_to_vid(f), []).append(f)

    v2audio: Dict[str, List[str]] = {}
    for f in glob.glob(str(audio_dir / f"*{file_ext}")):
        v2audio.setdefault(file_to_vid(f), []).append(f)

    text_out, audio_out = [], []

    for vid in order:
        t_files = v2text.get(vid)
        a_files = v2audio.get(vid)
        if not t_files or not a_files:
            raise RuntimeError(f"Missing chunks for version_id={vid} in text or audio.")

        T = np.stack([np.load(f, allow_pickle=False) for f in t_files], axis=0)
        A = np.stack([np.load(f, allow_pickle=False) for f in a_files], axis=0)
        text_out.append(T.mean(axis=0))
        audio_out.append(A.mean(axis=0))

    return np.stack(text_out, axis=0), np.stack(audio_out, axis=0)


def load_segment_pairs(
    text_dir: Path,
    audio_dir: Path,
    *,
    file_ext: str = ".npy",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load segment-level (chunk) embeddings and align them by exact basename
    "<version_id><chunk_id>.npy". Only the intersection of basenames is kept.

    Returns
    -------
    text_embeddings : (N, D_t)
    audio_embeddings: (N, D_a)
    ids             : list[str]  (aligned basenames without extension)
    """
    text_files = {os.path.basename(p): p for p in glob.glob(str(text_dir / f"*{file_ext}"))}
    audio_files = {os.path.basename(p): p for p in glob.glob(str(audio_dir / f"*{file_ext}"))}

    common = sorted(set(text_files.keys()) & set(audio_files.keys()))
    if not common:
        raise FileNotFoundError("No common segment basenames between text_dir and audio_dir.")

    T = [np.load(text_files[b], allow_pickle=False) for b in common]
    A = [np.load(audio_files[b], allow_pickle=False) for b in common]

    return np.stack(T, axis=0), np.stack(A, axis=0), [b[: -len(file_ext)] for b in common]


# ------------------------------------------------------------------------
# Plot + similarity utilities
# ------------------------------------------------------------------------


def pca_audio_text_embeddings(
    text_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    n: Optional[int] = 100,
    *,
    title: str = "Audio-Text PCA Embeddings",
    save_path: Optional[Path] = None,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Perform 2D PCA on paired audio/text embeddings, plot, and optionally save.

    Returns
    -------
    reduced : (2N_sel, 2)  (first N_sel rows = text; next N_sel rows = audio)
    """
    if text_embeddings.shape[0] != audio_embeddings.shape[0]:
        raise ValueError("text_embeddings and audio_embeddings must have same N (aligned pairs).")

    N = text_embeddings.shape[0]
    idx = np.arange(N)
    if n is not None and n < N:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(idx, size=n, replace=False)

    T = text_embeddings[idx]
    A = audio_embeddings[idx]
    X = np.concatenate([T, A], axis=0)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    labels = (["Text"] * T.shape[0]) + (["Audio"] * A.shape[0])
    colors = {"Text": "tab:blue", "Audio": "tab:red"}

    plt.figure(figsize=(8, 6))
    for label in ("Text", "Audio"):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.7, s=18, c=colors[label])
    plt.legend()
    plt.title(f"{title} (n={T.shape[0]})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

    return reduced


def audio_text_similarity(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    type_data: str = "aggregated",
    fig_path: Optional[Path] = None,
    bins: int = 20,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute cosine similarity for aligned pairs, plot histogram optionally.
    """
    if audio_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError("audio_embeddings and text_embeddings must have same N (aligned pairs).")

    def _safe_norm(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
        return np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)

    A = audio_embeddings.astype(np.float64, copy=False)
    T = text_embeddings.astype(np.float64, copy=False)
    A = A / _safe_norm(A)
    T = T / _safe_norm(T)
    scores = np.sum(A * T, axis=1)

    avg = float(np.mean(scores))
    std = float(np.std(scores))
    logger.info(f"[{type_data}] cosine similarity: avg={avg:.4f}, std={std:.4f}")

    if fig_path is not None:
        fig_path = Path(fig_path)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.hist(scores, bins=bins, alpha=0.85)
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.title(f"Audio–Text Cosine Similarity — {type_data}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

    return avg, std, scores


# ------------------------------------------------------------------------
# High-level runners
# ------------------------------------------------------------------------


def run_pca_aggregated(
    text_dir: Path,
    audio_dir: Path,
    *,
    ids_source_dir: Optional[Path] = None,
    n: Optional[int] = 100,
    fig_path: Optional[Path] = None,
    title: str = "Audio-Text PCA Embeddings",
) -> None:
    """
    Build df_version_ids_to_keep (>=2 chunks), aggregate by version_id, then PCA plot.
    """
    ids_dir = ids_source_dir or text_dir
    df_ids = get_ids_to_keep(ids_dir, Path("."))
    T, A = load_and_aggregate_test_set(df_ids, text_dir, audio_dir)
    pca_audio_text_embeddings(T, A, n=n, save_path=fig_path, title=title)


def run_similarity_aggregated(
    text_dir: Path,
    audio_dir: Path,
    *,
    ids_source_dir: Optional[Path] = None,
    fig_path: Optional[Path] = None,
    bins: int = 20,
) -> Dict[str, float]:
    """
    Aggregate by version_id (>=2 chunks) and compute similarity stats+hist.
    """
    ids_dir = ids_source_dir or text_dir
    df_ids = get_ids_to_keep(ids_dir, Path("."))
    T, A = load_and_aggregate_test_set(df_ids, text_dir, audio_dir)
    avg, std, _ = audio_text_similarity(A, T, type_data="aggregated", fig_path=fig_path, bins=bins)
    return {"avg": avg, "std": std}


def run_similarity_segments(
    text_dir: Path,
    audio_dir: Path,
    *,
    fig_path: Optional[Path] = None,
    bins: int = 20,
) -> Dict[str, float]:
    """
    Segment-level similarity: align by identical basenames "<version><chunk>.npy".
    """
    T, A, _ids = load_segment_pairs(text_dir, audio_dir)
    avg, std, _ = audio_text_similarity(A, T, type_data="segments", fig_path=fig_path, bins=bins)
    return {"avg": avg, "std": std}
