from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, Sequence


def plot_tsne(
    audio_embeddings: Sequence[Sequence[float]],
    lyrics_embeddings: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]] = None,
    perplexity: int = 30,
    random_state: int = 42,
    output_tag: str = "clap",
) -> None:
    """
    Plot a t-SNE visualization of audio and lyrics embeddings.

    Parameters
    ----------
    audio_embeddings : array-like (n_samples, dim)
        Embeddings derived from audio (e.g., CLAP, Whisper).
    lyrics_embeddings : array-like (n_samples, dim)
        Embeddings derived from lyrics (same number of samples and dimension as audio).
    labels : Optional[Sequence[str]], default=None
        Labels to annotate individual points (applied only to audio embeddings for clarity).
    perplexity : int, default=30
        Perplexity parameter for t-SNE (controls neighborhood size).
    random_state : int, default=42
        Random seed for reproducibility.
    output_tag : str, default="clap"
        Tag used in the saved output filename: `tsne_embeddings_<output_tag>.png`.

    Returns
    -------
    None
        Saves the t-SNE figure as a PNG and displays it.
    """
    # Convert to numpy arrays
    audio_embeddings = np.array(audio_embeddings)
    lyrics_embeddings = np.array(lyrics_embeddings)

    # Sanity check: embeddings must have the same shape
    if audio_embeddings.shape != lyrics_embeddings.shape:
        raise ValueError(f"Mismatched embedding sizes: {audio_embeddings.shape} vs {lyrics_embeddings.shape}")

    # Concatenate embeddings for joint projection
    X = np.concatenate([audio_embeddings, lyrics_embeddings], axis=0)
    types = ["Audio"] * len(audio_embeddings) + ["Lyrics"] * len(lyrics_embeddings)

    # Fit t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=random_state,
    )
    X_tsne = tsne.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    for label in set(types):
        idxs = [i for i, t in enumerate(types) if t == label]
        ax.scatter(
            X_tsne[idxs, 0],
            X_tsne[idxs, 1],
            label=label,
            alpha=0.7,
        )

    # Annotate if labels are provided
    if labels is not None:
        for i, txt in enumerate(labels):
            ax.annotate(
                txt,
                (X_tsne[i, 0], X_tsne[i, 1]),
                fontsize=8,
                alpha=0.5,
            )

    # Finalize figure
    ax.set_title("t-SNE of Audio and Lyrics Embeddings")
    ax.legend()
    plt.tight_layout()

    # Save figure
    out_path = f"tsne_embeddings_{output_tag}.png"
    plt.savefig(out_path, dpi=200)
    plt.show()
