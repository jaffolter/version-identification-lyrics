import numpy as np
from pathlib import Path
from typing import Dict
import pickle


def get_embeddings(path: Path, get_single_embedding: bool = False) -> Dict[str, np.ndarray]:
    """
    Load precomputed embeddings from a file.

        The file must contain a mapping from track identifiers (version_id) to one
        or more embeddings. Supported formats:
        - `.npz`: serialized numpy arrays
        - `.pkl`: pickled Python dictionary

        Behavior
        --------
        - If `get_single_embedding=True`, multiple embeddings per version_id are
        averaged into a single vector.
        - If `get_single_embedding=False`, all embeddings are returned as-is
        (useful for chunked audio).

        Args
        ----
        path : Path
            Path to the `.npz` or `.pkl` file containing the embeddings.
        get_single_embedding : bool, optional
            Whether to average multiple embeddings per track into one vector.
            Default = False.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping version_id (as string) → embedding(s).
            Values are:
            - np.ndarray of shape (d,) if `get_single_embedding=True`
            - np.ndarray of shape (n, d) if multiple embeddings are kept
    """
    # Load numpy array from .npz file
    if str(path).endswith(".npz"):
        data = np.load(path)

    elif str(path).endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)

    if get_single_embedding:
        embeddings = {str(k): np.mean(v, axis=0) for k, v in data.items()}

    # Ensure keys are strings
    embeddings = {str(k): v for k, v in data.items()}

    return embeddings


def save_embeddings(output_path: Path, embeddings: Dict[str, np.ndarray]):
    """
    Save embeddings to a file.

    The embeddings can be saved in either `.npz` (compressed numpy) or `.pkl` (pickle) format.

    Args
    ----
    output_path : Path
        Path to the output file (must end with .npz or .pkl).
    embeddings : Dict[str, np.ndarray]
        Dictionary mapping track identifiers (version_id) to their embeddings.
    """
    if str(output_path).endswith(".npz"):
        np.savez_compressed(output_path, **embeddings)
    elif str(output_path).endswith(".pkl"):
        with open(output_path, "wb") as f:
            pickle.dump(embeddings, f)
    else:
        raise ValueError("Unsupported file format. Please use .npz or .pkl.")
