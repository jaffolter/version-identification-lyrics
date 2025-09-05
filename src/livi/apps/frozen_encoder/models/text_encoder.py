import os
import pickle  # (unused right now, but kept since you had it)
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Union, Sequence

import numpy as np
import pandas as pd
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from livi.apps.retrieval_eval.data import convert_to_dict
from livi.core.data.utils.io_toolbox import save_embeddings
from loguru import logger

# -------------- Internal Helpers --------------


def get_detailed_instruct(query: str) -> str:
    return f"Instruct: Retrieve semantically similar text.\nQuery: {query}"


class TextEncoder:
    """
    Minimal wrapper around `SentenceTransformer` with optional chunking.

    Chunking behavior
    -----------------
        - If an input is a `str` and chunking is enabled, we construct chunks via:
        `self.chunker.create_documents([text])` and take the unique
        `doc.page_content` values.
        - If an input is `list[str]`, it is treated as pre-chunked and used as-is.
    """

    def __init__(self, model_name: str, chunking: bool = False):
        """
        Initialize the encoder and (optionally) a text chunker.

        Parameters
        ----------
        model_name : str
            A Sentence-Transformers model identifier.
        chunking : bool, default=False
            Whether to enable automatic text chunking.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

        self.chunker: Optional[RecursiveCharacterTextSplitter] = None
        if chunking:
            self.chunker = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=0,
                length_function=len,
                is_separator_regex=True,
            )

    def encode(self, inputs: List[str], batch_size: Optional[int] = 8) -> np.ndarray:
        """
        Encode a batch of strings.

        Parameters
        ----------
        inputs : list of str
            Raw text strings to encode.
        batch_size : int, optional
            Batch size for model inference.

        Returns
        -------
        np.ndarray
            Array of shape (N, D) with L2-normalized embeddings.
        """
        kwargs = {"task": "retrieval.query", "prompt": "retrieval.query"} if "jina" in self.model_name else {}
        inputs = [get_detailed_instruct(text) for text in inputs] if "instruct" in self.model_name else inputs

        return self.model.encode(inputs, normalize_embeddings=True, device=self.device, batch_size=batch_size, **kwargs)

    def encode_chunks(
        self,
        inputs: Sequence[Union[str, List[str]]],
        get_single_embedding: Optional[bool] = False,
        batch_size: Optional[int] = 8,
    ) -> np.ndarray:
        """
        Encode chunked input.
        - If the item is a `str`, it is chunked using `self.chunker`
        - If the item is `list[str]`, it is treated as already chunked.

        Parameters
        ----------
        inputs : Sequence[Union[str, list[str]]]
            Texts to encode; either raw strings or pre-chunked lists of strings.
        get_single_embedding : bool, optional
            If True, mean-pool chunk embeddings into a single (1, D) vector
            per ID.
        batch_size : int, optional
            Batch size for encoding.

        Returns
        -------
        np.ndarray
            Array of shape (N, D) with L2-normalized embeddings, with N the number of chunks.
        """
        try:
            # Need to chunk input text if it's a string
            if isinstance(inputs, str):
                if self.chunker is None:
                    raise ValueError(
                        "Chunker is not defined. Re-create TextEncoder with chunking=True "
                        "or pass pre-chunked lists instead of strings."
                    )
                docs = self.chunker.create_documents([inputs])
                chunks = list({doc.page_content for doc in docs})
            elif isinstance(inputs, list):
                chunks = inputs
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")

            # handle empty-chunk edge case
            if not chunks:
                logger.warning(f"Empty chunks.")
                return

            # Encode chunks and optionally mean-pool
            embs = self.encode(chunks, batch_size=batch_size).astype(np.float32)
            if get_single_embedding:
                embs = np.mean(embs, axis=0, keepdims=True)

        except Exception:
            logger.warning(f"Encoding failed.")
            return
        return embs


# --------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------


@lru_cache(maxsize=2)
def _get_cached_text_encoder(
    model_name: str = "Alibaba-NLP/gte-multilingual-base", chunking: bool = False
) -> TextEncoder:
    """
    Return a cached `TextEncoder` instance.

    Parameters
    ----------
    model_name : str, default="Alibaba-NLP/gte-multilingual-base"
        Sentence-Transformers model name to load.
    chunking : bool, default=False
        Whether the cached encoder should be initialized with chunking.

    Returns
    -------
    TextEncoder
        Cached encoder instance (per unique argument combination).
    """
    return TextEncoder(model_name=model_name, chunking=chunking)


def encode_text(
    inputs: Sequence[Union[str, List[str]]],
    text_encoder: Optional[TextEncoder],
    model_name: Optional[str],
    chunking: Optional[bool] = False,
    batch_size: Optional[int] = 8,
    get_single_embedding: Optional[bool] = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Encode texts with optional chunking.

    Parameters
    ----------
    inputs : Sequence[Union[str, list[str]]]
        Raw strings or pre-chunked lists of strings.
    ids : list of str
        Unique IDs associated with `inputs` (required if `chunking=True`).
    text_encoder : TextEncoder, optional
        Pre-initialized encoder to reuse. If None, a cached encoder is built.
    model_name : str, optional
        Model name used when constructing a cached encoder (if `text_encoder` is None).
    chunking : bool, default=False
        If True, return per-ID chunk embeddings as a dict; otherwise return a single array.
    batch_size : int, optional
        Batch size for inference.
    get_single_embedding : bool, optional
        If True (and `chunking=True`), mean-pool chunk embeddings into (1, D) per ID.

    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
        - If `chunking=False`: array of shape (N, D).
        - If `chunking=True`: mapping ID -> (num_chunks, D) or (1, D) if pooled.
    """
    enc = text_encoder or _get_cached_text_encoder(
        model_name=model_name or "Alibaba-NLP/gte-multilingual-base", chunking=chunking
    )

    if chunking != enc.chunker:
        enc = TextEncoder(model_name=model_name or enc.model_name, chunking=chunking)

    if chunking:
        return enc.encode_chunks(inputs, batch_size=batch_size, get_single_embedding=get_single_embedding)
    else:
        return enc.encode(list(inputs), batch_size=batch_size)


def encode_chunks_multiple_tracks(
    ids: List[str],
    inputs: Sequence[Union[str, List[str]]],
    text_encoder: Optional[TextEncoder],
    model_name: Optional[str],
    batch_size: Optional[int] = 8,
    get_single_embedding: Optional[bool] = False,
) -> Dict[str, np.ndarray]:
    """
    Encode texts by ID with optional chunking.

    For each item:
    - If the item is a `str`, it is chunked using `self.chunker` and
        unique `page_content` entries are used.
    - If the item is `list[str]`, it is treated as already chunked.

    Parameters
    ----------
    ids : list of str
        Unique IDs, one per input item.
    inputs : Sequence[Union[str, list[str]]]
        Texts to encode; either raw strings or pre-chunked lists of strings.
    batch_size : int, optional
        Batch size for encoding.
    get_single_embedding : bool, optional
        If True, mean-pool chunk embeddings into a single (1, D) vector
        per ID.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from ID to:
        - (num_chunks, D) array for chunked inputs, or
        - (1, D) array if `get_single_embedding=True`.
        Empty or failed items return an empty (0, 0) array.
    """
    if len(ids) != len(inputs):
        raise ValueError("`ids` and `inputs` must have the same length.")

    text_encoder = text_encoder or _get_cached_text_encoder(
        model_name=model_name or "Alibaba-NLP/gte-multilingual-base", chunking=True
    )

    results: Dict[str, np.ndarray] = {}

    for id_, item in tqdm(zip(ids, inputs), total=len(ids), desc="Encoding (chunked)"):
        embs = text_encoder.encode_chunks(item, batch_size=batch_size, get_single_embedding=get_single_embedding)
        if embs is None:
            embs = np.zeros((0, 0), dtype=np.float32)
        results[id_] = embs

    return results


def encode_text_dataset(
    metadata_path: Path,
    output_path: Path,
    col_text: str,
    col_id: str,
    model_name: Optional[str],
    chunking: Optional[bool] = False,
    batch_size: Optional[int] = 8,
    get_single_embedding: Optional[bool] = False,
) -> None:
    """
    Encode a CSV text column and save embeddings to disk.

    Behavior
    --------
    - If `chunking=False`: saves a single embedding per ID
        (id -> (D,) in a dict via `convert_to_dict`).
    - If `chunking=True`: saves per-ID chunk matrices
        (id -> (num_chunks, D)).
    - If `get_single_embedding=True` and `chunking=True`: mean-pools chunk
        embeddings to a single (1, D) vector per ID.

    Parameters
    ----------
    metadata_path : pathlib.Path
        Path to the CSV containing the text and ID columns.
    output_path : pathlib.Path
        Destination path for the serialized embeddings (e.g., .npz/.pkl).
    col_text : str
        Column name containing the input texts (raw or chunked lists as strings).
    col_id : str
        Column name containing unique identifiers.
    model_name : str, optional
        Encoder model name used if a cached encoder is constructed.
    chunking : bool, default=False
        Whether to compute and store chunk-level embeddings.
    batch_size : int, optional
        Batch size for encoding.
    get_single_embedding : bool, default=False
        If True (with `chunking=True`), mean-pool chunk embeddings per ID.

    Raises
    ------
    FileNotFoundError
        If `metadata_path` does not exist.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    texts = df[col_text].astype(str).tolist()
    ids = df[col_id].astype(str).tolist()

    enc = _get_cached_text_encoder(model_name=model_name or "Alibaba-NLP/gte-multilingual-base", chunking=chunking)

    # ensure chunker state matches the request
    if chunking != enc.chunker is None:
        enc = TextEncoder(model_name=model_name or enc.model_name, chunking=chunking)

    if chunking:
        embs = encode_chunks_multiple_tracks(
            ids,
            texts,
            text_encoder=enc,
            model_name=enc.model_name,
            batch_size=batch_size,
            get_single_embedding=get_single_embedding,
        )
    else:
        embs = encode_text(
            texts,
            text_encoder=enc,
            model_name=enc.model_name,
            chunking=chunking,
            batch_size=batch_size,
            get_single_embedding=get_single_embedding,
        )

    mapping = embs if chunking else convert_to_dict(embs, ids)
    save_embeddings(output_path, mapping)
