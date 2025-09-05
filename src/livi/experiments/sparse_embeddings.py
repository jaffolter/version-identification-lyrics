# text_encoders.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from FlagEmbedding import BGEM3FlagModel
from sklearn.feature_extraction.text import TfidfVectorizer
from bm25s import BM25HF, tokenize

from livi.apps.retrieval_eval.data import convert_to_dict
from livi.core.data.utils.io_toolbox import save_embeddings


# ---------- Base interface ----------


class TextEncoderBase(ABC):
    """
    Common interface for all encoders/indexers.
    """

    @abstractmethod
    def encode(self, texts: List[str]): ...

    def encode_dataset(
        self,
        metadata_path: Path,
        output_path: Path,
        col_text: str,
        col_id: Optional[str],
    ) -> None:
        """Default CSV → embeddings dumper (can be overridden)."""
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(metadata_path)
        texts = df[col_text].astype(str).tolist()
        ids = df[col_id].astype(str).tolist()

        enc = self.encode(texts)
        mapping = convert_to_dict(enc, ids)
        save_embeddings(output_path, mapping)


# ---------- BGE-M3 (sparse only) ----------


class BGESparseEncoder(TextEncoderBase):
    """
    BAAI/bge-m3 sparse lexical weights (dict per doc).
    encode() -> List[dict], encode_dataset() saves {id -> dict}.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=[self.device])

    def encode(self, texts: List[str]):
        out = self.model.encode_queries(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        return out["lexical_weights"]  # List[dict]


# ---------- BGE-M3 (hybrid: dense + sparse + colbert) ----------


class BGEHybridEncoder(TextEncoderBase):
    """
    BAAI/bge-m3 dense + sparse + colbert.
    encode() -> Dict[str, List], encode_dataset() saves {id -> {'dense','sparse','colbert'}}.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=[self.device])

    def encode(self, texts: List[str]):
        return self.model.encode_queries(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)


# ---------- TF-IDF ----------


class TfidfEncoder(TextEncoderBase):
    """
    TF-IDF encoder with optional `max_features`.
    encode() -> np.ndarray of shape (N, D)
    """

    def __init__(self, max_features: Optional[int] = None):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def encode(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.fit_transform(texts)
        return X.toarray()  # careful on very large corpora


# ---------- BM25 indexer ----------


class BM25Indexer(TextEncoderBase):
    """
    BM25 index builder (not an embedding encoder).
    Use build_index(...) or build_index_dataset(...).
    """

    def __init__(self):
        pass

    def encode(self, texts: List[str]):
        retriever = BM25HF(corpus=texts)
        retriever.index(tokenize(texts))
        return retriever

    def encode_dataset(
        self,
        metadata_path: Path,
        output_path: Path,
        col_text: str,
        hub_repo: Optional[str] = None,
    ) -> None:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(metadata_path)
        texts = df[col_text].astype(str).tolist()

        retriever = self.encode(texts)
        save_embeddings(output_path, retriever)

        if not hub_repo:
            raise ValueError("`hub_repo` must be provided (e.g., 'user/dataset_bm25').")
        retriever.save_to_hub(hub_repo)


# ---------- factory ----------


def get_text_encoder(
    backend: str,
    *,
    model_name: Optional[str] = None,
    use_fp16: bool = True,
    device: Optional[str] = None,
) -> TextEncoderBase:
    """
    Choose a concrete encoder/indexer by backend name.
    - 'bge-sparse'
    - 'bge-hybrid'
    - 'tfidf'
    - 'bm25'
    """
    backend = backend.lower()
    if backend == "bge-sparse":
        return BGESparseEncoder(model_name=model_name or "BAAI/bge-m3", use_fp16=use_fp16, device=device)
    if backend == "bge-hybrid":
        return BGEHybridEncoder(model_name=model_name or "BAAI/bge-m3", use_fp16=use_fp16, device=device)
    if backend == "tfidf":
        return TfidfEncoder()
    if backend == "bm25":
        return BM25Indexer()
    raise ValueError(f"Unknown backend '{backend}'.")
