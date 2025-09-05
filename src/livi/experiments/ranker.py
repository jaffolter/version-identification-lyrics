from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from bm25s import BM25, tokenize
from FlagEmbedding import BGEM3FlagModel
from livi.apps.retrieval_eval.data import get_input_data_evaluator
from livi.core.data.utils.io_toolbox import get_embeddings


# ---------------------------
# Utilities
# ---------------------------


def _mask_self_matches(sim: np.ndarray, queries: List[str], corpus: List[str]) -> None:
    """In-place: set diagonal-like query==corpus matches to -inf."""
    for i, qid in enumerate(queries):
        for j, cid in enumerate(corpus):
            if qid == cid:
                sim[i, j] = -np.inf


# ---------------------------
# Base ranker (shared logic)
# ---------------------------


class BaseRanker:
    """
    Core evaluation pipeline shared by all rankers.
    Subclasses implement `compute_similarity_matrix()` and set/return a (Q x C) np.ndarray.
    """

    def __init__(
        self,
        queries_ids: List[str],
        corpus_ids: List[str],
        relevant_docs: Dict[str, Sequence[str]],
        k: Optional[int] = 100,
    ):
        self.queries_ids = queries_ids
        self.corpus_ids = corpus_ids
        self.relevant_docs = relevant_docs

        self.k = min(k if k is not None else 100, len(corpus_ids))

    # ---- Subclasses must implement this ----
    def compute_similarity_matrix(self) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # ---- Shared ranking ----
    def rank_results(self, sim_matrix: np.ndarray, save_path: Optional[Path] = None) -> List[Dict]:
        """
        Convert a similarity matrix (Q x C) into a ranked list of results with relevance labels.
        """
        sim = torch.from_numpy(sim_matrix)
        nb_queries = sim.shape[0]

        # For each query: top-k indices by descending similarity
        top_vals, top_idx = torch.topk(sim, self.k, dim=1, largest=True, sorted=True)
        top_vals, top_idx = top_vals.cpu().tolist(), top_idx.cpu().tolist()

        ranked_results: List[Dict] = []
        for qi in range(nb_queries):
            qid = self.queries_ids[qi]
            positives = set(self.relevant_docs.get(qid, []))
            rank = 1
            for ci, score in zip(top_idx[qi], top_vals[qi]):
                cid = self.corpus_ids[ci]
                ranked_results.append(
                    {
                        "query_id": qid,
                        "corpus_id": cid,
                        "score": float(score),
                        "rank": rank,
                        "relevant": int(cid in positives),
                    }
                )
                rank += 1

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(ranked_results).to_csv(save_path, index=False)

        return ranked_results

    def get_rank_first_relevant_item(self, sim_matrix: np.ndarray) -> pd.DataFrame:
        """
        For each query, find the first (best-rank) relevant item from the *full* ranking.
        Returns a DataFrame with columns: query_id, corpus_id, rank, score.
        """
        rows: List[Dict] = []

        for qid in self.queries_ids:
            q_idx = self.queries_ids.index(qid)
            scores = torch.from_numpy(sim_matrix[q_idx])
            relevant_ids = set(self.relevant_docs.get(qid, []))

            # full descending sort – stable
            _, sorted_idx = torch.topk(scores, k=scores.numel(), largest=True, sorted=True)

            for rank, idx in enumerate(sorted_idx.tolist(), start=1):
                if self.corpus_ids[idx] in relevant_ids:
                    rows.append(
                        {
                            "query_id": qid,
                            "corpus_id": self.corpus_ids[idx],
                            "rank": rank,
                            "score": float(scores[idx]),
                        }
                    )
                    break

        return pd.DataFrame(rows)

    def get_fp(self, ranked_results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """
        False positives from ranked_results: keep rank==1 results where relevant==0.
        """
        df = pd.DataFrame(ranked_results)
        return df[(df["relevant"] == 0) & (df["rank"] == 1)].copy()

    def get_fp_fn(
        self, df_ranked_first_item: pd.DataFrame, df_fn: pd.DataFrame, save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Join false positives with first false negatives.
        Assumes df_ranked_first_item are the first relevant items (rename "FN" to whatever your pipeline produces).
        """
        # keep only error cases (first relevant item is not at rank 1)
        df_ranked_first_item = df_ranked_first_item[df_ranked_first_item["rank"] != 1]
        df = df_ranked_first_item.merge(df_fn, on="query_id", how="left", suffixes=("_fp", "_fn"))

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)

        return df

    def compute_metrics(
        self,
        ranked_results: Sequence[Mapping[str, Any]],
        df_ranked_first_item: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Compute MR1, HR@1/10/100, and MAP@10 from ranked results (list of dicts) and first relevant ranks.
        """
        df = pd.DataFrame(ranked_results)

        # HR@1
        df_hit1 = df[(df["rank"] == 1) & (df["relevant"] == 1)]
        hr1 = len(df_hit1) / len(self.queries_ids) if len(self.queries_ids) else 0.0

        # MR1 (guard against empty)
        mr1 = float(df_ranked_first_item["rank"].mean()) if not df_ranked_first_item.empty else 0.0

        # HR@10 / HR@100
        hr10 = (
            df[(df["rank"] <= 10) & (df["relevant"] == 1)]["query_id"].nunique() / len(self.queries_ids)
            if len(self.queries_ids)
            else 0.0
        )
        hr100 = (
            df[(df["rank"] <= 100) & (df["relevant"] == 1)]["query_id"].nunique() / len(self.queries_ids)
            if len(self.queries_ids)
            else 0.0
        )

        # MAP@10
        ap_list: List[float] = []
        for qid in self.queries_ids:
            top_hits = df[df["query_id"] == qid].sort_values("rank").head(10)
            positives = set(self.relevant_docs.get(qid, []))
            num_correct = 0
            sum_prec = 0.0
            for k, hit in enumerate(top_hits.itertuples(), start=1):
                if hit.corpus_id in positives:
                    num_correct += 1
                    sum_prec += num_correct / k
            ap_list.append(sum_prec / min(10, max(1, len(positives))) if num_correct > 0 else 0.0)
        map10 = float(np.mean(ap_list)) if ap_list else 0.0

        metrics = {
            "MR1": mr1,
            "HR1": hr1,
            "HR10": hr10,
            "HR100": hr100,
            "MAP10": map10,
        }

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([metrics]).sort_values(by="HR1", ascending=False).to_csv(save_path, index=False)

        return metrics


# ---------------------------
# Specialized rankers
# ---------------------------


class DenseRanker(BaseRanker):
    """Cosine similarity on dense embeddings dict: id -> (D,) np.ndarray."""

    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        queries_ids: Optional[List[str]] = None,
        corpus_ids: Optional[List[str]] = None,
        relevant_docs: Optional[Dict[str, Sequence[str]]] = None,
        query_embeddings: Optional[Dict[str, np.ndarray]] = None,
        k: Optional[int] = 100,
    ):
        super().__init__(queries_ids, corpus_ids, relevant_docs, k)
        self.embeddings = embeddings
        self.query_embeddings = query_embeddings

    def compute_similarity_matrix(self) -> np.ndarray:
        query_embed = self.query_embeddings or self.embeddings
        q = np.vstack([query_embed[qid] for qid in self.queries_ids])
        c = np.vstack([self.embeddings[cid] for cid in self.corpus_ids])
        qt = F.normalize(torch.from_numpy(q).float(), p=2, dim=1)
        ct = F.normalize(torch.from_numpy(c).float(), p=2, dim=1)
        sim = (qt @ ct.T).numpy()
        _mask_self_matches(sim, self.queries_ids, self.corpus_ids)
        return sim


class SparseRanker(BaseRanker):
    """
    Lexical matching using a provided sparse model:
        - sparse_embeddings: id -> sparse structure (e.g., lexical weights dict)
        - model must expose: compute_lexical_matching_score(queries, corpus) -> np.ndarray
    """

    def __init__(
        self,
        queries_ids: List[str],
        corpus_ids: List[str],
        relevant_docs: Dict[str, Sequence[str]],
        sparse_embeddings: Dict[str, Any],
        model: BGEM3FlagModel,
        k: Optional[int] = 100,
    ):
        super().__init__(queries_ids, corpus_ids, relevant_docs, k)
        self.sparse_embeddings = sparse_embeddings
        self.model = model

    def compute_similarity_matrix(self) -> np.ndarray:
        q_sparse = [self.sparse_embeddings[qid] for qid in self.queries_ids]
        c_sparse = [self.sparse_embeddings[cid] for cid in self.corpus_ids]
        sim = self.model.compute_lexical_matching_score(q_sparse, c_sparse)
        _mask_self_matches(sim, self.queries_ids, self.corpus_ids)
        return sim


class BGERanker(BaseRanker):
    """
    Fusion of dense + sparse + colbert for BGE-M3 style embeddings:
        embeddings[id] -> {"dense": (D,), "sparse": <lexical>, "colbert": (T, Dcol)}
    model must provide:
        - compute_lexical_matching_score(sparse_q, sparse_c) -> float/np
        - colbert_score(colbert_q, colbert_c) -> tensor(float)
    """

    def __init__(
        self,
        queries_ids: List[str],
        corpus_ids: List[str],
        relevant_docs: Dict[str, Sequence[str]],
        embeddings: Dict[str, Dict[str, Any]],
        model: BGEM3FlagModel,
        k: Optional[int] = 100,
    ):
        super().__init__(queries_ids, corpus_ids, relevant_docs, k)
        self.embeddings = embeddings
        self.model = model

    def compute_similarity_matrix(self) -> np.ndarray:
        sim_rows: List[List[float]] = []
        for qid in self.queries_ids:
            row: List[float] = []
            q = self.embeddings[qid]
            for cid in self.corpus_ids:
                c = self.embeddings[cid]
                dense = float(np.dot(q["dense"], c["dense"]))
                sparse = float(self.model.compute_lexical_matching_score(q["sparse"], c["sparse"]))
                colbert = float(self.model.colbert_score(q["colbert"], c["colbert"]).item())
                score = (dense + sparse + colbert) / 3.0
                row.append(score)
            sim_rows.append(row)

        sim = np.asarray(sim_rows, dtype=float)
        _mask_self_matches(sim, self.queries_ids, self.corpus_ids)
        return sim


class BM25Ranker(BaseRanker):
    """
    BM25 similarity: requires a pre-built retriever with .retrieve(tokens, k) API,
    and a mapping id->text to map back scores over the full corpus order.
    """

    def __init__(
        self,
        queries_ids: List[str],
        corpus_ids: List[str],
        relevant_docs: Dict[str, Sequence[str]],
        bm25_model: BM25,
        id_to_lyrics: Dict[str, str],
        k: Optional[int] = 100,
    ):
        super().__init__(queries_ids, corpus_ids, relevant_docs, k)
        self.id_to_lyrics = id_to_lyrics
        self.model = bm25_model

    def compute_similarity_matrix(self) -> np.ndarray:
        sim_rows: List[List[float]] = []
        for qid in self.queries_ids:
            tokens = tokenize(self.id_to_lyrics[qid])  # consistent tokenizer usage
            docs, scores = self.model.retrieve(tokens, k=len(self.corpus_ids))

            doc_to_score = {d["text"]: float(s) for d, s in zip(docs[0], scores[0])}
            row = [doc_to_score.get(self.id_to_lyrics[cid], 0.0) for cid in self.corpus_ids]
            sim_rows.append(row)

        sim = np.asarray(sim_rows, dtype=float)
        _mask_self_matches(sim, self.queries_ids, self.corpus_ids)
        return sim


class ChunkedRanker(BaseRanker):
    """
    Chunked similarity:
      chunked_embeddings[id] -> np.ndarray of shape (num_chunks, dim)
      Score(query, doc) = r-mean-max over chunk cosine sims.
    """

    def __init__(
        self,
        queries_ids: List[str],
        corpus_ids: List[str],
        relevant_docs: Dict[str, Sequence[str]],
        chunked_embeddings: Dict[str, np.ndarray],
        k: Optional[int] = 100,
    ):
        super().__init__(queries_ids, corpus_ids, relevant_docs, k)
        self.chunked_embeddings = chunked_embeddings

    @staticmethod
    def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
        return a @ b.T

    @staticmethod
    def _r_mean_max(sim_m: torch.Tensor) -> torch.Tensor:
        return torch.max(sim_m, dim=1).values.mean()

    def compute_similarity_matrix(self) -> np.ndarray:
        rows: List[List[float]] = []
        for qid in self.queries_ids:
            q_chunks = torch.as_tensor(self.chunked_embeddings[qid], dtype=torch.float32)
            row: List[float] = []
            for cid in self.corpus_ids:
                c_chunks = torch.as_tensor(self.chunked_embeddings[cid], dtype=torch.float32)
                sim_m = self._cosine_sim(q_chunks, c_chunks)
                score = float(self._r_mean_max(sim_m))
                row.append(score)
            rows.append(row)

        sim = np.asarray(rows, dtype=float)
        _mask_self_matches(sim, self.queries_ids, self.corpus_ids)
        return sim


class FusionRanker(BaseRanker):
    """
    Fuse two precomputed similarity matrices (dense vs sparse/BM25).
    `fuse_method`: "mean" (softmax-mean) or "max".
    """

    def __init__(
        self,
        queries_ids: List[str],
        corpus_ids: List[str],
        relevant_docs: Dict[str, Sequence[str]],
        sim_matrix_a: np.ndarray,
        sim_matrix_b: np.ndarray,
        fuse_method: str = "mean",
        k: Optional[int] = 100,
    ):
        super().__init__(queries_ids, corpus_ids, relevant_docs, k)
        self.A = sim_matrix_a
        self.B = sim_matrix_b
        self.fuse_method = fuse_method

    def compute_similarity_matrix(self) -> np.ndarray:
        if self.A.shape != self.B.shape:
            raise ValueError("Similarity matrices A and B must have the same shape.")

        if self.fuse_method == "mean":
            m = nn.Softmax(dim=1)
            A_sm = m(torch.from_numpy(self.A))
            B_sm = m(torch.from_numpy(self.B))
            sim = ((A_sm + B_sm) / 2.0).numpy()
        elif self.fuse_method == "max":
            sim = np.maximum(self.A, self.B)
        else:
            raise ValueError(f"Unknown fuse_method: {self.fuse_method}")

        _mask_self_matches(sim, self.queries_ids, self.corpus_ids)
        return sim


# ---- Orchestrator ---------------------------------------------------


def run_evaluation(
    eval_type: str,
    path_metadata: Path,
    path_embeddings: Optional[Path] = None,
    path_embeddings_2: Optional[Path] = None,
    *,
    get_single_embedding: bool = True,
    col_id: str = "version_id",
    text_id: str = "lyrics",
    k: int = 100,
    hub_repo: Optional[str] = None,
    fusion_sparse_type: Optional[str] = None,  # "sparse" | "bm25"
    path_ranked_csv: Optional[Path] = None,
    path_fp_tp_csv: Optional[Path] = None,
    path_metrics_csv: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Run a retrieval evaluation pipeline end-to-end.

    This function:
      1) loads your metadata CSV and derives:
         - queries_ids
         - corpus_ids
         - relevant_docs (mapping query_id -> list of positives)
         - id_to_lyrics (for text-based models such as BM25/Rerankers)
      2) builds the appropriate ranker based on `eval_type`
      3) computes similarity matrix, ranks, error tables, and metrics
      4) optionally saves CSV artifacts

    Parameters
    ----------
    eval_type : str
        One of: "dense", "sparse", "bge", "bm25", "chunked", "fusion".
    path_metadata : Path
        CSV with at least columns `col_id` and `text_id`.
    path_embeddings : Optional[Path]
        Path to embeddings (for most eval types). For "fusion", typically dense here.
    path_embeddings_2 : Optional[Path]
        Second embeddings path (used for "fusion" paired with `fusion_sparse_type`,
        e.g., sparse embeddings), or a second dense if you wish (but code assumes
        sparse/bm25 for the second leg).
    get_single_embedding : bool, default True
        If True and the file contains multiple vectors per id (e.g., chunks),
        average them into a single vector. Passed to `get_embeddings`.
    col_id : str, default "version_id"
        ID column in metadata.
    text_id : str, default "lyrics"
        Text column in metadata (used for BM25, reranking, etc.).
    k : int, default 100
        Top-K to keep when ranking.
    hub_repo : Optional[str]
        HF Hub repo id for BM25 index (e.g. "user/dataset_bm25") when eval_type="bm25"
        or "fusion" with `fusion_sparse_type="bm25"`.
    fusion_sparse_type : Optional[str]
        For eval_type="fusion", choose "sparse" (BGE lexical) or "bm25".
    path_ranked_csv : Optional[Path]
        Where to save ranked results CSV. If None, not saved.
    path_fp_tp_csv : Optional[Path]
        Where to save (first-relevant vs FP) join CSV. If None, not saved.
    path_metrics_csv : Optional[Path]
        Where to save a one-row metrics CSV. If None, not saved.

    Returns
    -------
    Dict[str, float]
        Metrics dict with keys: MR1, HR1, HR10, HR100, MAP10.
    """
    eval_type = eval_type.lower()

    # ---- Load & validate metadata -----------------------------------
    metadata = pd.read_csv(path_metadata, dtype={col_id: str})
    req_cols = {col_id, text_id}
    if not req_cols.issubset(metadata.columns):
        raise ValueError(f"Metadata CSV must contain columns: {sorted(req_cols)}")

    queries_ids, corpus_ids, relevant_docs = get_input_data_evaluator(metadata)
    id_to_lyrics = dict(zip(metadata[col_id].astype(str), metadata[text_id].astype(str)))

    # ---- For most eval types we need embeddings ---------------------
    embeddings = None
    embeddings_2 = None
    if eval_type not in {"bm25", "fusion"}:
        if path_embeddings is None:
            raise ValueError("`path_embeddings` is required for this evaluation type.")
        embeddings = get_embeddings(path_embeddings, get_single_embedding=get_single_embedding)

    if eval_type == "fusion":
        if path_embeddings is None or path_embeddings_2 is None:
            raise ValueError("For fusion, both `path_embeddings` and `path_embeddings_2` are required.")
        if fusion_sparse_type not in {"sparse", "bm25"}:
            raise ValueError("`fusion_sparse_type` must be 'sparse' or 'bm25' for fusion.")
        embeddings = get_embeddings(path_embeddings, get_single_embedding=get_single_embedding)
        if fusion_sparse_type == "sparse":
            embeddings_2 = get_embeddings(path_embeddings_2, get_single_embedding=False)  # sparse is dict-like
        # bm25 branch doesn't load embeddings_2

    # ---- Build ranker -----------------------------------------------
    if eval_type == "dense":
        ranker = DenseRanker(embeddings, queries_ids, corpus_ids, relevant_docs, k=k)

    elif eval_type == "sparse":
        # BGE sparse model for lexical matching
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        ranker = SparseRanker(queries_ids, corpus_ids, relevant_docs, embeddings, model, k=k)

    elif eval_type == "bge":
        # BGE full: embeddings must be dict[id] -> {"dense","sparse","colbert"}
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        ranker = BGERanker(queries_ids, corpus_ids, relevant_docs, embeddings, model, k=k)

    elif eval_type == "bm25":
        if not hub_repo:
            raise ValueError("`hub_repo` is required for BM25 evaluation.")
        bm25_model = BM25HF.load_from_hub(hub_repo, load_corpus=True)
        ranker = BM25Ranker(queries_ids, corpus_ids, relevant_docs, bm25_model, id_to_lyrics, k=k)

    elif eval_type == "chunked":
        # chunked embeddings: dict[id] -> (num_chunks, dim) ndarray
        ranker = ChunkedRanker(queries_ids, corpus_ids, relevant_docs, embeddings, k=k)

    elif eval_type == "fusion":
        # A: dense similarity from `embeddings`
        dense_ranker = DenseRanker(queries_ids, corpus_ids, relevant_docs, embeddings, k=k)
        sim_matrix_a = dense_ranker.compute_similarity_matrix()

        # B: either sparse or bm25
        if fusion_sparse_type == "sparse":
            sparse_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            sparse_ranker = SparseRanker(queries_ids, corpus_ids, relevant_docs, embeddings_2, sparse_model, k=k)
            sim_matrix_b = sparse_ranker.compute_similarity_matrix()
        else:  # "bm25"
            if not hub_repo:
                raise ValueError("`hub_repo` is required for fusion with BM25.")
            bm25_model = BM25.load_from_hub(hub_repo, load_corpus=True)
            bm25_ranker = BM25Ranker(queries_ids, corpus_ids, relevant_docs, bm25_model, id_to_lyrics, k=k)
            sim_matrix_b = bm25_ranker.compute_similarity_matrix()

        ranker = FusionRanker(
            queries_ids, corpus_ids, relevant_docs, sim_matrix_a, sim_matrix_b, fuse_method="mean", k=k
        )

    else:
        raise ValueError(f"Unknown eval_type: {eval_type!r}")

    # ---- Pipeline ----------------------------------------------------
    sim_matrix = ranker.compute_similarity_matrix()
    ranked_results = ranker.rank_results(sim_matrix, save_path=path_ranked_csv)

    first_relevant_items = ranker.get_rank_first_relevant_item(sim_matrix)
    fp = ranker.get_fp(ranked_results)
    _ = ranker.get_fp_fn(first_relevant_items, fp, save_path=path_fp_tp_csv)

    metrics = ranker.compute_metrics(ranked_results, first_relevant_items)
    return metrics
