from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Mapping, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def filter_ids(
    valid_ids: List[str], ids: Union[List[str], Dict[str, List[str]]]
) -> Union[List[str], Dict[str, List[str]]]:
    """
    Filter ids to only include keys/values present in embeddings.

    - If ids is a list: keep only ids that exist in embeddings.
    - If ids is a dict: keep only keys in embeddings and filter each value list
      to ids that exist in embeddings.
    """

    if isinstance(ids, list):
        return [i for i in ids if i in valid_ids]

    elif isinstance(ids, dict):
        return {k: [v for v in vs if v in valid_ids] for k, vs in ids.items() if k in valid_ids}

    return ids


class Ranker:
    """
    Class for evaluating retrieval models on Version Identification task.
    """

    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        queries_ids: Optional[List[str]] = None,
        corpus_ids: Optional[List[str]] = None,
        relevant_docs: Optional[Dict[str, Sequence[str]]] = None,
        query_embeddings: Optional[Dict[str, np.ndarray]] = None,
        k: Optional[int] = 100,
    ):
        # Data setup
        valid_ids = list(embeddings.keys())
        self.queries_ids = filter_ids(valid_ids, queries_ids)
        self.corpus_ids = filter_ids(valid_ids, corpus_ids)
        self.relevant_docs = filter_ids(valid_ids, relevant_docs)

        # Number of top results to keep per query
        self.k = min(k if k is not None else 100, len(self.corpus_ids))

        print(
            f"{len(embeddings)} embeddings, {len(self.queries_ids) if self.queries_ids else 'all'} queries, {len(self.corpus_ids) if self.corpus_ids else 'all'} corpus"
        )

        # Precomputed embeddings
        self.embeddings = embeddings
        self.query_embeddings = query_embeddings

    # ---- Subclasses must implement this ----
    def compute_similarity_matrix(self) -> np.ndarray:
        query_embed = self.query_embeddings or self.embeddings
        q = np.vstack([query_embed[qid] for qid in self.queries_ids])
        c = np.vstack([self.embeddings[cid] for cid in self.corpus_ids])
        qt = F.normalize(torch.from_numpy(q).float(), p=2, dim=1)
        ct = F.normalize(torch.from_numpy(c).float(), p=2, dim=1)
        sim = (qt @ ct.T).numpy()
        _mask_self_matches(sim, self.queries_ids, self.corpus_ids)
        return sim

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


# ---- Orchestrator ---------------------------------------------------


def run_evaluation(
    path_metadata: Path,
    path_embeddings: Optional[Path] = None,
    path_embeddings_2: Optional[Path] = None,
    *,
    get_single_embedding: bool = True,
    col_id: str = "version_id",
    text_id: str = "lyrics",
    k: int = 100,
    path_metrics: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Run a retrieval evaluation pipeline end-to-end.

    This function:
        1) loads metadata CSV and derives:
            - queries_ids
            - corpus_ids
            - relevant_docs (mapping query_id -> list of positives)
        2) builds the ranker
        3) computes similarity matrix, ranks, error tables, and metrics
        4) saves outputs (if paths provided)

    Parameters
    ----------
    path_metadata : Path
        CSV with at least columns `col_id` and `text_id`.
    path_embeddings : Optional[Path]
        Path to embeddings (for most eval types). For "fusion", typically dense here.
    path_embeddings_2 : Optional[Path]
        Second embeddings path (used for query embeddings when model is e5-large-instruct
    get_single_embedding : bool, default True
        If True and the file contains multiple vectors per id (e.g., chunks),
        average them into a single vector. Passed to `get_embeddings`.
    col_id : str, default "version_id"
        ID column in metadata.
    text_id : str, default "lyrics"
        Text column in metadata (used for BM25, reranking, etc.).
    k : int, default 100
        Top-K to keep when ranking.
    path_metrics : Optional[Path]
        Where to save a one-row metrics CSV. If None, not saved.

    Returns
    -------
    Dict[str, float]
        Metrics dict with keys: MR1, HR1, HR10, HR100, MAP10.
    """
    # ---- Load & validate metadata -----------------------------------
    metadata = pd.read_csv(path_metadata, dtype={col_id: str})
    req_cols = {col_id, text_id}
    if not req_cols.issubset(metadata.columns):
        raise ValueError(f"Metadata CSV must contain columns: {sorted(req_cols)}")

    md5_to_id = dict(zip(metadata["md5_encoded"], metadata[col_id]))
    queries_ids, corpus_ids, relevant_docs = get_input_data_evaluator(metadata)

    # ---- Load embeddings ---------------------
    embeddings = None

    if path_embeddings is None:
        raise ValueError("`path_embeddings` is required for this evaluation type.")

    embeddings = get_embeddings(path_embeddings, get_single_embedding=get_single_embedding)
    # Rename keys to match ids in metadata
    embeddings = {md5_to_id[k]: v for k, v in embeddings.items() if k in md5_to_id}
    print(f"{embeddings.keys()}")

    if path_embeddings_2:
        query_embeddings = get_embeddings(path_embeddings_2, get_single_embedding=get_single_embedding)
    else:
        query_embeddings = None

    # ---- Build ranker -----------------------------------------------
    ranker = Ranker(
        embeddings=embeddings,
        queries_ids=queries_ids,
        corpus_ids=corpus_ids,
        relevant_docs=relevant_docs,
        query_embeddings=query_embeddings,
        k=k,
    )

    # ---- Pipeline ----------------------------------------------------
    sim_matrix = ranker.compute_similarity_matrix()
    ranked_results = ranker.rank_results(sim_matrix)

    first_relevant_items = ranker.get_rank_first_relevant_item(sim_matrix)

    metrics = ranker.compute_metrics(ranked_results, first_relevant_items, save_path=path_metrics)
    return metrics
