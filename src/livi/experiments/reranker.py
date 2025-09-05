from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
from FlagEmbedding import FlagReranker

from livi.apps.retrieval_eval.data import get_input_data_evaluator


class Reranker:
    """
    Rerank pre-ranked retrieval results with a cross-encoder-like scorer, then
    compute evaluation artifacts (FTP/FP) directly from the reranked results.

    Parameters
    ----------
    queries_ids : List[str]
        Query (anchor) IDs in evaluation order.
    corpus_ids : List[str]
        Candidate corpus IDs.
    relevant_docs : Dict[str, List[str]]
        Mapping: query_id -> list of relevant corpus_ids.
    ranked_docs : pd.DataFrame
        Initial ranking as a DataFrame with columns at least:
        ["query_id", "corpus_id", "rank", "relevant", "score"] (score can be placeholder).
    model : object, optional
        Reranker model. Must expose one of:
            - .compute_score([query, doc], normalize=True)  (e.g., FlagEmbedding/BGE M3 Reranker)
            - .predict([[query, doc], ...], convert_to_tensor=True).tolist()  (SBERT-style)
        You can keep more paths if you need (see rerank_results_bge_torch stub).

    Notes
    -----
    - Call `filter_results_to_rerank()` first to build `self.ranked_docs_ids`
        from the provided `ranked_docs`.
    - Then call the reranking method (e.g., `rerank_results_bge()`).
    - Afterwards, call `get_first_true_positives_from_ranked()` and `get_fp_from_ranked()`,
        then `compute_metrics()` if you need metrics.
    """

    def __init__(
        self,
        queries_ids: List[str],
        corpus_ids: List[str],
        relevant_docs: Dict[str, List[str]],
        ranked_docs: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        *,
        model: Optional[object] = None,
        column_id: Optional[str] = "version_id",
        text_id: Optional[str] = "lyrics",
    ) -> None:
        self.queries_ids = list(map(str, queries_ids))
        self.corpus_ids = list(map(str, corpus_ids))
        self.relevant_docs = {str(k): list(map(str, v)) for k, v in relevant_docs.items()}
        self.ranked_docs = ranked_docs.copy()  # keep a copy

        # Reranker model
        self.model = model

        # Text lookup (id -> lyrics). If no metadata, assume caller fills it later.
        self.id_to_lyrics: Dict[str, str] = {}
        if metadata is not None:
            if not {column_id, text_id}.issubset(metadata.columns):
                raise ValueError(f"`metadata` must contain columns: {sorted({column_id, text_id})}.")
            self.id_to_lyrics = dict(zip(map(str, metadata[column_id]), metadata[text_id].astype(str)))

    # ---------------------------------------------------------------------
    # Prep: restrict & map the candidate set we will rerank
    # ---------------------------------------------------------------------
    def filter_results_to_rerank(self) -> Dict[str, List[str]]:
        """
        Keep only queries with at least one relevant doc and candidates that are in `self.corpus_ids`.
        Also builds `ranked_docs_ids` = {query_id: [candidate_ids...]}.
        """
        if self.ranked_docs is None or self.ranked_docs.empty:
            raise ValueError("`ranked_docs` is empty. Provide a non-empty DataFrame.")

        df = self.ranked_docs.copy()
        req = {"query_id", "corpus_id", "relevant", "rank"}
        if not req.issubset(df.columns):
            raise ValueError(f"`ranked_docs` must contain columns: {sorted(req)}")

        # Drop queries with no relevant doc at all (based on `relevant` column)
        grouped = df.groupby("query_id")["relevant"].apply(list)
        to_keep = grouped[~grouped.apply(lambda x: all(r is False for r in x))].index
        df = df[df["query_id"].isin(to_keep)]

        # Keep only candidates existing in corpus_ids
        df = df[df["corpus_id"].isin(self.corpus_ids)]

        # Build mapping for reranking
        ranked_docs_ids = df.sort_values(["query_id", "rank"]).groupby("query_id")["corpus_id"].apply(list).to_dict()

        return ranked_docs_ids

    # ---------------------------------------------------------------------
    # Reranking variants (choose one based on your model)
    # ---------------------------------------------------------------------
    def rerank(self, ranked_docs_ids: Dict[str, List[str]]) -> List[Dict]:
        """
        Rerank using a BGE-style reranker exposing:
            score = model.compute_score([query_text, doc_text], normalize=True)
        """
        if ranked_docs_ids is None:
            raise RuntimeError("Call filter_results_to_rerank() before reranking.")

        if not self.id_to_lyrics:
            raise RuntimeError("`id_to_lyrics` is empty. Provide metadata or set it yourself.")

        if self.model is None or not hasattr(self.model, "compute_score"):
            raise RuntimeError("`model` must provide `.compute_score([q, d], normalize=True)`.")

        reranked_results: List[Dict] = []
        for qid in tqdm(self.queries_ids, total=len(self.queries_ids), desc="Reranking (BGE)"):
            if qid not in self.ranked_docs_ids:
                continue

            q_rel = set(self.relevant_docs.get(qid, []))
            q_candidates = ranked_docs_ids[qid]
            q_text = self.id_to_lyrics.get(qid, "")

            # score each candidate
            scored = []
            for cid in q_candidates:
                c_text = self.id_to_lyrics.get(cid, "")
                score = self.model.compute_score([q_text, c_text], normalize=True)
                scored.append((cid, float(score)))

            # sort desc by score
            scored.sort(key=lambda x: x[1], reverse=True)

            for rank, (cid, s) in enumerate(scored, start=1):
                reranked_results.append(
                    {
                        "query_id": qid,
                        "corpus_id": cid,
                        "rank": rank,
                        "relevant": int(cid in q_rel),
                        "score": s,
                    }
                )

        return reranked_results

    def get_rank_first_relevant_item(self, reranked_results: np.ndarray) -> pd.DataFrame:
        """
        First relevant item per query from ranked_results.
        """
        reranked_results = pd.DataFrame(reranked_results)

        rank_first_relevant_items: List[Dict] = []

        for qid in self.queries_ids:
            df_relevant = reranked_results[
                (reranked_results["query_id"] == qid) & (reranked_results["relevant"] == 1)
            ].sort_values(by="rank", ascending=True)

            if not df_relevant.empty:
                rank_first_relevant_items.append(
                    {
                        "query_id": qid,
                        "corpus_id": df_relevant.iloc[0]["corpus_id"],
                        "rank": df_relevant.iloc[0]["rank"],
                        "relevant": True,
                        "score": df_relevant.iloc[0]["score"],
                    }
                )

        return pd.DataFrame(rank_first_relevant_items)

    def compute_metrics(
        self, ranked_results: np.ndarray, rank_first_relevant_items: pd.DataFrame, save_path: Path
    ) -> Dict[str, float]:
        df = pd.DataFrame(ranked_results)

        # HR@1
        df_hit1 = df[(df["rank"] == 1) & (df["relevant"] == 1)]
        hr1 = len(df_hit1) / len(self.queries_ids)

        # MR1
        mr1 = float(rank_first_relevant_items["rank"].mean())

        # HR@10 / HR@100
        hr10 = df[(df["rank"] <= 10) & (df["relevant"] == 1)]["query_id"].nunique() / len(self.queries_ids)
        hr100 = df[(df["rank"] <= 100) & (df["relevant"] == 1)]["query_id"].nunique() / len(self.queries_ids)

        # MAP@10
        ap_list = []
        for qid in self.queries_ids:
            top_hits = df[df["query_id"] == qid].sort_values("rank").head(10)
            positives = set(self.relevant_docs.get(qid, []))
            num_correct = 0
            sum_prec = 0.0
            for k, hit in enumerate(top_hits.itertuples(), start=1):
                if hit.corpus_id in positives:
                    num_correct += 1
                    sum_prec += num_correct / k
            ap = sum_prec / min(10, max(1, len(positives))) if num_correct > 0 else 0.0
            ap_list.append(ap)
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


def run_reranker(
    path_metadata: Path,
    path_ranked_input: Path,
    *,
    out_metrics: Optional[Path] = None,
    id_col: str = "version_id",
    text_col: str = "lyrics",
) -> Dict[str, float]:
    """
    Run the cross-encoder reranking pipeline end-to-end.

    Parameters
    ----------
    path_metadata : Path
        CSV containing at least `id_col` and `text_col`.
    path_ranked_input : Path
        CSV of initial ranking with columns: query_id, corpus_id, rank, relevant (bool/int) and optionally score.
    out_metrics_json : Optional[Path]
        Where to save the metrics JSON (if provided).
    id_col : str, default "version_id"
        Name of ID column in metadata.
    text_col : str, default "lyrics"
        Name of text column in metadata.
    reranker_name : Optional[str]
        If provided, load a FlagEmbedding reranker model by name.
    model : Optional[object]
        If provided, use this object directly as the reranker (must expose .compute_score([q,d], normalize=True)).

    Returns
    -------
    Dict[str, float]
        Metrics dict: {"MR1","HR1","HR10","HR100","MAP10"}.
    """
    # Load data
    metadata = pd.read_csv(path_metadata, dtype=str)
    if not {id_col, text_col}.issubset(metadata.columns):
        raise ValueError(f"Metadata CSV must contain '{id_col}' and '{text_col}' columns.")

    ranked_df = pd.read_csv(path_ranked_input, dtype={"query_id": str, "corpus_id": str})

    queries_ids, corpus_ids, relevant_docs = get_input_data_evaluator(metadata)

    # Load  model
    model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    # Build and run pipeline
    reranker = Reranker(
        queries_ids=queries_ids,
        corpus_ids=corpus_ids,
        relevant_docs=relevant_docs,
        ranked_docs=ranked_df,
        metadata=metadata,
        model=model,
        column_id=id_col,
        text_id=text_col,
    )

    ranked_docs_ids = reranker.filter_results_to_rerank()
    reranked_results = reranker.rerank(ranked_docs_ids)
    first_relevant_items = reranker.get_rank_first_relevant_item(reranked_results)
    metrics = reranker.compute_metrics(reranked_results, first_relevant_items, out_metrics)

    return metrics
