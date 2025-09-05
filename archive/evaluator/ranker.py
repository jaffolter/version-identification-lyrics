import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import bm25s
import torch.nn as nn


class Ranker:
    def __init__(
        self,
        queries_ids,
        corpus_ids,
        relevant_docs,
        embeddings=None,
        sparse_embeddings=None,
        query_embeddings=None,
        chunked_embeddings=None,
        model=None,
        bm25_model=None,
        k=100,
        model_name=None,
        dataset_name=None,
        metadata=None,
        experiment_type=None,
    ):
        self.queries_ids = queries_ids
        self.corpus_ids = corpus_ids
        self.relevant_docs = relevant_docs
        # Embedding model
        self.model = model
        self.bm25_model = bm25_model
        # Embeddings
        self.query_embeddings = query_embeddings
        self.embeddings = embeddings
        self.sparse_embeddings = sparse_embeddings
        self.chunked_embeddings = chunked_embeddings
        #self.id_to_lyrics = dict(zip(metadata["version_id"], metadata["lyrics"]))
        self.id_to_lyrics = dict()
        # k value for ranking
        self.k = min(k, len(corpus_ids))  # Ensure k does not exceed corpus size
        # Results
        self.sim_matrix = None
        self.first_true_positives = None
        self.ranked_results = None
        self.ranked_results_dict = None
        # Metrics
        self.metrics = None
        # Dataframes
        self.df_fp = None
        # Model and dataset names
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.experiment_type = experiment_type

    def get_similarity_matrix(self) -> np.array:
        # Stack query embeddings into a matrix E: shape (Q, D)
        if self.query_embeddings:
            query_E = np.vstack([self.query_embeddings[v] for v in self.queries_ids])
        else:
            query_E = np.vstack([self.embeddings[v] for v in self.queries_ids])
        query_E_tensor = torch.from_numpy(query_E).float()  # shape (N, D)
        query_E_normalized = F.normalize(
            query_E_tensor, p=2, dim=1
        )  # L2 normalize each row

        # Stack corpus embeddings into a matrix E: shape (C, D)
        E = np.vstack([self.embeddings[v] for v in self.corpus_ids])
        E_tensor = torch.from_numpy(E).float()
        E_normalized = F.normalize(E_tensor, p=2, dim=1)

        # Compute cosine similarity matrix as dot product of normalized rows
        cosine_sim_matrix = query_E_normalized @ E_normalized.T
        cosine_sim_matrix = cosine_sim_matrix.numpy()

        # Mask self-matches
        # (i.e., query i with corpus item i)
        for i, qid in enumerate(self.queries_ids):
            for j, cid in enumerate(self.corpus_ids):
                if qid == cid:
                    cosine_sim_matrix[i, j] = -np.inf
        self.sim_matrix = cosine_sim_matrix
        return cosine_sim_matrix

    def get_sparse_similarity_matrix(self) -> np.array:
        # Retrieve the corpus and query embeddings
        query_E = [self.sparse_embeddings[v] for v in self.queries_ids]
        corpus_E = [self.sparse_embeddings[v] for v in self.corpus_ids]

        sim_matrix = self.model.compute_lexical_matching_score(query_E, corpus_E)
        # Mask self-matches
        # (i.e., query i with corpus item i)
        for i, qid in enumerate(self.queries_ids):
            for j, cid in enumerate(self.corpus_ids):
                if qid == cid:
                    sim_matrix[i, j] = -np.inf

        self.sim_matrix = sim_matrix
        return sim_matrix

    def get_bge_similarity_matrix(self):
        sim_matrix = []
        for qid in tqdm(self.queries_ids, total=len(self.queries_ids), desc=f"Ranking"):
            res_query = []
            embedding_q = self.embeddings[qid]
            for cid in self.corpus_ids:
                embedding_c = self.embeddings[cid]

                dense_score = embedding_q["dense"] @ embedding_c["dense"].T
                sparse_score = self.model.compute_lexical_matching_score(
                    embedding_q["sparse"], embedding_c["sparse"]
                )
                colbert_score = self.model.colbert_score(
                    embedding_q["colbert"], embedding_c["colbert"]
                ).item()

                # print(f"Query: {qid}, Corpus: {cid}, Dense Score: {dense_score}, Sparse Score: {sparse_score}, Colbert Score: {colbert_score}")
                # Combine the scores
                combined_score = (dense_score + sparse_score + colbert_score) / 3.0
                # similarity_scores = scores['colbert+sparse+dense']
                res_query.append(combined_score)

                # Reshape the flat list into a matrix
            sim_matrix.append(res_query)
        sim_matrix = np.array(sim_matrix)

        # Mask self-matches
        for i, qid in enumerate(self.queries_ids):
            for j, cid in enumerate(self.corpus_ids):
                if qid == cid:
                    sim_matrix[i, j] = -np.inf

        self.sim_matrix = sim_matrix
        return sim_matrix

    def get_bm25_similarity_matrix(self) -> np.array:
        """
        Compute a similarity matrix using BM25S scores between queries and corpus lyrics.
        """
        sim_matrix = []

        for qid in self.queries_ids:
            query_tokens = bm25s.tokenize(self.id_to_lyrics[qid])

            docs, scores = self.bm25_model.retrieve(
                query_tokens, k=len(self.corpus_ids)
            )

            new_docs = [x["text"] for x in docs[0]]

            # Map retrieved doc to score
            doc_to_score = dict(zip(new_docs, list(scores[0])))

            # For every corpus ID, retrieve its score or assign 0 if not retrieved
            new_scores = [
                doc_to_score.get(self.id_to_lyrics[cid], 0.0) for cid in self.corpus_ids
            ]

            sim_matrix.append(new_scores)

        sim_matrix = np.array(sim_matrix)

        # Mask self-matches
        for i, qid in enumerate(self.queries_ids):
            for j, cid in enumerate(self.corpus_ids):
                if qid == cid:
                    sim_matrix[i, j] = -np.inf

        self.sim_matrix = sim_matrix

        return sim_matrix

    def get_similarity_matrix_fused(self, fuse_method="mean") -> np.array:
        m = nn.Softmax(dim=1)

        # We compute similarity matrix using the dense method
        dense_sim_matrix = self.get_similarity_matrix()

        # We compute similarity matrix using the sparse method
        sparse_sim_matrix = (
            self.get_sparse_similarity_matrix()
            if self.sparse_embeddings
            else self.get_bm25_similarity_matrix()
        )

        assert dense_sim_matrix.shape == sparse_sim_matrix.shape, (
            "The two similarity matrices must have the same shape."
        )

        # We first convert the scores to softmax for comparison
        dense_sim_matrix_softmax = m(torch.from_numpy(dense_sim_matrix))
        sparse_sim_matrix_softmax = m(torch.from_numpy(sparse_sim_matrix))

        # We fuse the two similarity matrices
        if fuse_method == "mean":
            self.sim_matrix = np.mean(
                [dense_sim_matrix_softmax, sparse_sim_matrix_softmax], axis=0
            )
        elif fuse_method == "max":
            self.sim_matrix = np.maximum(dense_sim_matrix, sparse_sim_matrix)

    def rank_results(
        self,
    ) -> list[dict]:
        cosine_sim_matrix = torch.from_numpy(self.sim_matrix)
        nb_queries = cosine_sim_matrix.shape[0]
        
        print(f"Ranking {nb_queries} queries with {len(self.corpus_ids)} corpus documents...")

        # Get top-k values for each row (query)
        pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
            cosine_sim_matrix,
            self.k,
            dim=1,
            largest=True,
            sorted=True,
        )
        pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
        pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

        # Store the results in a list of dictionaries
        # key is the query id, value is a list of tuples (doc_id, score)
        ranked_results = []
        ranked_results_dict = {query_id: [] for query_id in self.queries_ids}

        for query_itr in range(nb_queries):
            query_id = self.queries_ids[query_itr]
            query_relevant_docs = self.relevant_docs[query_id]
            rank = 1

            for doc_idx, score in zip(
                pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]
            ):
                corpus_id = self.corpus_ids[doc_idx]
                relevant = corpus_id in query_relevant_docs

                ranked_results.append(
                    {
                        "query_id": query_id,
                        "corpus_id": corpus_id,
                        "score": score,
                        "rank": rank,
                        "relevant": relevant,
                    }
                )
                ranked_results_dict[query_id].append([query_id, corpus_id])
                rank += 1

        self.ranked_results = ranked_results
        self.ranked_results_dict = ranked_results_dict

        # Save the ranked results to a CSV file
        df_ranked = pd.DataFrame(ranked_results)

        os.makedirs(
            f"new_results/ranking/{self.dataset_name.replace('/', '_')}",
            exist_ok=True,
        )
        df_ranked.to_csv(
            f"new_results/ranking/{self.dataset_name.replace('/', '_')}/{self.model_name.replace('/', '_')}_{self.experiment_type}.csv",
            index=False,
        )

    def get_first_true_positives(
        self,
    ) -> dict:
        first_true_positives = []

        for query_id in self.queries_ids:
            q_idx = self.queries_ids.index(query_id)
            scores = torch.from_numpy(self.sim_matrix[q_idx])
            relevant_set = set(self.relevant_docs[query_id])

            # full descending sort – stable
            _, sorted_idx = torch.topk(
                scores, k=scores.numel(), largest=True, sorted=True
            )

            for rank, c_idx in enumerate(sorted_idx.tolist(), start=1):
                corpus_id = self.corpus_ids[c_idx]
                if corpus_id in relevant_set:
                    first_true_positives.append(
                        {
                            "rank": rank,
                            "query_id": query_id,
                            "relevant": True,
                            "corpus_id": corpus_id,
                            "score": scores[c_idx].item(),
                        }
                    )
                    break

        self.first_true_positives = first_true_positives

    def get_fp(self) -> dict:
        df_ranked = pd.DataFrame(self.ranked_results)
        df_fp = df_ranked[(df_ranked["rank"] == 1) & (df_ranked["relevant"] == 0)]

        df_tp = pd.DataFrame(self.first_true_positives)
        df_tp = df_tp[df_tp.query_id.isin(df_fp.query_id)]
        df_tp = df_tp.rename(
            columns={
                "corpus_id": "tp_id",
                "score": "tp_score",
                "rank": "tp_rank",
            }
        ).drop(columns=["relevant"])

        df_fp = df_fp.rename(
            columns={
                "corpus_id": "fp_id",
                "score": "fp_score",
                "rank": "fp_rank",
            }
        ).drop(columns=["relevant"])

        df = df_fp.merge(df_tp, on="query_id", how="left")
        self.df_fp = df

        # Save
        os.makedirs(
            f"new_results/fp_tp/{self.dataset_name.replace('/', '_')}",
            exist_ok=True,
        )
        df.to_csv(
            f"new_results/fp_tp/{self.dataset_name.replace('/', '_')}/{self.model_name.replace('/', '_')}_{self.experiment_type}.csv",
            index=False,
        )
        return df

    def compute_metrics(self) -> dict:
        df = pd.DataFrame(self.ranked_results)

        df_relevant = df[(df["rank"] == 1) & (df["relevant"] == 1)]
        hr = len(df_relevant) / len(self.queries_ids)

        df_tp = pd.DataFrame(self.first_true_positives)
        mr1 = float(df_tp["rank"].mean())

        # HR@10
        df_relevant_10 = df[(df["rank"] <= 10) & (df["relevant"] == 1)]
        hr10 = df_relevant_10["query_id"].nunique() / len(self.queries_ids)

        # HR@100
        df_relevant_100 = df[(df["rank"] <= 100) & (df["relevant"] == 1)]
        hr100 = df_relevant_100["query_id"].nunique() / len(self.queries_ids)

        # MAP@10
        ap_list = []

        for query_id in self.queries_ids:
            top_hits = (
                df[df["query_id"] == query_id]
                .sort_values(by="rank", ascending=True)
                .head(10)
            )

            query_relevant_docs = self.relevant_docs[query_id]

            num_correct = 0
            sum_precisions = 0.0

            for idx, hit in enumerate(top_hits.itertuples(), start=1):  # 1-based rank
                if hit.corpus_id in query_relevant_docs:
                    num_correct += 1
                    sum_precisions += num_correct / idx  # precision@k

            if num_correct > 0:
                ap = sum_precisions / min(10, len(query_relevant_docs))
            else:
                ap = 0.0

            ap_list.append(ap)

        map10 = np.mean(ap_list)

        # TP/FP metrics
        df = self.df_fp
        df["rank_diff"] = np.abs(df["fp_rank"] - df["tp_rank"])
        df["score_diff"] = np.abs(df["fp_score"] - df["tp_score"])

        mean_rank_diff = np.nanmean(df["rank_diff"])
        mean_score_diff = np.nanmean(df["score_diff"])

        metrics = {
            "mr1": mr1,
            "hr": hr,
            "hr10": hr10,
            "hr100": hr100,
            "map10": map10,
            "nb_tp": len(df_relevant),
            "nb_fp": len(self.queries_ids) - len(df_relevant),
            "Mean Rank Diff": mean_rank_diff,
            "Mean Score Diff": mean_score_diff,
        }

        self.metrics = metrics
        return metrics

    def get_similarity_matrix_chunked(self):
        def compute_cosine_similarity_matrix(a, b):
            a = F.normalize(a, p=2, dim=1)
            b = F.normalize(b, p=2, dim=1)
            return torch.matmul(a, b.T)

        def r_mean_max(sim_m):
            max_sim = torch.max(sim_m, dim=1).values
            return torch.mean(max_sim)

        sim_matrix = []

        for query_id in tqdm(self.queries_ids, desc="Ranking"):
            query_chunks = torch.Tensor(self.chunked_embeddings[query_id])
            scores = []
            for corpus_id in self.corpus_ids:
                corpus_chunks = torch.Tensor(self.chunked_embeddings[corpus_id])
                similarity_matrix = compute_cosine_similarity_matrix(
                    query_chunks, corpus_chunks
                )
                score = r_mean_max(similarity_matrix)  # now correct
                scores.append(score.item())
            sim_matrix.append(scores)

        sim_matrix = np.array(sim_matrix)

        # Mask self-matches
        for i, qid in enumerate(self.queries_ids):
            for j, cid in enumerate(self.corpus_ids):
                if qid == cid:
                    sim_matrix[i, j] = -np.inf

        self.sim_matrix = sim_matrix
        return sim_matrix




# ---- IO helpers -----------------------------------------------------


def load_kv_pickle_or_npz(path: Path) -> Dict[str, Any]:
    """Load dict-like embeddings saved as .pkl or .npz (npz: key->array)."""
    if str(path).endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    elif str(path).endswith(".npz"):
        npz = np.load(path, allow_pickle=True)
        return {k: npz[k] for k in npz.files}
    else:
        raise ValueError(f"Unsupported embedding file: {path}")


def load_relevants(path: Path) -> Dict[str, List[str]]:
    """
    Expect CSV with columns: query_id, corpus_id (one row per relevant pair).
    Builds mapping query_id -> list of relevant corpus_ids.
    """
    df = pd.read_csv(path)
    if not {"query_id", "corpus_id"}.issubset(df.columns):
        raise ValueError("Relevants CSV must have columns: query_id, corpus_id")
    rel: Dict[str, List[str]] = {}
    for qid, grp in df.groupby("query_id"):
        rel[qid] = grp["corpus_id"].astype(str).tolist()
    return rel


def load_ids(path: Path, col: str) -> List[str]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"IDs CSV must have column: {col}")
    return df[col].astype(str).tolist()
