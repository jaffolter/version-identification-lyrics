import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Reranker:
    def __init__(
        self,
        queries_ids,
        corpus_ids,
        relevant_docs,
        ranked_docs,
        model=None,
        model_name=None,
        reranker_name=None,
        dataset_name=None,
        metadata=None,
        experiment_type=None,
    ):
        self.queries_ids = queries_ids
        self.corpus_ids = corpus_ids
        self.relevant_docs = relevant_docs
        self.ranked_docs = ranked_docs  # df
        self.ranked_docs_ids = None

        # Embedding model
        self.model = model
        # Metadata
        self.id_to_lyrics = dict(zip(metadata["version_id"], metadata["lyrics"]))
        # Results
        self.reranked_results = None
        self.first_true_positives = None
        # Metrics
        self.metrics = None
        # Dataframes
        self.df_fp = None
        # Model and dataset names
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.experiment_type = experiment_type
        # Reranker
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            reranker_name, trust_remote_code=True
        )
        self.max_length = 1024
        prompt = You are given the lyrics of a song (query) and the lyrics of another song (passage). 
        If the passage shows strong lyrical overlap, clear semantic similarity, or paraphrased or translated patterns indicating it is a cover of the query, return 'Yes'. 
        Do not answer 'Yes' just because of similar themes or mood without lyrical evidence. Return only 'Yes' or 'No'.
        
        sep = "\n"
        self.prompt_inputs = self.tokenizer(
            prompt, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        self.sep_inputs = self.tokenizer(
            sep, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        self.yes_loc = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        """

    def filter_results_to_rerank(self):
        """
        Filter the results to only include the top-k results for each query.
        """
        df = self.ranked_docs

        grouped = df.groupby("query_id").agg({"relevant": list})
        grouped["to_remove"] = grouped["relevant"].apply(
            lambda x: all(r is False for r in x)
        )
        to_remove = grouped[grouped["to_remove"]].index

        df_filtered = df[~df["query_id"].isin(to_remove)]
        print(len(df_filtered), "results after removing queries with no relevant docs")
        df_filtered = df_filtered[df_filtered["corpus_id"].isin(self.corpus_ids)]
        print(f"Filtered results: {len(df_filtered)}")

        self.ranked_docs = df_filtered

        grouped = df_filtered.groupby("query_id").agg(
            {
                "corpus_id": list,
            }
        )

        self.ranked_docs_ids = dict(zip(grouped.index, grouped["corpus_id"]))

    def rerank_results_bge(self):
        reranked_results = []
        for idx, query_id in tqdm(
            enumerate(self.queries_ids), total=len(self.queries_ids), desc=f"Reranking"
        ):
            torch.cuda.empty_cache()
            if str(query_id) not in self.ranked_docs_ids:
                continue

            query_relevant_docs = self.relevant_docs[query_id]
            query_ranked_docs = self.ranked_docs_ids[query_id]

            scores_query = []
            lyrics_query = self.id_to_lyrics[query_id]

            #  Create data to rerank
            for corpus_id in query_ranked_docs:
                sentence_pairs = [lyrics_query, self.id_to_lyrics[corpus_id]]

                score = self.model.compute_score(sentence_pairs, normalize=True)
                scores_query.append(score)

            # Step 2: Zip corpus IDs with their scores
            results_with_scores = list(
                zip(
                    query_ranked_docs,
                    scores_query,
                )
            )

            # Step 3: Sort by score (descending)
            sorted_results = sorted(
                results_with_scores, key=lambda x: x[1], reverse=True
            )

            # Step 4: Build final ranked output
            for rank, (corpus_id, score) in enumerate(sorted_results, start=1):
                relevant = corpus_id in query_relevant_docs

                reranked_results.append(
                    {
                        "query_id": query_id,
                        "corpus_id": corpus_id,
                        "rank": rank,
                        "relevant": relevant,
                        "score": score,
                    }
                )
        self.reranked_results = reranked_results

        # Save the reranked results to a CSV file
        df_reranked = pd.DataFrame(reranked_results)

        return reranked_results

    def tokenize_pairs(self, pairs):
        inputs = []
        for query, passage in pairs:
            query_inputs = self.tokenizer(
                f"A: {query}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=self.max_length * 3 // 4,
                truncation=True,
            )
            passage_inputs = self.tokenizer(
                f"B: {passage}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
            )
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs["input_ids"],
                self.sep_inputs + passage_inputs["input_ids"],
                truncation="only_second",
                max_length=self.max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item["input_ids"] = item["input_ids"] + self.sep_inputs + self.prompt_inputs
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)
        return self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=self.max_length + len(self.sep_inputs) + len(self.prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    def rerank_results_bge_torch(self):
        reranked_results = []
        for idx, query_id in tqdm(
            enumerate(self.queries_ids),
            total=len(self.queries_ids),
            desc=f"Reranking",
        ):
            if query_id not in self.ranked_docs_ids:
                continue

            query_relevant_docs = self.relevant_docs[query_id]
            query_ranked_docs = self.ranked_docs_ids[query_id]

            res_query = []
            lyrics_query = self.id_to_lyrics[query_id]

            for corpus_id in query_ranked_docs:
                #  Create data to rerank
                sentence_pairs = [[lyrics_query, self.id_to_lyrics[corpus_id]]]

                with torch.no_grad():
                    # Tokenize the batch
                    inputs = self.tokenize_pairs(sentence_pairs)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    # Compute scores
                    outputs = self.model(**inputs, return_dict=True)
                    scores = outputs.logits[:, -1, self.yes_loc].view(-1).float()

                    res_query.append(
                        scores.cpu()
                    )  # store on CPU to avoid filling GPU memory

            # Step 2: Zip corpus IDs with their scores
            results_with_scores = list(
                zip(
                    query_ranked_docs,
                    res_query,
                )
            )
            # Step 3: Sort by score (descending)
            sorted_results = sorted(
                results_with_scores, key=lambda x: x[1], reverse=True
            )
            # Step 4: Build final ranked output
            for rank, (corpus_id, score) in enumerate(sorted_results, start=1):
                relevant = corpus_id in query_relevant_docs

                reranked_results.append(
                    {
                        "query_id": query_id,
                        "corpus_id": corpus_id,
                        "rank": rank,
                        "relevant": relevant,
                        "score": score,
                    }
                )
        self.reranked_results = reranked_results

        # Save the reranked results to a CSV file
        df_reranked = pd.DataFrame(reranked_results)

        return reranked_results

    def rerank_results(self):
        reranked_results = []
        for idx, query_id in tqdm(
            enumerate(self.queries_ids), total=len(self.queries_ids), desc=f"Reranking"
        ):
            if query_id not in self.ranked_docs_ids:
                continue

            query_relevant_docs = self.relevant_docs[query_id]
            query_ranked_docs = self.ranked_docs_ids[query_id]

            #  Create data to rerank
            sentence_pairs = [
                [self.id_to_lyrics[query_id], self.id_to_lyrics[corpus_id]]
                for corpus_id in query_ranked_docs
            ]

            # Step 1: Compute scores from reranker (normalized if needed)
            # scores = self.reranker.compute_score(
            # self.ranked_results_lyrics_dict[query_id], normalize=True
            # )
            scores = self.model.predict(sentence_pairs, convert_to_tensor=True).tolist()

            # Step 2: Zip corpus IDs with their scores
            results_with_scores = list(
                zip(
                    query_ranked_docs,
                    scores,
                )
            )

            # Step 3: Sort by score (descending)
            sorted_results = sorted(
                results_with_scores, key=lambda x: x[1], reverse=True
            )

            # Step 4: Build final ranked output
            for rank, (corpus_id, score) in enumerate(sorted_results, start=1):
                relevant = corpus_id in query_relevant_docs

                reranked_results.append(
                    {
                        "query_id": query_id,
                        "corpus_id": corpus_id,
                        "rank": rank,
                        "relevant": relevant,
                        "score": score,
                    }
                )

        self.reranked_results = reranked_results

        # Save the reranked results to a CSV file
        df_reranked = pd.DataFrame(reranked_results)

        return reranked_results

    def get_first_true_positives_reranked(self):
        """
        Get the first true positives for each query in the reranked results.
        """
        first_true_positives = []

        # Retrieve the reranked results as a DataFrame
        df_reranked = pd.DataFrame(self.reranked_results)

        for query_id in self.queries_ids:
            df_relevant = df_reranked[
                (df_reranked["query_id"] == query_id) & (df_reranked["relevant"] == 1)
            ]

            if not df_relevant.empty:
                row = df_relevant.iloc[0]
                first_true_positives.append(
                    {
                        "rank": row["rank"],
                        "query_id": query_id,
                        "relevant": True,
                        "corpus_id": row["corpus_id"],
                        "score": row["score"],
                    }
                )

        self.first_true_positives = first_true_positives

    def get_fp(self) -> dict:
        df_ranked = pd.DataFrame(self.reranked_results)
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

        return df

    def compute_metrics(self) -> dict:
        df = pd.DataFrame(self.reranked_results)

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
        # df["rank_diff"] = np.abs(df["fp_rank"] - df["tp_rank"])
        # df["score_diff"] = np.abs(df["fp_score"] - df["tp_score"])

        # mean_rank_diff = np.nanmean(df["rank_diff"])
        # mean_score_diff = np.nanmean(df["score_diff"])

        metrics = {
            "mr1": mr1,
            "hr": hr,
            "hr10": hr10,
            "hr100": hr100,
            "map10": map10,
            "nb_tp": len(df_relevant),
            "nb_fp": len(self.queries_ids) - len(df_relevant),
            # "Mean Rank Diff": mean_rank_diff,
            # "Mean Score Diff": mean_score_diff,
        }

        self.metrics = metrics
        return metrics
