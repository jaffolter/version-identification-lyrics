import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import evaluate
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import os

nltk.download("punkt_tab")

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


class LyricsEvaluator:
    def __init__(
        self,
        df_fp,
        metadata,
        model_name=None,
        dataset_name=None,
    ):
        self.metadata = metadata
        self.id_to_lyrics = dict(zip(metadata["version_id"], metadata["lyrics"]))

        self.metrics_fp = None

        self.df_fp = df_fp

        self.model_name = model_name
        self.dataset_name = dataset_name

    def get_lyrics(self) -> pd.DataFrame:
        """
        Get the lyrics dataframe for the first true positives.
        """

        df = (
            self.df_fp.merge(
                self.metadata[["version_id", "lyrics", "deezer_id"]].rename(
                    columns={"lyrics": "fp_lyrics", "deezer_id": "fp_deezer_id"}
                ),
                left_on="fp_id",
                right_on="version_id",
                how="left",
            )
            .drop(columns=["version_id"])
            .merge(
                self.metadata[["version_id", "lyrics", "deezer_id"]].rename(
                    columns={"lyrics": "tp_lyrics", "deezer_id": "tp_deezer_id"}
                ),
                left_on="tp_id",
                right_on="version_id",
                how="left",
            )
            .drop(columns=["version_id"])
            .merge(
                self.metadata[["version_id", "lyrics", "deezer_id"]].rename(
                    columns={"lyrics": "q_lyrics", "deezer_id": "q_deezer_id"}
                ),
                left_on="query_id",
                right_on="version_id",
                how="left",
            )
            .drop(columns=["version_id"])
        )
        df = df.rename(
            columns={
                "fp_id": "FP ID",
                "tp_id": "FN ID",
                "fp_score": "FP Score",
                "tp_score": "FN Score",
                "fp_rank": "FP Rank",
                "tp_rank": "FN Rank",
                "query_id": "Query ID",
                "fp_lyrics": "FP Lyrics",
                "tp_lyrics": "FN Lyrics",
                "q_lyrics": "Query Lyrics",
                "fp_deezer_id": "FP Deezer ID",
                "tp_deezer_id": "FN Deezer ID",
                "q_deezer_id": "Query Deezer ID",
            }
        )

        df = df[
            [
                "Query ID",
                "FP ID",
                "FN ID",
                "FP Score",
                "FN Score",
                "FN Rank",
                "Query Deezer ID",
                "FP Deezer ID",
                "FN Deezer ID",
                "Query Lyrics",
                "FP Lyrics",
                "FN Lyrics",
            ]
        ]
        self.df_fp_lyrics = df
        os.makedirs(f"new_results/fp_tp_lyrics/{self.dataset_name}", exist_ok=True)
        df.to_csv(
            f"new_results/fp_tp_lyrics/{self.dataset_name}/lyrics.csv", index=False
        )

        return df

    def compute_metrics_lyrics(self, vectorizer) -> dict:
        """
        Compute the metrics for the lyrics.
        """

        df = self.df_fp_lyrics
        to_add_metrics = []

        for _, row in df.iterrows():
            # Compute the similarity between the lyrics
            fp_lyrics = row["FP Lyrics"]
            tp_lyrics = row["FN Lyrics"]
            q_lyrics = row["Query Lyrics"]

            fp_tokens = word_tokenize(fp_lyrics)
            tp_tokens = word_tokenize(tp_lyrics)
            q_tokens = word_tokenize(q_lyrics)

            set_fp = set(fp_tokens)
            set_tp = set(tp_tokens)
            set_q = set(q_tokens)

            # ---- BLEU ----
            bleu_fp = bleu.compute(predictions=[fp_lyrics], references=[[q_lyrics]])[
                "bleu"
            ]
            bleu_tp = bleu.compute(predictions=[tp_lyrics], references=[[q_lyrics]])[
                "bleu"
            ]

            # ---- ROUGE ----
            rouge_fp = float(
                rouge.compute(
                    predictions=[fp_lyrics], references=[q_lyrics], use_stemmer=True
                )["rougeL"]
            )
            rouge_tp = float(
                rouge.compute(
                    predictions=[tp_lyrics], references=[q_lyrics], use_stemmer=True
                )["rougeL"]
            )

            # ---- Jaccard ----
            jac_fp = (
                len(set_fp & set_q) / len(set_fp | set_q) if set_fp | set_q else 0.0
            )
            jac_tp = (
                len(set_tp & set_q) / len(set_tp | set_q) if set_tp | set_q else 0.0
            )

            # ---- TF/IDF ----
            q_vec = vectorizer.transform([q_lyrics])
            fp_vec = vectorizer.transform([fp_lyrics])
            tp_vec = vectorizer.transform([tp_lyrics])
            cos_fp = cosine_similarity(q_vec, fp_vec)[0, 0]
            cos_tp = cosine_similarity(q_vec, tp_vec)[0, 0]

            to_add_metrics.append(
                {
                    "bleu_fp": bleu_fp,
                    "bleu_fn": bleu_tp,
                    "delta_bleu": bleu_fp - bleu_tp,
                    "rouge_fp": rouge_fp,
                    "rouge_fn": rouge_tp,
                    "delta_rouge": rouge_fp - rouge_tp,
                    "jac_fp": jac_fp,
                    "jac_fn": jac_tp,
                    "delta_jac": jac_fp - jac_tp,
                    "tfidf_fp": cos_fp,
                    "tfidf_fn": cos_tp,
                    "delta_tfidf": cos_fp - cos_tp,
                }
            )

        # Add the metrics to the dataframe
        df_metrics = pd.DataFrame(to_add_metrics)
        # print(df_metrics.head())
        df = df.join(df_metrics)
        self.df_fp_lyrics = df
        return df
