from livi.apps.retrieval_eval.ranker import DenseRanker
from livi.core.data.utils.io_toolbox import get_embeddings
from pathlib import Path

import pandas as pd
from loguru import logger


def get_similarity_transcription_editorial(
    path_embeddings_transcription: Path,
    path_embeddings_editorial: Path,
):
    embeddings_transcription = get_embeddings(path_embeddings_transcription)
    embeddings_editorial = get_embeddings(path_embeddings_editorial)

    ranker = DenseRanker(embeddings=embeddings_transcription, query_embeddings=embeddings_editorial)

    # Get similarity matrix
    similarity_matrix = ranker.compute_similarity_matrix()

    # Compute average cosine similarity (average over diagonal elements)
    avg_cosine_similarity = similarity_matrix.diagonal().mean()
    logger.info(f"Average cosine similarity: {avg_cosine_similarity}")
