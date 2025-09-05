import pandas as pd
import pickle
import torch
import numpy as np
from typing import Tuple, Dict, List


def get_input_data_evaluator(df: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    Get the input data for the evaluator:
    - queries_ids: the ids of the queries
    - corpus_ids: the ids of the corpus
    - relevant_docs_dict: a dictionary with the relevant documents for each query
    The relevant documents are the documents that have the same clique_id as the query

    Args:
        df (pd.DataFrame): the dataset to process
    Returns:
        queries_ids (list): the ids of the queries
        corpus_ids (list): the ids of the corpus
        relevant_docs_dict (dict): a dictionary with the relevant documents for each query
    """
    # Get the queries and corpus ids
    queries_ids = df["version_id"]
    corpus_ids = list(df["version_id"])

    # Mapping from version_id to clique_id
    version_to_clique = dict(zip(df["version_id"], df["clique_id"]))

    # Filter out the queries that have no relevant documents
    # and create a dictionary with the relevant documents for each query
    # (versions with the same clique_id)
    relevant_docs_dict = {}
    filtered_queries = []

    for query_idx in queries_ids:
        clique_idx = version_to_clique[query_idx]

        relevant_docs = df[df["clique_id"] == clique_idx]["version_id"].tolist()

        # Delete the query from the relevant documents
        relevant_docs.remove(query_idx)

        # If there are no relevant documents, remove the query from the queries_ids
        if len(relevant_docs) == 0:
            continue

        filtered_queries.append(query_idx)
        relevant_docs_dict[query_idx] = relevant_docs

    return filtered_queries, corpus_ids, relevant_docs_dict


def convert_to_dict(embeddings: List, version_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    Convert the embeddings to a dictionary with version_id as the key.

    Args:
        embeddings (list): A list of tuples containing version_id and embedding.
        version_ids (list): A list of version_ids corresponding to the embeddings.

    Returns:
        dict: A dictionary with version_id as the key and embedding as the value.
    """
    return {version_id: embedding for version_id, embedding in zip(version_ids, embeddings)}
