from sentence_transformers import SentenceTransformer
import os
import pickle
from FlagEmbedding import BGEM3FlagModel
from sklearn.feature_extraction.text import TfidfVectorizer
import bm25s
from bm25s.hf import BM25HF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


def convert_to_dict(embeddings, version_ids):
    """
    Convert the embeddings to a dictionary with version_id as the key.

    Args:
        embeddings (list): A list of tuples containing version_id and embedding.
        version_ids (list): A list of version_ids corresponding to the embeddings.

    Returns:
        dict: A dictionary with version_id as the key and embedding as the value.
    """
    return {
        version_id: embedding for version_id, embedding in zip(version_ids, embeddings)
    }


def compute_embeddings_pre_trained(
    model_name, dataset, ids, lyrics, device, is_query=False, transcribed=None
):
    """
    Compute the embeddings for the given lyrics using a pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model.
        dataset (str): The name of the dataset.
        ids (list): A list of ids corresponding to the lyrics.
        lyrics (list): A list of lyrics to compute embeddings for.
        device (str): The device to use for computation
        is_query (bool): Whether the embeddings are for queries or not, and filenames
            will be adjusted accordingly.

    Returns:
        embeddings (list): A list of computed embeddings.
    """
    dataset = dataset + "_big" if dataset == "discogs_vi" else dataset
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Additional logic for specific models
    kwargs = {}
    if "jina" in model_name:
        # Add prompt for Jina models
        kwargs = {
            "task": "retrieval.query",
            "prompt": "retrieval.query",
        }

    # Create embeddings
    embeddings = model.encode(
        lyrics, normalize_embeddings=True, batch_size=8, device=device, **kwargs
    )

    # Create a dict with key = version_id
    embeddings_dict = convert_to_dict(embeddings, ids)

    save_dir = os.path.join("src/evaluator/embeddings", model_name.replace("/", "_"))
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{dataset}_query_embeddings" if is_query else f"{dataset}_embeddings"

    filename += f"{transcribed}.pkl" if transcribed else ".pkl"

    # Save the embeddings to a file
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(embeddings_dict, f)

    return embeddings


def compute_embeddings_sparse(dataset, ids, lyrics):
    """
    Compute the embeddings for the given lyrics using a sparse model.

    Args:
        dataset (str): The name of the dataset.
        ids (list): A list of ids corresponding to the lyrics.
        lyrics (list): A list of lyrics to compute embeddings for.

    Returns:
        embeddings (list): A list of computed embeddings.
    """
    model_name = "BAAI/bge-m3"
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    # Create embeddings
    embeddings = model.encode_queries(
        lyrics, return_dense=False, return_sparse=True, return_colbert_vecs=False
    )
    embeddings = embeddings["lexical_weights"]
    embeddings_dict = convert_to_dict(embeddings, ids)

    save_dir = os.path.join("src/evaluator/embeddings", model_name.replace("/", "_"))
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{dataset}_embeddings.pkl"

    # Save the embeddings to a file
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(embeddings_dict, f)


def compute_embeddings_bgem3(dataset, ids, lyrics):
    """
    Compute the embeddings for the given lyrics using a sparse model.

    Args:
        dataset (str): The name of the dataset.
        ids (list): A list of ids corresponding to the lyrics.
        lyrics (list): A list of lyrics to compute embeddings for.

    Returns:
        embeddings (list): A list of computed embeddings.
    """
    model_name = "BAAI/bge-m3-full"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
        devices=[device],
    )

    # Create embeddings
    embeddings = model.encode_queries(
        lyrics, return_dense=True, return_sparse=True, return_colbert_vecs=True
    )

    embeddings_dict = {}
    for idx, value in enumerate(ids):
        embeddings_dict[value] = {
            "dense": embeddings["dense_vecs"][idx],
            "sparse": embeddings["lexical_weights"][idx],
            "colbert": embeddings["colbert_vecs"][idx],
        }

    save_dir = os.path.join("src/evaluator/embeddings", model_name.replace("/", "_"))
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{dataset}_embeddings.pkl"

    # Save the embeddings to a file
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(embeddings_dict, f)


def compute_embeddings_tdidf(dataset, ids, lyrics):
    """
    Compute the embeddings for the given lyrics using a TF-IDF model.

    Args:
        dataset (str): The name of the dataset.
        ids (list): A list of ids corresponding to the lyrics.
        lyrics (list): A list of lyrics to compute embeddings for.

    Returns:
        embeddings (list): A list of computed embeddings.
    """

    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(lyrics).toarray()

    # Create a dict with key = version_id
    embeddings_dict = convert_to_dict(embeddings, ids)

    save_dir = os.path.join("src/evaluator/embeddings", "tfidf")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{dataset}_embeddings.pkl"

    # Save the embeddings to a file
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(embeddings_dict, f)

    return embeddings


def compute_index_bm25(dataset, ids, lyrics):
    """
    Compute the BM25 index for the given lyrics.

    Args:
        dataset (str): The name of the dataset.
        ids (list): A list of ids corresponding to the lyrics.
        lyrics (list): A list of lyrics to compute embeddings for.

    Returns:
        index (BM25): The BM25 index.
    """
    retriever = BM25HF(corpus=lyrics)
    retriever.index(bm25s.tokenize(lyrics))

    user = "joanne-affolter"
    retriever.save_to_hub(f"{user}/{dataset}_bm25")


# -------------------------------------------------------------
# Chunked embeddings
# -------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=True,
)


def chunk_lyrics(lyrics):
    docs = text_splitter.create_documents([lyrics])
    unique_chunks = list({doc.page_content for doc in docs})
    return unique_chunks


def encode_chunks(model, chunks, device):
    with torch.no_grad():
        return model.encode(chunks, normalize_embeddings=True, device=device)


def compute_chunked_embeddings_pre_trained(
    model_name, dataset, ids, lyrics, device, is_query=False, is_transcribed=False
):
    """
    Compute and store chunked embeddings row-by-row.
    """
    
    dataset = dataset + "_big" if dataset == "discogs_vi" else dataset
    assert len(ids) == len(lyrics), "IDs and lyrics must match in length."

    # Load the model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model = model.to(device)

    save_dir = f"src/evaluator/embeddings/{model_name.replace('/', '_')}"
    os.makedirs(save_dir, exist_ok=True)

    all_embeddings = {}

    for idx, (id_, lyric) in tqdm(
        enumerate(zip(ids, lyrics)), total=len(ids), desc="Processing lyrics"
    ):
        try:
            if type(lyric) == str:
                chunks = chunk_lyrics(lyric)
            else :
                chunks = lyric

            embeddings = encode_chunks(model, chunks, device)
            all_embeddings[id_] = embeddings

        except Exception as e:
            all_embeddings.append(None)  # Optional: pad with None

    # Save to a single .npy file as an object array
    file_prefix = (
        "query_chunked_embeddings" if is_query 
        else "chunked_embeddings" if not is_transcribed 
        else "chunked_embeddings_transcription"
    )
    path_file = os.path.join(save_dir, f"{dataset}_{file_prefix}.pkl")

    with open(path_file, "wb") as f:
        pickle.dump(all_embeddings, f)


def encode_chunks_sparse(model, chunks):
    with torch.no_grad():
        return model.encode_queries(
            chunks, return_dense=False, return_sparse=True, return_colbert_vecs=False
        )


def compute_chunked_embeddings_sparse(dataset, ids, lyrics):
    """
    Compute and store chunked embeddings row-by-row.
    """
    model_name = "BAAI/bge-m3"
    # Load the model
    model = BGEM3FlagModel(model_name, use_fp16=True)

    save_dir = os.path.join("src/evaluator/embeddings", model_name.replace("/", "_"))
    os.makedirs(save_dir, exist_ok=True)

    all_embeddings = {}

    for idx, (id_, lyric) in tqdm(
        enumerate(zip(ids, lyrics)), total=len(ids), desc="Processing lyrics"
    ):
        try:
            chunks = chunk_lyrics(lyric)
            embeddings = encode_chunks_sparse(model, chunks)
            all_embeddings[id_] = embeddings

        except Exception as e:
            all_embeddings.append(None)  # Optional: pad with None

    # Save to a single .npy file as an object array
    path_file = os.path.join(save_dir, f"{dataset}_chunked_embeddings.pkl")

    with open(path_file, "wb") as f:
        pickle.dump(all_embeddings, f)
