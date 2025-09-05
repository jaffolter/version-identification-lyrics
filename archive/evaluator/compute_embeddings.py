import pickle
import torch
from tqdm import tqdm
import pandas as pd

from src.evaluator.embeddings import (
    compute_embeddings_pre_trained,
    compute_embeddings_sparse,
    compute_index_bm25,
    compute_embeddings_bgem3,
    compute_chunked_embeddings_pre_trained,
    compute_chunked_embeddings_sparse,
)
import os

from huggingface_hub import login
import dotenv

dotenv.load_dotenv()


dataset_name = os.environ.get("DATASET", "shs100kok")
transcribed = os.environ.get("TRANSCRIBED", "True").lower() == "true"
type_transcription = os.environ.get("TYPE_TRANSCRIPTION", "transcription_vocal_chunked").lower()

ht_token = os.getenv("HUGGINGFACE_TOKEN", "")
print(f"Using dataset: {dataset_name}")
print(f"Transcribed: {transcribed}")
print(f"Type of transcription: {type_transcription}")


# -------------------------------------------------------
# Load datasets
# -------------------------------------------------------
path_df = (
    f"src/data/benchmarks/{dataset_name}_final.csv"
)
df = pd.read_csv(path_df)
ids = df["version_id"].tolist()

if not transcribed:
    text = df["lyrics"].tolist()
elif type_transcription == "transcription":
    text = df["transcription"].tolist()
elif type_transcription == "transcription_vocal_chunked":
    df["transcription_vocal"] = df["transcription_vocal"].apply(lambda x : x.split("\n\n"))
    text = df["transcription_vocal"].tolist()
elif type_transcription == "transcription_vocal":
    text = df["transcription_vocal"].tolist()
else:
    text = df["transcription_vocal_english"].tolist()

# -------------------------------------------------------
# Compute dense embeddings
# -------------------------------------------------------

model_names = [
    "Alibaba-NLP/gte-multilingual-base",
    #"intfloat/multilingual-e5-small",
    #"intfloat/multilingual-e5-base",
    #"intfloat/multilingual-e5-large",
    #"intfloat/multilingual-e5-large-instruct",
    #"sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    #"jinaai/jina-embeddings-v3",
]

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_detailed_instruct(query: str) -> str:
    return f"Instruct: Retrieve semantically similar text.\nQuery: {query}"
"""

for model_name in tqdm(
    model_names, total=len(model_names), desc="Computing embeddings"
):
    torch.cuda.empty_cache()
    print(f"Computing embeddings for model {model_name}")
    compute_embeddings_pre_trained(
        model_name=model_name,
        dataset=dataset_name,
        ids=ids,
        lyrics=text,
        device=device,
        is_query=False,
        transcribed=f"_{type_transcription}" if transcribed else None,
    )

    if "instruct" in model_name:
        compute_embeddings_pre_trained(
            model_name=model_name,
            dataset=dataset_name,
            ids=ids,
            lyrics=[get_detailed_instruct(x) for x in text],
            device=device,
            is_query=True,
            transcribed=f"_{type_transcription}" if transcribed else None,
        )

if transcribed:
    print("Transcribed dataset detected, skipping sparse embeddings and BM25 index.")
    exit(0)



# -------------------------------------------------------
# Compute sparse embeddings
# -------------------------------------------------------
compute_embeddings_sparse(
    dataset=dataset_name,
    ids=ids,
    lyrics=text,
)

# -------------------------------------------------------
# Compute BM25 embeddings
# -------------------------------------------------------
compute_index_bm25(
    dataset=dataset_name,
    ids=ids,
    lyrics=text,
)


# -------------------------------------------------------
# Compute BGE-M3 embeddings
# -------------------------------------------------------
compute_embeddings_bgem3(
    dataset=dataset_name,
    ids=ids,
    lyrics=text,
)
 

 """
# -------------------------------------------------------
# Compute chunked embeddings
# -------------------------------------------------------

for model_name in tqdm(
    model_names, total=len(model_names), desc="Computing chunked embeddings"
):
    torch.cuda.empty_cache()
    print(f"Computing chunked embeddings for model {model_name}")
    compute_chunked_embeddings_pre_trained(
        model_name=model_name, dataset=dataset_name, ids=ids, lyrics=text, device=device
    )

    if "instruct" in model_name:
        compute_chunked_embeddings_pre_trained(
            model_name=model_name,
            dataset=dataset_name,
            ids=ids,
            lyrics=[get_detailed_instruct(x) for x in text],
            device=device,
            is_query=True,
        )
"""
# -------------------------------------------------------
# Compute sparse embeddings
# -------------------------------------------------------
compute_chunked_embeddings_sparse(
    dataset=dataset_name,
    ids=ids,
    lyrics=text,
)

"""