import datasets
from datasets import Dataset, DatasetDict

import os
import pickle
import torch
from tqdm import tqdm
import pandas as pd

from src.evaluator.data_loader import get_input_data_evaluator, get_embeddings
from src.evaluator.ranker import Ranker
from src.evaluator.parser import parse_reranker, parse_sparse, parse_audio

from FlagEmbedding import BGEM3FlagModel
import bm25s
from bm25s.hf import BM25HF

from huggingface_hub import login

import os
import numpy as np




dataset_name = os.environ.get("DATASET", "shs100kok")  # covers80, shs100kok, discogs_vi_mini, discogs_vi
experiment_type = os.environ.get("EXPERIMENT_TYPE", "transcription_vocal")    #transcription_vocal, transcription_vocal_chunked
print(f"Experiment type: {experiment_type}")
print(f"Using dataset: {dataset_name}")

embedding_filename = {
    "baseline": "embeddings",
    "chunking": "chunked_embeddings",
    "transcription_vocal_chunked": "chunked_embeddings",
    "transcription": "embeddings_transcription",
    "transcription_vocal": "embeddings_transcription_vocal",
    "transcription_vocal_english": "embeddings_transcription_vocal_english",
}

# -------------------------------------------------------
# Load datasets
# -------------------------------------------------------


def load_data(dataset_name, detail=None):
    path = f"src/data/benchmarks/{dataset_name}" 
    path += f"_{detail}.csv" if detail else ".csv"
    df = pd.read_csv(path)
    print(df)
    df["version_id"] = df["version_id"].astype(str)
    ds = Dataset.from_pandas(df)
    return df, ds



for detail in ["q1"]: 
    #df_covers80, ds_covers80 = load_data("covers80")
    df_shs100k, ds_shs100k = load_data("shs100kok")
    #df_discogs_vi_mini, ds_discogs_vi_mini = load_data("discogs_vi_mini")
    #df_discogs_vi, ds_discogs_vi = load_data("discogs_vi", detail=detail)

    ds_mapping = {
        #"covers80": ds_covers80,
        "shs100kok": ds_shs100k,
        #"discogs_vi_mini": ds_discogs_vi_mini,
        #"discogs_vi": ds_discogs_vi,
    }

    df_mapping = {
        #"covers80": df_covers80,
        "shs100kok": df_shs100k,
        #"discogs_vi_mini": df_discogs_vi_mini,
        #"discogs_vi": df_discogs_vi,
    }

    input_data_evaluator_dict = {
        #"covers80": get_input_data_evaluator(ds_covers80),
        "shs100kok": get_input_data_evaluator(ds_shs100k),
        #"discogs_vi_mini": get_input_data_evaluator(ds_discogs_vi_mini),
        #"discogs_vi": get_input_data_evaluator(ds_discogs_vi),
    }

    # Filter for the selected dataset
    ds_mapping = {k: v for k, v in ds_mapping.items() if k == dataset_name}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------
    # Models to evaluate
    # -------------------------------------------------------

    model_names = [
        "Alibaba-NLP/gte-multilingual-base",
        #"sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        #"intfloat/multilingual-e5-small",
        #"intfloat/multilingual-e5-base",
        #"intfloat/multilingual-e5-large",
        #"intfloat/multilingual-e5-large-instruct",
        #"jinaai/jina-embeddings-v3",
    ]


    sparse_model_bge = None
    dense_model_names = None
    bm25_index = None

    if "sparse" in experiment_type:
        sparse_model_bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

        bm25_index = {}
        for dataset_name, dataset in ds_mapping.items():
            bm25_model = BM25HF.load_from_hub(
                f"joanne-affolter/{dataset_name}_bm25", load_corpus=True
            )
            bm25_index[dataset_name] = bm25_model

        dense_model_names = model_names.copy()

        model_names = [
            "BAAI/bge-m3",
            "BM25",
            "BAAI/bge-m3-full",  # colbert+sparse+dense
        ]


    # -------------------------------------------------------
    # Baseline
    # -------------------------------------------------------

    results = {
        "covers80": [],
        "shs100kok": [],
        "discogs_vi": [],
        "discogs_vi_mini": [],
    }
    print(ds_mapping)

    for model_name in tqdm(model_names, total=len(model_names), desc="Evaluating models"):
        print(f"\n\nEvaluating {model_name}")
        for dataset_name, dataset in ds_mapping.items():
            print(f"\n\nEvaluating {model_name} on {dataset_name}...")

            # Load the embeddings
            if "BM25" in model_name:
                embeddings = None

            elif "bge" in model_name:
                file_name = f"src/evaluator/embeddings/{model_name.replace('/', '_')}/{dataset_name}_{embedding_filename['baseline']}.pkl"
                embeddings = get_embeddings(file_name)
                print("   ", len(embeddings.keys()), "embeddings loaded")

            elif "transcription_vocal_chunked" in experiment_type:
                dataset_name_tmp = dataset_name + "_big" if dataset_name == "discogs_vi" else dataset_name
                file_name = f"src/evaluator/embeddings/{model_name.replace('/', '_')}/{dataset_name_tmp}_{embedding_filename['transcription_vocal_chunked']}.pkl"
                print(f"   Loading chunked embeddings from {file_name}...")
                embeddings = get_embeddings(file_name)
                # Take the average of the embeddings
                embeddings = {str(k): np.mean(v, axis=0) for k, v in embeddings.items()}
                print("   ", len(embeddings.keys()), "embeddings loaded")

            else:
                dataset_name_tmp = dataset_name + "_big" if dataset_name == "discogs_vi" else dataset_name
                file_name = f"src/evaluator/embeddings/{model_name.replace('/', '_')}/{dataset_name_tmp}_{embedding_filename[experiment_type]}.pkl"
                print(f"   Loading embeddings from {file_name}...")

                embeddings = get_embeddings(file_name)
                print("   ", len(embeddings.keys()), "embeddings loaded")

            if "instruct" in model_name:
                file_name = f"src/evaluator/embeddings/{model_name.replace('/', '_')}/{dataset_name}_query_{embedding_filename[experiment_type]}.pkl"

                query_embeddings = get_embeddings(file_name)

            queries_ids, corpus_ids, relevant_docs = input_data_evaluator_dict[dataset_name]
            
            # Filter queries_ids and corpus_ids based on the embeddings
            queries_ids = [q for q in queries_ids if str(q) in embeddings]
            corpus_ids = [c for c in corpus_ids if str(c) in embeddings]
            relevant_docs = {
                q: [doc for doc in docs if str(doc) in embeddings]
                for q, docs in relevant_docs.items()
            }
            
            print(f"   {len(queries_ids)} queries, {len(corpus_ids)} corpus documents, {len(relevant_docs)} relevant documents")

            ranker = Ranker(
                queries_ids=queries_ids,
                corpus_ids=corpus_ids,
                relevant_docs=relevant_docs,
                metadata=df_mapping[dataset_name],
                embeddings=embeddings if "chunking" not in experiment_type or experiment_type=="transcription_vocal_chunked" else None,
                sparse_embeddings=embeddings if "sparse" in experiment_type else None,
                chunked_embeddings=embeddings if "chunking" in experiment_type else None,
                query_embeddings=query_embeddings if "instruct" in model_name else None,
                model=sparse_model_bge if "sparse" in experiment_type else None,
                bm25_model=bm25_index[dataset_name]
                if "sparse" in experiment_type
                else None,
                k=100,
                model_name=model_name,
                dataset_name=dataset_name,
                experiment_type=experiment_type,
            )
            print(ranker)

            if "chunking" in experiment_type and "chunking_t" not in experiment_type:
                ranker.get_similarity_matrix_chunked()

            elif "bge" in model_name and "full" not in model_name:
                ranker.get_sparse_similarity_matrix()

            elif "bge" in model_name and "full" in model_name:
                ranker.get_bge_similarity_matrix()

            elif "BM25" in model_name:
                ranker.get_bm25_similarity_matrix()

            else:
                ranker.get_similarity_matrix()

            ranker.rank_results()
            ranker.get_first_true_positives()
            ranker.get_fp()

            ranker.compute_metrics()

            results[dataset_name].append(
                {
                    "model_name": model_name,
                    "MR1": ranker.metrics.get("mr1"),
                    "HR1": ranker.metrics.get("hr"),
                    "HR10": ranker.metrics.get("hr10"),
                    "HR100": ranker.metrics.get("hr100"),
                    "MAP10": ranker.metrics.get("map10"),
                    # "# TP": ranker.metrics.get("nb_tp"),
                    # "# FP": ranker.metrics.get("nb_fp"),
                    "Mean Rank Diff": ranker.metrics.get("Mean Rank Diff"),
                    "Mean Score Diff": ranker.metrics.get("Mean Score Diff"),
                }
            )
            
            
            print(results[dataset_name])


    # -------------------------------------------------------
    # Sparse fusion models
    # -------------------------------------------------------
    if "sparse" in experiment_type:
        model_names.remove("BAAI/bge-m3-full")
        for dense_model in tqdm(dense_model_names, total=len(dense_model_names)):
            for sparse_model in tqdm(
                model_names, total=len(model_names), desc="Evaluating fusion"
            ):
                for dataset_name, dataset in ds_mapping.items():
                    for fuse_method in ["mean", "max"]:
                        print(
                            f"\nEvaluating {dense_model} + {sparse_model} on {dataset_name} with {fuse_method} fusion"
                        )

                        # Load the dense embeddings
                        file_name = f"new_data/{dense_model.replace('/', '_')}/{dataset_name}_{embedding_filename['baseline']}.pkl"
                        embeddings = get_embeddings(file_name)

                        if "instruct" in dense_model:
                            file_name = f"new_data/{dense_model.replace('/', '_')}/{dataset_name}_query_{embedding_filename['baseline']}.pkl"
                            query_embeddings = get_embeddings(file_name)
                        # Load the sparse embeddings
                        if "bge" in sparse_model:
                            file_name = f"new_data/{sparse_model.replace('/', '_')}/{dataset_name}_{embedding_filename['baseline']}.pkl"
                            sparse_embeddings = get_embeddings(file_name)
                        else:
                            sparse_embeddings = None

                        queries_ids, corpus_ids, relevant_docs = input_data_evaluator_dict[
                            dataset_name
                        ]

                        ranker = Ranker(
                            queries_ids=queries_ids,
                            corpus_ids=corpus_ids,
                            relevant_docs=relevant_docs,
                            metadata=df_mapping[dataset_name],
                            embeddings=embeddings,
                            sparse_embeddings=sparse_embeddings,
                            query_embeddings=query_embeddings
                            if "instruct" in dense_model
                            else None,
                            model=sparse_model_bge if "bge" in sparse_model else None,
                            bm25_model=bm25_index[dataset_name]
                            if "BM25" in sparse_model
                            else None,
                            k=100,
                            model_name=f"{dense_model}+{sparse_model}+{fuse_method}",
                            dataset_name=dataset_name,
                            experiment_type=experiment_type,
                        )

                        ranker.get_similarity_matrix_fused(fuse_method=fuse_method)
                        ranker.rank_results()

                        ranker.get_first_true_positives()
                        ranker.get_fp()

                        ranker.compute_metrics()

                        results[dataset_name].append(
                            {
                                "model_name": f"{dense_model}+{sparse_model}+{fuse_method}",
                                "MR1": ranker.metrics.get("mr1"),
                                "HR1": ranker.metrics.get("hr"),
                                "HR10": ranker.metrics.get("hr10"),
                                "HR100": ranker.metrics.get("hr100"),
                                "MAP10": ranker.metrics.get("map10"),
                                "# TP": ranker.metrics.get("nb_tp"),
                                "# FP": ranker.metrics.get("nb_fp"),
                                "Mean Rank Diff": ranker.metrics.get("Mean Rank Diff"),
                                "Mean Score Diff": ranker.metrics.get("Mean Score Diff"),
                            }
                        )


    # -------------------------------------------------------
    # Write results to CSV
    # -------------------------------------------------------
    for dataset_name, value in results.items():
        if len(value) == 0:
            continue
        df = pd.DataFrame(value).sort_values(by="HR1", ascending=False)

        os.makedirs(f"src/evaluator/results/benchmarking/{dataset_name}", exist_ok=True)
        df.to_csv(
            f"src/evaluator/results/benchmarking/{dataset_name}/{experiment_type}_{detail}.csv",
            index=False,
        )

        print(f"\nDataset: {dataset_name}")
        print(df)
        print("_" * 100)
    
            
        if experiment_type != "baseline":
            # init_df = pd.read_csv(f"src/evaluator/results/benchmarking/{dataset_name}/transcription_vocal.csv")
            init_df = pd.read_csv(f"src/evaluator/results/benchmarking/{dataset_name}/baseline.csv")

            if experiment_type == "sparse":
                parsed_df = parse_sparse(df, init_df)
            elif experiment_type == "audio":
                parsed_df = parse_audio(df, init_df)
            else:
                # Parse results (compare with baseline)
                parsed_df = parse_reranker(df, init_df)

            os.makedirs(f"src/evaluator/results/benchmarking_parsed/{dataset_name}", exist_ok=True)
            parsed_df.to_csv(
                f"src/evaluator/results/benchmarking_parsed/{dataset_name}/{experiment_type}.csv",
                index=False,
            )

    """
    dataset_name = "shs100k"

    df = pd.read_csv(f"src/evaluator/results/benchmarking/{dataset_name}/audio.csv")
    init_df = pd.read_csv(f"src/evaluator/results/benchmarking/{dataset_name}/chunking_t.csv") #baseline, , transcription_vocal, chunking_t


    parsed_df = parse_audio(df, init_df)
    print(parsed_df)
    """




