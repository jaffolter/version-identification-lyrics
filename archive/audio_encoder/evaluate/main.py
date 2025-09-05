import ast
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.webdataset import WebDataset
from src.evaluation.vi_evaluator import Ranker, get_input_data_evaluator, load_metadata, parse_audio
from src.inference.main import Session

class WhisperEvaluation: 
        def __init__(
            self, 
            checkpoint_path: str, 
            config_path: str, 
            session_name: str, 
        ):
            self.session_name = session_name
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Instantiate the inference session
            self.session = Session(config_path=config_path, checkpoint_path=checkpoint_path) 

            # Create a folder to save results
            self.results_folder = f"src/results/whisper"
            os.makedirs(self.results_folder, exist_ok=True)
        
        def compute_embeddings_benchmarks(
            self, 
            dataset_name: str, 
            path_metadata: str, 
            path_audio_folder: str, 
            compute_vocal_segments: bool=True,
            get_single_embedding: bool=True
        ) -> None:
            """
            Compute embeddings from the model for a given dataset and audio folder. 
            Save the embeddings to a file. 
            
            Args:
                dataset_name (str): Name of the dataset.
                path_metadata (str): Path to the metadata file.
                path_audio_folder (str): Path to the folder containing audio files.
                compute_vocal_segments (bool): Whether to compute vocal segments or not.
                get_single_embedding (bool): Whether to get a single embedding or not.
            """
            # Get audio embedding
            df = pd.read_csv(path_metadata)
            print(len(df))

            results = {}
            for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Computing Embedding - {dataset_name}"):
                version_id = row['version_id']
                md5_encoded = row['md5_encoded']
                path_audio = os.path.join(path_audio_folder, f"{md5_encoded}.mp3")
                
                if not os.path.exists(path_audio):
                    print(f"Audio file {path_audio} does not exist. Skipping...")
                    continue
                
                if not compute_vocal_segments:
                    vocal_segments = ast.literal_eval(row["vocal_segments"])
                else:
                    vocal_segments = None
                    
                # Inference pass from the model 
                try:
                    audio_embedding = self.session.inference_from_raw_audio_no_trained(
                        audio_path=path_audio,
                        vocal_segments=vocal_segments,
                        get_single_embedding=get_single_embedding
                    )
                    
                    print(audio_embedding.shape)
                    
                    # Store the embedding in the results dictionary
                    results[version_id] = audio_embedding
                    
                except Exception as e:
                    print(f"Error processing version_id {version_id}: {e}")
                    continue
                    
                
            # Save audio embeddings
            path_save_audio = os.path.join(self.results_folder, "benchmark_audio_embeddings", f"{dataset_name}.npz")
            os.makedirs(os.path.dirname(path_save_audio), exist_ok=True)
            
            safe_results = {str(k): v.detach().cpu().numpy() for k, v in results.items()}
            np.savez_compressed(path_save_audio, **safe_results)

            logger.info(f"Saved audio embeddings to {path_save_audio}")
            
        def evaluate_version_identification(
            self, 
            dataset_name: str, 
            details: Optional[str] = None
        ) -> None:
            
            # Prepare data for evaluation
            df, ds = load_metadata(dataset_name, details)
            
            print(len(df))
            queries_ids, corpus_ids, relevant_docs = get_input_data_evaluator(ds)

            # Load embeddings
            path_audio_embeddings = f"{self.results_folder}/benchmark_audio_embeddings/{dataset_name}.npz"
            print(path_audio_embeddings)
            embeddings = np.load(path_audio_embeddings)
            
            # Loop over the dict and average 
            #new_dict = {}
            
            #for key, value in embeddings.items():
                #print(key, value.shape)
                #new_dict[key] = value.mean(axis=0)
                
            #embeddings = new_dict
            print(len(embeddings.keys()))

            # Filter queries_ids and corpus_ids based on the embeddings
            queries_ids = [q for q in queries_ids if str(q) in embeddings]
            corpus_ids = [c for c in corpus_ids if str(c) in embeddings]
            relevant_docs = {
                q: [doc for doc in docs if str(doc) in embeddings]
                for q, docs in relevant_docs.items()
            }
            
            print("_"*100)
            print(f"   {len(queries_ids)} queries, {len(corpus_ids)} corpus documents, {len(relevant_docs)} relevant documents")

            ranker = Ranker(
                queries_ids=queries_ids,
                corpus_ids=corpus_ids,
                relevant_docs=relevant_docs,
                embeddings=embeddings,
                k=100,
            )
            
            ranker.get_similarity_matrix()
            ranker.rank_results()
            ranker.get_first_true_positives()
            ranker.get_fp()

            ranker.compute_metrics()
            
            results = [{
                "Title": "Results",
                "model_name": "LIE",
                "MR1": ranker.metrics.get("mr1"),
                "HR1": ranker.metrics.get("hr"),
                "HR10": ranker.metrics.get("hr10"),
                "MAP10": ranker.metrics.get("map10"),
            }]
            
            df = pd.DataFrame(results)
            
            print(df)
            df.to_csv('src/results/RESSSSSSSS2.csv', index=False)
            
            # Compute delta with respect to the baseline
            baselines = {
                "Clean Lyrics" : f"src/evaluation/baselines/{dataset_name}/baseline.csv",
                "Transcription on Full Lyrics": f"src/evaluation/baselines/{dataset_name}/transcription_vocal.csv",
                "Transcription on Chunked Lyrics + Average": f"src/evaluation/baselines/{dataset_name}/chunking_t.csv",
                "Audio Baseline (CoverHunter)": f"src/evaluation/baselines/{dataset_name}/coverhunter.csv",
            }
            
            results = df.copy()
            for col in ["MR1", "HR1", "HR10", "MAP10"]:
                results[col].apply(lambda x: f"{x:.3f}")
            results = results.drop(columns=["model_name"])
            res_df = [results]
            
            for baseline_name, baseline_path in baselines.items():
                if dataset_name == "discogs_vi" and baseline_name == "Clean Lyrics":
                    continue
                baseline_df = pd.read_csv(baseline_path)
                parsed_df = parse_audio(df.copy(), baseline_df)
                parsed_df['Title'] = baseline_name
                
                res_df.append(parsed_df)
                
            final_df = pd.concat(res_df, ignore_index=True)
            print(df)
            
            # Save results to CSV
            path_save_csv = f"{self.results_folder}/csv/version_identification_results_{dataset_name}.csv"
            os.makedirs(os.path.dirname(path_save_csv), exist_ok=True)
            final_df.to_csv(path_save_csv, index=False)
            
            
    
class ModelEvaluation: 
    def __init__(
        self, 
        checkpoint_path: str, 
        config_path: str, 
        session_name: str, 
    ):
        self.session_name = session_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Instantiate the inference session
        self.session = Session(config_path=config_path, checkpoint_path=checkpoint_path) 

        # Create a folder to save results
        self.results_folder = f"src/results/{self.session_name}"
        os.makedirs(self.results_folder, exist_ok=True)
    
    def load_test_set(
        self, folder_test_set: str = "src/data/test", shard_start_end: str="000000..000167"
    ) -> DataLoader:
        """
        Load the test set from the specified folder and shard range.
        
        Args:
            folder_test_set (str): Path to the folder containing the test set shards.
            shard_start_end (str): Range of shards to load, e.g., "000000..000167".
        Returns:
            DataLoader: DataLoader for the test set.
        """
        shardurl = os.path.join(folder_test_set, "shard-{" + shard_start_end + "}.tar")
        
        # Load the dataset using WebDataset
        dataset = WebDataset(shardurl, window=100, batch_size=32) 
        
        return DataLoader(
            dataset,
            batch_size=None, 
            num_workers=0,   
            shuffle=False,   
        )
    
    def compute_embeddings_from_test_set(
        self,
        dataloader: DataLoader, 
        get_single_embedding: bool=False , 
        compute_text_embeddings: bool=False, 
    ) -> None:
        """
        Compute embeddings from the test set using the session's inference method.
        
        Args:
            dataloader (DataLoader): DataLoader for the test set.
            get_single_embedding (bool): Whether to get a single embedding or not.
            compute_text_embeddings (bool): Whether to compute text embeddings or not.
        """
        
        for idx, (audio, text, ids) in enumerate(tqdm(dataloader, desc="Computing embeddings from test set")):
            audio_embeddings = self.session.inference_from_extracted_features(
                audio_features=audio,
                get_single_embedding=get_single_embedding, 
            )
            
            for idx, sample_id in enumerate(ids):
                path_save_audio = f"{self.results_folder}/test_audio_embeddings/{sample_id}.npy"
                os.makedirs(os.path.dirname(path_save_audio), exist_ok=True)

                if not os.path.exists(path_save_audio):
                    np.save(path_save_audio, audio_embeddings[idx].cpu().numpy())

                if compute_text_embeddings:
                    path_save_text = f"src/results/test_text_embeddings/{sample_id}.npy"
                    os.makedirs(os.path.dirname(path_save_text), exist_ok=True)

                    if not os.path.exists(path_save_text):
                        np.save(path_save_text, text[idx].numpy())
            
    def compute_embeddings_benchmarks(
        self, 
        dataset_name: str, 
        path_metadata: str, 
        path_audio_folder: str, 
        compute_vocal_segments: bool=True,
        get_single_embedding: bool=True
    ) -> None:
        """
        Compute embeddings from the model for a given dataset and audio folder. 
        Save the embeddings to a file. 
        
        Args:
            dataset_name (str): Name of the dataset.
            path_metadata (str): Path to the metadata file.
            path_audio_folder (str): Path to the folder containing audio files.
            compute_vocal_segments (bool): Whether to compute vocal segments or not.
            get_single_embedding (bool): Whether to get a single embedding or not.
        """
        # Get audio embedding
        df = pd.read_csv(path_metadata)

        results = {}
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Computing Embedding - {dataset_name}"):
            version_id = row['version_id']
            md5_encoded = row['md5_encoded']
            path_audio = os.path.join(path_audio_folder, f"{md5_encoded}.mp3")
            
            if not os.path.exists(path_audio):
                print(f"Audio file {path_audio} does not exist. Skipping...")
                continue
            
            if not compute_vocal_segments:
                vocal_segments = ast.literal_eval(row["vocal_segments"])
            else:
                vocal_segments = None
                
            # Inference pass from the model 
            try:
                audio_embedding = self.session.inference_from_raw_audio(
                    audio_path=path_audio,
                    vocal_segments=vocal_segments,
                    get_single_embedding=get_single_embedding
                )
                
                # Store the embedding in the results dictionary
                results[version_id] = audio_embedding
                
            except Exception as e:
                print(f"Error processing version_id {version_id}: {e}")
                continue
                
            
        # Save audio embeddings
        path_save_audio = os.path.join(self.results_folder, "benchmark_audio_embeddings", f"{dataset_name}.npz")
        os.makedirs(os.path.dirname(path_save_audio), exist_ok=True)
        
        safe_results = {str(k): v.detach().cpu().numpy() for k, v in results.items()}
        np.savez_compressed(path_save_audio, **safe_results)

        logger.info(f"Saved audio embeddings to {path_save_audio}")

    def load_local_embeddings_test_set(
        self,
        n: Optional[int] = 100 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the text and audio embeddings from the test set.
        
        Args:
            n (int): Number of samples to load.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Text and audio embeddings as numpy arrays.
        """
        text_embeddings = []
        audio_embeddings = []

        text_files = glob.glob("src/results/test_text_embeddings/*.npy")
        audio_files = glob.glob(f"{self.results_folder}/test_audio_embeddings/*.npy")
        
        for file in text_files[:n] if n is not None else text_files:
            embedding = np.load(file)
            text_embeddings.append(embedding)
            
        for file in audio_files[:n] if n is not None else audio_files:
            embedding = np.load(file)
            audio_embeddings.append(embedding)
        
        return np.array(text_embeddings), np.array(audio_embeddings)

    
    def get_version_ids_to_keep(self):
        """ 
        Retrieve version_ids that have at least two chunks in the test set.
        
        Returns:
            pd.DataFrame: DataFrame containing version_ids and chunk_ids.
        """
        # Retrieve all ids in the test set: ids = [version_id + chunk_id]
        text_files = glob.glob("src/results/test_text_embeddings/*.npy")
        ids = [os.path.basename(f).replace('.npy', '') for f in text_files]
        
        df = pd.DataFrame({
            "ids": ids,
        })
        
        # Derive version_id and chunk_id from id
        df['version_id'] = df['ids'].apply(lambda x: x[:-1])
        df['chunk_id'] = df['ids'].apply(lambda x: x[-1])
        
        # Only keep version_ids that have at least two chunks in the test set
        grouped = df.groupby('version_id').size().reset_index(name='count')    
        grouped_f = grouped[grouped['count'] > 1]
        df = df[df['version_id'].isin(grouped_f['version_id'])].reset_index(drop=True)    

        return df

    def load_local_embeddings_test_set_and_aggregate(
        self, df_version_ids_to_keep: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        Load the text and audio embeddings from the test set and aggregate them by version_id.
        Args:
            df_version_ids_to_keep (pd.DataFrame): DataFrame containing version_ids to keep.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Aggregated text and audio embeddings.
        """
        version_to_text_embeddings: Dict[str, List[np.ndarray]] = {}
        version_to_audio_embeddings: Dict[str, List[np.ndarray]] = {}

        # Load all available text and audio embedding files
        text_files = glob.glob("src/results/test_text_embeddings/*.npy")
        audio_files = glob.glob(f"{self.results_folder}/test_audio_embeddings/*.npy")
        
        # Filter by version_id: this allows us to only keep versions for which we have at least 
        # two chunks in the test set
        version_ids = set(df_version_ids_to_keep["version_id"])

        for file in text_files:
            # Retrieve version_id from the filename: version_id + chunk_id
            basename = os.path.basename(file).replace('.npy', '')
            version_id = basename[:-1]  
            
            # Load the embedding if the version_id is in the set of version_ids
            if version_id in version_ids:
                emb = np.load(file)
                version_to_text_embeddings.setdefault(version_id, []).append(emb)

        for file in audio_files:    # same for audio
            basename = os.path.basename(file).replace('.npy', '')
            version_id = basename[:-1] 
            if version_id in version_ids:
                emb = np.load(file)
                version_to_audio_embeddings.setdefault(version_id, []).append(emb)

        # Average the embeddings per version_id
        text_embeddings = []
        audio_embeddings = []

        for k in version_to_text_embeddings.keys():
            if k in version_to_audio_embeddings:
                text_embeddings.append(np.mean(version_to_text_embeddings[k], axis=0))
                audio_embeddings.append(np.mean(version_to_audio_embeddings[k], axis=0))
            else:
                print(f"Warning: No audio embeddings found for version_id {k}")
    
        return np.array(text_embeddings), np.array(audio_embeddings)

    def pca_audio_text_embeddings(
        self, 
        text_embeddings: np.ndarray, 
        audio_embeddings: np.ndarray, 
        n: int = 100,
    ) -> None:
        """
        Perform PCA on the audio and text embeddings from the test set, plot the results and save the figure.
        
        Args:
            text_embeddings (np.ndarray): Text embeddings.
            audio_embeddings (np.ndarray): Audio embeddings.
            n (int): Number of samples to plot.
        """
        pca = PCA(n_components=2)
        
        embeddings = np.concatenate((text_embeddings, audio_embeddings), axis=0)
        labels = ["Text" for _ in range(text_embeddings.shape[0])] + \
                 ["Audio" for _ in range(audio_embeddings.shape[0])]
        colors = {"Text": "blue", "Audio": "red"}
                 
        reduced = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(8, 6))
        for label in set(labels):
            idx = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label, alpha=0.7, color=colors[label])
    
        plt.legend()
        plt.title(f"Audio-Text PCA Embeddings - n={n}")
        plt.grid(True)
        plt.tight_layout()
        
        path_save_fig = f"{self.results_folder}/figures/pca_audio_text_embeddings_{n}.png"
        os.makedirs(os.path.dirname(path_save_fig), exist_ok=True)
        plt.savefig(path_save_fig)

    
    def audio_text_similarity(
        self, audio_embeddings: np.ndarray, text_embeddings: np.ndarray, type_data: str = "local"
    ) ->None:
        """
        Compute the cosine similarity between audio and text embeddings, plot the distribution of scores, and save
        the results to a CSV file.
        
        Args:
            audio_embeddings (np.ndarray): Audio embeddings.
            text_embeddings (np.ndarray): Text embeddings.
            type_data (str): Type of data, e.g., "local" or "global".
        """
        
        def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
        scores = [cosine_similarity(audio_embeddings[i], text_embeddings[i]) for i in range(len(audio_embeddings))]

        # Compute average and standard deviation of cosine similarity scores and store 
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        df = pd.DataFrame({
            "Average Cosine Similarity": [avg_score],
            "Standard Deviation of Cosine Similarity": [std_score]
        })
        path_save_csv = f"{self.results_folder}/csv/audio_text_similarity_scores_{type_data}.csv"
        os.makedirs(os.path.dirname(path_save_csv), exist_ok=True)
        df.to_csv(path_save_csv, index=False)
        
        # Plot the distribution of scores
        plt.figure(figsize=(8, 6))
        plt.hist(scores, bins=20, color='blue', alpha=0.7)
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.title("Distribution of Audio-Text Cosine Similarity Scores - " + type_data)
        plt.grid(True)
        
        path_save_fig = f"{self.results_folder}/figures/audio_text_similarity_distribution_{type_data}.png"
        plt.savefig(path_save_fig)
        
        
    def evaluate_version_identification(
        self, 
        dataset_name: str, 
        details: Optional[str] = None
    ) -> None:
        
        # Prepare data for evaluation
        df, ds = load_metadata(dataset_name, details)
        queries_ids, corpus_ids, relevant_docs = get_input_data_evaluator(ds)

        # Load embeddings
        path_audio_embeddings = f"{self.results_folder}/benchmark_audio_embeddings/{dataset_name}.npz"
        embeddings = np.load(path_audio_embeddings)
        
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
            embeddings=embeddings,
            k=100,
        )
        
        ranker.get_similarity_matrix()
        ranker.rank_results()
        ranker.get_first_true_positives()
        ranker.get_fp()

        ranker.compute_metrics()
        
        results = [{
            "Title": "Results",
            "model_name": "LIE",
            "MR1": ranker.metrics.get("mr1"),
            "HR1": ranker.metrics.get("hr"),
            "HR10": ranker.metrics.get("hr10"),
            "MAP10": ranker.metrics.get("map10"),
        }]
        
        df = pd.DataFrame(results)
        
        print(df)
        df.to_csv('src/results/RESSSSSSSS.csv', index=False)
        
        # Compute delta with respect to the baseline
        baselines = {
            "Clean Lyrics" : f"src/evaluation/baselines/{dataset_name}/baseline.csv",
            "Transcription on Full Lyrics": f"src/evaluation/baselines/{dataset_name}/transcription_vocal.csv",
            "Transcription on Chunked Lyrics + Average": f"src/evaluation/baselines/{dataset_name}/chunking_t.csv",
            "Audio Baseline (CoverHunter)": f"src/evaluation/baselines/{dataset_name}/coverhunter.csv",
        }
        
        results = df.copy()
        for col in ["MR1", "HR1", "HR10", "MAP10"]:
            results[col].apply(lambda x: f"{x:.3f}")
        results = results.drop(columns=["model_name"])
        res_df = [results]
        
        for baseline_name, baseline_path in baselines.items():
            if dataset_name == "discogs_vi" and baseline_name == "Clean Lyrics":
                continue
            baseline_df = pd.read_csv(baseline_path)
            parsed_df = parse_audio(df.copy(), baseline_df)
            parsed_df['Title'] = baseline_name
            
            res_df.append(parsed_df)
            
        final_df = pd.concat(res_df, ignore_index=True)
        print(df)
        
        # Save results to CSV
        path_save_csv = f"{self.results_folder}/csv/version_identification_results_{dataset_name}.csv"
        os.makedirs(os.path.dirname(path_save_csv), exist_ok=True)
        final_df.to_csv(path_save_csv, index=False)
        
        