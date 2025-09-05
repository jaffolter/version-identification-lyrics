"""
WebDataset loader for the Audio Encoder dataset.

------------------------------------------------------------
Important:
    - The dataset MUST be created beforehand as a WebDataset 
      (sharded .tar archives). 
    - Use the CLI command below to generate it:

        poetry run livi-data create-audio-encoder-dataset \
            --split train \
            --path-out "data/audio_encoder_dataset/train/shard-%06d.tar" \
            --metadata-path "data/audio_encoder_dataset/metadata/train.csv" \
            --mel-path "data/audio_encoder_dataset/mel" \
            --lyrics-embeddings-path "data/audio_encoder_dataset/lyrics_embeddings" \
            --shard-size 1000

    - Runtime parameters (e.g., `shardurl`, `window`, `batch_size`)
      should not be hardcoded. They can be configured in:

        livi/apps/audio_encoder/config/livi.yaml
------------------------------------------------------------
"""

import io
import os
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Iterator

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from webdataset.handlers import warn_and_continue


class WebDataset(IterableDataset):
    """
    PyTorch `IterableDataset` for loading WebDataset shards containing
    precomputed audio features (Mel spectrograms) and lyrics embeddings.

    Each sample in the WebDataset is expected to contain:
        - "features.npy": audio features (e.g., log-Mel spectrograms).
        - "text.npy": lyrics-informed embeddings.
        - "__key__": unique identifier (UID) of the sample.

    Args:
        shardurl (str):
            Glob pattern or path to the WebDataset shard(s),
            e.g. "<DATASET_PATH>/train/shard-{000000..000099}.tar".
            The dataset path can be defined in `livi/apps/audio_encoder/config/livi.yaml` via the `data.data_dir` key.
        window (int):
            Number of shards to shuffle at once (controls memory usage).
            Configurable via `livi.yaml`.
        batch_size (int):
            Number of samples per batch.
            Configurable via `livi.yaml`.
    """

    def __init__(self, shardurl: str, window: int, batch_size: int):
        self.shardurl = shardurl
        self.window = window
        self.batch_size = batch_size
        self.shuffle_buffer = window  # buffer size for in-memory sample shuffling

        # Build the streaming pipeline
        self.pipeline = wds.DataPipeline(
            # 1. List shards from the URL/pattern
            wds.SimpleShardList(self.shardurl),
            # 2. Shuffle shard order
            wds.shuffle(self.window),
            # 3. Split shards across workers (for multi-worker DataLoader)
            wds.split_by_worker,
            # 4. Read .tar files and yield raw samples
            wds.tarfile_to_samples(),
            # 5. Shuffle samples in memory
            wds.shuffle(self.shuffle_buffer),
            # 6. Decode samples: load .npy payloads into torch tensors
            wds.map(
                lambda x: {
                    "mel": torch.from_numpy(np.load(io.BytesIO(x["features.npy"]))).float().detach(),
                    "target": torch.from_numpy(np.load(io.BytesIO(x["text.npy"]))).float().detach(),
                    "id": x["__key__"],  # keep UID for traceability
                },
                handler=warn_and_continue,  # skip missing/broken samples gracefully
            ),
            # 7. Convert to tuple format (mel, target, id)
            wds.to_tuple("mel", "target", "id"),
            # 8. Batch samples
            wds.batched(self.batch_size),
        )

    def __iter__(self) -> Iterator:
        """
        Return an iterator over the dataset.

        Yields:
            Tuple[torch.Tensor, torch.Tensor, str]:
                Batched audio features, batched text embeddings, and IDs.
        """
        return iter(self.pipeline)


def make_loader(cfg: DictConfig, split: str) -> DataLoader:
    """
    Create a DataLoader for a given dataset split (train/val/test) using WebDataset shards.

    Args:
        cfg (DictConfig): Hydra configuration with dataset parameters.
            Expected fields:
                - data_dir (str): Root directory of dataset shards.
                - window (int): Number of samples in a rolling window for batching.
                - batch_size (int): Batch size.
                - num_workers (int): Number of worker processes for data loading.
                - last_shard_train / last_shard_val / last_shard_test (str):
                    Last shard filename ID for each split (e.g., '000999').
        split (str): Dataset split, one of ["train", "val", "test"].

    Returns:
        DataLoader: PyTorch DataLoader wrapping the WebDataset pipeline.
    """
    if split == "train":
        last_shard_id = cfg.data.last_shard_train
    elif split == "val":
        last_shard_id = cfg.data.last_shard_val
    elif split == "test":
        last_shard_id = cfg.data.last_shard_test
    else:
        raise ValueError(f"Unknown split: {split}")

    # Example: "shard-{000000..000999}.tar"
    shard_pattern = f"000000..{last_shard_id}"
    shard_url = os.path.join(cfg.data.data_dir, split, f"shard-{{{shard_pattern}}}.tar")

    dataset = WebDataset(
        shard_url,
        window=cfg.data.window,
        batch_size=cfg.data.batch_size,
    )

    return DataLoader(
        dataset,
        batch_size=None,  # WebDataset handles batching internally
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
        shuffle=False,  # WebDataset handles shuffling internally
    )
