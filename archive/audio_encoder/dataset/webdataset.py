import io

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from webdataset.handlers import warn_and_continue


class WebDataset(IterableDataset):
    """ 
    A PyTorch IterableDataset that reads from a WebDataset tar archive.
    
    Args:
        shardurl (str): URL or path to the WebDataset tar archive.
        window (int): Number of shards to shuffle at once.
        batch_size (int): Number of samples per batch.
    """
    def __init__(self, shardurl: str, window: int, batch_size: int):
        self.shardurl = shardurl
        self.window = window
        self.shuffle_buffer = window  # Buffer size for shuffling samples in memory
        self.batch_size = batch_size  # Batch size for the dataset

        self.pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.shardurl),     # Create a list of shards from the shard URL
            wds.shuffle(self.window),               # Shuffle the shards
            wds.split_by_worker,                    # Split the shards by worker
            wds.tarfile_to_samples(),               # Convert tar files to samples
            wds.shuffle(self.shuffle_buffer),       # Shuffle samples in memory
            wds.map(                                # Decode the samples
                lambda x: {
                    "features": torch.from_numpy(np.load(io.BytesIO(x["features.npy"]))).float().detach(),
                    "text": torch.from_numpy(np.load(io.BytesIO(x["text.npy"]))).float().detach(),
                    "id": x["__key__"],
                }, handler=warn_and_continue        # Handle missing files
            ),
            wds.to_tuple("features", "text", "id"),       # Convert to tuple of features and text
            wds.batched(self.batch_size),           # Batch the samples
        )

    def __iter__(self) -> iter:
        """
        Returns an iterator over the dataset.
        
        Yields:
            Tuple[torch.Tensor, torch.Tensor]: A batch of features and text tensors.
        """
        return iter(self.pipeline)