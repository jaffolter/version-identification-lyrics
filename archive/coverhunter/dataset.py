from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from loguru import logger

from coverhunter.src.cqt import shorter


class AudioFeatDataset(Dataset[Tuple[str, torch.Tensor]]):
    """
    PyTorch Dataset for extracting fixed-length chunks of CQT features.

    Each item returned is a tuple: (version_id, cqt_tensor).

    Parameters
    ----------
    hp : dict
        Hyperparameters. Must contain key "mean_size" for the `shorter()` function.
    ds : HuggingFace Dataset
        Must provide columns "version_id" and "cqt_path" (paths to .npy files).
    infer_frame : int
        Target number of frames to extract per sample.
    """

    def __init__(self, hp: dict, ds, infer_frame: int) -> None:
        # Save hyperparameters
        self._hp = hp

        # Store (id, path) pairs so we can load them later
        self._data: List[Tuple[str, Path]] = list(zip(ds["version_id"], ds["cqt_path"]))

        # Number of frames to extract per sample
        self.chunk_len = int(infer_frame)

    def __len__(self) -> int:
        """Return total number of examples in the dataset."""
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[Optional[str], Optional[torch.Tensor]]:
        """
        Load and process a single item.

        Steps:
        1. Load the CQT feature from disk (expects .npy file).
        2. Randomly pick a start index (if longer than `chunk_len`).
        3. Extract a chunk of length `chunk_len`.
        4. If too short, pad with constant value (-100).
        5. Optionally downsample with `shorter()`.
        6. Convert to torch.Tensor and return with version_id.

        If an error occurs, returns (None, None).
        """
        version_id, cqt_path = self._data[idx]

        try:
            # ----------------------------------------------------------
            # Step 1: Load the CQT matrix (shape = [time, frequency_bins])
            # ----------------------------------------------------------
            cqt_feat = np.load(cqt_path)
            feat_len = cqt_feat.shape[0]  # number of frames

            # ----------------------------------------------------------
            # Step 2: Choose a random start index for chunk extraction
            # ----------------------------------------------------------
            if feat_len > self.chunk_len:
                # random.randint is inclusive → ensure proper range
                start = random.randint(0, feat_len - self.chunk_len)
            else:
                # If sequence is shorter than chunk, start at 0
                start = 0

            # ----------------------------------------------------------
            # Step 3: Slice the chunk
            # ----------------------------------------------------------
            cqt_chunk = cqt_feat[start : start + self.chunk_len]

            # ----------------------------------------------------------
            # Step 4: Pad if chunk is too short
            # Example: if we asked for 100 frames but file has only 80,
            # we pad with -100 for the missing 20 frames.
            # ----------------------------------------------------------
            if cqt_chunk.shape[0] < self.chunk_len:
                pad_amount = self.chunk_len - cqt_chunk.shape[0]
                cqt_chunk = np.pad(
                    cqt_chunk,
                    pad_width=((0, pad_amount), (0, 0)),  # pad in time dimension only
                    mode="constant",
                    constant_values=-100,
                )

            # ----------------------------------------------------------
            # Step 5: Downsample (if mean_size is set)
            # `shorter` typically reduces the time resolution
            # to make sequences more compact.
            # ----------------------------------------------------------
            mean_size = self._hp.get("mean_size", None)
            if mean_size is not None:
                cqt_chunk = shorter(cqt_chunk, mean_size)

            # ----------------------------------------------------------
            # Step 6: Convert to torch.Tensor
            # ----------------------------------------------------------
            tensor = torch.from_numpy(cqt_chunk).float()

            return version_id, tensor

        except Exception as e:
            # Catch-all to avoid crashing training when a file is corrupted
            logger.warning(
                "[AudioFeatDataset] Failed to load idx=%d path=%s: %s",
                idx,
                cqt_path,
                e,
            )
            return None, None


# ----------------------------------------------------------------------
# Utility collate_fn for DataLoader
# ----------------------------------------------------------------------
def skip_none_collate(batch: Sequence[Tuple[Optional[str], Optional[torch.Tensor]]]) -> Tuple[List[str], torch.Tensor]:
    """
    Collate function that drops (None, None) samples.
    This way, corrupted/missing files won't break the batch.

    Returns
    -------
    version_ids : list of str
    batch_tensor : torch.Tensor
        Stacked tensor of shape [batch_size, time, freq].
    """
    # Filter out failed items
    items = [(vid, feat) for vid, feat in batch if vid is not None and feat is not None]

    if not items:
        # Entire batch failed (rare) → return empty placeholders
        return [], torch.empty(0)

    version_ids, tensors = zip(*items)
    return list(version_ids), torch.stack(tensors, dim=0)
