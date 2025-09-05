from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from coverhunter.src.utils import load_hparams
from coverhunter.src.model import Model
from coverhunter.dataset import AudioFeatDataset, skip_none_collate


# ---------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------
def generate_embeddings(
    model: Model,
    data_loader: DataLoader,
    device: torch.device,
    dataset_dir: str | os.PathLike[str],
    dataset_name: str,
    *,
    empty_cache_each_batch: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, np.ndarray]:
    """
    Run inference over `data_loader` and collect embeddings into a dict.

    Parameters
    ----------
    model : Model
        Model exposing `inference(tensor)` -> (embeddings, aux).
    data_loader : DataLoader
        Must yield tuples: (version_ids: list[str], cqt_feat: torch.Tensor[B, T, F]).
    device : torch.device
        "cuda", "cpu", "mps", etc.
    dataset_dir : str | Path
        Output directory where <dataset_name>_embeddings.npz will be saved.
    dataset_name : str
        Used in the output filename.
    empty_cache_each_batch : bool, default False
        If True and using CUDA, calls `torch.cuda.empty_cache()` per batch. Useful
        for tight GPU memory but slightly slower.
    dtype : torch.dtype, default torch.float32
        Cast features to this dtype before inference (if needed).

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping {version_id -> embedding_vector}.
    """
    out_dir = Path(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_name}_embeddings.npz"

    model.eval()
    embeddings: Dict[str, np.ndarray] = {}

    # Use inference_mode for better perf and to avoid autograd overhead
    with torch.inference_mode():
        for step, (version_ids, cqt_feat) in enumerate(tqdm(data_loader, desc="Generating embeddings", unit="batch")):
            # Skip entirely empty batches (e.g., all items failed and were dropped by collate)
            if not version_ids or cqt_feat.numel() == 0:
                logger.warning("Skipping empty batch at step %d", step)
                continue

            try:
                # Optional cache trim (can help if you see OOMs on long runs)
                if empty_cache_each_batch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Move features to device & dtype
                x = cqt_feat.to(device=device, dtype=dtype, non_blocking=True)

                # Model inference
                # Expectation: model.inference(x) -> (embeddings[B, D], aux)
                embed, _ = model.inference(x)

                # Sanity check shapes
                if embed.ndim != 2 or embed.shape[0] != len(version_ids):
                    raise ValueError(
                        f"Mismatch between embeddings shape {embed.shape} and batch ids {len(version_ids)}"
                    )

                # Detach -> CPU -> numpy (float32 to keep file small & consistent)
                embed = embed.detach().to("cpu", dtype=torch.float32).numpy()

                # Accumulate into dict
                for v_id, vec in zip(version_ids, embed, strict=False):
                    if v_id in embeddings:
                        logger.warning("Duplicate version_id encountered: %s (overwriting)", v_id)
                    embeddings[v_id] = vec  # vec is 1D np.ndarray of dim D

            except Exception as e:
                logger.exception("Error processing batch %d: %s", step, e)
                # Continue to next batch; partial results are OK
                continue

    # Save compressed NPZ (keys are version_ids; values are vectors)
    try:
        np.savez_compressed(out_file, **embeddings)
        logger.info("Saved embeddings: %s (%d items)", out_file, len(embeddings))
    except Exception as e:
        logger.exception("Failed saving embeddings to %s: %s", out_file, e)

    return embeddings


# ---------------------------------------------------------------------
# High-level orchestration: load model, build dataset/loader, run inference
# ---------------------------------------------------------------------
def get_coverhunter_embeddings(
    ds: Mapping[str, Iterable],  # supports HF datasets or dict-like with columns
    model_path: str | os.PathLike[str],
    dataset_dir: str | os.PathLike[str],
    dataset_name: str,
    *,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    empty_cache_each_batch: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load CoverHunter model + hyperparameters, build a DataLoader on `AudioFeatDataset`,
    and generate embeddings.

    Parameters
    ----------
    ds : Mapping-like
        Provides columns "version_id" and "cqt_path".
    model_path : str | Path
        Directory that contains:
          - config/hparams.yaml
          - checkpoints/ (folder with model weights)
    dataset_dir : str | Path
        Output directory for the saved NPZ file.
    dataset_name : str
        Name used in the output filename (<dataset_name>_embeddings.npz).
    batch_size : Optional[int]
        Overrides the batch size from hyperparameters if provided.
    num_workers : Optional[int]
        Overrides dataloader workers from hyperparameters if provided.
    pin_memory : bool, default True
        Enable pinned memory to speed up host->GPU transfer (CUDA only).
    persistent_workers : Optional[bool]
        If None, will be set automatically to True when num_workers > 0.
    empty_cache_each_batch : bool, default False
        Pass-through to `generate_embeddings`.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping {version_id -> embedding_vector}.
    """
    model_path = Path(model_path)

    # ---------------------------
    # Load hyperparameters (hp)
    # ---------------------------
    hp_path = model_path / "config" / "hparams.yaml"
    model_hp = load_hparams(str(hp_path))

    # Resolve device from hp; fall back gracefully if CUDA not available
    hp_device = str(model_hp.get("device", "cuda"))
    if hp_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(hp_device)
    logger.info("Using device: %s", device)

    # ---------------------------
    # Initialize and load model
    # ---------------------------
    model = Model(model_hp).to(device)
    checkpoint_dir = model_path / "checkpoints"
    model.load_model_parameters(str(checkpoint_dir), device=device)
    logger.info("Model initialized and parameters loaded from %s", checkpoint_dir)

    # ----------------------------------------
    # Derive inference frame length (in frames)
    # chunk_frame can be int or list[int] in your hp
    # ----------------------------------------
    chunk_frame = model_hp.get("chunk_frame", 0)
    mean_size = model_hp.get("mean_size", 1)
    if isinstance(chunk_frame, list):
        infer_frame = int(chunk_frame[0]) * int(mean_size)
    else:
        infer_frame = int(chunk_frame) * int(mean_size)
    if infer_frame <= 0:
        raise ValueError(f"Invalid infer_frame computed from hp: {infer_frame}")
    logger.info("Inference chunk length (frames): %d", infer_frame)

    # ---------------------------
    # Dataset and DataLoader
    # ---------------------------
    dataset = AudioFeatDataset(model_hp, ds, infer_frame)
    logger.info("Dataset initialized with %d items", len(dataset))

    # Choose defaults from hp, allow overrides via function args
    bs = int(batch_size or model_hp.get("batch_size", 16))
    nw = int(num_workers if num_workers is not None else model_hp.get("num_workers", 0))
    if persistent_workers is None:
        persistent_workers = nw > 0  # good default when workers are used

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,  # keep order stable for reproducibility
        num_workers=nw,
        collate_fn=skip_none_collate,  # drop failed items safely
        pin_memory=(pin_memory and device.type == "cuda"),
        persistent_workers=persistent_workers,
    )
    logger.info("DataLoader ready: batch_size=%d, workers=%d", bs, nw)

    # ---------------------------
    # Run inference + save
    # ---------------------------
    embeddings = generate_embeddings(
        model=model,
        data_loader=loader,
        device=device,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        empty_cache_each_batch=empty_cache_each_batch,
    )

    return embeddings
