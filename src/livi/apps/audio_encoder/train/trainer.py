"""
Trainer for the LIVI audio encoder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Callable

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb

from livi.apps.audio_encoder.data.dataset import make_loader
from livi.apps.audio_encoder.models.whisper_encoder import WhisperEncoder
from livi.apps.audio_encoder.train.loss import MSECosineLoss
from livi.apps.audio_encoder.train.val_metrics import retrieval_metrics
from livi.apps.audio_encoder.utils.seed import set_seed
from livi.apps.audio_encoder.train.scheduler import get_linear_warmup_scheduler


# --------------------------------------------------------------------
# Environment setup (safe defaults; tweak as needed)
# --------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_float32_matmul_precision("high")


# --------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------
class Trainer:
    """
    Wraps data, model, optimization, training/validation loops, and logging.

    Attributes:
        cfg: Hydra/OmegaConf configuration.
        device: torch.device on which to run (e.g., "cuda:0").
        model: Trainable projection head (takes Whisper hidden states → audio embeddings).
        whisper: Frozen Whisper encoder producing hidden states from Mel inputs.
        train_loader/val_loader: Iterators over WebDataset shards.
        optimizer: AdamW optimizer.
        scheduler: LR scheduler (LambdaLR).
        loss_fn: MSECosineLoss (pairwise-structure MSE + instance cosine).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # ----- device & seed -----
        self.device = torch.device(cfg.model.device)
        torch.cuda.set_device(self.device)
        set_seed(cfg.seed)

        # ----- model -----
        self.model = instantiate(cfg.model).to(self.device)
        # Frozen whisper encoder used as a feature extractor
        self.whisper = WhisperEncoder(
            model_name=cfg.model.whisper_model_name,
            device=cfg.model.device,
            compile=cfg.model.compile,
        )

        # ----- data -----
        self.train_loader = make_loader(cfg, "train")
        self.val_loader = make_loader(cfg, "val")

        # Steps per epoch used by schedulers/logging (avoid relying on len(dataloader))
        self.len_train_loader = max(1, cfg.data.total_train_samples // cfg.data.batch_size)
        self.len_val_loader = max(1, cfg.data.total_val_samples // cfg.data.batch_size)
        self.total_training_steps = cfg.train.epochs * self.len_train_loader

        # ----- loss -----
        self.loss_fn = MSECosineLoss().to(self.device)

        # ----- optimizer -----
        self.optimizer = torch.optim.AdamW(
            params=list(self.model.parameters()),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.opt_betas),
            weight_decay=cfg.optimizer.weight_decay,
            foreach=False,
            capturable=True,
            fused=True,
        )

        # ----- scheduler -----
        self.scheduler = get_linear_warmup_scheduler(self.optimizer, warmup_steps=cfg.optimizer.warmup_steps)

        # ----- logging -----
        self.log_every_n = int(cfg.wandb.log_steps)
        self.log_dir = str(cfg.wandb.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=self.log_dir,
        )

    # ----------------------------------------------------------------
    # Training step
    # ----------------------------------------------------------------
    def training_step(self, mel: torch.Tensor, target: torch.Tensor, epoch: int, step: int) -> None:
        """
        Single optimization step: forward → loss → backward → opt/sched update.

        Args:
            mel:   Mel spectrograms, shape (B, n_mels, T), torch.float32.
            target: Target lyrics embeddings, shape (B, D), torch.float32.
            epoch: Current epoch index (0-based).
            step:  Step index within the epoch (0-based).
        """
        self.model.train()

        mel = mel.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        # Forward: frozen Whisper → hidden states → projection head → embeddings
        with torch.no_grad():  # Whisper is frozen
            whisper_hidden_states = self.whisper(mel)
        pred = self.model(whisper_hidden_states)

        # Loss and backward
        loss = self.loss_fn(pred, target)
        loss.backward()

        # Optimizer/scheduler
        self.optimizer.step()
        self.scheduler.step()  # one LR update per optimizer step
        self.optimizer.zero_grad(set_to_none=True)

        # Logging
        global_step = epoch * self.len_train_loader + step
        if (step + 1) % self.log_every_n == 0:
            self._log_train_metrics(global_step, loss, target, pred)

    # ----------------------------------------------------------------
    # Logging helpers
    # ----------------------------------------------------------------
    @torch.no_grad()
    def _safe_mean(self, t: torch.Tensor) -> float:
        """Detach → float → mean on CPU (for logging)."""
        return t.detach().float().mean().cpu().item()

    @torch.no_grad()
    def _log_train_metrics(self, step: int, loss: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> None:
        """
        Log core training stats to Weights & Biases.

        Args:
            step:   Global step (for x-axis in W&B).
            loss:   Scalar tensor from the current step.
            target: Ground-truth text embeddings, (B, D).
            pred:   Predicted audio embeddings, (B, D).
        """
        grads = [p.grad for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        grad_norm = torch.norm(torch.stack([g.norm() for g in grads])).detach().cpu().item() if grads else 0.0

        lr = self.optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "train/loss": self._safe_mean(loss),
                "train/cosine_sim": self._safe_mean(F.cosine_similarity(pred, target)),
                "train/audio_norm": self._safe_mean(pred.norm(dim=1)),
                "train/grad_norm": grad_norm,
                "train/learning_rate": lr,
            },
            step=step,
        )

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Compute retrieval metrics on the validation set.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Dict[str, float]: Mean HR@k, MAP@k, and diagonal cosine similarity
            across all validation batches.
        """
        self.model.eval()
        metrics_list: list[Dict[str, float]] = []

        for step, (mel, target, _) in enumerate(
            tqdm(self.val_loader, total=self.len_val_loader, desc=f"Validating {epoch + 1}")
        ):
            if step >= 2:
                break
            mel = mel.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            whisper_hidden_states = self.whisper(mel)
            pred = self.model(whisper_hidden_states)

            batch_metrics = retrieval_metrics(pred, target, topk=self.cfg.metrics.topk, map_k=10)
            metrics_list.append(batch_metrics)

        # Aggregate
        keys = metrics_list[0].keys()
        metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=(epoch + 1))
        return metrics

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------
    def save(self, epoch: int) -> None:
        """
        Save model weights to <checkpoint_dir>/<wandb_run>/epoch_<epoch>.pth

        Args:
            epoch: Current epoch index (0-based).
        """
        run_name = wandb.run.name if wandb.run else "default_run"
        out_path = Path(self.cfg.model.checkpoint_dir) / run_name / f"epoch_{epoch}.pth"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_path)
        logger.info(f"Saved checkpoint: {out_path}")

    # ----------------------------------------------------------------
    # Training loop (epochs)
    # ----------------------------------------------------------------
    def train(self) -> None:
        """
        Full training loop over `cfg.train.epochs` epochs:
            - Iterate train loader → call `training_step`
            - Validate at epoch end
            - Save a checkpoint
        """
        for epoch in range(self.cfg.train.epochs):
            for step, (mel, target, _) in enumerate(
                tqdm(self.train_loader, total=self.len_train_loader, desc=f"Epoch {epoch + 1}")
            ):
                if step >= 2:
                    break
                try:
                    self.training_step(mel, target, epoch, step)
                except Exception as e:
                    logger.warning(f"[Train] step={step} failed: {e}")

            try:
                self.validate(epoch)
            except Exception as e:
                logger.warning(f"[Validate] epoch={epoch} failed: {e}")

            self.save(epoch)
            logger.info(f"Epoch {epoch + 1} complete.")
