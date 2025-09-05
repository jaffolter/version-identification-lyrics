import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.webdataset import WebDataset
from models.lie import WhisperEncoder, WhisperEncoderTrainable
from train.losses import CosineLoss, InfoNCELoss, MSECosineLoss
from train.scheduler import cosine_with_warmup
from utils.retrieval import retrieval_metrics
from transformers import get_cosine_schedule_with_warmup


# Environment setup ---------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Utility functions ---------------------------------------------
def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): The seed value to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(cfg: DictConfig, split: str) -> DataLoader:
    """
    Create a DataLoader for the specified split (train, val, test) using WebDataset.
    
    Args:
        cfg (DictConfig): Configuration dictionary containing dataset parameters.
        split (str): The dataset split to load ('train', 'val', 'test').
    Returns:
        DataLoader: A DataLoader instance for the specified dataset split.
    """
    
    # Last shard ID/filename based on the split
    last_shard_id = (
        cfg.data.last_shard_train
        if split == "train"
        else cfg.data.last_shard_val
        if split == "val"
        else cfg.data.last_shard_test
    )
    # Shard URL pattern
    # e.g., "000000..000999.tar" for train -> from file 000000.tar to 000999.tar
    file = f"000000..{last_shard_id}"
    shardurl = os.path.join(cfg.data.data_dir, split, "shard-{" + file + "}.tar")
    
    # Load the dataset using WebDataset
    dataset = WebDataset(shardurl, window=cfg.data.window, batch_size=cfg.data.batch_size) 
    
    return DataLoader(
        dataset,
        batch_size=None, # Set to None to use WebDataset's batch_size
        num_workers=cfg.data.num_workers,       # > 0 for parallel data loading
        pin_memory=True,
        persistent_workers=True if cfg.data.num_workers > 0 else False,
        shuffle=False,  # Set to False to use WebDataset's batch_size
    )

class Trainer:
    """
    Trainer class for training the model. Defines the training loop, validation, and logging.
    """
    
    def __init__(self, cfg: DictConfig):
        """ 
        Initializes the Trainer with the given configuration (in conf/config.yaml).
        
        Args:
            cfg (DictConfig): Configuration object containing model, optimizer, data, training parameters
        """
        
        # Configuration setup --------------------------------------------------
        self.cfg = cfg
        
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        
        set_seed(cfg.seed)

        # Model setup ---------------------------------------------
        self.model = instantiate(cfg.model).to(self.device)

        # Whether or not to freeze the WhisperEncoder
        self.freeze_whisper = cfg.model.freeze_whisper
        if self.freeze_whisper:     
            self.whisper = WhisperEncoder(debug=False, pooling=self.cfg.model.pooling)  
        else:                       
            self.whisper = WhisperEncoderTrainable(debug=False, pooling=self.cfg.model.pooling)

        # Data -----------------------------------------------------
        self.train_loader = make_loader(cfg, "train")
        self.val_loader = make_loader(cfg, "val")
        
        self.len_train_loader = cfg.data.total_train_samples // cfg.data.batch_size
        self.len_val_loader = cfg.data.total_val_samples // cfg.data.batch_size
        self.T_max = cfg.train.epochs * self.len_train_loader
        
        # Optimizer setup ------------------------------------------
        
        # 1. Loss function
        self.loss_name = self.cfg.optimizer.loss
        if self.cfg.optimizer.loss == "contrastive":
            self.loss_fn = InfoNCELoss().to(self.device)
        elif self.cfg.optimizer.loss == "cosine":
            self.loss_fn = CosineLoss().to(self.device)
        elif self.cfg.optimizer.loss == "mse":
            self.loss_fn = MSECosineLoss().to(self.device)

        # 2. Optimizer
        # Only captures Whisper parameters if not frozen
        params = list(self.model.parameters())
        if not self.freeze_whisper:
            params += list(self.whisper.parameters())

        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.opt_betas),
            weight_decay=cfg.optimizer.weight_decay,  
            foreach=False,  # forces one big kernel instead of many small ops
            capturable=True,  # keeps it on-device, no CPU sync
            fused=True,  
        )

        # 3. Learning Rate Scheduler
        # https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        # The 1cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate 
        # and then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate.
        # This policy was initially described in the paper Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.
        """self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.optimizer.max_lr,  # max learning rate
            epochs=cfg.train.epochs,  
            steps_per_epoch=self.len_train_loader,       
            anneal_strategy='cos',     # cosine annealing
        )"""
        
        total_training_steps = cfg.train.epochs * self.len_train_loader
        """self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=total_training_steps, 
            eta_min=self.cfg.optimizer.min_lr
        )"""
        """warmup_steps = int(0.05 * total_training_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )"""
        #self.warmup_scheduler = LambdaLR(self.optimizer, lambda step: min(1.0, step / cfg.optimizer.warmup_steps))
        #self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.cosine_with_warmup)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: cosine_with_warmup(step, self.cfg, self.len_train_loader))
        
        
        # Training setup ------------------------------------------
        
        # 1. Mixed Precision Training (AMP)
        self.use_amp = cfg.model.use_amp
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
            
        # 2. Gradient Accumulation
        self.accum_steps = cfg.train.accumulation_steps

        # 3. Gradient Clipping
        self.grad_clip = cfg.train.grad_clip
        self.grad_clip_norm = cfg.train.grad_clip_max_norm


        # Logging setup -------------------------------------------
        self.log_every_n = cfg.wandb.log_steps
        self.log_logit_scale = cfg.wandb.log_logit_scale
        
        self.log_dir = cfg.wandb.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize WandB
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=self.log_dir,
        )


    def training_step(self, audio: torch.Tensor, text: torch.Tensor, epoch: int, step: int) -> None:
        """ 
        Performs a single training step.
        
        Args:
            audio (torch.Tensor): Audio features tensor.
            text (torch.Tensor): Text embeddings tensor.
            epoch (int): Current epoch number.
            step (int): Current step number within the epoch.
        
        """
        # Forward pass ---------------------------------------------------------
        # Get audio and text features
        audio_features = audio.to(self.device, non_blocking=True)
        text_embeddings = text.to(self.device, non_blocking=True)

        # Enable autocast for mixed precision training
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            
            # Get Whisper outputs 
            if self.freeze_whisper:  # Frozen WhisperEncoder -> Just works as a feature extractor    
                whisper_outputs = self.whisper.forward(audio_features)
            else:   # Trainable WhisperEncoder -> Forward pass through the model    
                whisper_outputs = self.whisper(audio_features)
                
            # Forward pass through the model : Pooling + Projection MLP
            if self.loss_name == "contrastive":
                audio_embeddings, logit_scale = self.model(whisper_outputs)
            else:
                audio_embeddings = self.model(whisper_outputs)
                logit_scale = None

            # Loss calculation
            # Note: we divide by self.accum_steps to average the loss over the accumulation steps
            loss = self.loss_fn(audio_embeddings, text_embeddings, logit_scale) / self.accum_steps      
            
        # Backward pass --------------------------------------------------------
        # Note: scaler activated only if self.use_amp is True
        self.scaler.scale(loss).backward()

        # We only update the model parameters every `self.accum_steps` steps to simulate a larger batch size.
        if (step + 1) % self.accum_steps == 0 or (step + 1) == self.len_train_loader:
            global_step = epoch * self.len_train_loader + step
            
            # Gradient clipping to prevent exploding gradients
            if self.grad_clip:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                                
                params = [p for p in self.model.parameters() if p.requires_grad]
                if not self.freeze_whisper:  # We only clip gradients if Whisper is trainable
                    params += [p for p in self.whisper.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)
            
            # Log training metrics to wandb every `self.log_every_n` steps
            # Note: `self.log_every_n` should be a multiple of `self.accum_steps`
            if (step + 1) % self.log_every_n == 0:  
                self.log_train_metrics(global_step, loss, text_embeddings, audio_embeddings, logit_scale)
        
            # Optimizer step 
            # Note: scaler.step() is used for mixed precision training, only used if self.use_amp is True
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler step
            self.scheduler.step(global_step)

            # Reset gradients
            self.optimizer.zero_grad(set_to_none=True)

    
    @torch.no_grad()
    def log_train_metrics(self, step: int, loss: float, txt: torch.Tensor, aud: torch.Tensor, logit_scale: Optional[float]) -> None:
        """
        Logs training metrics to WandB.
        
        Args:
            step (int): Current training step.
            loss (float): Loss value from the training step.
            txt (torch.Tensor): Text embeddings tensor.
            aud (torch.Tensor): Audio embeddings tensor.
            logit_scale (float): Logit scale value from the model.
        """
        
        def safe_tensor_stat(t: torch.Tensor) -> float:
            """
            Ensure tensor is on CPU and detached before computing statistics.
            This avoids issues with tensors on GPU or requiring gradients.
            
            Args:
                t (torch.Tensor): Input tensor.
                
            Returns:
                float: The mean value of the tensor, safely converted to a float.
            """
            
            return t.detach().float().mean().cpu().item()

        # Retrieve gradients from model parameters
        grads = [p.grad for p in self.model.parameters() if p.grad is not None and p.requires_grad]

        # Gradient norm
        grad_norm = torch.norm(torch.stack([g.norm() for g in grads])).detach().cpu().item() if grads else 0.0

        # Percentage of NaNs in gradients (to check for exploding or vanishing gradients)
        total_grad_elems = sum(g.numel() for g in grads)
        nan_grad_elems = sum(torch.isnan(g).sum().item() for g in grads)
        nan_grad_percent = 100.0 * nan_grad_elems / total_grad_elems if total_grad_elems > 0 else 0.0

        # Current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]

        log_dict = {
            "train/loss": safe_tensor_stat(loss),
            "train/cosine_sim": safe_tensor_stat(F.cosine_similarity(aud, txt)),    # Average cosine similarity between audio and text embeddings
            "train/aud_embedding_norm": safe_tensor_stat(aud.norm(dim=1)),
            "train/txt_embedding_norm": safe_tensor_stat(txt.norm(dim=1)),
            "train/grad_norm": grad_norm,
            "train/grad_nan_percent": nan_grad_percent,
            "train/learning_rate": current_lr,
        }
        
        if self.log_logit_scale:
            log_dict["train/logit_scale"] = safe_tensor_stat(logit_scale)
            log_dict["train/temperature"] = 1.0 / safe_tensor_stat(logit_scale)
            

        wandb.log(log_dict, step=step)

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """ 
        Validates the model on the validation set and logs metrics.
        Args:
            epoch (int): Current epoch number.
        
        Returns:
            Dict: A dictionary containing validation metrics such as cosine similarity, recall, and mean average precision.
        """
        self.model.eval()
        validation_metrics = []

        for step, (audio, text, id) in enumerate(
            tqdm(self.val_loader, total=self.len_val_loader, desc=f"Validating {epoch + 1}")
        ):
            #if step > 5:  # Limit validation to first 5 batches for quick testing
                #break
            # Get data
            audio_features = audio.to(self.device, non_blocking=True)
            text_embeddings = text.to(self.device, non_blocking=True)

            # Forward pass
            if self.freeze_whisper:  # Frozen WhisperEncoder -> Just works as a feature extractor
                whisper_outputs = self.whisper.forward(audio_features)
            else:  # Trainable WhisperEncoder -> Forward pass through the model
                whisper_outputs = self.whisper(audio_features)
            
            if self.loss_name == "contrastive":
                audio_embeddings, _ = self.model(whisper_outputs)
            else: 
                audio_embeddings = self.model(whisper_outputs)
                
            # Compute validation metrics
            metrics = retrieval_metrics(audio_embeddings, text_embeddings, topk=self.cfg.metrics.topk, map_k=10)
            validation_metrics.append(metrics)
            
        # Aggregate metrics across all validation batches
        metrics = {k: np.mean([m[k] for m in validation_metrics]) for k in validation_metrics[0].keys()}
        
        # Log validation metrics to WandB
        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=(epoch + 1))

        return metrics

    def save(self, epoch: int) -> None:
        """
        Saves the model checkpoint to a specified directory.
        Args:
            epoch (int): Current epoch number.
        """
        
        # Define file path for saving the model checkpoint
        run_name = wandb.run.name if wandb.run else "default_run"
        save_dir = f"src/checkpoints/{run_name}/{self.cfg.model.pooling}_{self.cfg.optimizer.loss}_frozen_{self.cfg.model.freeze_whisper}_epoch_{epoch}.pth"
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
            
        torch.save(self.model.state_dict(), save_dir)

    def train(self):
        """
        Main training loop that iterates over epochs and steps, performing training and validation.
        """
        
        for epoch in range(self.cfg.train.epochs):
            self.model.train()
            
            # training step ------------------------
            for step, (audio, text, id) in enumerate(
                tqdm(self.train_loader, total=self.len_train_loader, desc=f"Epoch {epoch + 1}")
            ):
                #if step > 5 : 
                    #break
                try:
                    self.training_step(audio, text, epoch, step)
                except Exception as e:
                    logger.warning(f"[Step {step}] Training failed: {e}")
                
            # validation ------------------------
            try:
                val_metrics = self.validate(epoch)
            except Exception as e:
                logger.warning(f"Validation failed: {e}")

            # save model -------
            self.save(epoch)
            logger.info(f"Epoch {epoch + 1} completed. Model checkpoint saved.")



# ========== Hydra entry-point ===================================================
# Note: This is the entry-point for running the training script with Hydra configuration management.
if __name__ == "__main__":
    import hydra

    @hydra.main(version_base="1.3", config_path="conf", config_name="config")
    def main(cfg: DictConfig):
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        trainer = Trainer(cfg)
        trainer.train()

    main()
