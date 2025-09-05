import os, random, numpy as np, torch, warnings
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from loguru import logger
from datetime import datetime

from data.webdataset import WebDataset
from train.losses import InfoNCELoss
from utils.retrieval import retrieval_metrics

from bitsandbytes.optim import PagedAdamW8bit
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(cfg, split):
    last_shard_id = (
        cfg.data.last_shard_train if split == "train"
        else cfg.data.last_shard_val if split == "val"
        else cfg.data.last_shard_test
    )
    file = f"000000..{last_shard_id}"
    shardurl = os.path.join(cfg.data.data_dir, split, "shard-{" + file + "}.tar")
    dataset = WebDataset(shardurl, window=cfg.data.window)()
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        set_seed(cfg.seed)

        self.model = instantiate(cfg.model).to(self.device)
        self.loss_fn = InfoNCELoss().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = cfg.optimizer.lr,
            betas = tuple(cfg.optimizer.opt_betas),
            weight_decay = cfg.optimizer.weight_decay
        )

        self.len_train_loader = cfg.data.total_train_samples // cfg.data.batch_size
        self.T_max = cfg.train.epochs * self.len_train_loader
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.T_max)

        self.train_loader = make_loader(cfg, "train")
        self.val_loader = make_loader(cfg, "val")

        self.log_every_n = cfg.wandb.log_steps
        self.accum_steps = cfg.train.accumulation_steps
        self.grad_clip = cfg.train.grad_clip
        self.grad_clip_norm = cfg.train.grad_clip_max_norm

        self.log_dir = cfg.wandb.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True), dir=self.log_dir)
        
        self.patience = cfg.train.early_stop_patience
        self.best_score = 0.0
        self.best_epoch = -1

    def training_step(self, audio, text, step):
        audio_features = audio.to(self.device, non_blocking=True)
        text_embeddings = text.to(self.device, non_blocking=True)
        txt, aud, logit_scale = self.model(audio_features, text_embeddings)
        loss = self.loss_fn(aud, txt, logit_scale) / self.accum_steps
        loss.backward()

        if (step % self.accum_steps == 0) or (step == self.len_train_loader):
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return loss, txt, aud, logit_scale
    
    def training_step_profiler(self, audio, text, step):
        with record_function("load_data"):
            audio_features = audio.to(self.device, non_blocking=True)
            text_embeddings = text.to(self.device, non_blocking=True)
        
        with record_function("model_forward"):
            txt, aud, logit_scale = self.model(audio_features, text_embeddings)
        
        with record_function("compute_loss"):
            loss = self.loss_fn(aud, txt, logit_scale) / self.accum_steps
        
        with record_function("backward"):
            loss.backward()

            if (step % self.accum_steps == 0) or (step == self.len_train_loader):
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                    
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        return loss, txt, aud, logit_scale

    @torch.no_grad()
    def log_train_metrics(self, step, loss, txt, aud, logit_scale):
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        grad_norm = torch.norm(torch.stack([g.norm() for g in grads])).detach().cpu().item()

        log_dict = {
            "train/loss": loss.detach().cpu().item(),
            "train/cosine_sim": F.cosine_similarity(aud, txt).mean().detach().cpu().item(),
            "train/aud_embedding_norm": aud.norm(dim=1).mean().detach().cpu().item(),
            "train/txt_embedding_norm": txt.norm(dim=1).mean().detach().cpu().item(),
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "train/logit_scale": logit_scale.detach().cpu().item(),
            "train/grad_norm": grad_norm,
        }
        wandb.log(log_dict, step=step)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        all_audio, all_text = [], []

        for audio, text in tqdm(self.val_loader, desc="Validating"):
            audio = audio.to(self.device, non_blocking=True)
            text = text.to(self.device, non_blocking=True)
            txt, aud, _ = self.model(audio, text)
            all_audio.append(F.normalize(aud, dim=-1))
            all_text.append(F.normalize(txt, dim=-1))

        a = torch.cat(all_audio)
        t = torch.cat(all_text)
        metrics = retrieval_metrics(a, t, topk=self.cfg.metrics.topk, map_k=10)
        
        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=(epoch + 1))
        
        return metrics

    def early_stopping(self, val_metrics, epoch):
        cosine_sim = val_metrics["cosine_sim"]
        improved = cosine_sim > self.best_score
        if improved:
            self.best_score, self.best_epoch = cosine_sim, epoch
            ckpt = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_cosine_sim": self.best_score,
            }

            filename = f"model_epoch_{epoch}_cos_{cosine_sim:.4f}.pth"
            path = os.path.join(self.log_dir, "checkpoints", filename)
            os.makedirs(os.path.join(self.log_dir, "checkpoints"), exist_ok=True)
            torch.save(ckpt, path)

        # stop if no improvement for `patience` epochs
        if self.patience and (epoch - self.best_epoch) >= self.patience:
            return True
        
        return False
        
    def train(self):
        for epoch in range(self.cfg.train.epochs):
            self.model.train()
            
            # profiler -------------------------
            if epoch == 0 and self.cfg.wandb.profile:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=1, warmup=2, active=8),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.log_dir, "profiler")),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                
                    for step, (audio, text) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"), 1):
                        loss, txt, aud, logit_scale = self.training_step_profiler(audio, text, step)

                        if step % self.log_every_n == 0:
                            global_step = epoch * self.len_train_loader + step
                            self.log_train_metrics(global_step, loss, txt, aud, logit_scale)
            
            # normal step -----------------------
            else : 
                for step, (audio, text) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"), 1):
                    loss, txt, aud, logit_scale = self.training_step(audio, text, step)

                    if step % self.log_every_n == 0:
                        global_step = epoch * self.len_train_loader + step
                        self.log_train_metrics(global_step, loss, txt, aud, logit_scale)

            # validation ------------------------                  
            val_metrics = self.validate(epoch)

            # checkpoint & early stopping -------
            stop = self.early_stopping(val_metrics, epoch)
            if stop : 
                logger.warning(f"Early stopping: no cosine_sim improvement for {self.patience} epochs.")
                return
            
               


# ========== Hydra entry-point ===================================================
if __name__ == "__main__":
    import hydra

    @hydra.main(version_base="1.3", config_path="conf", config_name="config")
    def main(cfg: DictConfig):
        trainer = Trainer(cfg)
        trainer.train()

    main()
