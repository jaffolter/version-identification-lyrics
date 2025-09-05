import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.webdataset import WebDataset
from models.lie import WhisperEncoder, WhisperEncoderTrainable
from train.losses import CosineLoss, InfoNCELoss, MSECosineLoss
from utils.retrieval import retrieval_metrics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps, min_lr=1.0e-6):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def make_loader(cfg, split):
    last_shard_id = (
        cfg.data.last_shard_train
        if split == "train"
        else cfg.data.last_shard_val
        if split == "val"
        else cfg.data.last_shard_test
    )
    file = f"000000..{last_shard_id}"
    shardurl = os.path.join(cfg.data.data_dir, split, "shard-{" + file + "}.tar")
    dataset = WebDataset(shardurl, window=cfg.data.window, batch_size=cfg.data.batch_size)  # ()
    return DataLoader(
        dataset,
        batch_size=None,  # cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.data.num_workers > 0 else False,
        shuffle=False,
    )


def check_model_devices(model):
    for name, param in model.named_parameters():
        if param.device != torch.device("cuda:0"):
            raise RuntimeError(f"Parameter {name} is on {param.device}")
    for name, buffer in model.named_buffers():
        if buffer.device != torch.device("cuda:0"):
            raise RuntimeError(f"Buffer {name} is on {buffer.device}")


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        # logger.info(f"Using device: {self.device}")
        # torch.set_default_device(self.device)
        set_seed(cfg.seed)

        self.model = instantiate(cfg.model).to(self.device)
        # self.model = torch.compile(self.model, mode="reduce-overhead")

        self.freeze_whisper = cfg.model.freeze_whisper

        if self.freeze_whisper:
            self.whisper = WhisperEncoder(debug=False, pooling=self.cfg.model.pooling)  # .to(self.device)
        else:
            self.whisper = WhisperEncoderTrainable(debug=False, pooling=self.cfg.model.pooling)
        # logger.info(f"Model: {self.model.__class__.__name__} on {next(self.model.parameters()).device}")

        # logger.info(f"Model: {self.model}")

        check_model_devices(self.model)

        # logger.info(f"Whisper: {self.whisper.__class__.__name__} on {next(self.whisper.parameters()).device}")

        # check_model_devices(self.whisper)

        if self.cfg.optimizer.loss == "contrastive":
            self.loss_fn = InfoNCELoss().to(self.device)
        elif self.cfg.optimizer.loss == "cosine":
            self.loss_fn = CosineLoss().to(self.device)
        elif self.cfg.optimizer.loss == "mse":
            self.loss_fn = MSECosineLoss().to(self.device)

        params = list(self.model.parameters())
        if not self.freeze_whisper:
            params += list(self.whisper.parameters())
        self.optimizer = torch.optim.AdamW(
            # self.model.parameters(),
            params,
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.opt_betas),
            weight_decay=cfg.optimizer.weight_decay,  # None, #
            foreach=False,  # ← forces one big kernel instead of many small ops
            capturable=True,  # ← keeps it on-device, no CPU sync
            fused=True,  # (requires PT ≥ 2.2 + Ampere+ GPU, otherwise just drop)
        )

        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                assert p.device.type == "cuda", f"{p.shape} still on {p.device}"

        self.scaler = torch.amp.GradScaler("cuda", enabled=True)

        self.len_train_loader = cfg.data.total_train_samples // cfg.data.batch_size
        self.T_max = cfg.train.epochs * self.len_train_loader
        #self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.T_max, eta_min=1.0e-6)
        # self.warmup_scheduler = LambdaLR(self.optimizer, lambda step: min(1.0, step / cfg.optimizer.warmup_steps))
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.cosine_with_warmup)
        # self.scheduler = cosine_lr(self.optimizer, cfg.optimizer.lr, cfg.optimizer.warmup_steps, self.T_max)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-4,               # or higher if you're using AMP
            total_steps=30000,         # match total training steps
            pct_start=0.1,             # warmup % (10% of steps)
            anneal_strategy='cos',     # cosine annealing
            div_factor=25,             # start at max_lr / 25
            final_div_factor=1e4       # end at max_lr / 1e4
        )

        self.train_loader = make_loader(cfg, "train")
        self.val_loader = make_loader(cfg, "val")

        self.log_every_n = cfg.wandb.log_steps
        self.accum_steps = cfg.train.accumulation_steps
        self.grad_clip = cfg.train.grad_clip
        self.grad_clip_norm = cfg.train.grad_clip_max_norm

        self.log_dir = cfg.wandb.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=self.log_dir,
            # mode="disabled",
        )

        self.patience = cfg.train.early_stop_patience
        self.best_score = 0.0
        self.best_epoch = -1

        self.logit_scale = self.model.logit_scale  # assumes logit_scale is nn.Parameter
        self.logit_scale.requires_grad_(False)

    def training_step(self, audio, text, epoch, step):
        # start = perf_counter()
        audio_features = audio.to(self.device, non_blocking=True)
        text_embeddings = text.to(self.device, non_blocking=True)

        if self.freeze_whisper:
            whisper_outputs = self.whisper.forward(audio_features)
        else:
            whisper_outputs = self.whisper(audio_features)
        audio_embeddings, logit_scale = self.model(whisper_outputs)

        loss = self.loss_fn(audio_embeddings, text_embeddings, logit_scale) / self.accum_steps
        loss.backward()

        global_step = 0

        if (step + 1) % self.accum_steps == 0:
            if self.grad_clip:
                params = [p for p in self.model.parameters() if p.requires_grad]
                if self.freeze_whisper:
                    params += [p for p in self.whisper.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            if (step + 1) % self.log_every_n == 0:  # and (step % self.accum_steps == 0):
                global_step = epoch * self.len_train_loader + step
                self.log_train_metrics(global_step, loss, text_embeddings, audio_embeddings, logit_scale)

            self.optimizer.step()
            # self.scheduler.step()
            # self.scheduler(epoch * self.len_train_loader + step)

            # global_step = epoch * self.len_train_loader + step
            # self.scheduler(global_step)
            #if global_step < self.cfg.optimizer.warmup_steps:
                #self.warmup_scheduler.step()
            #else:
                #self.scheduler.step()
            self.scheduler.step()

            global_step += 1

            #if global_step == self.cfg.optimizer.warmup_steps:
                #self.logit_scale.requires_grad_(True)
                #logger.info("Logit scale is now trainable.")

            self.optimizer.zero_grad(set_to_none=True)

        # Update logit scale
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        #with torch.no_grad():
            #logit_scale.data.clamp_(0, math.log(100))

        # end = perf_counter()
        # logger.info(f"Step took {end - start:.4f} seconds")

    def training_step_amp(self, audio, text, epoch, step):
        audio_features = audio.to(self.device, non_blocking=True)
        text_embeddings = text.to(self.device, non_blocking=True)

        # torch.compiler.cudagraph_mark_step_begin()

        if self.freeze_whisper:
            whisper_outputs = self.whisper.forward(audio_features)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            if not self.freeze_whisper:
                whisper_outputs = self.whisper(audio_features)
            audio_embeddings, logit_scale = self.model(whisper_outputs)

            loss = self.loss_fn(audio_embeddings, text_embeddings, logit_scale) / self.accum_steps

        self.scaler.scale(loss).backward()

        global_step = 0

        if (step + 1) % self.accum_steps == 0 or (step + 1) == self.len_train_loader:
            if self.grad_clip:
                params = [p for p in self.model.parameters() if p.requires_grad]
                if self.freeze_whisper:
                    params += [p for p in self.whisper.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)

            if (step + 1) % self.log_every_n == 0:
                global_step = epoch * self.len_train_loader + step
                self.log_train_metrics(global_step, loss, text_embeddings, audio_embeddings, logit_scale)

            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if global_step < self.cfg.optimizer.warmup_steps:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()

            global_step += 1

            if global_step == self.cfg.optimizer.warmup_steps:
                self.logit_scale.requires_grad_(True)
                logger.info("Logit scale is now trainable.")

            self.optimizer.zero_grad(set_to_none=True)

        # Update logit scale
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            logit_scale.data.clamp_(0, math.log(100))

        # end = perf_counter()
        # logger.info(f"Step took {end - start:.4f} seconds")

    def training_step_profiler(self, audio, text, epoch, step):
        # start = perf_counter()
        with record_function("load_data"):
            audio_features = audio.to(self.device, non_blocking=True)
            text_embeddings = text.to(self.device, non_blocking=True)

        with record_function("whisper"):
            if self.freeze_whisper:
                whisper_outputs = self.whisper.forward(audio_features)
            else:
                whisper_outputs = self.whisper(audio_features)
        with record_function("model_forward"):
            audio_embeddings, logit_scale = self.model(whisper_outputs)

        with record_function("compute_loss"):
            loss = self.loss_fn(audio_embeddings, text_embeddings, logit_scale) / self.accum_steps

        with record_function("backward"):
            loss.backward()

            if (step + 1) % self.accum_steps == 0:
                if (step + 1) % self.log_every_n == 0:
                    global_step = epoch * self.len_train_loader + step
                    self.log_train_metrics(global_step, loss, text_embeddings, audio_embeddings, logit_scale)

                self.optimizer.step()
                # self.scheduler.step()
                global_step = epoch * self.len_train_loader + step
                self.scheduler(global_step)

                if global_step == self.cfg.optimizer.warmup_steps:
                    self.logit_scale.requires_grad_(True)
                    logger.info("Logit scale is now trainable.")

                self.optimizer.zero_grad(set_to_none=True)

        # end = perf_counter()
        # logger.info(f"Step took {end - start:.4f} seconds")

    @torch.no_grad()
    def log_train_metrics(self, step, loss, txt, aud, logit_scale):
        def safe_tensor_stat(t):
            return t.detach().float().mean().cpu().item()

        grads = [p.grad for p in self.model.parameters() if p.grad is not None]

        # Gradient norm
        grad_norm = torch.norm(torch.stack([g.norm() for g in grads])).detach().cpu().item() if grads else 0.0

        # Percentage of NaNs in gradients
        total_grad_elems = sum(g.numel() for g in grads)
        nan_grad_elems = sum(torch.isnan(g).sum().item() for g in grads)
        nan_grad_percent = 100.0 * nan_grad_elems / total_grad_elems if total_grad_elems > 0 else 0.0

        # Current learning rate (assuming one param group)
        current_lr = self.optimizer.param_groups[0]["lr"]

        log_dict = {
            "train/loss": safe_tensor_stat(loss),
            "train/cosine_sim": safe_tensor_stat(F.cosine_similarity(aud, txt)),
            "train/aud_embedding_norm": safe_tensor_stat(aud.norm(dim=1)),
            "train/txt_embedding_norm": safe_tensor_stat(txt.norm(dim=1)),
            "train/logit_scale": safe_tensor_stat(logit_scale),
            "train/temperature": 1.0 / safe_tensor_stat(logit_scale),
            "train/grad_norm": grad_norm,
            "train/grad_nan_percent": nan_grad_percent,
            "train/learning_rate": current_lr,
        }

        wandb.log(log_dict, step=step)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        all_audio, all_text = [], []

        for audio, text in tqdm(self.val_loader, desc="Validating"):
            audio_features = audio.to(self.device, non_blocking=True)
            text_embeddings = text.to(self.device, non_blocking=True)

            whisper_outputs = self.whisper.forward(audio_features)
            audio_embeddings, _ = self.model(whisper_outputs)

            all_audio.append(F.normalize(audio_embeddings, dim=-1))
            all_text.append(F.normalize(text_embeddings, dim=-1))

        a = torch.cat(all_audio)
        t = torch.cat(all_text)
        metrics = retrieval_metrics(a, t, topk=self.cfg.metrics.topk, map_k=10)

        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=(epoch + 1))

        return metrics

    """def early_stopping(self, val_metrics, epoch):
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
    """

    def save(self, epoch):
        save_dir = f"src/checkpoints/{self.cfg.model.pooling}_{self.cfg.optimizer.loss}_whisper_frozen_{self.cfg.model.freeze_whisper}_epoch_{epoch}_v2.pth"
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
        torch.save(self.model.state_dict(), save_dir)

    def train(self):
        for epoch in range(self.cfg.train.epochs):
            self.model.train()

            # profiler -------------------------
            if epoch == 0 and self.cfg.wandb.profile:
                os.makedirs(os.path.join(self.log_dir, "profiler"), exist_ok=True)
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=1, warmup=2, active=20, repeat=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.log_dir, "profiler")),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                    # logger.info(f"Starting epoch {epoch + 1} with profiler enabled.")
                    for step, (audio, text) in enumerate(
                        tqdm(self.train_loader, total=self.len_train_loader, desc=f"Epoch {epoch + 1}")
                    ):
                        try:
                            self.training_step_profiler(audio, text, epoch, step)
                            prof.step()  # Step the profiler after each batch
                        except Exception as e:
                            logger.warning(f"[Step {step}] Training with profiler failed: {e}")
                            continue

            elif self.cfg.model.use_amp:
                # logger.info(f"Starting epoch {epoch + 1} with AMP enabled.")
                for step, (audio, text) in enumerate(
                    tqdm(self.train_loader, total=self.len_train_loader, desc=f"Epoch {epoch + 1}")
                ):
                    try:
                        self.training_step_amp(audio, text, epoch, step)
                    except Exception as e:
                        logger.warning(f"[Step {step}] Training with AMP failed: {e}")
                        continue
            # normal step -----------------------
            else:
                # logger.info(f"Starting epoch {epoch + 1} without profiler.")
                for step, (audio, text) in enumerate(
                    tqdm(self.train_loader, total=self.len_train_loader, desc=f"Epoch {epoch + 1}")
                ):
                    try:
                        self.training_step(audio, text, epoch, step)
                    except Exception as e:
                        logger.warning(f"[Step {step}] Training failed: {e}")
                        continue
                    # logger.info(f"Step {step + 1}/{self.len_train_loader} completed.")
                    # break  # Skip logging for now

                    # if step % self.log_every_n == 0:
                    # global_step = epoch * self.len_train_loader + step
                    # self.log_train_metrics(global_step, loss, txt, aud, logit_scale)

            # validation ------------------------
            """
            try:
                val_metrics = self.validate(epoch)
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
                continue
            """
            # checkpoint & early stopping -------
            # stop = self.early_stopping(val_metrics, epoch)
            """stop = self.early_stopping(epoch)
            if stop:
                logger.warning(f"Early stopping: no cosine_sim improvement for {self.patience} epochs.")
                return
            """
            logger.info(f"Epoch {epoch + 1} completed. Saving model checkpoint.")
            self.save(epoch)


# ========== Hydra entry-point ===================================================
if __name__ == "__main__":
    import hydra

    @hydra.main(version_base="1.3", config_path="conf", config_name="config")
    def main(cfg: DictConfig):
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        trainer = Trainer(cfg)
        trainer.train()

    main()
