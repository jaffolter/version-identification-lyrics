# ───────────────────────────── imports ─────────────────────────────
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch, torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from bitsandbytes.optim import PagedAdamW8bit
from lion_pytorch import Lion  # pip install lion-pytorch   (optional)
from torch.utils.data import RandomSampler, DataLoader

""
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

from data.dataset import LIEDataset, CliqueBatchSampler, LIEDatasetLight
from .losses import InfoNCELoss
from utils.retrieval import all_gather, retrieval_metrics
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True


# ───────────────────────────── LightningModule ────────────────────
class LitLIE(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = instantiate(cfg.model)
        self.loss_fn = InfoNCELoss()
        self.validation_outputs = []

    def forward(self, audio, text, valid_frames=None):
        return self.model(audio, text, valid_frames)

    """def training_step(self, batch, _):
        audio, text = batch["audio"], batch["text_embed"]
        txt, aud, logit_scale = self(audio, text)
        loss = self.loss_fn(aud, txt, logit_scale, self.device)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=audio.size(0), sync_dist=True
        )
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                {
                    "train/loss": loss.item(),
                    "global_step": self.global_step,
                }
            )
        return loss"""

    def training_step(self, batch, _):
        try:
            audio, text, valid_frames = batch["audio"], batch["text_embed"], batch["valid_frames"]
        except:
            audio, text, valid_frames = (
                batch["audio"],
                batch["text_embed"],
                None,
            )  # valid_frames is not used in this version
        txt, aud, logit_scale, alpha = self(audio, text, valid_frames)
        loss = self.loss_fn(aud, txt, logit_scale, alpha, self.device)

        # Get optimizer & LR
        if self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = 0.0  # fallback if not available yet

        # Compute additional metrics
        cosine_sim_diag = F.cosine_similarity(aud, txt).mean()  # .item()
        # grad_norm = sum(p.grad.norm().item() for p in self.parameters() if p.grad is not None)
        # grad_norm = torch.stack([p.grad.norm() for p in self.parameters() if p.grad is not None]).sum()
        # Log to terminal/progress bar/logger
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=audio.size(0), sync_dist=True
        )
        self.log("train/lr", current_lr, on_step=True, prog_bar=False, sync_dist=True)
        self.log("train/cosine_diag", cosine_sim_diag, on_step=True, prog_bar=False, sync_dist=True)
        # self.log("train/grad_norm", grad_norm, on_step=True, prog_bar=False, sync_dist=True)

        # Log to W&B (manual control)
        """if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    "train/cosine_diag": cosine_sim_diag,
                    # "train/grad_norm": grad_norm,
                    "global_step": self.global_step,
                }
            )"""

        return loss

    def validation_step(self, batch, _):
        try:
            audio, text, valid_frames = batch["audio"], batch["text_embed"], batch["valid_frames"]
        except:
            audio, text, valid_frames = (
                batch["audio"],
                batch["text_embed"],
                None,
            )
        txt, aud, _, _ = self(audio, text, valid_frames)
        txt, aud = map(lambda t: F.normalize(t, dim=-1), (txt, aud))

        self.validation_outputs.append({"a": aud, "t": txt})
        return

    def on_validation_epoch_end(self):
        a = torch.cat([o["a"] for o in self.validation_outputs], dim=0)
        t = torch.cat([o["t"] for o in self.validation_outputs], dim=0)

        if self.trainer.world_size > 1:
            a, t = all_gather(a), all_gather(t)

        metrics = retrieval_metrics(a, t, topk=self.hparams.train.topk, map_k=10)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=True, sync_dist=True)

        # if isinstance(self.logger, WandbLogger):
        # self.logger.experiment.log({f"val/{k}": v.item() for k, v in metrics.items()}, step=self.global_step)

        self.validation_outputs.clear()

    # ───────────── optimiser / scheduler factory ─────────────
    def configure_optimizers(self):
        p = self.hparams.train  # shortcut

        if p.opt_name == "paged_adamw8bit":
            optimizer = PagedAdamW8bit(
                self.parameters(), lr=p.lr, betas=tuple(p.opt_betas), weight_decay=p.weight_decay
            )
        elif p.opt_name == "lion":
            optimizer = Lion(self.parameters(), lr=p.lr, weight_decay=p.weight_decay)
        else:  # AdamW
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=p.lr, betas=tuple(p.opt_betas), weight_decay=p.weight_decay
            )

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=p.t0_restart, eta_min=p.lr_min)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ───────────────────────────── dataloader helper ──────────────────


def make_loader(cfg, split):
    print(f"Creating {split} dataset with batch size {cfg.train.batch_size}...")
    dataset = LIEDatasetLight(
        text_emb_file=cfg.data.text_emb_file,
        metadata_file=cfg.data.metadata_file,
        split=split,
        sampling_rate=cfg.data.sr,
    )

    if split == "train":
        print(f"Creating CliqueBatchSampler for {split} split...")
        """sampler = CliqueBatchSampler(
            dataset,
            batch_size=cfg.train.batch_size,
        )"""
        # sampler = RandomSampler(dataset)

        print(f"Creating DataLoader with CliqueBatchSampler for {split} split...")
        return torch.utils.data.DataLoader(
            dataset,
            # sampler=sampler,
            # num_workers=cfg.train.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
            batch_size=cfg.train.batch_size,
        )
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )"""
    else:
        print(f"Creating DataLoader for {split} split...")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            # num_workers=cfg.train.num_workers,
            # pin_memory=True,
        )
