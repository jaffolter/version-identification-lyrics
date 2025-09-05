import os, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from data.dataset import LIEDatasetLight
from train.losses import InfoNCELoss
from utils.retrieval import retrieval_metrics

from bitsandbytes.optim import PagedAdamW8bit
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(cfg, split):
    dataset = LIEDatasetLight(
        text_emb_file=cfg.data.text_emb_file,
        metadata_file=cfg.data.metadata_file,
        split=split,
        sampling_rate=cfg.data.sr,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=(split == "train"),
    )


def train(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.model).to(device)
    loss_fn = InfoNCELoss()

    # Choose optimizer
    if cfg.train.opt_name == "paged_adamw8bit":
        optimizer = PagedAdamW8bit(
            model.parameters(), lr=cfg.train.lr, betas=tuple(cfg.train.opt_betas), weight_decay=cfg.train.weight_decay
        )
    elif cfg.train.opt_name == "lion":
        optimizer = Lion(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, betas=tuple(cfg.train.opt_betas), weight_decay=cfg.train.weight_decay
        )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.train.t0_restart, eta_min=cfg.train.lr_min)
    scaler = torch.cuda.amp.GradScaler()

    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val")

    wandb_logger = None
    if cfg.log.logger == "wandb":
        wandb_logger = wandb.init(
            project=cfg.log.project, config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.log_dir
        )

    best_hr1 = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress):
            audio = batch["audio"].to(device)
            text = batch["text_embed"].to(device)
            valid_frames = batch.get("valid_frames", None)
            if valid_frames is not None:
                valid_frames = valid_frames.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                txt, aud, logit_scale = model(audio, text, valid_frames)
                loss = loss_fn(aud, txt, logit_scale, device)

            scaler.scale(loss).backward()

            # Gradient clipping
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + step / len(train_loader))

            cosine_sim_diag = F.cosine_similarity(aud, txt).mean().item()
            total_loss += loss.item()

            if wandb_logger:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/sim": cosine_sim_diag,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

            progress.set_postfix(loss=loss.item(), cos_sim=cosine_sim_diag)

        # ─── Validation ─────────────────────────
        model.eval()
        all_txt, all_aud = [], []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                text = batch["text_embed"].to(device)
                valid_frames = batch.get("valid_frames", None)
                if valid_frames is not None:
                    valid_frames = valid_frames.to(device)

                txt, aud, *_ = model(audio, text, valid_frames)
                txt = F.normalize(txt, dim=-1)
                aud = F.normalize(aud, dim=-1)

                all_txt.append(txt)
                all_aud.append(aud)

        t = torch.cat(all_txt, dim=0)
        a = torch.cat(all_aud, dim=0)
        metrics = retrieval_metrics(a, t, topk=cfg.train.topk, map_k=10)

        if wandb_logger:
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=epoch)

        print(f"Val HR@1: {metrics['HR@1']:.4f}")

        # Save best checkpoint
        if metrics["HR@1"] > best_hr1:
            best_hr1 = metrics["HR@1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                os.path.join(cfg.log_dir, "best_checkpoint.pth"),
            )
            print("Best model saved!")


# ─── Entry Point with Hydra ─────────────────────
if __name__ == "__main__":
    import hydra

    @hydra.main(version_base="1.3", config_path="conf", config_name="config")
    def main(cfg: DictConfig):
        train(cfg)

    main()
