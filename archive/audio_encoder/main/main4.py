import hydra, lightning as L, torch
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from train.trainer import LitLIE, make_loader

torch.set_float32_matmul_precision("high")  # <─ matmul hint


# ──────────────────────────────────────────────────────────────
@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    model = LitLIE(cfg)

    if cfg.extras.compile:
        model = torch.compile(model)  # one single compile() call

    train_dl = make_loader(cfg, "train")
    val_dl = make_loader(cfg, "val")
    # print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    # ── logger ────────────────────────────────────────────────
    logger = None
    if cfg.log.logger == "wandb":
        logger = WandbLogger(
            project=cfg.log.project,
            save_dir=cfg.log_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── callbacks ────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(monitor="val/HR@1", mode="max", save_top_k=3),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ── trainer ──────────────────────────────────────────────
    trainer = L.Trainer(
        enable_progress_bar=True,
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.train.epochs,
        default_root_dir=cfg.log_dir,
        **cfg.trainer,
    )

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
