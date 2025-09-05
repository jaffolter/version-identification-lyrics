import os, random, numpy as np, torch, warnings
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from loguru import logger
from datetime import datetime

# ─── your modules ────────────────────────────────
from data.webdataset import WebDataset
from train.losses import InfoNCELoss
from utils.retrieval import retrieval_metrics

# optional optimisers
from bitsandbytes.optim import PagedAdamW8bit
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.profiler import profile, record_function, ProfilerActivity
from time import perf_counter

# -------------------------------------------------
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# ──────────────────────────────


# ========== helpers ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    dataset = WebDataset(shardurl, window=cfg.data.window)()
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
        # prefetch_factor=None,  # cfg.data.prefetch_factor,
        # collate_fn=collate_fn
    )


# ========== main ==============================================================
def train(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    precision = cfg.train.precision.lower()
    if precision not in ("fp16-mixed", "bf16-mixed", "fp32"):
        raise ValueError("train.precision must be one of: fp16-mixed | bf16-mixed | fp32")

    use_amp = device.type == "cuda" and precision != "fp32"
    logger.info(f"Using device: {device}, precision: {precision}, AMP enabled: {use_amp}")

    amp_dtype = {
        "fp16-mixed": torch.float16,
        "bf16-mixed": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]

    # model --------------------------------------------------------------------
    model = instantiate(cfg.model)
    if cfg.train.compile:
        try:
            model = torch.compile(model, mode="auto")
        except Exception as e:
            warnings.warn(f"torch.compile() failed: {e}; continuing without.")
    model = model.to(device)

    # data ---------------------------------------------------------------------
    train_loader, val_loader = make_loader(cfg, "train"), make_loader(cfg, "train")

    # optimiser & loss ----------------------------------------------------------------
    opt_kwargs = dict(
        lr=cfg.optimizer.lr, betas=tuple(cfg.optimizer.opt_betas), weight_decay=cfg.optimizer.weight_decay
    )
    optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    len_train_data = cfg.data.total_train_samples // cfg.data.batch_size
    T_max = cfg.train.epochs * len_train_data
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

    accumulation_steps = cfg.train.accumulation_steps

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_fn = InfoNCELoss()

    # W&B ----------------------------------------------------------------------
    os.makedirs(cfg.wandb.log_dir, exist_ok=True)
    wandb_runner = wandb.init(
        project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.wandb.log_dir
    )
    log_every_n = cfg.wandb.log_steps

    # early-stopping ----------------------------------------------
    best_score, best_epoch = 0.0, -1
    patience = cfg.train.early_stop_patience

    # ==========================================================================
    for epoch in range(cfg.train.epochs):
        # -------------------- TRAIN ------------------------------------------
        model.train()

        prog = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        # logger.info(f"Number of batches to process in epoch {epoch + 1}: {len_train_data}")

        if epoch == 0 and cfg.wandb.profile:
            # start = perf_counter()
            
                for step, (audio, text) in enumerate(prog, 1):
                    # t_step_start = perf_counter()

                    with record_function("load_data"):
                        # start = perf_counter()
                        audio_features = audio.to(device, non_blocking=True)
                        text_embeddings = text.to(device, non_blocking=True)
                        # end = perf_counter()
                        # logger.info(f"Data loaded in {end - start:.4f} sec")

                    optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        with record_function("model_forward"):
                            # start = perf_counter()
                            txt, aud, logit_scale = model(audio_features, text_embeddings)
                            # end = perf_counter()
                            # logger.info(f"Model forward pass done in {end - start:.4f} sec")

                        with record_function("compute_loss"):
                            # start = perf_counter()
                            loss = loss_fn(aud, txt, logit_scale)
                            # end = perf_counter()
                            # logger.info(f"Loss computed in {end - start:.4f} sec")

                    with record_function("backward"):
                        # start = perf_counter()
                        # scaler.scale(loss).backward()

                        # if cfg.train.grad_clip:
                        # scaler.unscale_(optimizer)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip_max_norm)

                        # scaler.step(optimizer)
                        # scaler.update()
                        # scheduler.step()
                        # end = perf_counter()
                        # logger.info(f"Backward pass done in {end - start:.4f} sec")$loss.backward()
                        loss = loss / accumulation_steps
                        loss.backward()
                        if step % accumulation_steps == 0 or step == len_train_data:
                            if cfg.train.grad_clip:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), max_norm=cfg.train.grad_clip_max_norm
                                )

                            optimizer.step()
                            scheduler.step()
                    prof.step()
            # end = perf_counter()
            # logger.info(f"Profiling took {end - start:.4f} sec")
        else:
            for step, (audio, text) in enumerate(prog, 1):
                # t_step_start = perf_counter()

                # 1. Load data -------------------------------------------------
                # start = perf_counter()
                audio_features, text_embeddings = (
                    audio.to(device, non_blocking=True),
                    text.to(device, non_blocking=True),
                )

                # end = perf_counter()
                # logger.info(f"Data loaded in {end - start:.4f} sec")

                optimizer.zero_grad(set_to_none=True)

                # 2. Forward --------------------------------------------------------------------------------
                # start = perf_counter()
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    txt, aud, logit_scale = model(audio_features, text_embeddings)
                    # end = perf_counter()
                    # logger.info(f"Model forward pass done in {end - start:.4f} sec")

                    # start = perf_counter()
                    loss = loss_fn(aud, txt, logit_scale)
                    # end = perf_counter()
                    # logger.info(f"Loss computed in {end - start:.4f} sec")

                # 3. Backward --------------------------------------------------------------------------------
                # start = perf_counter()
                # scaler.scale(loss).backward()

                # gradient clipping
                # if cfg.train.grad_clip:
                #    scaler.unscale_(optimizer)
                #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip_max_norm)

                # scaler.step(optimizer)
                # scaler.update()
                # scheduler.step()

                # end = perf_counter()
                # logger.info(f"Backward pass done in {end - start:.4f} sec")
                loss = loss / accumulation_steps  # Normalize loss for accumulation
                loss.backward()

                if step % accumulation_steps == 0 or step == len_train_data:
                    if cfg.train.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip_max_norm)

                    optimizer.step()
                    scheduler.step()

            # end = perf_counter()
            # logger.info(f"Full step took {end - start:.4f} sec")

            # 4. Logging --------------------------------------------------------------------------------
        if step % log_every_n == 0:
            # s#tart = perf_counter()
            # start = perf_counter()
            global_step = epoch * len_train_data + step

            # start = perf_counter()
            cos_sim = F.cosine_similarity(aud, txt).mean().detach().cpu().item()

            # end = perf_counter()
            # logger.info(f"Cosine similarity computed in {end - start:.4f} sec")

            grad_norm = None
            if cfg.train.grad_clip:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                grad_norm = torch.norm(torch.stack([g.norm() for g in grads])).detach().cpu().item()

            logit_scale_val = logit_scale.item() if isinstance(logit_scale, torch.Tensor) else logit_scale

            aud_norm = aud.norm(dim=1).mean().detach().cpu().item()

            txt_norm = txt.norm(dim=1).mean().detach().cpu().item()

            log_dict = {
                "train/loss": loss.detach().cpu().item(),
                "train/cosine_sim": cos_sim,
                "train/aud_embedding_norm": aud_norm,
                "train/txt_embedding_norm": txt_norm,
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }

            if grad_norm is not None:
                log_dict["train/grad_norm"] = grad_norm

            if logit_scale_val is not None:
                log_dict["train/logit_scale"] = logit_scale_val

            wandb.log(log_dict, step=global_step)

            # logger.info(
            #    f"[Step {global_step}] Loss: {loss.detach().cpu().item():.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
            # )

            # end = perf_counter()
            # logger.info(f"Logging took {end - start:.4f} sec")

            # logger.info(f"Full step took {perf_counter() - t_step_start:.4f} sec")
        # -------------------- VALIDATION --------------------------------------
        model.eval()
        prog_val = tqdm(val_loader, ncols=100, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            a_all, t_all = [], []
            for step, (audio, text) in enumerate(prog_val, 1):
                audio_features, text_embeddings = (
                    audio.to(device, non_blocking=True),
                    text.to(device, non_blocking=True),
                )

                txt, aud, _ = model(audio_features, text_embeddings)
                a_all.append(F.normalize(aud, dim=-1))
                t_all.append(F.normalize(txt, dim=-1))

        a = torch.cat(a_all)
        t = torch.cat(t_all)

        metrics = retrieval_metrics(a, t, topk=cfg.metrics.topk, map_k=10)
        prefixed_metrics = {f"val/{k}": v for k, v in metrics.items()}
        global_step = epoch * cfg.data.total_val_samples + step
        wandb.log(prefixed_metrics, step=global_step)

        # logger.info("Validation metrics:", ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))

        # checkpoint & early stop --------------------------------------------
        cosine_sim = metrics["cosine_sim"]
        improved = cosine_sim > best_score
        if improved:
            best_score, best_epoch = cosine_sim, epoch
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_cosine_sim": best_score,
            }
            # current date
            filename = f"model_epoch_{epoch}_cos_{cosine_sim:.4f}.pth"
            path = os.path.join(cfg.wandb.log_dir, "checkpoints", filename)
            os.makedirs(os.path.join(cfg.wandb.log_dir, "checkpoints"), exist_ok=True)
            torch.save(ckpt, path)

        # stop if no improvement for `patience` epochs
        if patience and (epoch - best_epoch) >= patience:
            logger.warning(f"Early stopping: no cosine_sim improvement for {patience} epochs.")
            break


# ========= Hydra entry-point ===================================================
if __name__ == "__main__":
    import hydra

    @hydra.main(version_base="1.3", config_path="conf", config_name="config")
    def main(cfg: DictConfig):
        print(cfg)
        train(cfg)

    main()
