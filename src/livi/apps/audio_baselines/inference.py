import sys, os, importlib
from omegaconf import OmegaConf
import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from tqdm import tqdm
import pandas as pd
import glob
import torchaudio
from time import perf_counter
import random

from .utils import pytorch_utils, audio_utils
from livi.config import settings
from pathlib import Path
from loguru import logger

ACCEPTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")
ROOT = Path(__file__).resolve().parent


def run_inference(
    dataset: str = "covers80",
    model_name: str = "dvinet",
    path_in: str = None,
    path_out: str = None,
    config: str = None,
    checkpoint: str = None,
    device: str = "cuda",
    hop_size: float = 5.0,
    win_len: float = -1,
    inference_time: bool = False,
    truncate: int = 1,
):
    """Main inference routine, callable from CLI (Typer)."""
    # Définir les chemins par défaut si non fournis
    if path_in is None:
        path_in = str(Path(settings.AUDIO_DIR) / dataset)
    if path_out is None:
        path_out = str(Path(settings.EMBEDDINGS_DIR) / "audio_baselines" / model_name / dataset)
    if config is None:
        config = f"{ROOT}/config/{model_name}.yaml"
    if checkpoint is None:
        checkpoint = f"{ROOT}/checkpoints/{model_name}.ckpt"
    if win_len <= 0:
        win_len = None

    # Init pytorch/Fabric
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(False)
    fabric = Fabric(
        accelerator=device,
        devices=1,
        num_nodes=1,
        strategy=DDPStrategy(broadcast_buffers=False),
        precision="32",
    )
    fabric.launch()

    # Load conf
    logger.info("Load model conf...")
    conf = OmegaConf.load(config)

    # Init model
    logger.info("Init model...")

    module = importlib.import_module("livi.apps.audio_baselines.models." + conf.model.name)
    with fabric.init_module():
        model = module.Model(conf.model, sr=conf.data.samplerate)
    model = fabric.setup(model)

    # Load model
    logger.info("Load checkpoint...")
    state = pytorch_utils.get_state(model, None, None, conf, None, None, None)
    fabric.load(checkpoint, state)
    model, _, _, conf, _, _, _ = pytorch_utils.set_state(state)
    model.eval()

    # Print number of parameters
    logger.info(f"Model: {conf.model.name}")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    filenames = []
    for path, dirs, files in os.walk(path_in):
        for file in files:
            torch.cuda.empty_cache()
            # Filter audio files
            _, ext = os.path.splitext(file)
            if ext.lower() not in ACCEPTED_AUDIO_EXTENSIONS:
                continue
            # Get full filename
            fn_in = os.path.join(path, file)

            fn_out = os.path.join(path_out, os.path.relpath(fn_in, path_in))
            fn_out = os.path.splitext(fn_out)[0] + ".pt"
            path_out_dir, _ = os.path.split(fn_out)
            filenames.append([fn_in, path_out_dir, fn_out])

    avg_preprocessing = []
    avg_inference = []
    avg_total = []

    logger.info(f"Found {len(filenames)} files to process.")

    # Select a random sample of 200 tracks with seed 42
    if inference_time:
        random.seed(42)
        filenames = random.sample(filenames, 200)
        logger.info("Computing inference time on a random sample of 200 tracks.")

    with torch.inference_mode():
        for fn_in, path_out_dir, fn_out in tqdm(filenames, ascii=True, ncols=100, desc="Extract embeddings"):
            start = perf_counter()

            # Load mono audio
            x = audio_utils.load_audio(fn_in, sample_rate=model.sr, n_channels=1)

            if truncate:
                duration = x.shape[1] / model.sr
                if duration > 240:
                    x = x[:, : int(model.sr * 240)]

            if x is None:
                continue
            end = perf_counter()

            # Compute embeddings
            start2 = perf_counter()
            try:
                z = model(x, shingle_hop=hop_size, shingle_len=win_len)
                z = z.squeeze(0).cpu()

            except Exception as e:
                logger.error("ERROR")
            end2 = perf_counter()

            avg_preprocessing.append(end - start)
            avg_inference.append(end2 - start2)
            avg_total.append(end2 - start)

            # Save
            os.makedirs(path_out_dir, exist_ok=True)
            torch.save(z, fn_out)

    if inference_time:
        logger.info(
            f"Avg preprocessing time: {sum(avg_preprocessing) / len(avg_preprocessing):.2f} seconds (std: {torch.std(torch.tensor(avg_preprocessing)):.2f})"
        )
        logger.info(
            f"Avg inference time: {sum(avg_inference) / len(avg_inference):.2f} seconds (std: {torch.std(torch.tensor(avg_inference)):.2f})"
        )
        logger.info(
            f"Avg total time: {sum(avg_total) / len(avg_total):.2f} seconds (std: {torch.std(torch.tensor(avg_total)):.2f})"
        )

    logger.info("Done.")
