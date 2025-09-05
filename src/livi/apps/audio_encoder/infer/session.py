import torch
from omegaconf import OmegaConf
from loguru import logger
from livi.apps.audio_encoder.data.factory import load_model
from livi.apps.audio_encoder.models.whisper_encoder import WhisperEncoder
from livi.core.data.preprocessing.vocal_detector import get_cached_vocal_detector, extract_vocals
from livi.core.data.preprocessing.whisper_feature_extractor import get_cached_feature_extractor, extract_mel
from livi.core.data.utils.audio_toolbox import load_audio
from typing import List, Optional

from livi.utils.time import record_time
from loguru import logger
import numpy as np
from pathlib import Path
from typing import Dict


class Session:
    """
    Main class to run inference on the audio model.
    Given the path of an audio file, it :
    - loads the audio
    - extract vocal chunks
    - extract mel spectrograms via Whisper feature extractor
    - Run a pass through the audio encoder
    - Outputs the audio embeddings aligned with lyrics-informed embeddings
    """

    def __init__(self, checkpoint_path: str, config_path: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = OmegaConf.load(config_path)

        # Model
        self.model = load_model(self.cfg, checkpoint_path, self.device)
        self.whisper = WhisperEncoder(
            model_name=self.cfg.model.whisper_model_name,
            device=self.device,
            compile=self.cfg.model.compile,
        )

        # Preprocessing
        self.vocal_detector = get_cached_vocal_detector(
            vocal_threshold=self.cfg.data.vocal_threshold,
            mean_vocalness_threshold=self.cfg.data.mean_vocalness_threshold,
            sample_rate=self.cfg.data.sr,
            chunk_sec=self.cfg.data.chunk_sec,
            max_total_pad_sec=self.cfg.data.max_total_pad_sec,
        )
        self.feature_extractor = get_cached_feature_extractor(
            sample_rate=self.cfg.data.sr,
            model_name=self.cfg.model.whisper_model_name,
        )

    def get_model_size(self):
        """
        Returns the number of parameters of the model.
        """
        nb_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters in the model: {nb_params}")
        logger.info(f"Model size: {nb_params / 1e6:.2f}M parameters")

    def inference(self, audio_path: str, get_global_embedding: bool = True) -> torch.Tensor:
        waveform = load_audio(audio_path, target_sample_rate=self.cfg.data.sr)

        is_vocal, mean_vocalness, _, _, chunks_audio = extract_vocals(audio_path, waveform, self.vocal_detector)

        if not is_vocal:
            logger.warning(f"Audio file {audio_path} is detected as non-vocal: {mean_vocalness:.4f}")
            return

        # Extract mel spectrograms
        mel = extract_mel(chunks_audio, self.feature_extractor)
        mel = mel.to(dtype=torch.float32, device=self.device)

        # Forward pass
        with torch.no_grad():
            whisper_hidden_states = self.whisper(mel)
            embeddings = self.model(whisper_hidden_states)

        if get_global_embedding:
            return embeddings.mean(dim=0, keepdim=True)

        return embeddings

    def estimate_inference_time(
        self,
        audio_paths: List[str],
        start_after: Optional[int] = 5,
    ) -> None:
        """
        Estimate the inference time for a random sample of tracks.
        Args:
            audio_paths (List[str]): List of audio file paths to estimate inference time for.
            start_after (int): Number of tracks to wait before starting the timer
            (first inference steps are longer than after with torch.compile)
        """
        pre_times, inf_times, total_times = [], [], []
        for idx, audio_path in enumerate(audio_paths):
            with record_time(total_times, idx, start_after):
                with record_time(pre_times, idx, start_after):
                    waveform = load_audio(audio_path, target_sample_rate=self.cfg.data.sr)

                    # Extract vocal segments
                    is_vocal, mean_vocalness_score, _, _, chunks_audio = extract_vocals(
                        audio_path, waveform, self.vocal_detector
                    )
                    if not is_vocal:
                        logger.warning(
                            f"{audio_path} not enough vocal content (mean vocalness={mean_vocalness_score:.3f}) → passing."
                        )
                        continue
                    # Extract mel spectrograms
                    mel = extract_mel(chunks_audio, self.feature_extractor)
                    mel = mel.to(dtype=torch.float32, device=self.device)

                with record_time(inf_times, idx, start_after):
                    # Forward pass
                    with torch.no_grad():
                        whisper_hidden_states = self.whisper(mel)
                        embeddings = self.model(whisper_hidden_states)

        def mean_std(xs):
            return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))

        pre_mean, pre_std = mean_std(pre_times)
        inf_mean, inf_std = mean_std(inf_times)
        tot_mean, tot_std = mean_std(total_times)

        logger.info(f"Preproc: {pre_mean:.4f}s (±{pre_std:.4f})")
        logger.info(f"Infer  : {inf_mean:.4f}s (±{inf_std:.4f})")
        logger.info(f"Total  : {tot_mean:.4f}s (±{tot_std:.4f})")


# -----------------------------
# High-level runner functions
# -----------------------------


def run_inference_single(
    checkpoint_path: Path,
    config_path: Path,
    audio_path: Path,
    *,
    get_global_embedding: bool = True,
) -> np.ndarray:
    """
    Run inference on a single audio file.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the audio model checkpoint.
    config_path : Path
        Path to the Hydra/OmegaConf config used by the model.
    audio_path : Path
        Path to the input audio file (.mp3, .wav, etc.).
    get_global_embedding : bool, default True
        If True, returns a single vector (mean pooling over chunks). If False,
        returns chunk-level embeddings (N_chunks, D).

    Returns
    -------
    np.ndarray
        The embedding array.
    """
    session = Session(str(checkpoint_path), str(config_path))
    embedding = session.inference(str(audio_path), get_global_embedding=get_global_embedding)

    # Convert to numpy
    if embedding is None:
        return None  # or raise ValueError("No vocals detected...")
    else:
        return embedding.detach().cpu().numpy()


def run_inference(
    checkpoint_path: Path,
    config_path: Path,
    audio_dir: Path,
    path_out: Optional[Path] = None,
    get_global_embedding: bool = True,
) -> np.ndarray:
    """
    Run audio encoder inference on all audio files in a directory and save
    a mapping {basename -> embedding} to disk.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the audio model checkpoint.
    config_path : Path
        Path to the OmegaConf YAML
    audio_dir : Path
        Directory to scan recursively for audio files.
    path_out : Path, optional
        Destination .npz path.
    get_global_embedding : bool, default True
        If True, returns a single vector (mean pooling over chunks). If False,

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from file stem (basename without extension) to its embedding array.
        Files for which no embedding is produced (e.g., no vocals) are skipped.
    """
    session = Session(str(checkpoint_path), str(config_path))
    path_out.parent.mkdir(parents=True, exist_ok=True)

    embeddings: Dict[str, np.ndarray] = {}
    errors: List[str] = []
    for audio_path in sorted(audio_dir.glob("**/*.mp3")):
        torch.cuda.empty_cache()
        embedding = session.inference(str(audio_path), get_global_embedding=get_global_embedding)
        filename = audio_path.stem
        if embedding is None:
            errors.append(str(audio_path))
            logger.warning(f"Skipping {audio_path} (no embedding produced)")
        else:
            embeddings[filename] = embedding.detach().cpu().numpy()

    if errors:
        logger.warning(f"Skipped {len(errors)} files (no embedding produced). Examples:\n" + "\n".join(errors[:5]))

    # Save in .npz
    np.savez(path_out, **embeddings)
    return embeddings


def run_estimate_time(
    checkpoint_path: Path,
    config_path: Path,
    audio_dir: Path,
    *,
    sample_size: int = 200,
    start_after: int = 5,
    seed: int = 42,
) -> None:
    """
    Estimate average preprocessing/inference/total times over a sample of files.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the audio model checkpoint.
    config_path : Path
        Path to the Hydra/OmegaConf config used by the model.
    audio_dir : Path
        Directory to find audio files (e.g., 'data/').
    sample_size : int, default 200
        Number of files to sample for timing.
    start_after : int, default 5
        Warm-up iterations to exclude from averages.
    seed : int, default 42
        RNG seed for sampling.
    """
    files = sorted(audio_dir.glob("**/*.mp3"))
    if not files:
        raise FileNotFoundError(f"No files matched glob: {audio_dir}")

    rng = np.random.default_rng(seed)
    if len(files) > sample_size:
        files = list(rng.choice(files, size=sample_size, replace=False))
    else:
        files = list(files)

    logger.info(f"Timing on {len(files)} files (warm-up skip: {start_after}).")
    session = Session(str(checkpoint_path), str(config_path))

    # Log model size
    session.get_model_size()

    # Estimate inference time
    session.estimate_inference_time([str(p) for p in files], start_after=start_after)
