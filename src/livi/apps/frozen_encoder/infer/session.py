import torch
from omegaconf import OmegaConf
from loguru import logger
from livi.core.data.preprocessing.vocal_detector import get_cached_vocal_detector, extract_vocals
from livi.core.data.utils.audio_toolbox import load_audio

from livi.apps.frozen_encoder.models.text_encoder import _get_cached_text_encoder, encode_text
from livi.apps.frozen_encoder.models.transcriber import _get_cached_transcriber, transcribe
from typing import List, Optional

from livi.utils.time import record_time
from loguru import logger
import numpy as np
from pathlib import Path
from typing import Dict


class Session:
    """
    Main class to run inference on the frozen encoder.
    Given the path of an audio file, it :
    - loads the audio
    - extract vocal chunks
    - extract mel spectrograms via Whisper feature extractor
    - Transcribe via Whisper
    - Compute text embeddings via pre-trained multilingual text encoder
    - Outputs lyrics-informed embeddings
    """

    def __init__(self, config_path: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = OmegaConf.load(config_path)

        # Transcriber / Text Encoder
        self.transcriber = _get_cached_transcriber(
            model_name=self.cfg.transcriber.model_name,
            device=self.device,
            dtype_fp16_on_cuda=torch.float16 if self.device.type == "cuda" else torch.float32,
            sampling_rate=self.cfg.data.sr,
            num_beams=self.cfg.transcriber.num_beams,
            condition_on_prev_tokens=self.cfg.transcriber.condition_on_prev_tokens,
            compression_ratio_threshold=self.cfg.transcriber.compression_ratio_threshold,
            temperature=self.cfg.transcriber.temperature,
            logprob_threshold=self.cfg.transcriber.logprob_threshold,
            return_timestamps=self.cfg.transcriber.return_timestamps,
            remove_phrases=self.cfg.transcriber.remove_phrases,
            repeat_threshold=self.cfg.transcriber.repeat_threshold,
            min_words_per_chunk=self.cfg.transcriber.min_words_per_chunk,
        )
        self.text_encoder = _get_cached_text_encoder(
            model_name=self.cfg.text_encoder.model_name,
            chunking=self.cfg.text_encoder.chunking,
        )
        # Vocal Segments Extraction
        self.vocal_detector = get_cached_vocal_detector(
            vocal_threshold=self.cfg.data.vocal_threshold,
            mean_vocalness_threshold=self.cfg.data.mean_vocalness_threshold,
            sample_rate=self.cfg.data.sr,
            chunk_sec=self.cfg.data.chunk_sec,
            max_total_pad_sec=self.cfg.data.max_total_pad_sec,
        )

    def inference(self, audio_path: str) -> torch.Tensor:
        waveform = load_audio(audio_path, target_sample_rate=self.cfg.data.sr)

        # Extract vocal segments
        is_vocal, mean_vocalness_score, _, _, chunks_audio = extract_vocals(audio_path, waveform, self.vocal_detector)
        if not is_vocal:
            logger.warning(
                f"{audio_path} not enough vocal content (mean vocalness={mean_vocalness_score:.3f}) → passing."
            )
            return

        with torch.no_grad():
            # Transcribe
            transcriptions = transcribe(
                chunks_audio, translate=self.cfg.transcriber.translate, transcriber=self.transcriber
            )
            if self.cfg.text_encoder.chunking:
                inputs = [x for x in transcriptions[0] if x]
            else:
                inputs = transcriptions[-1]

            embeddings = encode_text(
                inputs,
                text_encoder=self.text_encoder,
                model_name=self.cfg.text_encoder.model_name,
                chunking=self.cfg.text_encoder.chunking,
                batch_size=self.cfg.text_encoder.batch_size,
                get_single_embedding=self.cfg.text_encoder.get_single_embedding,
            )

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
        pre_times, transc_times, encoding_times, total_times = [], [], [], []

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

                with record_time(transc_times, idx, start_after):
                    # Transcribe with Whisper
                    transcriptions = transcribe(
                        chunks_audio, translate=self.cfg.transcriber.translate, transcriber=self.transcriber
                    )
                    if self.cfg.text_encoder.chunking:
                        inputs = [x for x in transcriptions[0] if x]
                    else:
                        inputs = transcriptions[-1]

                with record_time(encoding_times, idx, start_after):
                    # Encode with text encoder
                    embeddings = encode_text(
                        inputs,
                        text_encoder=self.text_encoder,
                        model_name=self.cfg.text_encoder.model_name,
                        chunking=self.cfg.text_encoder.chunking,
                        batch_size=self.cfg.text_encoder.batch_size,
                        get_single_embedding=self.cfg.text_encoder.get_single_embedding,
                    )

        def mean_std(xs):
            return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))

        pre_mean, pre_std = mean_std(pre_times)
        transc_mean, transc_std = mean_std(transc_times)
        encoding_mean, encoding_std = mean_std(encoding_times)
        tot_mean, tot_std = mean_std(total_times)

        logger.info(f"Preproc: {pre_mean:.4f}s (±{pre_std:.4f})")
        logger.info(f"Transc : {transc_mean:.4f}s (±{transc_std:.4f})")
        logger.info(f"Encoding: {encoding_mean:.4f}s (±{encoding_std:.4f})")
        logger.info(f"Total  : {tot_mean:.4f}s (±{tot_std:.4f})")


# ---------------------------------
# High-level runner helper methods
# ---------------------------------
def run_inference(
    config_path: Path,
    audio_dir: Path,
    path_out: Optional[Path] = None,
) -> np.ndarray:
    """
    Run frozen-encoder inference on all audio files in a directory and save
    a mapping {basename -> embedding} to disk.

    Parameters
    ----------
    config_path : Path
        Path to the OmegaConf YAML
    audio_dir : Path
        Directory to scan recursively for audio files.
    path_out : Path, optional
        Destination .npz path.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from file stem (basename without extension) to its embedding array.
        Files for which no embedding is produced (e.g., no vocals) are skipped.
    """
    session = Session(str(config_path))
    path_out.parent.mkdir(parents=True, exist_ok=True)

    embeddings: Dict[str, np.ndarray] = {}
    for audio_path in sorted(audio_dir.glob("**/*.mp3")):
        emb = session.inference(str(audio_path))
        filename = audio_path.stem
        embeddings[filename] = emb

    # Save in .npz
    np.savez(path_out, **embeddings)
    return embeddings


def run_inference_single(
    config_path: Path,
    audio_path: Path,
) -> np.ndarray:
    """
    Run frozen-encoder inference on a single audio file.

    Parameters
    ----------
    config_path : Path
        Path to Hydra/OmegaConf config (contains data/preproc settings).
    audio_path : Path
        Path to the input audio file.

    Returns
    -------
    np.ndarray
        Embedding array.
    """
    session = Session(str(config_path))
    return session.inference(str(audio_path))


def run_estimate_time(
    config_path: Path,
    audio_dir: Path,
    *,
    sample_size: int = 200,
    start_after: int = 5,
    seed: int = 42,
) -> None:
    """
    Estimate average preprocessing/transcription/text-encoding/total times.

    Parameters
    ----------
    checkpoint_path : Path
        Placeholder for symmetry (unused).
    config_path : Path
        Path to Hydra/OmegaConf config.
    audio_dir : Path
        Directory where audio files live; recursively scans for *.mp3.
    sample_size : int, default 200
        Random sample size.
    start_after : int, default 5
        Warm-up iterations to skip.
    seed : int, default 42
        RNG seed for sampling.
    """
    files = sorted(audio_dir.glob("**/*.mp3"))
    if not files:
        raise FileNotFoundError(f"No files found under: {audio_dir}")

    rng = np.random.default_rng(seed)
    if len(files) > sample_size:
        files = list(rng.choice(files, size=sample_size, replace=False))
    else:
        files = list(files)

    logger.info(f"Timing on {len(files)} files (warm-up skip: {start_after}).")
    session = Session(str(config_path))
    session.estimate_inference_time([str(p) for p in files], start_after=start_after)
