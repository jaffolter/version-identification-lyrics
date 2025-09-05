import torch
from omegaconf import OmegaConf
from loguru import logger
from livi.apps.audio_encoder.data.factory import load_model
from livi.apps.audio_encoder.models.whisper_encoder import WhisperEncoder
from livi.core.data.preprocessing.vocal_detector import get_cached_vocal_detector, extract_vocals
from livi.core.data.preprocessing.whisper_feature_extractor import get_cached_feature_extractor, extract_mel
from livi.core.data.utils.audio_toolbox import load_audio
import torch.nn.functional as F


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

    def inference(self, audio_path: str, get_global_embedding: bool = True) -> torch.Tensor:
        waveform = load_audio(audio_path, target_sample_rate=self.cfg.data.sr)

        # Extract vocal segments
        _, _, _, _, chunks_audio = extract_vocals(audio_path, waveform, self.vocal_detector)

        # Extract mel spectrograms
        mel = extract_mel(chunks_audio, self.feature_extractor)
        mel = mel.to(dtype=torch.float32, device=self.device)

        # Forward pass
        with torch.no_grad():
            whisper_hidden_states = self.whisper(mel)
            embeddings = F.adaptive_avg_pool1d(whisper_hidden_states.transpose(1, 2), 1).squeeze(-1)

        if get_global_embedding:
            return embeddings.mean(dim=0)

        return embeddings
