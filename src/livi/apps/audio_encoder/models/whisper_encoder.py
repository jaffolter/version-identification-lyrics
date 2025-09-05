from typing import Optional
from loguru import logger

import torch
import torch.nn.functional as F
import contextlib

from torch import Tensor
from transformers import WhisperModel


# --------------------- Whisper Encoder (Frozen) ---------------------
# Implements the forward pass of Whisper encoder (frozen), serving as a feature extractor.


class WhisperEncoder:
    """
    Frozen Whisper encoder (no decoder), used as a feature extractor.

    What it does
    ------------
    - Takes precomputed log-Mel spectrograms (not raw audio).
      Input should be produced by `WhisperProcessor(...).feature_extractor`:
        shape = (B, 80, T)
        B = batch size
        80 = Mel bins (fixed for Whisper)
        T = time frames
    - Runs them through the Whisper encoder to obtain frame-level hidden states.

    Output
    ------
    Tensor of shape (B, L, C):
        B = batch size
        L = encoder sequence length
        C = embedding dimension of the Whisper encoder
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        device: Optional[str] = "cuda:0",
        compile: Optional[bool] = False,
    ):
        """
        Args:
            model_name (str): Name of the Whisper model to load from Hugging Face Hub.
            device (str | None): Device to run on. If None, picks "cuda:0" if available, else "cpu".
        """
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        # Load Whisper and keep only the encoder
        self.model = WhisperModel.from_pretrained(model_name).to(self.device).encoder

        # Freeze encoder parameters & set eval mode (we don't train this component)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # Optional: compile for faster inference (PyTorch ≥ 2.0)
        if compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile disabled for Whisper encoder: {e}")

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input log-Mel features of shape (B, 80, T).

        Returns:
            Tensor: Frame-level hidden states of shape (B, L, C).
        """
        # AMP gives a nice speedup on GPU; skip autocast on CPU for safety
        use_autocast = self.device.type == "cuda"
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()

        with torch.no_grad(), autocast_ctx:
            hidden_states = self.model(x).last_hidden_state  # (B, L, C)
            return hidden_states.detach().float()
