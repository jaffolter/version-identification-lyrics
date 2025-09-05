import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from livi.apps.audio_encoder.models.projection import Projection
from livi.apps.audio_encoder.models.attention_pooling import AttentionPooling

# ——————————————————— LIVI Audio Encoder ——————————————————————
# Implements the trainable part of the audio encoder, processes hidden states extracted from frozen
# Whisper Encoder via a learnable attention-pooling layer, followed by a projection MLP.


class LiviAudioEncoder(nn.Module):
    """
    This module takes frame-level representations from a frozen Whisper encoder
    and maps them into the target lyrics-informed embedding space.

    Workflow:
        Input: audio features of shape (B, T, C)
          - B = batch size
          - T = sequence length (frames)
          - C = Whisper encoder dimension (e.g., 1280)

        1. AttentionPooling:
            - Introduces a learnable [CLS] token.
            - Aggregates sequence information into a single vector.

        2. Projection (MLP):
            - Projects pooled embedding into target embedding space (e.g., 768-dim).

        3. Normalization:
            - L2-normalizes the output to unit length.

        Output: audio embedding of shape (B, dim_embed)
    """

    def __init__(
        self,
        dim_whisper: int = 1280,
        dim_hiddens: List[int] = [2048, 1024],
        dim_embed: int = 768,
        num_heads: int = 1,
        mlp_ratio: int = 2,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_scale: float = 1e-4,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initialize the LiviAudioEncoder.

        Args:
            dim_whisper (int, default=1280): Input dimension from Whisper encoder.
            dim_hiddens (List[int], default=[2048, 1024]): Hidden layer dimensions for projection MLP.
            dim_embed (int, default=768): Output embedding dimension.
            num_heads (int, default=1): Number of attention heads in pooling.
            mlp_ratio (int, default=2): Expansion ratio for hidden dim in pooling MLP.
            qkv_bias (bool, default=False): Whether to use bias in QKV projections.
            qk_scale (float, optional): Override scaling factor for QK dot product.
            drop (float, default=0.0): Dropout rate for projections.
            attn_drop (float, default=0.0): Dropout rate for attention weights.
            init_scale (float, default=1e-4): Initialization scale for residual scaling params.
            device (torch.device | None): Device to place the module on. Defaults to CUDA if available.
        """
        super().__init__()

        # Attention pooling to condense sequence embeddings into CLS embedding
        self.pooling = AttentionPooling(
            dim=dim_whisper,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            init_scale=init_scale,
        )

        # Projection MLP to map CLS embedding → target embedding space
        self.audio_proj = Projection(d_in=dim_whisper, d_out=dim_embed, hidden=dim_hiddens)

        # Device assignment
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio (Tensor): Input tensor of shape (B, T, C),
                where B = batch size, T = sequence length, C = Whisper dim.

        Returns:
            Tensor: Normalized audio embeddings of shape (B, dim_embed).
        """
        # Aggregate sequence into CLS embedding
        audio = self.pooling(audio)  # (B, C)

        # Project into target embedding space
        audio_embedding = self.audio_proj(audio)  # (B, dim_embed)

        # Normalize for cosine similarity
        audio_embedding = F.normalize(audio_embedding, p=2, dim=-1)

        return audio_embedding
