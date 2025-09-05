import torch
import torch.nn as nn
import torch.nn.functional as F


class MSECosineLoss(nn.Module):
    """
    MSE–Cosine Loss for training the audio encoder in LIVI.

    The training objective contains two components: L_total = α * L_cos + (1 - α) * L_MSE
    - Instance-level alignment: each audio embedding a_i should be close
    to its paired lyrics embedding t_i under cosine similarity.
    - Geometry preservation: the similarity structure of the audio space
    should reflect that of the fixed lyrics-informed space.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """
        Args:
            alpha (float): Weight of the cosine alignment term L_cos.
            (1 - alpha) is applied to the geometry term L_MSE.
            Set to 0.5 by default.
        """
        super().__init__()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, audio_embeds: torch.Tensor, lyrics_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined loss.

        Args:
            audio_embeds (torch.Tensor): (B, D) audio embeddings a_i
            lyrics_embeds  (torch.Tensor): (B, D) target lyrics embeddings t_i

        Returns:
            torch.Tensor: scalar loss, L_total
        """
        if audio_embeds.shape != lyrics_embeds.shape:
            raise ValueError("Audio and lyrics embeddings must have the same shape.")

        # Normalize embeddings
        audio_out = F.normalize(audio_embeds, dim=-1)
        lyrics_out = F.normalize(lyrics_embeds, dim=-1)

        # (1) Instance-level alignment: L_cos
        target = torch.ones(audio_out.size(0), device=audio_out.device)
        L_cos = self.cosine_loss(audio_out, lyrics_out, target)

        # (2) Geometry preservation: L_MSE
        audio_similarity = torch.matmul(audio_out, audio_out.T)
        lyrics_similarity = torch.matmul(lyrics_out, lyrics_out.T)
        L_mse = self.mse_loss(audio_similarity, lyrics_similarity)

        # (3) Weighted combination
        return self.alpha * L_cos + (1 - self.alpha) * L_mse
