
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """ 
    Implements a batch-wise InfoNCE loss for contrastive learning.
    This loss encourages the model to maximize the similarity between positive pairs
    (audio and text embeddings) while minimizing the similarity with negative pairs.
    
    Normally, loss is symetric (i.e., audio to text and text to audio),
    but here we only implement the audio to text direction, since the text embedding space is frozen. 
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: nn.Parameter,
    ) -> torch.Tensor:
        """
        Args:
            audio_embeds (torch.Tensor): (B, D) - audio embeddings
            text_embeds (torch.Tensor): (B, D) - text embeddings
            logit_scale (nn.Parameter): Scaling factor for logits

        Returns:
            torch.Tensor: scalar loss
        """

        assert audio_embeds.shape == text_embeds.shape, "Embeddings must be same shape"

        # Normalize features
        audio_out = F.normalize(audio_embeds, dim=-1)
        text_out = F.normalize(text_embeds, dim=-1)

        # Compute logits_per_audio: the dot product of audio and text embeddings
        # scaled by logit_scale
        logits_per_audio = torch.matmul(audio_out, text_out.T) * logit_scale
        
        # targets: the indices of the positive pairs (e.g. the diagonal of the matrix)
        targets = torch.arange(audio_out.size(0), device=audio_out.device)

        # Compute the InfoNCE loss (cross-entropy loss)
        loss = F.cross_entropy(logits_per_audio, targets)

        return loss


class CosineLoss(nn.Module):
    """ 
    Implements a batch-wise Cosine Similarity loss.
    This loss encourages the model to maximize the cosine similarity between
    audio and text embeddings.
    """
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(
        self,
        audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        _unused: Optional[nn.Parameter] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_embeds (torch.Tensor): (B, D) - audio embeddings
            text_embeds (torch.Tensor): (B, D) - text embeddings
        Returns:
            torch.Tensor: scalar loss
        """

        assert audio_embeds.shape == text_embeds.shape, "Embeddings must be same shape"

        # Normalize features
        audio_out = F.normalize(audio_embeds, dim=-1)
        text_out = F.normalize(text_embeds, dim=-1)

        # Targets: the indices of the positive pairs
        target = torch.ones(audio_out.size(0), device=audio_out.device)
        
        # Compute the Cosine Embedding Loss
        loss = self.loss_fn(audio_out, text_out, target)
        
        return loss


class MSECosineLoss(nn.Module):
    """
    Combines two loss components to align audio and text embeddings:

    1. MSE Loss:
       Minimizes the difference between the pairwise cosine similarity matrices 
       of the audio and text embeddings within a batch. This encourages the 
       structure of the audio embedding space to mirror that of the text space.

    2. Cosine Similarity Loss:
       Encourages each individual audio embedding to be closely aligned 
       with its corresponding text embedding using cosine similarity.

    This combined loss helps ensure both local alignment (pairwise) and 
    global structural similarity between the two modalities.
    """
    def __init__(self) -> None:
        super().__init__()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        _unused: Optional[nn.Parameter] = None,
    ) -> torch.Tensor:
        """"
        Args:
            audio_embeds (torch.Tensor): (B, D) - audio embeddings
            text_embeds (torch.Tensor): (B, D) - text embeddings

        Returns:
            torch.Tensor: scalar loss
        """
        assert audio_embeds.shape == text_embeds.shape, "Embeddings must be same shape"

        # Normalize features
        audio_out = F.normalize(audio_embeds, dim=-1)
        text_out = F.normalize(text_embeds, dim=-1)

        # 1. Cosine Similarity Loss: Audio to text alignment
        # Force each audio embedding to be aligned with its
        # corresponding text embedding via cosine similarity
        target = torch.ones(audio_out.size(0), device=audio_out.device)
        loss = self.cosine_loss(audio_out, text_out, target)

        # 2. MSE Loss: Structure preservation
        # Minimize the difference between the pairwise cosine similarity matrix of audio 
        # and that of text (i.e., match audio space to the geometry of the text space)
        audio_similarity = torch.matmul(audio_out, audio_out.T)
        text_similarity = torch.matmul(text_out, text_out.T)

        mse_loss = self.mse_loss(audio_similarity, text_similarity)

        alpha = 0.5  # Weighting factor for the two losses
        loss = alpha * loss + (1 - alpha) * mse_loss
        
        return loss
