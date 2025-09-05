import math
import os
from typing import List, Optional

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from loguru import logger
from timm.layers import trunc_normal_
from torch import Tensor
from torchinfo import summary
from transformers import WhisperModel

torch._dynamo.config.suppress_errors = True

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_float32_matmul_precision("high")

# ——————————————————— Attention Pooling ——————————————————————
# Code taken from:
# https://github.com/facebookresearch/deit/blob/main/patchconvnet_models.py#L156
# Use an attention pooling layer to aggregate features from the Whisper model.


class Mlp(nn.Module):
    """
    MLP block with two linear layers and dropout (optional).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.ReLU,
        drop: float = 0.0,
    ):
        """
        Args:
            in_features (int): Input feature dimension.
            hidden_features (Optional[int]): Hidden feature dimension. If None, defaults to in_features.
            out_features (Optional[int]): Output feature dimension. If None, defaults to in_features.
            act_layer (nn.Module): Activation layer to use.
            drop (float): Dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Rotary Position Embedding (RoPE) implementation
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation in PyTorch."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            embedding_dims (int): Dimension of the embeddings.
            min_timescale (int): Minimum timescale for RoPE.
            max_timescale (int): Maximum timescale for RoPE.
            dtype (torch.dtype): Data type for the embeddings.
        """
        super().__init__()
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")

        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.compute_dtype = dtype

        half_embedding_dim = embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / embedding_dims
        timescale = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)
        self.register_buffer("timescale", timescale, persistent=False)

    def forward(self, inputs: torch.Tensor, position: torch.Tensor):
        """Applies RoPE."""
        position = position.unsqueeze(-1).unsqueeze(-1)
        sinusoid_inp = position / self.timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        first_half, second_half = torch.chunk(inputs.to(torch.float32), 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat(
            (first_part.to(self.compute_dtype), second_part.to(self.compute_dtype)),
            dim=-1,
        )


class Learned_Aggregation_Layer(nn.Module):
    """
    Attention pooling layer: Applies a self-attention mechanism where only the CLS token
    serves as the query, enabling it to aggregate information from all other tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Args:
            dim (int): Dimension of the input features.
            num_heads (int, optional): Number of attention heads. Defaults to 1.
            qkv_bias (bool, optional): Whether to use bias in the QKV linear layers. Defaults to False.
            qk_scale (Optional[float], optional): Scaling factor for the QK dot product. Defaults to None.
            attn_drop (float, optional): Dropout rate for the attention weights. Defaults to 0.0.
            proj_drop (float, optional): Dropout rate for the output projection. Defaults to 0.0.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.ql = nn.Linear(dim, dim, bias=qkv_bias)
        self.kl = nn.Linear(dim, dim, bias=qkv_bias)
        self.vl = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_prob = proj_drop

        # --- Rotary Embedding ---
        self.rotary_emb = RotaryEmbedding(embedding_dims=head_dim)

        self.use_sdpa = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        H = self.num_heads
        E = C // H

        q = x[:, 0].unsqueeze(1)  # CLS token as query
        k = x  # Full hidden states as keys and values
        v = x

        q = self.ql(q).view(B, 1, H, E)  # shape: (B, 1, H, E)
        q = rearrange(q, "b t h e -> b h t e")

        # Add rotary embeddings to keys
        k_pos = repeat(torch.arange(k.shape[1]), "l -> b l", b=k.shape[0]).to(x.device)
        k = self.kl(k).view(B, T, H, E)
        k = self.rotary_emb(k, position=k_pos)
        k = rearrange(k, "b t h e -> b h t e")

        v = self.vl(v).view(B, T, H, E)
        v = rearrange(v, "b t h e -> b h t e")

        # Compute self-attention
        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.proj_drop_prob,
                )
        else:
            attn = (q @ k.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = rearrange(x, "b h t e -> b t (h e)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Layer_scale_init_Block_only_token(nn.Module):
    """
    Block that applies self-attention and MLP to the CLS token to aggregate information into a single embedding.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        Attention_block=Learned_Aggregation_Layer,
        Mlp_block=Mlp,
        init_values: float = 1e-4,
    ):
        """
        Args:
            dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to input dimension.
            qkv_bias (bool, optional): Whether to use bias in the QKV linear layers. Defaults to False.
            qk_scale (Optional[float], optional): Scaling factor for the QK dot product. Defaults to None.
            drop (float, optional): Dropout rate for the output projection.
            attn_drop (float, optional): Dropout rate for the attention weights. Defaults to 0.0.
            Attention_block (nn.Module, optional): Attention block to use. Defaults to Learned_Aggregation_Layer.
            Mlp_block (nn.Module, optional): MLP block to use. Defaults to Mlp.
            init_values (float, optional): Initial values for the layer scale parameters. Defaults to 1e-4.
        """
        super().__init__()

        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Attention and MLP blocks
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU, drop=drop)

        # Parameters for residual connections
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor, x_cls: torch.Tensor) -> torch.Tensor:
        u = torch.cat((x_cls, x), dim=1)

        # (Normalization -> Attention) + Residual Connection
        x_cls2 = x_cls + self.gamma_1 * self.attn(self.norm1(u))

        # (Normalization -> MLP) + Residual Connection
        x_cls3 = x_cls + self.gamma_2 * self.mlp(self.norm2(x_cls2))

        return x_cls3


class AttentionPooling(nn.Module):
    """
    Attention pooling layer that aggregates features from the input sequence using a self-attention mechanism.
    The CLS token serves as the query, allowing it to gather information from all other tokens in the sequence.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        mlp_ratio: int = 2,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_scale: float = 1e-4,
        rescale: float = 0.02,
    ):
        """
        Args:
            dim (int): Dimension of the input features.
            num_heads (int, optional): Number of attention heads. Defaults to 1.
            mlp_ratio (int, optional): Ratio of MLP hidden dimension to input dimension.
            qkv_bias (bool, optional): Whether to use bias in the QKV linear layers
            qk_scale (Optional[float], optional): Scaling factor for the QK dot product. Defaults to None.
            drop (float, optional): Dropout rate for the output projection. Defaults to 0
            attn_drop (float, optional): Dropout rate for the attention weights. Defaults to 0.0.
            init_scale (float, optional): Initial scale for the layer scale parameters. Defaults to 1e-4.
        """
        super().__init__()

        # Full attention block (self-attention + MLP)
        self.attn = Layer_scale_init_Block_only_token(
            dim=int(dim),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            Attention_block=Learned_Aggregation_Layer,
            Mlp_block=Mlp,
            init_values=init_scale,
        )

        # Normalization layer
        self.norm_layer = nn.LayerNorm(dim)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(dim)))

        # Initialize the CLS token
        self.rescale = rescale
        trunc_normal_(self.cls_token, std=self.rescale)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.rescale)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Expand the CLS token to match the batch size
        cls_token = self.cls_token.expand(B, -1, -1)

        # We initialize the CLS token with the mean of the input features (over the sequence length)
        cls_token = cls_token + x.mean(dim=1, keepdim=True)

        # Apply the attention block to the CLS token and the input features
        cls_token = self.attn(x, cls_token)

        # Concatenate back the CLS token with the input features
        x = torch.cat((cls_token, x), dim=1)

        # Apply normalization to the output
        x = self.norm_layer(x)

        # Return only the CLS token as the pooled output
        return x[:, 0]


# ——————————————————— Projection ——————————————————————


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


class DenseGeneral(nn.Module):
    """
    Generalized dense layer that supports multiple input and output shapes.
    This layer is similar to a fully connected layer but allows for more flexible input and output dimensions.
    """

    def __init__(
        self,
        in_shapes: tuple[int, ...],
        out_features: tuple[int, ...],
        axis: tuple[int, ...] = (-1,),
    ):
        """
        Args:
            in_shapes (tuple[int, ...]): Input shapes of the layer.
            out_features (tuple[int, ...]): Output feature dimensions of the layer.
            axis (tuple[int, ...], optional): Axes along which to perform the dense operation. Defaults to (-1,).
        """
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis
        self.kernel_shape = self.in_shapes + self.out_features

        self.weight = nn.Parameter(torch.empty(self.kernel_shape))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass of the dense layer.
        """
        inputs = inputs.contiguous()
        norm_axis = _normalize_axes(self.axis, inputs.ndim)
        kernel_contract_axes = tuple(range(len(norm_axis)))

        output = torch.tensordot(
            inputs.to(self.weight.dtype),
            self.weight,
            dims=(norm_axis, kernel_contract_axes),
        ).to(inputs.dtype)
        return output


class Projection(nn.Module):
    """
    Projection layer that applies a series of linear layers with batch normalization and ReLU activation.
    This layer is used to project input features to a desired output dimension.
    """

    def __init__(self, d_in: int = 1024, d_out: int = 768, hidden: List[int] | None = [768]):
        """
        Args:
            d_in (int, optional): Input dimension. Defaults to 1024.
            d_out (int, optional): Output dimension. Defaults to 768.
            hidden (List[int] | None, optional): List of hidden layers dimension. Defaults to [768].
        """
        super().__init__()
        hidden = hidden or []
        layers, dim = [], d_in

        for h in hidden:
            layers += [
                DenseGeneral(in_shapes=(dim,), out_features=(h,)),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ]
            dim = h

        layers.append(DenseGeneral(in_shapes=(dim,), out_features=(d_out,)))
        self.net = nn.Sequential(*layers)

    def summary(self) -> None:
        example_input = torch.zeros(2, 1024)
        summary(self, input_data=example_input)

    def forward(self, x):
        return self.net(x)


# ——————————————————— Whisper Encoder ——————————————————————


class WhisperEncoder:
    """
    Extract features (mel-spectrograms) from raw audio waveforms using the Whisper model.
    Designed for integration into a PyTorch pipeline, this component functions purely as a feature extractor
    and is not trained as part of the model.
    """

    def __init__(self, debug: bool = False, pooling: str = "mean", device: Optional[str] = "cuda:0"):
        """
        Args:
            debug (bool, optional): Whether to enable debug mode (hence logging). Defaults to False.
            pooling (str, optional): Pooling method to use. Defaults to "mean". Options are "mean" or "attention".
            device (Optional[str], optional): Device to run the model on. Defaults to "cuda:0" if available.
        """
        self.debug = debug
        self.pooling = pooling
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        # Only keep the encoder part of the Whisper model
        self.model = WhisperModel.from_pretrained("openai/whisper-large-v3-turbo").to(self.device).encoder

        # Freeze the model parameters and set to evaluation mode
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Compile the model for faster inference
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"Failed to compile Whisper model: {e}")

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            # In any case, use AMP for faster inference (even if the full model is not trained is AMP)
            with torch.autocast("cuda", dtype=torch.float16):
                # Forward pass through the Whisper model, retrieve the last hidden state (dim: N, T, C)
                hidden_states = self.model(x).last_hidden_state

                # Apply mean pooling if specified, otherwise return the hidden states directly
                if self.pooling == "mean":
                    output = F.adaptive_avg_pool1d(hidden_states.transpose(1, 2), 1).squeeze(-1)
                else:
                    output = hidden_states

        return output.detach().float()


class WhisperEncoderTrainable(nn.Module):
    """
    Trainable Whisper encoder that extracts features from audio waveforms.
    This class is designed to be used in a PyTorch pipeline and can be trained as part
    of a larger model.
    """

    def __init__(self, debug: bool = False, pooling: str = "mean", device: Optional[str] = "cuda:0"):
        """
        Args:
            debug (bool, optional): Whether to enable debug mode (hence logging). Defaults to False.
            pooling (str, optional): Pooling method to use. Defaults to "mean". Options are "mean" or "attention".
            device (Optional[str], optional): Device to run the model on. Defaults to "cuda:0" if available.
        """
        super().__init__()

        self.debug = debug
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.pooling = pooling

        self.model = WhisperModel.from_pretrained("openai/whisper-large-v3-turbo").to(self.device).encoder

    def forward(self, x: Tensor) -> Tensor:
        # Forward pass through the Whisper model, retrieve the last hidden state (dim: N, T, C)
        hidden_states = self.model(x).last_hidden_state

        # Apply mean pooling if specified, otherwise return the hidden states directly
        if self.pooling == "mean":
            output = F.adaptive_avg_pool1d(hidden_states.transpose(1, 2), 1).squeeze(-1)
        else:
            output = hidden_states

        return output.float()


# ——————————————————— LIE (Lyrics-Informed Embeddings)  ——————————————————————
class LightLIE(nn.Module):
    """
    Light Lyrics-Informed Embeddings (LIE) model that learns lyrics-informed audio embeddings.
    This model applies pooling on hidden states from the Whisper model and projects them to a lower-dimensional space.

    Possible pooling methods include:
    - "mean": Average pooling over the sequence length.
    - "attention": Attention pooling that aggregates information from all tokens using a self-attention mechanism
    """

    def __init__(
        self,
        dim_whisper: int = 1280,
        dim_hiddens: List[int] = [2048, 1024],
        dim_embed: int = 768,
        pooling: str = "attention",
        num_heads: int = 1,
        mlp_ratio: int = 2,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_scale: float = 1e-4,
        device: torch.device | None = None,
        debug: bool = True,
        use_logit_scale: bool = False,
        **kwargs,
    ):
        """
        Args:
            dim_whisper (Optional[int]): Dimension of the Whisper model's hidden states. Defaults to 1280.
            dim_hiddens (Optional[List[int]]): List of hidden dimensions for the projection layers. Defaults to [2048, 1536].
            dim_embed (int): Output embedding dimension, same as text embedding dimension. Defaults to 768.
            pooling (str): Pooling method to use. Options are "mean" or "attention". Defaults to "attention".
            num_heads (int): Number of attention heads for the attention pooling layer. Defaults to 1.
            mlp_ratio (float): Ratio of MLP hidden dimension to input dimension. Defaults to 2.0.
            qkv_bias (bool): Whether to use bias in the QKV linear layers. Defaults to False.
            qk_scale (Optional[float]): Scaling factor for the QK dot product. Defaults to None.
            drop (float): Dropout rate for the output projection. Defaults to 0.0.
            attn_drop (float): Dropout rate for the attention weights. Defaults to 0.0.
            init_scale (float): Initial scale for the layer scale parameters. Defaults to 1e-4.
            device (torch.device | None): Device to run the model on. If None, defaults to "cuda:0" if available.
            debug (bool): Whether to enable debug mode (hence logging). Defaults to True.
        """
        super().__init__()

        #  Initialize Attention Pooling Layer if specified
        self.pooling_str = pooling

        if pooling == "attention":
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

        # Projection Layer
        self.audio_proj = Projection(d_in=dim_whisper, d_out=dim_embed, hidden=dim_hiddens)

        #  Parameters
        self.debug = debug
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.use_logit_scale = use_logit_scale
        if use_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07), requires_grad=True, device=self.device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if self.pooling_str == "attention":
            audio = self.pooling(audio)

        audio_embedding = self.audio_proj(audio)

        # Normalize the audio embedding
        audio_embedding = F.normalize(audio_embedding, p=2, dim=-1)

        # Return the audio embedding, optionally with logit scale
        if self.use_logit_scale:
            return audio_embedding, self.logit_scale.exp()

        return audio_embedding
