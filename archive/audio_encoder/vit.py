import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from einops import rearrange, repeat
from torch import Tensor


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


class DenseGeneral(nn.Module):
    """
    PyTorch equivalent of flax.linen.DenseGeneral with shapes defined at init.

    Stores weights (`kernel`) in the same layout as Jax and uses torch.tensordot
    for the generalized matrix multiplication. Weight/bias shapes are calculated
    and parameters created during initialization based on config.
    `load_weights` validates shapes and copies data.

    Attributes:
        axis (Tuple[int, ...]): Input axis or axes to contract.
        in_shapes (Tuple[int, ...]): Sizes of the input dimensions specified by `axis`.
        out_features (Tuple[int, ...]): Shape of the output features (non-contracted dims).
        use_bias (bool): Whether to add a bias term.
        weight (nn.Parameter): The kernel parameter.
        bias (Optional[nn.Parameter]): The bias parameter (if use_bias=True).
    """

    def __init__(
        self,
        in_shapes: tuple[int, ...],
        out_features: tuple[int, ...],
        axis: tuple[int, ...] = (-1,),
        # weight_dtype: torch.dtype | None = None,
        # device: torch.device | None = None,
    ):
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis
        self.kernel_shape = self.in_shapes + self.out_features

        # factory_kwargs = {"device": device, "dtype": weight_dtype}
        self.weight = nn.Parameter(torch.empty(self.kernel_shape))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        norm_axis = _normalize_axes(self.axis, inputs.ndim)
        kernel_contract_axes = tuple(range(len(norm_axis)))

        output = torch.tensordot(
            inputs.to(self.weight.dtype),
            self.weight,
            dims=(norm_axis, kernel_contract_axes),
        ).to(inputs.dtype)
        return output


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation in PyTorch."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")
        print(
            f"Using RoPE with embedding dims: {embedding_dims}, min_timescale: {min_timescale}, max_timescale: {max_timescale}"
        )
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


class MlpBlock(nn.Module):
    """MLP block using DenseGeneral."""

    def __init__(self, dim_in: int, dim_out: int, intermediate_dim: int):
        super().__init__()

        self.wi_fused = DenseGeneral(
            in_shapes=(dim_in,),
            out_features=(2, intermediate_dim),
            axis=(-1,),
        )

        self.wo = DenseGeneral(
            in_shapes=(intermediate_dim,),
            out_features=(dim_out,),
            axis=(-1,),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        fused_x = self.wi_fused(x)

        gate = fused_x[..., 0, :]
        up = fused_x[..., 1, :]

        hidden = torch.mul(F.silu(gate), up)

        output = self.wo(hidden)
        return output


class Attention(nn.Module):
    def __init__(
        self,
        dim_embed,
        n_heads: int = 8,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa=True,
        head_dim=None,
    ):
        super().__init__()
        self.dim_embed = dim_embed
        self.n_heads = n_heads
        if head_dim is None:
            self.head_dim = dim_embed // n_heads
        else:
            self.head_dim = head_dim
        self.scale = qk_scale or self.head_dim**-0.5
        self.ql = DenseGeneral((dim_embed,), (n_heads, self.head_dim))  # , bias=qkv_bias)
        self.kl = DenseGeneral((dim_embed,), (n_heads, self.head_dim))  # , bias=qkv_bias)
        self.vl = DenseGeneral((dim_embed,), (n_heads, self.head_dim))  # , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = DenseGeneral((self.head_dim * n_heads,), (dim_embed,))
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.attn_mask = None

        # --- Rotary Embedding ---
        self.rotary_emb = RotaryEmbedding(embedding_dims=self.head_dim)

    def summary(self) -> None:
        example_input = torch.zeros(1, 4, self.dim_embed)
        summary(self, input_data=example_input)

    def forward(self, x, cross_attn: Tensor = None, mask: Optional[Tensor] = None) -> Tensor:
        if cross_attn is None:
            q, k, v = x, x, x
        else:
            k, v = cross_attn, cross_attn
            q = x
        """
        position = repeat(torch.arange(x.shape[1]), "l -> b l", b=x.shape[0]).to(x.device)

        r = "b t h e -> b h t e"
        q = rearrange(self.rotary_emb(self.ql(q), position=position), r)
        k = rearrange(self.rotary_emb(self.kl(k), position=position), r)
        v = rearrange(self.vl(v), r)
        """

        q_pos = repeat(torch.arange(q.shape[1]), "l -> b l", b=q.shape[0]).to(x.device)
        k_pos = repeat(torch.arange(k.shape[1]), "l -> b l", b=k.shape[0]).to(x.device)
        r = "b t h e -> b h t e"
        q = rearrange(self.rotary_emb(self.ql(q), position=q_pos), r)
        k = rearrange(self.rotary_emb(self.kl(k), position=k_pos), r)
        v = rearrange(self.vl(v), r)

        if mask is not None:
            # mask is a padding mask, with True for valid tokens and False for padding
            batch_size, seq_length = mask.shape
            attn_mask = torch.ones((batch_size, seq_length, seq_length), dtype=torch.bool)
            for i in range(batch_size):
                valid_len = seq_length - mask[i].sum().item()
                attn_mask[i, :valid_len, :valid_len] = False

            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.n_heads, seq_length, seq_length)
            self.attn_mask = attn_mask.to(q.device)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.proj_drop_prob,
                    attn_mask=self.attn_mask,
                )
        else:
            attn = (q @ k.transpose(-1, -2)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = rearrange(x, "b h t e -> b t (h e)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim_embed,
        n_heads,
        mlp_ratio=4.0,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=partial(nn.RMSNorm, eps=1e-5, dtype=torch.float32),
    ):
        super().__init__()
        self.dim_embed = dim_embed
        self.norm1 = norm_layer(dim_embed)
        self.attn = Attention(
            dim_embed,
            n_heads=n_heads,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = norm_layer(dim_embed)
        mlp_hidden_dim = int(dim_embed * mlp_ratio)
        self.mlp = MlpBlock(
            dim_in=dim_embed,
            dim_out=dim_embed,
            intermediate_dim=mlp_hidden_dim,
        )

    def _rescale_block(self, layer_id):
        scale = math.sqrt(2.0 * (layer_id + 1))  # avoid division by zero
        self.attn.proj.weight.data.div_(scale)
        self.mlp.wi_fused.weight.data.div_(scale)
        self.mlp.wo.weight.data.div_(scale)
        return

    def summary(self) -> None:
        example_input = torch.zeros(1, 4, self.dim_embed)
        summary(self, input_data=example_input)
        return

    def forward(self, x, cross_attn: Tensor = None, mask: Optional[Tensor] = None) -> Tensor:
        if cross_attn is None:
            # self attention
            x = x + self.attn(self.norm1(x), mask=mask)
        else:
            # cross attention
            x = x + self.attn(self.norm1(x), cross_attn=self.norm1(cross_attn), mask=mask)

        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Transformer with optional Cross Attention"""

    def __init__(
        self,
        dim_embed: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer=partial(nn.RMSNorm, eps=1e-5, dtype=torch.float32),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_embed = dim_embed

        # Attention Blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim_embed=dim_embed,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(dim_embed)
        self._rescale_blocks()

    def _rescale_blocks(self):
        for layer_id, layer in enumerate(self.blocks):
            layer._rescale_block(layer_id)
        return

    def get_num_layers(self):
        return len(self.blocks)

    def summary(self) -> None:
        example_input = torch.zeros(1, 4, self.dim_embed)
        summary(self, input_data=example_input)
        return

    def forward(self, x: Tensor = None, cross_attn: Tensor = None, mask: Optional[Tensor] = None) -> Tensor:
        kargs = {"cross_attn": cross_attn, "mask": mask} if cross_attn is not None else {"mask": mask}
        for blk in self.blocks:
            x = blk(x, **kargs)
        x = self.norm(x)
        return x
