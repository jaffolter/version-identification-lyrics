import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Dict
from torchinfo import summary
from loguru import logger

import laion_clap
from laion_clap.training.data import get_audio_features

from .vit import Transformer
from transformers import ClapModel, ClapProcessor
from transformers import WhisperFeatureExtractor, WhisperModel

import os
from torch.profiler import profile, record_function, ProfilerActivity

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ——————————————————— Projection ——————————————————————


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


class DenseGeneral(nn.Module):
    def __init__(
        self,
        in_shapes: tuple[int, ...],
        out_features: tuple[int, ...],
        axis: tuple[int, ...] = (-1,),
    ):
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis
        self.kernel_shape = self.in_shapes + self.out_features

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


class Projection(nn.Module):
    def __init__(self, d_in: int = 1024, d_out: int = 768, hidden: List[int] | None = [768]):
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


# ——————————————————— Local Pooling ——————————————————————
class AveragePooling(nn.Module):
    def __init__(self, kernel_size: int = 16, stride: int = 8, device: torch.device | None = None, debug: bool = True):
        super().__init__()

        self.pooling = nn.AvgPool1d(kernel_size, stride=stride)

        self.debug = debug
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: Tensor, valid_frames: Tensor) -> Tensor:
        # x : (B, S * T, D) -> (B, D, S * T)
        B, seq_len, D = x.shape

        if self.debug:
            logger.info(f"[AveragePooling] Sequence length (S*T): {seq_len=}")
            logger.info(f"[AveragePooling] Input shape (B, S * T, D): {x.shape=}")

        # ---- Apply average pooling to inputs ----
        x = x.permute(0, 2, 1)

        if self.debug:
            logger.info(f"[AveragePooling] Input shape (B, D, S * T): {x.shape=}")

        x = self.pooling(x)

        if self.debug:
            logger.info(f"[AveragePooling] Pooled shape (B, D, (S * T)'): {x.shape=}")

        # ---- Apply average pooling to valid frames ----
        idx = torch.arange(seq_len, device=self.device).view(1, 1, -1)  # (1, 1, S * T)
        if self.debug:
            logger.info(f"[AveragePooling] Idx (1, 1, S*T): {idx.shape=}")

        # Mask invalid frames
        valid_frames = valid_frames.to(self.device)
        vf = valid_frames.unsqueeze(-1).unsqueeze(-1)  # shape (B) -> (B, 1, 1)
        mask = idx >= vf
        pooled_mask = self.pooling(mask.float()).bool()  # Pass the mask to the pooling operation
        if self.debug:
            logger.info(f"[AveragePooling] Valid frames (B, 1, 1): {vf.shape=}")
            logger.info(f"[AveragePooling] Mask (B, 1, S*T): {mask.shape=}")
            logger.info(f"[AveragePooling] Pooled mask shape (B, 1, (S * T)'): {pooled_mask.shape=}")

        return x.permute(0, 2, 1), pooled_mask.squeeze()  # Remove the channel dimension, shape (B, (S * T)')


# ——————————————————— LAION Encoder ——————————————————————
class LaionEncoder(nn.Module):
    def __init__(
        self, checkpoint_clap: str, audio_config_clap: Dict, device: torch.device | None = None, debug: bool = True
    ):
        super().__init__()

        # Initialize the CLAP model
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        model.load_ckpt(checkpoint_clap)
        self.clap = model.model

        self.audio_config_clap = audio_config_clap
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug

    @torch.no_grad()
    def _clap_forward(self, x: Tensor) -> Tensor:
        feats = [
            get_audio_features(
                {},
                w,
                max_len=480_000,
                data_truncating="rand_trunc",
                data_filling="pad",
                audio_cfg=self.audio_config_clap,
                require_grad=False,
            )
            for w in x
        ]
        keys = feats[0].keys()
        inp = {k: torch.cat([d[k].unsqueeze(0) for d in feats]).to(self.device) for k in keys}

        hidden_states = self.clap.encode_audio(inp, self.device)["fine_grained_embedding"]  # (B, 1024, 1024)

        return hidden_states

    def _summary(self) -> None:
        example_input = torch.zeros(2, 3, 1024)
        summary(self, input_data=example_input)

    def forward(self, x: Tensor) -> Tensor:
        if self.debug:
            logger.info(f"[LaionEncoder] Inputs (B, S, L) {x.shape=}")

        x = x.to(self.device)

        # (B, S, L) -> (B * S, L)
        B, S, H = x.shape
        x = x.reshape(B * S, H)

        # Get CLAP hidden states : (B * S, T, D)
        hidden_states = self._clap_forward(x)

        if self.debug:
            logger.info(f"[LaionEncoder] CLAP hidden states (B * S, T, D) {hidden_states.shape=}")

        # (B * S, T, D) -> (B, S * T, D)
        hidden_states = hidden_states.view(B, -1, hidden_states.shape[-1])

        if self.debug:
            logger.info(f"[LaionEncoder] Flattened hidden states (B, S * T, D) {hidden_states.shape=}")

        return hidden_states


class LaionEncoderFrozen(nn.Module):
    def __init__(
        self, checkpoint_clap: str, audio_config_clap: Dict, device: torch.device | None = None, debug: bool = True
    ):
        super().__init__()

        # Initialize the CLAP model
        # Initialize the CLAP model
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        model.load_ckpt(checkpoint_clap)
        self.model = model.model

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.audio_config_clap = audio_config_clap

        # freeze weights
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def _clap_forward(self, x: Tensor) -> Tensor:
        x = x.to(torch.float32)
        x = x.to(self.device)
        feats = [
            get_audio_features(
                {},
                w,
                max_len=480_000,
                data_truncating="fusion",
                data_filling="pad",
                audio_cfg=self.audio_config_clap,
                require_grad=False,
            )
            for w in x
        ]

        audio_embed = self.model.get_audio_embedding(feats)

        if self.debug:
            logger.info(f"[LaionEncoderFrozen] Inputs shape: {x.shape}")
            logger.info(f"[LaionEncoderFrozen] CLAP audio features shape: {audio_embed.shape}")

        return audio_embed

    def _summary(self) -> None:
        example_input = torch.zeros(2, 3, 1024)
        summary(self, input_data=example_input)

    def forward(self, x: Tensor) -> Tensor:
        if self.debug:
            logger.info(f"[LaionEncoder] Inputs (B, S, L) {x.shape=}")

        x = x.to(self.device)

        audio_embed = self._clap_forward(x)
        return audio_embed


class WhisperEncoder(nn.Module):
    def __init__(self, device: torch.device | None = None, debug: bool = True):
        super().__init__()

        self.model = WhisperModel.from_pretrained("openai/whisper-large-v3-turbo")
        self.processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
        self.debug = debug
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.sample_rate = 16_000

        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _whisper_forward(self, x: Tensor) -> Tensor:
        # Process the audio input
        with torch.no_grad():
            inputs = self.processor(
                x, return_tensors="pt", sampling_rate=self.sample_rate, padding=True, device=self.device
            ).to(self.device, dtype=torch.bfloat16)

            # Get the model outputs
            outputs = self.model.encoder(**inputs).last_hidden_state
            outputs = outputs.mean(dim=1)

            return outputs

    def forward(self, x: Tensor) -> Tensor:
        # x = [a.cpu().numpy() for a in x]
        outputs = self._whisper_forward(x)
        return outputs


# ——————————————————— LIE (whole model) ——————————————————————
class LightLIE(nn.Module):
    def __init__(
        self,
        audio_encoder_name: str,
        checkpoint_clap: Optional[str] = None,  # Path to CLAP checkpoint
        audio_config_clap: Optional[Dict] = None,
        dim_clap: Optional[int] = 1024,  # CLAP embedding dimension
        dim_whisper: Optional[int] = 512,  # Whisper embedding dimension
        dim_hiddens: Optional[List[int]] = [2048, 1024],
        dim_embed: int = 768,  # Output embedding dimension
        device: torch.device | None = None,
        debug: bool = True,
        **kwargs,
    ):
        super().__init__()

        # ----- Audio branch -----
        # 1. Audio encoder (frozen)
        if audio_encoder_name == "whisper":
            self.audio_encoder = WhisperEncoder(device=device, debug=debug)
        elif audio_encoder_name == "clap":
            self.audio_encoder = LaionEncoderFrozen(
                checkpoint_clap=checkpoint_clap, audio_config_clap=audio_config_clap, device=device, debug=debug
            )

        # 2. MLP
        self.audio_proj = Projection(d_in=dim_clap, d_out=dim_embed, hidden=dim_hiddens)

        # ----- Text branch -----
        self.text_encoder = lambda t: t

        # ------ Parameters ------
        self.cls = nn.Parameter(torch.zeros(1, 1, dim_clap)).to(device)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        self.debug = debug
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, audio: Tensor, text: Tensor, valid_frames: Optional[Tensor] = None) -> Tensor:
        # ---- Audio branch ----
        # 1. Get audio hidden states from CLAP
        clap_embedding = self.audio_encoder(audio)
        audio_embedding = self.audio_proj(clap_embedding.to(self.device))

        # ---- Text branch ----
        text_embedding = self.text_encoder(text)

        if self.debug:
            logger.info(f"[LIE] Input {audio.shape=} {text.shape=} ")
            logger.info(f"[LIE] Audio embedding shape: {audio_embedding.shape}")
            logger.info(f"[LIE] Text embedding shape: {text_embedding.shape}")

        return audio_embedding, text_embedding, self.logit_scale.exp()


class LIE(nn.Module):
    def __init__(
        self,
        checkpoint_clap: str,
        audio_config_clap: Dict,
        dim_clap: int = 1024,  # CLAP embedding dimension
        dim_embed: int = 768,  # Output embedding dimension
        kernel_size_pooling: int = 16,
        stride_pooling: int = 8,
        depth_vit: int = 4,
        n_heads_vit: int = 4,
        mlp_ratio_vit: float = 2.0,
        device: torch.device | None = None,
        debug: bool = True,
    ):
        super().__init__()

        # ----- Audio branch -----
        # 1. CLAP audio encoder (frozen)
        self.audio_encoder = LaionEncoder(
            checkpoint_clap=checkpoint_clap, audio_config_clap=audio_config_clap, device=device, debug=debug
        )

        # 2. Local Average Pooling
        self.avg_pooling = AveragePooling(
            kernel_size=kernel_size_pooling, stride=stride_pooling, device=device, debug=debug
        )

        # 3. Transformer
        self.transformer = Transformer(
            dim_embed=dim_clap, depth=depth_vit, n_heads=n_heads_vit, mlp_ratio=mlp_ratio_vit
        )
        print(self.transformer.summary())

        # 4. Projection head
        self.audio_proj = Projection(d_in=dim_clap, d_out=dim_embed, hidden=[dim_embed])

        # ----- Text branch -----
        self.text_encoder = lambda t: t

        # ------ Parameters ------
        self.cls = nn.Parameter(torch.zeros(1, 1, dim_clap)).to(device)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        # self.alpha = nn.Parameter(torch.tensor(0.1))  # MSE loss weight

        self.debug = debug

    def forward(self, audio: Tensor, text: Tensor, valid_frames: Optional[Tensor] = None) -> Tensor:
        # ---- Audio branch ----
        # 1. Get audio hidden states from CLAP
        hidden_states = self.audio_encoder(audio)

        # 2. Apply average pooling
        hidden_states_pooled, valid_frames_pooled = self.avg_pooling(hidden_states, valid_frames)

        # 3. Add CLS token
        # Add average of hidden states to CLS token
        B = hidden_states_pooled.shape[0]
        cls_token = self.cls.expand(B, -1, -1) + hidden_states_pooled.mean(dim=1, keepdim=True)

        # Add masked token = True for valid frames (CLS token is always valid)
        cls_token_pad = torch.ones(B, 1, dtype=torch.bool, device=hidden_states_pooled.device)

        hidden_states_pooled_cls = torch.cat([cls_token, hidden_states_pooled], dim=1)
        valid_frames_pooled_cls = torch.cat(
            [cls_token_pad, valid_frames_pooled],
            dim=1,
        )

        # 4. Pass to Transformer
        output = self.transformer(hidden_states_pooled_cls, mask=valid_frames_pooled_cls)
        output_cls = output[:, 0, :]  # Get the CLS token output
        # print the type of output_cls
        if self.debug:
            logger.info(f"[LIE] Output CLS type: {type(output_cls)}")

        # 5. Apply projection head
        audio_embedding = self.audio_proj(output_cls)

        # ---- Text branch ----
        text_embedding = self.text_encoder(text)

        if self.debug:
            logger.info(f"[LIE] Input {audio.shape=} {text.shape=} {valid_frames.shape=}")
            logger.info(f"[LIE] Audio hidden states {hidden_states.shape}")
            logger.info(f"[LIE] Pooled audio hidden states {hidden_states_pooled.shape}")
            logger.info(f"[LIE] Valid frames after pooling {valid_frames_pooled.shape}")
            logger.info(f"[LIE] Pooled audio hidden states + CLS {hidden_states_pooled_cls.shape}")
            logger.info(f"[LIE] Valid frames after pooling + CLS: {valid_frames_pooled_cls.shape=}")
            logger.info(f"[LIE] Output after transformer: {output_cls.shape}")
            logger.info(f"[LIE] Audio embedding: {audio_embedding.shape}")
            logger.info(f"[LIE] Text embedding: {text_embedding.shape}")

        return audio_embedding, text_embedding, self.logit_scale.exp(), self.alpha
