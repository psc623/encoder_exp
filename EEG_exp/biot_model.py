"""Self-contained BIOT encoder compatible with the released EEG checkpoints.

The upstream BIOT repository imports ``linear_attention_transformer`` without
pinning a version.  This file implements the exact non-causal, global linear
attention path used by the released checkpoints, while preserving the upstream
module names so that checkpoint loading can remain strict.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size: int = 256, n_freq: int = 101) -> None:
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(
                f"BIOT positional encoding supports at most {self.pe.shape[1]} "
                f"tokens per channel, received {x.shape[1]}. Segment recordings "
                "into shorter windows."
            )
        return self.dropout(x + self.pe[:, : x.shape[1]])


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x)))


class Chunk(nn.Module):
    """Compatibility wrapper used by the upstream dependency."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)


def linear_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Non-causal Q(K^T V) attention from linear-attention-transformer."""

    dim_head = q.shape[-1]
    q = q.softmax(dim=-1) * (dim_head**-0.5)
    k = k.softmax(dim=-2)
    context = torch.einsum("bhnd,bhne->bhde", k, v)
    return torch.einsum("bhnd,bhde->bhne", q, context)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"Embedding dimension {dim} is not divisible by {heads} heads")
        self.heads = heads
        self.dim_head = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.heads, self.dim_head).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        q = self._split_heads(self.to_q(x))
        k = self._split_heads(self.to_k(x))
        v = self._split_heads(self.to_v(x))
        attention = linear_attention(q, k, v)
        batch, _, tokens, _ = attention.shape
        attention = attention.transpose(1, 2).reshape(batch, tokens, -1)
        return self.dropout(self.to_out(attention))


class SequentialSequence(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for attention, feed_forward in self.layers:
            x = x + attention(x)
            x = x + feed_forward(x)
        return x


class LinearAttentionTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        heads: int = 8,
        depth: int = 4,
        attention_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(depth):
            layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, SelfAttention(dim, heads, attention_dropout)),
                        PreNorm(dim, Chunk(FeedForward(dim))),
                    ]
                )
            )
        self.layers = SequentialSequence(layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class BIOTEncoder(nn.Module):
    """BIOT EEG encoder returning one 256-dimensional vector per window."""

    def __init__(
        self,
        emb_size: int = 256,
        heads: int = 8,
        depth: int = 4,
        n_channels: int = 18,
        n_fft: int = 200,
        hop_length: int = 100,
    ) -> None:
        super().__init__()
        if emb_size != 256:
            raise ValueError("Released BIOT EEG checkpoints require emb_size=256")
        self.emb_size = emb_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_embedding = PatchFrequencyEmbedding(emb_size, n_fft // 2 + 1)
        self.transformer = LinearAttentionTransformer(emb_size, heads, depth)
        self.positional_encoding = PositionalEncoding(emb_size)
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(torch.arange(n_channels, dtype=torch.long), requires_grad=False)
        self.register_buffer(
            "stft_window",
            torch.ones(n_fft, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: Tensor, n_channel_offset: int = 0) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, channels, time], received {tuple(x.shape)}")
        batch, channels, samples = x.shape
        if samples < self.n_fft:
            raise ValueError(f"Input has {samples} samples, fewer than n_fft={self.n_fft}")
        if n_channel_offset + channels > self.channel_tokens.num_embeddings:
            raise ValueError(
                f"Input requires channel IDs through {n_channel_offset + channels - 1}, "
                f"but embedding table has {self.channel_tokens.num_embeddings} entries"
            )

        flattened = x.reshape(batch * channels, samples)
        spectrum = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.stft_window.to(dtype=x.dtype),
            center=False,
            onesided=True,
            return_complex=True,
        ).abs()
        # [B*C, F, K] -> [B, C, K, F]
        spectrum = spectrum.transpose(1, 2).reshape(batch, channels, -1, spectrum.shape[1])
        token_embeddings = self.patch_embedding(spectrum)
        tokens_per_channel = token_embeddings.shape[2]

        channel_ids = self.index[n_channel_offset : n_channel_offset + channels]
        channel_embeddings = self.channel_tokens(channel_ids)[None, :, None, :]
        token_embeddings = token_embeddings + channel_embeddings

        # Positional positions restart at zero inside every channel, as upstream.
        token_embeddings = token_embeddings.reshape(batch * channels, tokens_per_channel, -1)
        token_embeddings = self.positional_encoding(token_embeddings)
        sentence = token_embeddings.reshape(batch, channels * tokens_per_channel, -1)
        return self.transformer(sentence).mean(dim=1)


class BIOTACClassifier(nn.Module):
    """BIOT encoder plus a newly initialized Alzheimer-vs-control head."""

    def __init__(self, encoder: BIOTEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.ELU(), nn.Linear(encoder.emb_size, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x)).squeeze(-1)


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping of parameter names to tensors")

    state = dict(checkpoint)
    prefixes = ("model.biot.", "biot.", "model.", "encoder.")
    for prefix in prefixes:
        prefixed = [key for key in state if key.startswith(prefix)]
        if prefixed:
            state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
            break
    return state


def load_pretrained_encoder(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> BIOTEncoder:
    """Strictly load one of the raw encoder checkpoints released with BIOT."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BIOT checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch 1.x compatibility for the upstream environment.
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state = _extract_state_dict(checkpoint)
    if "channel_tokens.weight" not in state:
        raise KeyError("Checkpoint has no channel_tokens.weight; it is not a BIOT encoder checkpoint")
    n_channels = int(state["channel_tokens.weight"].shape[0])
    encoder = BIOTEncoder(n_channels=n_channels)
    encoder.load_state_dict(state, strict=True)
    return encoder


def configure_encoder_trainability(encoder: BIOTEncoder, regime: str) -> None:
    """Configure frozen, last-two-block, full, or scratch training."""

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    if regime == "pretrained_frozen":
        return
    if regime == "pretrained_last2":
        for block in encoder.transformer.layers.layers[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        return
    if regime in {"pretrained_full", "scratch"}:
        for parameter in encoder.parameters():
            if parameter is not encoder.index:
                parameter.requires_grad = True
        return
    raise ValueError(f"Unknown training regime: {regime}")

