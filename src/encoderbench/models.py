"""Common attention head and capacity-controlled bridge architectures."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


def fixed_3d_position_encoding(grid: Sequence[int], width: int) -> torch.Tensor:
    """Deterministic sinusoidal encoding in D/H/W flatten order."""
    if len(grid) != 3 or math.prod(grid) < 1 or width < 1:
        raise ValueError(f"Invalid position-encoding shape: grid={grid}, width={width}")
    axes = [torch.linspace(-1.0, 1.0, int(size), dtype=torch.float32) for size in grid]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    result = torch.zeros(coordinates.shape[0], width, dtype=torch.float32)
    frequencies = torch.exp(
        torch.arange(0, max(2, width // 6 * 2), 2, dtype=torch.float32)
        * (-math.log(10_000.0) / max(1, width // 3))
    )
    cursor = 0
    for axis in range(3):
        room = width - cursor
        if room <= 0:
            break
        count = min(len(frequencies), (room + 1) // 2)
        phase = coordinates[:, axis:axis + 1] * frequencies[:count].unsqueeze(0) * math.pi
        encoded = torch.stack((phase.sin(), phase.cos()), dim=-1).reshape(len(coordinates), -1)
        take = min(room, encoded.shape[1])
        result[:, cursor:cursor + take] = encoded[:, :take]
        cursor += take
    return result


class AttentionPoolHead(nn.Module):
    """Train-only standardization, learned token attention, and binary head."""

    def __init__(self, width: int, mean: torch.Tensor, std: torch.Tensor, hidden: int = 128):
        super().__init__()
        if mean.shape != (width,) or std.shape != (width,):
            raise ValueError("Normalization statistics must have shape [width]")
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-6))
        self.attention = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.classifier = nn.Linear(width, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = (tokens.float() - self.mean) / self.std
        weights = torch.softmax(self.attention(normalized).squeeze(-1), dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * normalized, dim=1)
        return self.classifier(pooled)


class FactorizedLinearBridge(nn.Module):
    """Rank-limited linear mapping with no activation between projections."""

    def __init__(self, input_width: int, output_width: int = 2560, rank: int = 512):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_width)
        self.down = nn.Linear(input_width, rank, bias=False)
        self.up = nn.Linear(rank, output_width)
        self.output_norm = nn.LayerNorm(output_width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.up(self.down(self.input_norm(tokens))))


class ResamplerLayer(nn.Module):
    def __init__(self, width: int = 512, heads: int = 8, ffn_size: int = 2048):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.cross_query_norm = nn.LayerNorm(width)
        self.cross_source_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_size), nn.GELU(), nn.Linear(ffn_size, width))

    def forward(self, queries: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        normalized_queries = self.cross_query_norm(queries)
        normalized_source = self.cross_source_norm(source)
        queries = queries + self.cross_attention(
            normalized_queries, normalized_source, normalized_source, need_weights=False
        )[0]
        return queries + self.ffn(self.ffn_norm(queries))


class ResamplerBridge(nn.Module):
    """Two-layer 64-query self/cross-attention bridge."""

    def __init__(self, input_width: int, output_width: int = 2560, width: int = 512,
                 query_count: int = 64, layers: int = 2, heads: int = 8,
                 ffn_size: int = 2048):
        super().__init__()
        self.source_projection = nn.Linear(input_width, width)
        self.queries = nn.Parameter(torch.empty(query_count, width))
        self.layers = nn.ModuleList([ResamplerLayer(width, heads, ffn_size) for _ in range(layers)])
        self.output_projection = nn.Linear(width, output_width)
        self.output_norm = nn.LayerNorm(output_width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.source_projection.weight)
        nn.init.zeros_(self.source_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        source = self.source_projection(tokens)
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, source)
        return self.output_norm(self.output_projection(queries))


def build_bridge(kind: str, input_width: int, settings: dict[str, int]) -> nn.Module:
    if kind == "linear":
        return FactorizedLinearBridge(input_width, settings["llm_hidden_size"], settings["rank"])
    if kind == "resampler":
        return ResamplerBridge(input_width=input_width, output_width=settings["llm_hidden_size"],
                               width=settings["rank"], query_count=settings["query_count"],
                               layers=settings["resampler_layers"], heads=settings["attention_heads"],
                               ffn_size=settings["ffn_size"])
    raise ValueError("Bridge kind must be 'linear' or 'resampler'")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

