"""Common attention-pooling head and shared model utilities."""

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

    def __init__(self, width: int, mean: torch.Tensor, std: torch.Tensor, hidden: int = 128,
                num_classes: int = 2):
        super().__init__()
        if mean.shape != (width,) or std.shape != (width,):
            raise ValueError("Normalization statistics must have shape [width]")
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-6))
        self.attention = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = (tokens.float() - self.mean) / self.std
        weights = torch.softmax(self.attention(normalized).squeeze(-1), dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * normalized, dim=1)
        return self.classifier(pooled)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

