"""Projection and CTC heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear projection to match teacher dimensionality."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CTCHead(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, dim)
        Returns:
            log_probs: (batch, time, vocab)
        """
        logits = self.fc(self.dropout(x))
        return torch.log_softmax(logits, dim=-1)
