"""Conformer-based EMG encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio


@dataclass
class EncoderConfig:
    input_dim: int
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 4
    ffn_dim: int = 512
    depthwise_conv_kernel_size: int = 15
    dropout: float = 0.1
    subsample_factor: int = 4  # power of 2 recommended


class Conv1dSubsampler(nn.Module):
    """Temporal subsampling with strided 1D convolutions."""

    def __init__(self, input_dim: int, output_dim: int, factor: int = 4, kernel_size: int = 5):
        super().__init__()
        if factor < 1:
            raise ValueError("factor must be >= 1")
        layers = []
        in_dim = input_dim
        remaining = factor
        while remaining > 1:
            stride = 2
            layers.append(
                nn.Conv1d(
                    in_dim,
                    output_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            in_dim = output_dim
            remaining //= 2
        if not layers:
            # No subsampling; project to output_dim.
            layers.append(nn.Conv1d(in_dim, output_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, feat)
        x = x.transpose(1, 2)  # (batch, feat, time)
        x = self.net(x)
        return x.transpose(1, 2)  # (batch, time', output_dim)


class EMGConformerEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.subsample = Conv1dSubsampler(
            input_dim=cfg.input_dim,
            output_dim=cfg.d_model,
            factor=cfg.subsample_factor,
        )
        self.encoder = torchaudio.models.Conformer(
            input_dim=cfg.d_model,
            num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
            num_layers=cfg.num_layers,
            depthwise_conv_kernel_size=cfg.depthwise_conv_kernel_size,
            dropout=cfg.dropout,
        )
        self.subsample_factor = cfg.subsample_factor

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, time, feat)
            lengths: (batch,) original lengths before padding.
        Returns:
            encoder_out: (batch, time', d_model)
            out_lengths: (batch,) adjusted lengths
        """
        x = self.subsample(x)
        out_lengths = None
        if lengths is not None:
            out_lengths = torch.div(
                lengths, self.subsample_factor, rounding_mode="floor"
            )
        x, out_lengths = self.encoder(x, out_lengths)
        return x, out_lengths
