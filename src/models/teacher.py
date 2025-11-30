"""Frozen WavLM teacher wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel


@dataclass
class TeacherConfig:
    model_name: str = "microsoft/wavlm-base-plus"
    layer: int = 9
    device: str = "cpu"


class FrozenWavLM(torch.nn.Module):
    def __init__(self, cfg: TeacherConfig):
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model_name)
        self.model = WavLMModel.from_pretrained(cfg.model_name)
        self.model.to(cfg.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.layer = cfg.layer
        self.device = cfg.device

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """
        Args:
            waveform: (time,) or (batch, time) tensor on CPU.
            sampling_rate: audio sample rate.
        Returns:
            hidden: (batch, time', dim) tensor on CPU.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="longest",
        )
        input_values = inputs["input_values"].to(self.device)
        outputs = self.model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if self.layer >= len(hidden_states):
            raise ValueError(
                f"Requested layer {self.layer} but model returned {len(hidden_states)}"
            )
        hidden = hidden_states[self.layer].detach().cpu()
        return hidden
