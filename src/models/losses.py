"""Loss utilities for distillation + CTC training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_distill: float = 0.7
    lambda_ctc: float = 0.3


class DistillationCTCLoss(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        blank_id: int,
        weights: LossWeights,
        normalize_distill: bool = False,
    ):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        self.weights = weights
        self.normalize_distill = normalize_distill

    def forward(
        self,
        log_probs: torch.Tensor,
        logit_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        student_repr: torch.Tensor,
        teacher_repr: torch.Tensor | None,
        teacher_lengths: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            log_probs: (batch, time, vocab) log softmax output.
            logit_lengths: (batch,) lengths after subsampling (before padding).
            targets: (batch, target_len) token ids.
            target_lengths: (batch,) token lengths.
            student_repr: (batch, time, dim) encoder output.
            teacher_repr: (batch, time_teacher, dim_teacher) or None.
            teacher_lengths: (batch,) teacher lengths before padding, if available.
        """
        # CTC expects (time, batch, vocab).
        log_probs_t = log_probs.transpose(0, 1)
        # CTCLoss expects lengths on CPU even when running on MPS/CUDA.
        ctc_lengths = logit_lengths.cpu() if logit_lengths.is_cuda or logit_lengths.is_mps else logit_lengths
        ctc = self.ctc_loss(log_probs_t, targets, ctc_lengths, target_lengths)

        distill = torch.tensor(0.0, device=log_probs.device)
        if teacher_repr is not None:
            # Align teacher to student time resolution to avoid over-penalizing padding.
            student_len = student_repr.size(1)
            teacher_len = teacher_repr.size(1)
            device = log_probs.device

            teacher = teacher_repr
            aligned_teacher_lengths = teacher_lengths.to(device) if teacher_lengths is not None else None
            if teacher_len != student_len:
                teacher = F.interpolate(
                    teacher.transpose(1, 2),
                    size=student_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
                if aligned_teacher_lengths is not None and teacher_len > 0:
                    scale = float(student_len) / float(teacher_len)
                    aligned_teacher_lengths = torch.clamp(
                        torch.round(aligned_teacher_lengths.float() * scale).long(),
                        max=student_len,
                    )

            student_lengths = logit_lengths.to(device)
            valid_lengths = student_lengths
            if aligned_teacher_lengths is not None:
                valid_lengths = torch.minimum(valid_lengths, aligned_teacher_lengths)
            valid_lengths = torch.clamp(valid_lengths, max=student_len)

            mask = (
                torch.arange(student_len, device=device)
                .unsqueeze(0)
                .expand(student_repr.size(0), -1)
                < valid_lengths.unsqueeze(1)
            )
            student_for_loss = student_repr
            teacher_for_loss = teacher
            if self.normalize_distill:
                student_for_loss = F.layer_norm(student_for_loss, student_for_loss.shape[-1:])
                teacher_for_loss = F.layer_norm(teacher_for_loss, teacher_for_loss.shape[-1:])

            mse = torch.pow(student_for_loss - teacher_for_loss, 2)
            masked = mse * mask.unsqueeze(-1)
            denom = (mask.sum() * student_repr.size(-1)).clamp_min(1)
            distill = masked.sum() / denom

        total = self.weights.lambda_ctc * ctc + self.weights.lambda_distill * distill
        return {"total": total, "ctc": ctc, "distill": distill}
