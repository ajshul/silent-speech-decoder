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
    def __init__(self, vocab_size: int, blank_id: int, weights: LossWeights):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        self.weights = weights

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
            logit_lengths: (batch,) lengths after subsampling.
            targets: (batch, target_len) token ids.
            target_lengths: (batch,) token lengths.
            student_repr: (batch, time, dim) encoder output.
            teacher_repr: (batch, time_teacher, dim_teacher) or None.
        """
        # CTC expects (time, batch, vocab).
        log_probs_t = log_probs.transpose(0, 1)
        ctc = self.ctc_loss(log_probs_t, targets, logit_lengths, target_lengths)

        if teacher_repr is None:
            distill = torch.tensor(0.0, device=log_probs.device)
        else:
            # Align by truncating to the shorter of student/teacher.
            max_time = min(student_repr.size(1), teacher_repr.size(1))
            distill = F.mse_loss(
                student_repr[:, :max_time], teacher_repr[:, :max_time]
            )

        total = self.weights.lambda_ctc * ctc + self.weights.lambda_distill * distill
        return {"total": total, "ctc": ctc, "distill": distill}
