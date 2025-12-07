"""CTC decoding utilities: greedy and beam search with optional KenLM LM."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import torch

from src.data.vocab import Vocab

DecoderFn = Callable[[torch.Tensor, torch.Tensor], List[str]]


def _greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int, blank_bias: float = 0.0) -> List[str]:
    """Argmax per frame, collapse repeats/blanks."""
    if blank_bias != 0.0:
        log_probs = log_probs.clone()
        log_probs[:, :, blank_id] = log_probs[:, :, blank_id] + blank_bias
    preds = torch.argmax(log_probs, dim=-1)  # (batch, time)
    decoded: List[List[int]] = []
    for seq, length in zip(preds, lengths):
        tokens: List[int] = []
        prev = None
        for i in range(int(length)):
            t = int(seq[i])
            if t == blank_id:
                prev = t
                continue
            if t == prev:
                continue
            tokens.append(t)
            prev = t
        decoded.append(tokens)
    return decoded


def build_greedy_decoder(vocab: Vocab, blank_bias: float = 0.0) -> DecoderFn:
    def decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        token_seqs = _greedy_decode(log_probs, lengths, blank_id=vocab.blank_id, blank_bias=blank_bias)
        return [vocab.decode(seq) for seq in token_seqs]

    return decode


def build_beam_decoder(
    vocab: Vocab,
    lm_path: Path | None = None,
    beam_width: int = 50,
    alpha: float = 0.6,
    beta: float = 0.0,
    beam_prune_logp: float = -10.0,
    blank_bias: float = 0.0,
) -> DecoderFn:
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError as exc:
        raise ImportError("pyctcdecode is required for beam search decoding") from exc

    # Reorder labels so blank is first to satisfy pyctcdecode expectations.
    # Drop pad from the label set to avoid duplicate "" entries (blank + pad)
    # and merge its probability mass into blank before decoding.
    non_blank_pad_indices = [i for i in range(vocab.size) if i not in {vocab.blank_id, vocab.pad_id}]
    labels = [""] + [vocab.tokens[i] for i in non_blank_pad_indices]

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(lm_path) if lm_path else None,
        alpha=alpha,
        beta=beta,
    )

    def decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        log_probs_np = log_probs.detach().cpu().numpy()
        blank_logp = log_probs_np[:, :, vocab.blank_id]
        if blank_bias != 0.0:
            blank_logp = blank_logp + float(blank_bias)
        if vocab.pad_id != vocab.blank_id and 0 <= vocab.pad_id < log_probs_np.shape[-1]:
            pad_logp = log_probs_np[:, :, vocab.pad_id]
            blank_logp = np.logaddexp(blank_logp, pad_logp)

        # Permute vocab dimension once to match decoder labels (blank first).
        stacked = [blank_logp[:, :, None]]
        if non_blank_pad_indices:
            stacked.append(log_probs_np[:, :, non_blank_pad_indices])
        log_probs_np = np.concatenate(stacked, axis=2)
        hyps: List[str] = []
        for i, length in enumerate(lengths):
            lp = log_probs_np[i, : int(length)]
            hyp = decoder.decode(
                lp,
                beam_width=beam_width,
                beam_prune_logp=beam_prune_logp,
            )
            hyps.append(hyp)
        return hyps

    return decode


def build_decoder(
    method: str,
    vocab: Vocab,
    lm_path: Path | None = None,
    beam_width: int = 50,
    alpha: float = 0.6,
    beta: float = 0.0,
    beam_prune_logp: float = -10.0,
    blank_bias: float = 0.0,
) -> DecoderFn:
    if method.lower() == "beam":
        return build_beam_decoder(
            vocab=vocab,
            lm_path=lm_path,
            beam_width=beam_width,
            alpha=alpha,
            beta=beta,
            beam_prune_logp=beam_prune_logp,
            blank_bias=blank_bias,
        )
    return build_greedy_decoder(vocab, blank_bias=blank_bias)
