"""CTC decoding utilities: greedy and beam search with optional KenLM LM."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import torch

from src.data.vocab import Vocab

DecoderFn = Callable[[torch.Tensor, torch.Tensor], List[str]]


def _greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int) -> List[str]:
    """Argmax per frame, collapse repeats/blanks."""
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


def build_greedy_decoder(vocab: Vocab) -> DecoderFn:
    def decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        token_seqs = _greedy_decode(log_probs, lengths, blank_id=vocab.blank_id)
        return [vocab.decode(seq) for seq in token_seqs]

    return decode


def build_beam_decoder(
    vocab: Vocab,
    lm_path: Path | None = None,
    beam_width: int = 50,
    alpha: float = 0.6,
    beta: float = 0.0,
    beam_prune_logp: float = -10.0,
) -> DecoderFn:
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError as exc:
        raise ImportError("pyctcdecode is required for beam search decoding") from exc

    # Reorder labels so blank is first to satisfy pyctcdecode expectations.
    perm = [vocab.blank_id] + [i for i in range(vocab.size) if i != vocab.blank_id]
    labels = [vocab.tokens[i] for i in perm]
    labels[0] = ""  # blank token
    if vocab.pad_id != vocab.blank_id and vocab.pad_id in perm:
        labels[perm.index(vocab.pad_id)] = ""  # treat pad as silence

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(lm_path) if lm_path else None,
        alpha=alpha,
        beta=beta,
    )

    def decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        log_probs_np = log_probs.detach().cpu().numpy()
        # Permute vocab dimension once to match decoder labels.
        log_probs_np = log_probs_np[:, :, perm]
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
) -> DecoderFn:
    if method.lower() == "beam":
        return build_beam_decoder(
            vocab=vocab,
            lm_path=lm_path,
            beam_width=beam_width,
            alpha=alpha,
            beta=beta,
            beam_prune_logp=beam_prune_logp,
        )
    return build_greedy_decoder(vocab)
