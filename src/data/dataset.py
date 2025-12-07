"""Dataset and dataloader utilities for cached EMG/teacher features."""

from __future__ import annotations

import random
from dataclasses import dataclass
import functools
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.index_dataset import load_index
from src.data.text_normalizer import normalize_transcript
from src.data.vocab import Vocab

@dataclass
class SpecAugmentConfig:
    time_masks: int = 2
    time_mask_width: float = 0.05  # fraction of sequence length
    freq_masks: int = 2
    freq_mask_width: int = 8
    p: float = 0.0  # probability to apply


@dataclass
class ChannelDropoutConfig:
    """Randomly zeros entire EMG channels to encourage cross-channel robustness."""

    p: float = 0.0  # probability to apply per sample
    max_channels: int = 1  # maximum number of channels to drop


class SpecAugment:
    """Lightweight SpecAugment on log-mel features."""

    def __init__(self, cfg: SpecAugmentConfig):
        self.cfg = cfg

    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (T, F)
        if self.cfg.p <= 0 or random.random() > self.cfg.p:
            return feat

        t, f = feat.shape
        out = feat.clone()

        for _ in range(self.cfg.time_masks):
            width = int(self.cfg.time_mask_width * t)
            if width <= 0:
                continue
            start = random.randint(0, max(t - width, 0))
            out[start : start + width] = 0.0

        for _ in range(self.cfg.freq_masks):
            width = min(self.cfg.freq_mask_width, f)
            if width <= 0:
                continue
            start = random.randint(0, max(f - width, 0))
            out[:, start : start + width] = 0.0

        return out


class EMGFeatureDataset(Dataset):
    """Loads cached EMG + teacher features and tokenized transcripts."""

    def __init__(
        self,
        index_path: Path,
        features_root: Path,
        splits: Sequence[str],
        vocab: Vocab,
        subsets: Optional[Sequence[str]] = None,
        include_teacher: bool = True,
        strict: bool = True,
        channel_dropout_cfg: Optional[ChannelDropoutConfig] = None,
    ) -> None:
        df = load_index(index_path)
        df = df[df["split"].isin(splits)].reset_index(drop=True)
        if subsets:
            if "subset" not in df.columns:
                raise KeyError("Index missing 'subset' column; re-run indexing.")
            df = df[df["subset"].isin(subsets)].reset_index(drop=True)
        # Normalize transcripts and drop empty/heading-only rows.
        df["transcript_norm"] = df["transcript"].apply(normalize_transcript)
        df = df[df["transcript_norm"].astype(bool)].reset_index(drop=True)
        self.df = df
        self.features_root = features_root
        self.vocab = vocab
        self.include_teacher = include_teacher
        self.strict = strict
        self.channel_dropout_cfg = channel_dropout_cfg or ChannelDropoutConfig()

    def __len__(self) -> int:
        return len(self.df)

    def _load_emg(self, utterance_id: str) -> torch.Tensor:
        path = self.features_root / "emg" / f"{utterance_id}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.load(path)
        # Collapse channels+mels into a single feature dimension.
        feat = torch.from_numpy(arr).float()  # (T, C, n_mels)
        feat = self._maybe_channel_dropout(feat)
        t, c, m = feat.shape
        return feat.view(t, c * m)

    def _maybe_channel_dropout(self, feat: torch.Tensor) -> torch.Tensor:
        """Drop full channels with probability p to make the model robust to sensor dropouts."""
        cfg = self.channel_dropout_cfg
        if cfg.p <= 0 or feat.ndim != 3 or random.random() > cfg.p:
            return feat
        _, channels, _ = feat.shape
        if channels <= 1:
            return feat
        max_drop = min(max(1, cfg.max_channels), channels - 1)
        drop_n = random.randint(1, max_drop)
        drop_indices = random.sample(range(channels), k=drop_n)
        feat = feat.clone()
        feat[:, drop_indices, :] = 0.0
        return feat

    def _load_teacher(self, utterance_id: str) -> Optional[torch.Tensor]:
        path = self.features_root / "teacher" / f"{utterance_id}.npy"
        if not path.exists():
            if self.strict:
                raise FileNotFoundError(path)
            return None
        return torch.from_numpy(np.load(path)).float()

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        utterance_id = row["utterance_id"]
        emg = self._load_emg(utterance_id)
        teacher = self._load_teacher(utterance_id) if self.include_teacher else None
        # Fallback for older in-memory datasets without the normalized column.
        transcript = row["transcript_norm"] if "transcript_norm" in row else normalize_transcript(row["transcript"])
        tokens = torch.tensor(self.vocab.encode(transcript), dtype=torch.long)
        return {
            "utterance_id": utterance_id,
            "emg": emg,
            "emg_length": emg.shape[0],
            "teacher": teacher,
            "teacher_length": teacher.shape[0] if teacher is not None else 0,
            "transcript": transcript,
            "tokens": tokens,
            "token_length": len(tokens),
        }


def collate_batch(
    batch: List[Dict],
    spec_augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    pad_value: float = 0.0,
    vocab: Optional[Vocab] = None,
) -> Dict[str, torch.Tensor]:
    emg_seqs = []
    emg_lengths = []
    teacher_seqs = []
    teacher_lengths = []
    token_seqs = []
    token_lengths = []
    utterance_ids = []
    transcripts = []

    for sample in batch:
        emg = sample["emg"]
        if spec_augment is not None:
            emg = spec_augment(emg)
        emg_seqs.append(emg)
        emg_lengths.append(emg.shape[0])

        teacher = sample["teacher"]
        if teacher is not None:
            teacher_seqs.append(teacher)
            teacher_lengths.append(teacher.shape[0])

        tokens = sample["tokens"]
        token_seqs.append(tokens)
        token_lengths.append(len(tokens))

        utterance_ids.append(sample["utterance_id"])
        transcripts.append(sample["transcript"])

    emg_padded = pad_sequence(emg_seqs, batch_first=True, padding_value=pad_value)
    emg_lengths_tensor = torch.tensor(emg_lengths, dtype=torch.long)

    teacher_padded = None
    teacher_lengths_tensor = None
    if teacher_seqs:
        teacher_padded = pad_sequence(
            teacher_seqs, batch_first=True, padding_value=pad_value
        )
        teacher_lengths_tensor = torch.tensor(teacher_lengths, dtype=torch.long)

    token_padded = pad_sequence(
        token_seqs,
        batch_first=True,
        padding_value=vocab.pad_id if vocab else 0,
    )
    token_lengths_tensor = torch.tensor(token_lengths, dtype=torch.long)

    return {
        "utterance_id": utterance_ids,
        "transcript": transcripts,
        "emg": emg_padded,
        "emg_lengths": emg_lengths_tensor,
        "teacher": teacher_padded,
        "teacher_lengths": teacher_lengths_tensor,
        "tokens": token_padded,
        "token_lengths": token_lengths_tensor,
    }


def make_dataloader(
    index_path: Path,
    features_root: Path,
    splits: Sequence[str],
    subsets: Optional[Sequence[str]],
    vocab: Vocab,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    spec_augment_cfg: Optional[SpecAugmentConfig] = None,
    include_teacher: bool = True,
    strict: bool = True,
    max_items: Optional[int] = None,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    channel_dropout_cfg: Optional[ChannelDropoutConfig] = None,
) -> DataLoader:
    dataset = EMGFeatureDataset(
        index_path=index_path,
        features_root=features_root,
        splits=splits,
        vocab=vocab,
        subsets=subsets,
        include_teacher=include_teacher,
        strict=strict,
        channel_dropout_cfg=channel_dropout_cfg,
    )
    if max_items is not None:
        max_items = min(max_items, len(dataset))
        dataset = Subset(dataset, range(max_items))
    spec_aug = SpecAugment(spec_augment_cfg) if spec_augment_cfg else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=functools.partial(collate_batch, spec_augment=spec_aug, vocab=vocab),
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
