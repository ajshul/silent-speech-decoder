from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import (
    EMGFeatureDataset,
    SpecAugment,
    SpecAugmentConfig,
    collate_batch,
)
from src.data.vocab import Vocab


def _make_vocab() -> Vocab:
    tokens = ["<pad>", "<blank>", "<unk>", " ", "a", "b"]
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    return Vocab(tokens=tokens, token_to_id=token_to_id, pad_id=0, blank_id=1, unk_id=2)


def _write_index(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def test_dataset_and_collate_with_teacher(tmp_path: Path) -> None:
    features_root = tmp_path / "features"
    emg_dir = features_root / "emg" / "split" / "spk"
    teacher_dir = features_root / "teacher" / "split" / "spk"
    emg_dir.mkdir(parents=True)
    teacher_dir.mkdir(parents=True)

    emg1 = np.random.randn(10, 2, 3).astype(np.float32)
    emg2 = np.random.randn(8, 2, 3).astype(np.float32)
    np.save(emg_dir / "001.npy", emg1)
    np.save(emg_dir / "002.npy", emg2)

    teacher1 = np.random.randn(5, 4).astype(np.float32)
    teacher2 = np.random.randn(6, 4).astype(np.float32)
    np.save(teacher_dir / "001.npy", teacher1)
    np.save(teacher_dir / "002.npy", teacher2)

    index_path = tmp_path / "index.parquet"
    _write_index(
        index_path,
        [
            {
                "utterance_id": "split/spk/001",
                "split": "train",
                "emg_path": "dummy",
                "audio_path": "dummy",
                "transcript": "ab",
            },
            {
                "utterance_id": "split/spk/002",
                "split": "train",
                "emg_path": "dummy",
                "audio_path": "dummy",
                "transcript": "a",
            },
        ],
    )

    vocab = _make_vocab()
    ds = EMGFeatureDataset(
        index_path=index_path,
        features_root=features_root,
        splits=["train"],
        vocab=vocab,
        include_teacher=True,
    )

    batch = [ds[0], ds[1]]
    out = collate_batch(batch, spec_augment=None, vocab=vocab)

    assert out["emg"].shape[0] == 2  # batch
    assert out["emg"].shape[1] == emg1.shape[0]  # padded to longest
    assert out["emg"].shape[2] == emg1.shape[1] * emg1.shape[2]
    assert out["teacher"].shape[1] == teacher2.shape[0]
    assert out["tokens"].shape[1] == 2  # longest transcript length
    assert out["token_lengths"].tolist() == [2, 1]


def test_specaugment_masks_time_and_freq() -> None:
    cfg = SpecAugmentConfig(time_masks=1, time_mask_width=0.5, freq_masks=1, freq_mask_width=2, p=1.0)
    aug = SpecAugment(cfg)
    feat = torch.ones(10, 4)
    out = aug(feat)
    assert out.shape == feat.shape
    # At least one zeroed region in time or freq.
    assert (out == 0).any()
