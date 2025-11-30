from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import make_dataloader
from src.data.vocab import Vocab
from src.models.emg_encoder import EMGConformerEncoder, EncoderConfig
from src.models.heads import CTCHead, ProjectionHead
from src.models.losses import DistillationCTCLoss, LossWeights


def _prepare_features(tmp_path: Path) -> tuple[Path, Path]:
    features_root = tmp_path / "features"
    emg_dir = features_root / "emg" / "split" / "spk"
    teacher_dir = features_root / "teacher" / "split" / "spk"
    emg_dir.mkdir(parents=True)
    teacher_dir.mkdir(parents=True)

    emg = np.random.randn(6, 2, 3).astype(np.float32)
    teacher = np.random.randn(3, 8).astype(np.float32)
    np.save(emg_dir / "000.npy", emg)
    np.save(teacher_dir / "000.npy", teacher)

    index_path = tmp_path / "index.parquet"
    pd.DataFrame(
        [
            {
                "utterance_id": "split/spk/000",
                "split": "train",
                "emg_path": "dummy",
                "audio_path": "dummy",
                "transcript": "ab",
            }
        ]
    ).to_parquet(index_path, index=False)
    return index_path, features_root


def test_train_step_smoke(tmp_path: Path) -> None:
    index_path, features_root = _prepare_features(tmp_path)
    vocab = Vocab(tokens=["<pad>", "<blank>", "<unk>", "a", "b"], token_to_id={"<pad>": 0, "<blank>": 1, "<unk>": 2, "a": 3, "b": 4}, pad_id=0, blank_id=1, unk_id=2)

    loader = make_dataloader(
        index_path=index_path,
        features_root=features_root,
        splits=["train"],
        subsets=None,
        vocab=vocab,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        spec_augment_cfg=None,
        include_teacher=True,
    )
    batch = next(iter(loader))
    input_dim = batch["emg"].shape[-1]

    encoder = EMGConformerEncoder(
        EncoderConfig(
            input_dim=input_dim,
            d_model=8,
            num_layers=1,
            num_heads=2,
            ffn_dim=16,
            subsample_factor=2,
        )
    )
    projection = ProjectionHead(input_dim=8, output_dim=8, dropout=0.0)
    ctc_head = CTCHead(input_dim=8, vocab_size=vocab.size, dropout=0.0)
    loss_fn = DistillationCTCLoss(vocab_size=vocab.size, blank_id=vocab.blank_id, weights=LossWeights())

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projection.parameters()) + list(ctc_head.parameters()), lr=1e-3)

    encoder.train()
    projection.train()
    ctc_head.train()

    emg = batch["emg"]
    emg_lengths = batch["emg_lengths"]
    tokens = batch["tokens"]
    token_lengths = batch["token_lengths"]
    teacher = batch["teacher"]

    enc_out, enc_lengths = encoder(emg, emg_lengths)
    student_repr = projection(enc_out)
    log_probs = ctc_head(enc_out)

    losses = loss_fn(
        log_probs=log_probs,
        logit_lengths=enc_lengths,
        targets=tokens,
        target_lengths=token_lengths,
        student_repr=student_repr,
        teacher_repr=teacher,
    )
    losses["total"].backward()
    optimizer.step()

    assert losses["total"].item() >= 0
