import torch

from src.models.emg_encoder import EMGConformerEncoder, EncoderConfig
from src.models.heads import CTCHead, ProjectionHead
from src.models.losses import DistillationCTCLoss, LossWeights


def test_emg_conformer_shapes() -> None:
    cfg = EncoderConfig(
        input_dim=6,
        d_model=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        subsample_factor=2,
    )
    encoder = EMGConformerEncoder(cfg)
    x = torch.randn(2, 10, cfg.input_dim)
    lengths = torch.tensor([10, 8])
    out, out_lengths = encoder(x, lengths)
    assert out.shape[0] == 2
    assert out.shape[2] == cfg.d_model
    assert out_lengths.tolist() == [5, 4]


def test_heads_and_losses() -> None:
    batch = 2
    time = 6
    d_model = 8
    vocab_size = 5

    feats = torch.randn(batch, time, d_model)
    proj = ProjectionHead(d_model, 4)
    proj_out = proj(feats)
    assert proj_out.shape == (batch, time, 4)

    ctc_head = CTCHead(d_model, vocab_size)
    log_probs = ctc_head(feats)
    assert log_probs.shape == (batch, time, vocab_size)

    targets = torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)
    logit_lengths = torch.tensor([time, time], dtype=torch.long)
    teacher = torch.randn(batch, time, proj_out.shape[-1])

    loss_fn = DistillationCTCLoss(vocab_size=vocab_size, blank_id=0, weights=LossWeights())
    losses = loss_fn(
        log_probs=log_probs,
        logit_lengths=logit_lengths,
        targets=targets,
        target_lengths=target_lengths,
        student_repr=proj_out,
        teacher_repr=teacher,
    )
    assert set(losses.keys()) == {"total", "ctc", "distill"}
    assert losses["total"].item() >= 0
