"""Training loop for EMG-to-text distillation/CTC model."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import SpecAugmentConfig, make_dataloader
from src.data.vocab import Vocab
from src.models.emg_encoder import EMGConformerEncoder, EncoderConfig
from src.models.heads import CTCHead, ProjectionHead
from src.models.losses import DistillationCTCLoss, LossWeights

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict, path: Path) -> None:
    path.write_text(json.dumps(cfg, indent=2))


def _resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: Dict, input_dim: int, vocab_size: int, device: torch.device) -> Tuple[nn.Module, nn.Module, nn.Module]:
    enc_cfg = cfg["model"]["encoder"]
    encoder = EMGConformerEncoder(
        EncoderConfig(
            input_dim=input_dim,
            d_model=enc_cfg["d_model"],
            num_layers=enc_cfg["num_layers"],
            num_heads=enc_cfg["num_heads"],
            ffn_dim=enc_cfg["ffn_dim"],
            depthwise_conv_kernel_size=enc_cfg["depthwise_conv_kernel_size"],
            dropout=enc_cfg.get("dropout", 0.1),
            subsample_factor=enc_cfg.get("subsample_factor", 4),
        )
    )
    projection = ProjectionHead(
        input_dim=enc_cfg["d_model"],
        output_dim=cfg["model"]["projection_dim"],
        dropout=enc_cfg.get("dropout", 0.1),
    )
    ctc_head = CTCHead(
        input_dim=enc_cfg["d_model"],
        vocab_size=vocab_size,
        dropout=cfg["model"].get("ctc_dropout", 0.1),
    )
    encoder.to(device)
    projection.to(device)
    ctc_head.to(device)
    return encoder, projection, ctc_head


def save_checkpoint(
    run_dir: Path,
    epoch: int,
    step: int,
    encoder: nn.Module,
    projection: nn.Module,
    ctc_head: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None,
    cfg: Dict,
    is_best: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "encoder": encoder.state_dict(),
        "projection": projection.state_dict(),
        "ctc_head": ctc_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "config": cfg,
    }
    ckpt_path = run_dir / "last.pt"
    torch.save(payload, ckpt_path)
    if is_best:
        torch.save(payload, run_dir / "best.pt")
    save_config(cfg, run_dir / "config.json")


def train_one_epoch(
    epoch: int,
    encoder: nn.Module,
    projection: nn.Module,
    ctc_head: nn.Module,
    loss_fn: DistillationCTCLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None,
    loader: DataLoader,
    device: torch.device,
    grad_accum: int,
    clip_grad_norm: float,
    log_interval: int,
    writer: SummaryWriter | None,
    global_step: int,
) -> int:
    encoder.train()
    projection.train()
    ctc_head.train()

    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(loader):
        emg = batch["emg"].to(device)
        emg_lengths = batch["emg_lengths"].to(device)
        tokens = batch["tokens"].to(device)
        token_lengths = batch["token_lengths"].to(device)
        teacher = (
            batch["teacher"].to(device) if batch["teacher"] is not None else None
        )

        enc_out, enc_lengths = encoder(emg, emg_lengths)
        student_repr = projection(enc_out)
        log_probs = ctc_head(enc_out)

        losses = loss_fn(
            log_probs=log_probs,
            logit_lengths=enc_lengths.cpu(),
            targets=tokens,
            target_lengths=token_lengths,
            student_repr=student_repr,
            teacher_repr=teacher,
        )
        loss = losses["total"] / grad_accum
        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters())
                    + list(projection.parameters())
                    + list(ctc_head.parameters()),
                    max_norm=clip_grad_norm,
                )
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if writer and (global_step % log_interval == 0):
            writer.add_scalar("train/total_loss", losses["total"].item(), global_step)
            writer.add_scalar("train/ctc_loss", losses["ctc"].item(), global_step)
            writer.add_scalar("train/distill_loss", losses["distill"].item(), global_step)
        global_step += 1

    return global_step


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    projection: nn.Module,
    ctc_head: nn.Module,
    loss_fn: DistillationCTCLoss,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    encoder.eval()
    projection.eval()
    ctc_head.eval()

    totals = []
    ctc_losses = []
    distill_losses = []
    for batch in loader:
        emg = batch["emg"].to(device)
        emg_lengths = batch["emg_lengths"].to(device)
        tokens = batch["tokens"].to(device)
        token_lengths = batch["token_lengths"].to(device)
        teacher = batch["teacher"].to(device) if batch["teacher"] is not None else None

        enc_out, enc_lengths = encoder(emg, emg_lengths)
        student_repr = projection(enc_out)
        log_probs = ctc_head(enc_out)
        losses = loss_fn(
            log_probs=log_probs,
            logit_lengths=enc_lengths.cpu(),
            targets=tokens,
            target_lengths=token_lengths,
            student_repr=student_repr,
            teacher_repr=teacher,
        )
        totals.append(losses["total"].item())
        ctc_losses.append(losses["ctc"].item())
        distill_losses.append(losses["distill"].item())

    return {
        "total": float(np.mean(totals)) if totals else 0.0,
        "ctc": float(np.mean(ctc_losses)) if ctc_losses else 0.0,
        "distill": float(np.mean(distill_losses)) if distill_losses else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EMG-to-text model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Override checkpoint/log directory. Defaults to results/checkpoints/<run_name>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single epoch over a tiny subset for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    cfg = load_config(args.config)

    device = _resolve_device()
    logger.info("Using device: %s", device)
    set_seed(cfg["logging"].get("seed", 42))

    vocab = Vocab.from_json(Path(cfg["data"]["vocab"]))

    spec_cfg = None
    spec_section = cfg.get("augmentation", {}).get("specaugment")
    if spec_section and spec_section.get("p", 0) > 0:
        spec_cfg = SpecAugmentConfig(
            time_masks=spec_section.get("time_masks", 2),
            time_mask_width=spec_section.get("time_mask_width", 0.05),
            freq_masks=spec_section.get("freq_masks", 2),
            freq_mask_width=spec_section.get("freq_mask_width", 8),
            p=spec_section.get("p", 0.0),
        )

    train_loader = make_dataloader(
        index_path=Path(cfg["data"]["index"]),
        features_root=Path(cfg["data"]["features_root"]),
        splits=cfg["data"]["train_splits"],
        subsets=cfg["data"].get("train_subsets"),
        vocab=vocab,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=True,
        num_workers=cfg["optim"].get("num_workers", 0),
        spec_augment_cfg=spec_cfg,
        include_teacher=True,
    )
    val_loader = make_dataloader(
        index_path=Path(cfg["data"]["index"]),
        features_root=Path(cfg["data"]["features_root"]),
        splits=cfg["data"]["val_splits"],
        subsets=cfg["data"].get("val_subsets"),
        vocab=vocab,
        batch_size=max(1, cfg["optim"]["batch_size"] // 2),
        shuffle=False,
        num_workers=cfg["optim"].get("num_workers", 0),
        spec_augment_cfg=None,
        include_teacher=True,
    )

    # Infer input dimension from a sample.
    sample = next(iter(train_loader))
    input_dim = sample["emg"].shape[-1]
    teacher_dim = sample["teacher"].shape[-1] if sample["teacher"] is not None else cfg["features"]["teacher"]["dim"]

    encoder, projection, ctc_head = build_model(cfg, input_dim=input_dim, vocab_size=vocab.size, device=device)
    loss_fn = DistillationCTCLoss(
        vocab_size=vocab.size,
        blank_id=vocab.blank_id,
        weights=LossWeights(
            lambda_distill=cfg["loss"]["lambda_distill"],
            lambda_ctc=cfg["loss"]["lambda_ctc"],
        ),
    ).to(device)

    params = list(encoder.parameters()) + list(projection.parameters()) + list(ctc_head.parameters())
    lr = float(cfg["optim"]["lr"])
    weight_decay = float(cfg["optim"].get("weight_decay", 0.0))
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = None

    run_name = cfg["logging"].get("run_name", "run")
    run_dir = args.run_dir or Path("results/checkpoints") / run_name
    writer = SummaryWriter(log_dir=run_dir / "tb")

    max_epochs = 1 if args.dry_run else cfg["optim"].get("max_epochs", 1)
    best_val = float("inf")
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        global_step = train_one_epoch(
            epoch=epoch,
            encoder=encoder,
            projection=projection,
            ctc_head=ctc_head,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=train_loader,
            device=device,
            grad_accum=cfg["optim"].get("grad_accum", 1),
            clip_grad_norm=cfg["optim"].get("clip_grad_norm", 0.0),
            log_interval=cfg["logging"].get("log_interval", 10),
            writer=writer,
            global_step=global_step,
        )
        train_time = time.time() - start
        val_losses = evaluate(
            encoder=encoder,
            projection=projection,
            ctc_head=ctc_head,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
        )
        logger.info(
            "Epoch %d done in %.1fs | val total %.4f (ctc %.4f, distill %.4f)",
            epoch,
            train_time,
            val_losses["total"],
            val_losses["ctc"],
            val_losses["distill"],
        )
        writer.add_scalar("val/total_loss", val_losses["total"], epoch)
        writer.add_scalar("val/ctc_loss", val_losses["ctc"], epoch)
        writer.add_scalar("val/distill_loss", val_losses["distill"], epoch)

        is_best = val_losses["total"] < best_val
        if is_best:
            best_val = val_losses["total"]
        save_checkpoint(
            run_dir=run_dir,
            epoch=epoch,
            step=global_step,
            encoder=encoder,
            projection=projection,
            ctc_head=ctc_head,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            is_best=is_best,
        )

        if args.dry_run:
            break

    writer.close()


if __name__ == "__main__":
    main()
