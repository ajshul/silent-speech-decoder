"""Training loop for EMG-to-text distillation/CTC model."""

from __future__ import annotations

import argparse
import json
import logging
import math
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
from tqdm import tqdm

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


def build_scheduler(
    cfg: Dict, optimizer: optim.Optimizer, total_updates: int
) -> optim.lr_scheduler._LRScheduler | None:
    """
    Supports simple cosine annealing or linear warmup/decay schedulers.
    Accepts either a string name or a config dict with fields:
      - name/type: "cosine" or "linear"
      - warmup_steps: int (linear)
      - eta_min: float (cosine)
      - t_max/total_steps: int (cosine/linear decay horizon)
    """
    sched_cfg = cfg["optim"].get("scheduler")
    if not sched_cfg:
        return None

    name = sched_cfg if isinstance(sched_cfg, str) else sched_cfg.get("name", sched_cfg.get("type", ""))
    name = str(name).lower()
    total_updates = max(1, total_updates)

    if name in {"cosine", "cosineannealing", "cosine_annealing"}:
        t_max = int(sched_cfg.get("t_max", total_updates)) if isinstance(sched_cfg, dict) else total_updates
        eta_min = float(sched_cfg.get("eta_min", 0.0)) if isinstance(sched_cfg, dict) else 0.0
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if name in {"linear", "linear_warmup", "warmup"}:
        warmup_steps = int(sched_cfg.get("warmup_steps", 0)) if isinstance(sched_cfg, dict) else 0
        decay_steps = int(sched_cfg.get("total_steps", total_updates)) if isinstance(sched_cfg, dict) else total_updates

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown scheduler '{name}'")


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

    if len(loader) == 0:
        return global_step

    last_losses: Dict[str, torch.Tensor] | None = None

    def optimizer_step() -> None:
        nonlocal global_step, last_losses
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
        global_step += 1
        if writer and last_losses is not None and (global_step % log_interval == 0 or global_step == 1):
            writer.add_scalar("train/total_loss", last_losses["total"].item(), global_step)
            writer.add_scalar("train/ctc_loss", last_losses["ctc"].item(), global_step)
            writer.add_scalar("train/distill_loss", last_losses["distill"].item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(
        total=len(loader),
        desc=f"Epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )
    next_log = log_interval
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

        last_losses = loss_fn(
            log_probs=log_probs,
            logit_lengths=enc_lengths.cpu(),
            targets=tokens,
            target_lengths=token_lengths,
            student_repr=student_repr,
            teacher_repr=teacher,
        )
        loss = last_losses["total"] / grad_accum
        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            optimizer_step()

        # Lightweight progress updates: only refresh text every log_interval to avoid overhead.
        if (batch_idx + 1) >= next_log or (batch_idx + 1) == len(loader):
            if last_losses is not None:
                pbar.set_postfix(
                    total=f"{last_losses['total'].item():.3f}",
                    ctc=f"{last_losses['ctc'].item():.3f}",
                    distill=f"{last_losses['distill'].item():.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
            next_log += log_interval
        pbar.update(1)

    # Final step for leftover gradients when len(loader) is not divisible by grad_accum.
    if len(loader) % grad_accum != 0:
        optimizer_step()

    pbar.close()
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
    parser.add_argument(
        "--overfit-batches",
        type=int,
        default=0,
        help="Limit train/val to this many batches to validate the model can overfit.",
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

    train_limit = None
    val_limit = None
    shuffle_train = True
    if args.overfit_batches > 0:
        train_limit = args.overfit_batches * cfg["optim"]["batch_size"]
        val_limit = train_limit
        shuffle_train = False
        logger.info("Overfitting on %d batches (~%d items) for train/val.", args.overfit_batches, train_limit)

    train_loader = make_dataloader(
        index_path=Path(cfg["data"]["index"]),
        features_root=Path(cfg["data"]["features_root"]),
        splits=cfg["data"]["train_splits"],
        subsets=cfg["data"].get("train_subsets"),
        vocab=vocab,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=shuffle_train,
        num_workers=cfg["optim"].get("num_workers", 0),
        spec_augment_cfg=spec_cfg,
        include_teacher=True,
        max_items=train_limit,
        pin_memory=cfg["optim"].get("pin_memory", False),
        prefetch_factor=cfg["optim"].get("prefetch_factor"),
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
        max_items=val_limit,
        pin_memory=cfg["optim"].get("pin_memory", False),
        prefetch_factor=cfg["optim"].get("prefetch_factor"),
    )

    logger.info(
        "Train batches: %d | Val batches: %d | batch size: %d | grad_accum: %d",
        len(train_loader),
        len(val_loader),
        cfg["optim"]["batch_size"],
        cfg["optim"].get("grad_accum", 1),
    )

    # Infer input dimension from a sample.
    sample = next(iter(train_loader))
    input_dim = sample["emg"].shape[-1]
    teacher_dim = (
        sample["teacher"].shape[-1]
        if sample["teacher"] is not None
        else cfg["features"]["teacher"]["dim"]
    )

    encoder, projection, ctc_head = build_model(cfg, input_dim=input_dim, vocab_size=vocab.size, device=device)
    base_loss_weights = LossWeights(
        lambda_distill=float(cfg["loss"]["lambda_distill"]),
        lambda_ctc=float(cfg["loss"]["lambda_ctc"]),
    )
    loss_fn = DistillationCTCLoss(
        vocab_size=vocab.size,
        blank_id=vocab.blank_id,
        weights=base_loss_weights,
    ).to(device)
    distill_warmup_epochs = int(cfg["loss"].get("distill_warmup_epochs", 0))

    params = list(encoder.parameters()) + list(projection.parameters()) + list(ctc_head.parameters())
    lr = float(cfg["optim"]["lr"])
    weight_decay = float(cfg["optim"].get("weight_decay", 0.0))
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    grad_accum = cfg["optim"].get("grad_accum", 1)
    max_epochs = 1 if args.dry_run else cfg["optim"].get("max_epochs", 1)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_updates = max_epochs * updates_per_epoch
    scheduler = build_scheduler(cfg, optimizer, total_updates)

    run_name = cfg["logging"].get("run_name", "run")
    run_dir = args.run_dir or Path("results/checkpoints") / run_name
    writer = SummaryWriter(log_dir=run_dir / "tb")

    best_val = float("inf")
    best_epoch = 0
    global_step = 0
    early_cfg = cfg["optim"].get("early_stopping", {})
    patience = int(early_cfg.get("patience", 0)) if early_cfg else 0
    min_delta = float(early_cfg.get("min_delta", 0.0)) if early_cfg else 0.0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        warmup_scale = 1.0
        if distill_warmup_epochs > 0:
            warmup_scale = min(1.0, epoch / float(distill_warmup_epochs))
        loss_fn.weights = LossWeights(
            lambda_distill=base_loss_weights.lambda_distill * warmup_scale,
            lambda_ctc=base_loss_weights.lambda_ctc,
        )
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
            grad_accum=grad_accum,
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
            "Epoch %d done in %.1fs | val total %.4f (ctc %.4f, distill %.4f) | loss weights ctc %.2f distill %.2f",
            epoch,
            train_time,
            val_losses["total"],
            val_losses["ctc"],
            val_losses["distill"],
            loss_fn.weights.lambda_ctc,
            loss_fn.weights.lambda_distill,
        )
        writer.add_scalar("val/total_loss", val_losses["total"], epoch)
        writer.add_scalar("val/ctc_loss", val_losses["ctc"], epoch)
        writer.add_scalar("val/distill_loss", val_losses["distill"], epoch)
        writer.add_scalar("train/lambda_ctc", loss_fn.weights.lambda_ctc, epoch)
        writer.add_scalar("train/lambda_distill", loss_fn.weights.lambda_distill, epoch)

        is_best = val_losses["total"] < (best_val - min_delta)
        if is_best:
            best_val = val_losses["total"]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
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

        if patience and patience_counter >= patience:
            logger.info(
                "Early stopping at epoch %d (best epoch %d val %.4f)", epoch, best_epoch, best_val
            )
            break

    writer.close()


if __name__ == "__main__":
    main()
