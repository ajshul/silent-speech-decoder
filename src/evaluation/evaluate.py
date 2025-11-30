"""Evaluation script: load a checkpoint, decode EMG features, and compute WER/CER."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jiwer
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import make_dataloader
from src.data.vocab import Vocab
from src.models.emg_encoder import EMGConformerEncoder, EncoderConfig
from src.models.heads import CTCHead, ProjectionHead

logger = logging.getLogger(__name__)


def _resolve_device(force: str | None = None) -> torch.device:
    if force:
        return torch.device(force)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int) -> List[List[int]]:
    """Argmax decode per frame, collapse repeats and blanks."""
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


def decode_batch(
    encoder: torch.nn.Module,
    ctc_head: torch.nn.Module,
    batch: Dict,
    device: torch.device,
    blank_id: int,
    vocab: Vocab,
) -> Tuple[List[str], List[str], List[str]]:
    emg = batch["emg"].to(device)
    emg_lengths = batch["emg_lengths"].to(device)
    transcripts = batch["transcript"]
    utterance_ids = batch["utterance_id"]

    with torch.no_grad():
        enc_out, enc_lengths = encoder(emg, emg_lengths)
        log_probs = ctc_head(enc_out)
    token_seqs = greedy_decode(log_probs.cpu(), enc_lengths.cpu(), blank_id=blank_id)
    decoded = [vocab.decode(seq) for seq in token_seqs]
    return utterance_ids, transcripts, decoded


def compute_metrics(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    return {
        "wer": float(jiwer.wer(refs, hyps)),
        "cer": float(jiwer.cer(refs, hyps)),
    }


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device, cfg: Dict, vocab_size: int):
    payload = torch.load(ckpt_path, map_location=device)

    enc_cfg = cfg["model"]["encoder"]
    if "input_dim" not in enc_cfg:
        raise ValueError("encoder.input_dim must be set before loading the model.")

    encoder = EMGConformerEncoder(
        EncoderConfig(
            input_dim=enc_cfg["input_dim"],
            d_model=enc_cfg["d_model"],
            num_layers=enc_cfg["num_layers"],
            num_heads=enc_cfg["num_heads"],
            ffn_dim=enc_cfg["ffn_dim"],
            depthwise_conv_kernel_size=enc_cfg["depthwise_conv_kernel_size"],
            dropout=enc_cfg.get("dropout", 0.1),
            subsample_factor=enc_cfg.get("subsample_factor", 2),
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

    encoder.load_state_dict(payload["encoder"])
    projection.load_state_dict(payload["projection"])
    ctc_head.load_state_dict(payload["ctc_head"])

    encoder.to(device).eval()
    projection.to(device).eval()
    ctc_head.to(device).eval()
    return encoder, projection, ctc_head, cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--index", type=Path, help="Override index path (defaults to config).")
    parser.add_argument("--features-root", type=Path, help="Override features root (defaults to config).")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to evaluate (default: config train/val).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Optional subsets to filter (e.g., val test eval_silent).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, help="cpu/mps/cuda (auto if unset).")
    parser.add_argument("--output", type=Path, help="Where to store outputs (default: results/eval/<run_name>).")
    parser.add_argument("--run-name", type=str, help="Override run name for output folder.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    device = _resolve_device(args.device)
    ckpt_path = args.checkpoint
    payload_cfg = torch.load(ckpt_path, map_location="cpu")["config"]
    cfg = payload_cfg
    data_cfg = cfg["data"]
    index_path = args.index or Path(data_cfg["index"])
    features_root = args.features_root or Path(data_cfg["features_root"])
    splits = args.splits or data_cfg.get("val_splits", ["voiced_parallel_data"])
    subsets = args.subsets

    vocab = Vocab.from_json(Path(data_cfg["vocab"]))

    # Infer input dim from metadata saved in config if present, else from a sample batch.
    encoder_cfg = cfg["model"]["encoder"]
    if "input_dim" not in encoder_cfg:
        tmp_loader = make_dataloader(
            index_path=index_path,
            features_root=features_root,
            splits=splits,
            subsets=subsets,
            vocab=vocab,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            spec_augment_cfg=None,
            include_teacher=False,
        )
        sample = next(iter(tmp_loader))
        encoder_cfg["input_dim"] = sample["emg"].shape[-1]
        cfg["model"]["encoder"]["input_dim"] = encoder_cfg["input_dim"]

    encoder, _, ctc_head, _ = load_model_from_checkpoint(
        ckpt_path, device=device, cfg=cfg, vocab_size=vocab.size
    )

    loader: DataLoader = make_dataloader(
        index_path=index_path,
        features_root=features_root,
        splits=splits,
        subsets=subsets,
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        spec_augment_cfg=None,
        include_teacher=False,
    )

    run_name = args.run_name or cfg.get("logging", {}).get("run_name", "eval_run")
    out_dir = args.output or Path("results/eval") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_refs: List[str] = []
    all_hyps: List[str] = []
    records: List[Dict] = []

    logger.info(
        "Evaluating checkpoint %s on %s splits %s subsets %s",
        ckpt_path,
        index_path,
        splits,
        subsets if subsets else "all",
    )

    for batch in loader:
        utterance_ids, refs, hyps = decode_batch(
            encoder=encoder,
            ctc_head=ctc_head,
            batch=batch,
            device=device,
            blank_id=vocab.blank_id,
            vocab=vocab,
        )
        for uid, ref, hyp in zip(utterance_ids, refs, hyps):
            all_refs.append(ref)
            all_hyps.append(hyp)
            records.append({"utterance_id": uid, "ref": ref, "hyp": hyp})

    metrics = compute_metrics(all_refs, all_hyps)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "predictions.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("WER: %.4f | CER: %.4f | outputs: %s", metrics["wer"], metrics["cer"], out_dir)


if __name__ == "__main__":
    main()
