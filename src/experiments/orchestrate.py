from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from src.experiments.config_builder import (
    DecoderSetting,
    RunSpec,
    VOICED_BASE_CONFIG,
    build_silent_probe_configs,
    build_silent_stage2_configs,
    build_voiced_probe_configs,
    build_voiced_stage2_configs,
)

LOG = logging.getLogger(__name__)
CONFIG_OUT_DIR = Path("results/experiments/configs")
SUMMARY_JSON = Path("results/experiments/summary.json")
SUMMARY_CSV = Path("results/experiments/summary.csv")


def run_command(cmd: List[str], dry_run: bool) -> None:
    LOG.info("Running: %s", " ".join(str(x) for x in cmd))
    if dry_run:
        LOG.info("[dry-run] skipping execution")
        return
    subprocess.run(cmd, check=True)


def write_config(spec: RunSpec, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.yaml"
    with path.open("w") as f:
        yaml.safe_dump(spec.config, f, sort_keys=False)
    return path


def _checkpoint_path(run_name: str) -> Path:
    return Path("results/checkpoints") / run_name / "best.pt"


def _config_features(cfg: Dict) -> Dict:
    aug = cfg.get("augmentation", {})
    spec = aug.get("specaugment", {}) or {}
    channel = aug.get("channel_dropout", {}) or {}
    decoding = cfg.get("decoding", {}) or {}
    sched_cfg = cfg.get("optim", {}).get("scheduler")
    if isinstance(sched_cfg, dict):
        scheduler_name = sched_cfg.get("name") or sched_cfg.get("type")
    else:
        scheduler_name = sched_cfg
    return {
        "specaugment_p": spec.get("p", 0.0),
        "specaugment_time_masks": spec.get("time_masks"),
        "specaugment_freq_masks": spec.get("freq_masks"),
        "specaugment_time_width": spec.get("time_mask_width"),
        "specaugment_freq_width": spec.get("freq_mask_width"),
        "channel_dropout_p": channel.get("p", 0.0),
        "channel_dropout_max": channel.get("max_channels"),
        "lambda_ctc": cfg.get("loss", {}).get("lambda_ctc"),
        "lambda_distill": cfg.get("loss", {}).get("lambda_distill"),
        "distill_warmup_epochs": cfg.get("loss", {}).get("distill_warmup_epochs"),
        "subsample_factor": cfg.get("model", {}).get("encoder", {}).get("subsample_factor"),
        "dropout": cfg.get("model", {}).get("encoder", {}).get("dropout"),
        "scheduler": scheduler_name,
        "scheduler_cfg": sched_cfg,
        "batch_size": cfg.get("optim", {}).get("batch_size"),
        "max_epochs": cfg.get("optim", {}).get("max_epochs"),
        "lr": cfg.get("optim", {}).get("lr"),
        "weight_decay": cfg.get("optim", {}).get("weight_decay"),
        "decoding_default": decoding,
        "experiment_tags": cfg.get("experiment", {}).get("tags", []),
        "experiment_description": cfg.get("experiment", {}).get("description", ""),
        "probe_batches": cfg.get("experiment", {}).get("probe_batches"),
    }


def _load_yaml(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_trained(spec: RunSpec, config_path: Path, dry_run: bool, force: bool) -> Optional[Path]:
    ckpt = _checkpoint_path(spec.name)
    if ckpt.exists() and not force:
        LOG.info("Checkpoint already exists for %s, skipping train.", spec.name)
        return ckpt
    cmd = ["python", "-m", "src.training.train", "--config", str(config_path), "--run-dir", str(ckpt.parent)]
    if spec.init_checkpoint:
        cmd += ["--init-checkpoint", str(spec.init_checkpoint)]
    if spec.overfit_batches:
        cmd += ["--overfit-batches", str(spec.overfit_batches)]
    run_command(cmd, dry_run=dry_run)
    return ckpt if ckpt.exists() or dry_run else None


def evaluate_checkpoint(
    spec: RunSpec,
    decoder: DecoderSetting,
    ckpt_path: Path,
    eval_batch_size: int,
    dry_run: bool,
    force: bool,
    config_batch_size: Optional[int],
    lm_available: bool,
) -> Optional[Path]:
    eval_run_name = f"{spec.name}__{decoder.name}"
    eval_dir = Path("results/eval") / eval_run_name
    metrics_path = eval_dir / "metrics.json"
    lm_path = decoder.lm_path
    if decoder.use_lm and not lm_available:
        LOG.info("Skipping decoder %s for %s (LM unavailable).", decoder.name, spec.name)
        return None
    if metrics_path.exists() and not force:
        LOG.info("Eval already exists for %s (%s), skipping.", spec.name, decoder.name)
        return eval_dir
    effective_batch = eval_batch_size
    if config_batch_size is not None:
        effective_batch = min(effective_batch, max(1, config_batch_size))
        if effective_batch != eval_batch_size:
            LOG.info("Clamping eval batch size to %d (train batch %d).", effective_batch, config_batch_size)
    cmd = [
        "python",
        "-m",
        "src.evaluation.evaluate",
        "--checkpoint",
        str(ckpt_path),
        "--run-name",
        eval_run_name,
        "--batch-size",
        str(effective_batch),
        "--decoder",
        decoder.method,
    ]
    if decoder.method == "beam":
        cmd += ["--beam-width", str(decoder.beam_width or 50)]
        if decoder.alpha is not None:
            cmd += ["--alpha", str(decoder.alpha)]
        if decoder.beta is not None:
            cmd += ["--beta", str(decoder.beta)]
        if decoder.beam_prune_logp is not None:
            cmd += ["--beam-prune-logp", str(decoder.beam_prune_logp)]
    if decoder.blank_bias:
        cmd += ["--blank-bias", str(decoder.blank_bias)]
    if decoder.use_lm and lm_path:
        cmd += ["--lm-path", str(lm_path)]
    run_command(cmd, dry_run=dry_run)
    return eval_dir if eval_dir.exists() or dry_run else None


def summarize_eval(
    spec: RunSpec,
    decoder: DecoderSetting,
    config_path: Path,
    ckpt_path: Path,
    eval_dir: Path,
    duration_sec: Optional[float] = None,
) -> Dict:
    metrics_file = eval_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(metrics_file)
    metrics = json.loads(metrics_file.read_text())
    cfg = _load_yaml(config_path)
    features = _config_features(cfg)
    record = {
        "stage": spec.stage,
        "dataset": spec.dataset,
        "train_run": spec.name,
        "decoder_name": decoder.name,
        "decoder_type": decoder.method,
        "beam_width": decoder.beam_width,
        "alpha": decoder.alpha,
        "beta": decoder.beta,
        "beam_prune_logp": decoder.beam_prune_logp,
        "blank_bias": decoder.blank_bias,
        "lm_used": decoder.use_lm and decoder.lm_path is not None and decoder.lm_path.exists(),
        "lm_path": str(decoder.lm_path) if decoder.lm_path else None,
        "metrics": metrics,
        "wer": metrics.get("wer"),
        "cer": metrics.get("cer"),
        "insertions": metrics.get("error_breakdown", {}).get("insertions"),
        "deletions": metrics.get("error_breakdown", {}).get("deletions"),
        "substitutions": metrics.get("error_breakdown", {}).get("substitutions"),
        "insertion_rate": metrics.get("error_breakdown", {}).get("insertion_rate"),
        "deletion_rate": metrics.get("error_breakdown", {}).get("deletion_rate"),
        "substitution_rate": metrics.get("error_breakdown", {}).get("substitution_rate"),
        "num_samples": metrics.get("data", {}).get("num_samples"),
        "config_path": str(config_path),
        "checkpoint_path": str(ckpt_path),
        "eval_dir": str(eval_dir),
        "features": features,
        "tags": spec.tags,
        "description": spec.description,
        "overfit_batches": spec.overfit_batches,
        "init_checkpoint": str(spec.init_checkpoint) if spec.init_checkpoint else None,
        "eval_duration_sec": duration_sec,
    }
    # Pull default decoder info if present in the config to support visualizations.
    default_dec = cfg.get("decoding", {}) or {}
    record["config_decoder_default"] = default_dec
    # Keep run_name for easier plotting.
    record["run_name"] = metrics.get("run_name", Path(eval_dir).name)
    return record


def run_specs(
    specs: Sequence[RunSpec],
    dry_run: bool,
    force_train: bool,
    force_eval: bool,
    eval_batch_size: int,
    existing_records: Optional[Sequence[Dict]] = None,
    lm_available: bool = True,
    summary_path: Optional[Path] = None,
) -> List[Dict]:
    results: List[Dict] = []
    existing_records = list(existing_records or [])
    existing_keys = {(rec.get("train_run"), rec.get("decoder_name")) for rec in existing_records}
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        LOG.info("=== Running %s (%s/%s) ===", spec.name, spec.stage, spec.dataset)
        config_path = write_config(spec, CONFIG_OUT_DIR)
        config_batch_size = spec.config.get("optim", {}).get("batch_size")
        ckpt_path = ensure_trained(spec, config_path, dry_run=dry_run, force=force_train)
        if ckpt_path is None and not dry_run:
            LOG.warning("No checkpoint produced for %s, skipping eval.", spec.name)
            continue
        for decoder in spec.decoder_grid:
            key = (spec.name, decoder.name)
            if key in existing_keys and not force_eval:
                LOG.info("Record already present for %s (%s); skipping eval due to resume.", spec.name, decoder.name)
                continue
            eval_dir = evaluate_checkpoint(
                spec=spec,
                decoder=decoder,
                ckpt_path=ckpt_path if ckpt_path else Path("missing"),
                eval_batch_size=eval_batch_size,
                dry_run=dry_run,
                force=force_eval,
                config_batch_size=config_batch_size,
                lm_available=lm_available,
            )
            if eval_dir is None:
                continue
            if dry_run:
                continue
            try:
                record = summarize_eval(spec, decoder, config_path, ckpt_path, eval_dir)
                results.append(record)
                if summary_path:
                    interim = existing_records + results
                    write_summary(interim, summary_path, summary_path.with_suffix(".csv"))
            except FileNotFoundError as exc:
                LOG.warning("Failed to summarize %s (%s): %s", spec.name, decoder.name, exc)
    return results


def pick_best(records: Sequence[Dict], dataset: str, stage: Optional[str] = None) -> Optional[Dict]:
    """
    Select best record prioritizing CER (primary), then WER, then deletion_rate.
    This emphasizes insertion control/blank tuning for silent EMG while still
    considering overall correctness.
    """
    filtered = [r for r in records if r.get("dataset") == dataset and (stage is None or r.get("stage") == stage)]
    filtered = [r for r in filtered if r.get("cer") is not None]
    filtered.sort(key=lambda r: (r.get("cer", 1e6), r.get("wer", 1e6), r.get("deletion_rate", 0.0)))
    return filtered[0] if filtered else None


def write_summary(records: List[Dict], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, indent=2))
    fieldnames = [
        "stage",
        "dataset",
        "train_run",
        "run_name",
        "decoder_name",
        "decoder_type",
        "wer",
        "cer",
        "insertion_rate",
        "deletion_rate",
        "substitution_rate",
        "beam_width",
        "alpha",
        "beta",
        "beam_prune_logp",
        "blank_bias",
        "lm_used",
        "specaugment_p",
        "channel_dropout_p",
        "subsample_factor",
        "lambda_ctc",
        "lambda_distill",
        "scheduler",
        "tags",
        "overfit_batches",
        "init_checkpoint",
        "config_path",
        "checkpoint_path",
        "eval_dir",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            feats = rec.get("features", {})
            writer.writerow(
                {
                    "stage": rec.get("stage"),
                    "dataset": rec.get("dataset"),
                    "train_run": rec.get("train_run"),
                    "run_name": rec.get("run_name"),
                    "decoder_name": rec.get("decoder_name"),
                    "decoder_type": rec.get("decoder_type"),
                    "wer": rec.get("wer"),
                    "cer": rec.get("cer"),
                    "insertion_rate": rec.get("insertion_rate"),
                    "deletion_rate": rec.get("deletion_rate"),
                    "substitution_rate": rec.get("substitution_rate"),
                    "beam_width": rec.get("beam_width"),
                    "alpha": rec.get("alpha"),
                    "beta": rec.get("beta"),
                    "beam_prune_logp": rec.get("beam_prune_logp"),
                    "blank_bias": rec.get("blank_bias"),
                    "lm_used": rec.get("lm_used"),
                    "specaugment_p": feats.get("specaugment_p"),
                    "channel_dropout_p": feats.get("channel_dropout_p"),
                    "subsample_factor": feats.get("subsample_factor"),
                    "lambda_ctc": feats.get("lambda_ctc"),
                    "lambda_distill": feats.get("lambda_distill"),
                    "scheduler": feats.get("scheduler"),
                    "tags": ",".join(rec.get("tags", [])),
                    "overfit_batches": rec.get("overfit_batches"),
                    "init_checkpoint": rec.get("init_checkpoint"),
                    "config_path": rec.get("config_path"),
                    "checkpoint_path": rec.get("checkpoint_path"),
                    "eval_dir": rec.get("eval_dir"),
                }
            )


def best_probe_to_knobs(record: Dict) -> Dict:
    feats = record.get("features", {})
    return {
        "specaugment_p": feats.get("specaugment_p"),
        "specaugment_time_masks": feats.get("specaugment_time_masks"),
        "specaugment_freq_masks": feats.get("specaugment_freq_masks"),
        "specaugment_time_width": feats.get("specaugment_time_width"),
        "specaugment_freq_width": feats.get("specaugment_freq_width"),
        "channel_dropout_p": feats.get("channel_dropout_p"),
        "channel_dropout_max": feats.get("channel_dropout_max"),
        "lambda_ctc": feats.get("lambda_ctc"),
        "lambda_distill": feats.get("lambda_distill"),
        "distill_warmup_epochs": feats.get("distill_warmup_epochs"),
        "subsample_factor": feats.get("subsample_factor"),
        "scheduler": feats.get("scheduler"),
        "scheduler_cfg": feats.get("scheduler_cfg"),
        "decoder_type": record.get("decoder_type"),
        "beam_width": record.get("beam_width"),
        "alpha": record.get("alpha"),
        "beta": record.get("beta"),
        "beam_prune_logp": record.get("beam_prune_logp"),
        "blank_bias": record.get("blank_bias"),
        "lm_path": record.get("lm_path"),
        "dropout": feats.get("dropout"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage experiment orchestrator (probes -> full runs).")
    parser.add_argument(
        "--probe-batches",
        type=int,
        default=48,
        help="Batches to use for Stage 1 probes (larger slices improve CER separation).",
    )
    parser.add_argument(
        "--probe-batches-silent",
        type=int,
        default=24,
        help="Batches to use for Stage 1 silent probes (smaller slice is sufficient and saves time).",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Batch size to use for evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Only write configs and print commands, do not execute.")
    parser.add_argument("--force-train", action="store_true", help="Re-run training even if checkpoints exist.")
    parser.add_argument("--force-eval", action="store_true", help="Re-run evaluation even if metrics exist.")
    parser.add_argument(
        "--stage",
        choices=["all", "stage1", "stage2"],
        default="all",
        help="Which stages to execute. Stage 2 requires Stage 1 summaries.",
    )
    parser.add_argument("--summary-json", type=Path, default=SUMMARY_JSON, help="Where to write the summary JSON.")
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV, help="Where to write the summary CSV.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing summary (skips evals already recorded unless --force-eval).",
    )
    parser.add_argument(
        "--preflight-overfit",
        action="store_true",
        help="Run a single-batch overfit check on the base voiced config before probes.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    all_records: List[Dict] = []
    if args.resume and args.summary_json.exists():
        try:
            all_records = json.loads(args.summary_json.read_text())
            LOG.info("Loaded %d existing records from %s for resume.", len(all_records), args.summary_json)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to load summary for resume: %s", exc)

    lm_available = Path("results/lm/char_5gram.arpa").exists()

    if args.preflight_overfit and args.stage in {"all", "stage1"}:
        LOG.info("Running preflight overfit check on baseline voiced config.")
        base_config = VOICED_BASE_CONFIG
        overfit_cmd = [
            "python",
            "-m",
            "src.training.train",
            "--config",
            str(base_config),
            "--run-dir",
            str(Path("results/checkpoints") / "preflight_overfit"),
            "--overfit-batches",
            "1",
            "--dry-run",
        ]
        run_command(overfit_cmd, dry_run=args.dry_run)

    voiced_probe_batches = args.probe_batches
    silent_probe_batches = args.probe_batches_silent or args.probe_batches

    if args.stage in {"all", "stage1"}:
        voiced_probes = build_voiced_probe_configs(voiced_probe_batches)
        all_records.extend(
            run_specs(
                voiced_probes,
                dry_run=args.dry_run,
                force_train=args.force_train,
                force_eval=args.force_eval,
                eval_batch_size=args.eval_batch_size,
                existing_records=all_records,
                lm_available=lm_available,
                summary_path=args.summary_json,
            )
        )
        # Stage 1 silent probes wait for a voiced checkpoint; fill later once best voiced is known.

    best_voiced_probe = pick_best(all_records, dataset="voiced", stage="stage1")
    if args.stage in {"stage2", "all"}:
        if best_voiced_probe is None:
            LOG.info("No Stage 1 voiced results yet; running probes first to seed Stage 2.")
            voiced_probes = build_voiced_probe_configs(args.probe_batches)
            all_records.extend(
                run_specs(
                    voiced_probes,
                    dry_run=args.dry_run,
                    force_train=args.force_train,
                    force_eval=args.force_eval,
                    eval_batch_size=args.eval_batch_size,
                    existing_records=all_records,
                    lm_available=lm_available,
                    summary_path=args.summary_json,
                )
            )
            best_voiced_probe = pick_best(all_records, dataset="voiced", stage="stage1")
        if best_voiced_probe is None:
            LOG.warning("Unable to find a best voiced probe; aborting Stage 2.")
            write_summary(all_records, args.summary_json, args.summary_csv)
            return
        voiced_stage2_specs = build_voiced_stage2_configs(best_probe_to_knobs(best_voiced_probe))
        all_records.extend(
            run_specs(
                voiced_stage2_specs,
                dry_run=args.dry_run,
                force_train=args.force_train,
                force_eval=args.force_eval,
                eval_batch_size=args.eval_batch_size,
                existing_records=all_records,
                lm_available=lm_available,
                summary_path=args.summary_json,
            )
        )

        best_voiced_full = pick_best(all_records, dataset="voiced", stage="stage2")
        if best_voiced_full is None:
            LOG.warning("No Stage 2 voiced run available for silent fine-tune.")
            write_summary(all_records, args.summary_json, args.summary_csv)
            return
        best_voiced_ckpt = Path(best_voiced_full["checkpoint_path"])

        # Stage 1 silent probes driven by voiced checkpoint.
        silent_probes = build_silent_probe_configs(silent_probe_batches, init_checkpoint=best_voiced_ckpt)
        all_records.extend(
            run_specs(
                silent_probes,
                dry_run=args.dry_run,
                force_train=args.force_train,
                force_eval=args.force_eval,
                eval_batch_size=args.eval_batch_size,
                existing_records=all_records,
                lm_available=lm_available,
                summary_path=args.summary_json,
            )
        )
        best_silent_probe = pick_best(all_records, dataset="silent", stage="stage1")
        if best_silent_probe is None:
            LOG.warning("Silent probes did not produce metrics; skipping Stage 2 silent.")
            write_summary(all_records, args.summary_json, args.summary_csv)
            return
        silent_stage2_specs = build_silent_stage2_configs(
            best_probe=best_probe_to_knobs(best_silent_probe),
            init_checkpoint=best_voiced_ckpt,
        )
        all_records.extend(
            run_specs(
                silent_stage2_specs,
                dry_run=args.dry_run,
                force_train=args.force_train,
                force_eval=args.force_eval,
                eval_batch_size=args.eval_batch_size,
                existing_records=all_records,
                lm_available=lm_available,
                summary_path=args.summary_json,
            )
        )

    write_summary(all_records, args.summary_json, args.summary_csv)
    LOG.info("Summary written to %s and %s", args.summary_json, args.summary_csv)


if __name__ == "__main__":
    main()
