"""
Compare WER/CER across runs and summarize config choices.

Usage:
  python tools/viz_metrics.py --runs mps_fast_basic mps_fast_plus mps_fast_plus_cer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def _maybe_import_seaborn():
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        return None
    return sns


def load_run(run: str, eval_dir: Path, ckpt_dir: Path) -> Dict:
    metrics_path = eval_dir / run / "metrics.json"
    cfg_path = ckpt_dir / run / "config.json"
    if not metrics_path.exists() or not cfg_path.exists():
        raise FileNotFoundError(f"Missing metrics or config for run {run}")

    metrics = json.loads(metrics_path.read_text())
    cfg = json.loads(cfg_path.read_text())
    aug = cfg.get("augmentation", {}).get("specaugment", {})
    channel = cfg.get("augmentation", {}).get("channel_dropout", {})
    sched_cfg = cfg["optim"].get("scheduler")
    scheduler = sched_cfg if isinstance(sched_cfg, str) else (sched_cfg.get("name") if sched_cfg else "none")

    return {
        "run": run,
        "wer": metrics["wer"],
        "cer": metrics["cer"],
        "lambda_ctc": float(cfg["loss"].get("lambda_ctc", 0.0)),
        "lambda_distill": float(cfg["loss"].get("lambda_distill", 0.0)),
        "spec_p": float(aug.get("p", 0.0)),
        "channel_dropout": float(channel.get("p", 0.0)),
        "scheduler": str(scheduler),
        "subsample_factor": cfg["model"]["encoder"].get("subsample_factor", 2),
    }


def make_scatter(df: pd.DataFrame, out_path: Path) -> None:
    sns = _maybe_import_seaborn()
    fig, ax = plt.subplots(figsize=(7, 6))
    if sns:
        sns.scatterplot(
            data=df,
            x="cer",
            y="wer",
            hue="scheduler",
            size="lambda_ctc",
            style="spec_p",
            palette="Dark2",
            ax=ax,
        )
    else:
        sched_colors = {s: c for s, c in zip(df["scheduler"].unique(), plt.cm.Dark2.colors)}
        for _, row in df.iterrows():
            ax.scatter(
                row["cer"],
                row["wer"],
                label=row["scheduler"],
                s=80 * (row["lambda_ctc"] + 0.5),
                color=sched_colors.get(row["scheduler"], "gray"),
                alpha=0.8,
            )
    for _, row in df.iterrows():
        ax.text(row["cer"], row["wer"], row["run"], fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("CER")
    ax.set_ylabel("WER")
    ax.set_title("WER vs CER by run (size=lambda_ctc, style=SpecAug p)")
    if sns:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[: len(set(df['scheduler']))], labels[: len(set(df['scheduler']))], title="scheduler")
    ax.grid(True, alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run names under results/eval and results/checkpoints.")
    parser.add_argument("--eval-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--strict", action="store_true", help="Fail on missing runs instead of skipping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict] = []
    for run in args.runs:
        try:
            rows.append(load_run(run, args.eval_dir, args.ckpt_dir))
        except FileNotFoundError as exc:
            if args.strict:
                raise
            print(f"Skipping {run}: {exc}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("No runs loaded; nothing to plot.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    make_scatter(df, args.out_dir / "metrics_scatter.png")
    print(f"Wrote {csv_path} and metrics_scatter.png")


if __name__ == "__main__":
    sys.exit(main())
