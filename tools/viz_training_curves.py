"""
Plot training/validation curves from TensorBoard logs for multiple runs.

Usage:
  python tools/viz_training_curves.py --runs mps_fast_basic mps_fast_plus mps_cer_tuned
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

SCALARS = [
    "train/total_loss",
    "train/ctc_loss",
    "train/distill_loss",
    "val/total_loss",
    "train/lr",
]


def load_scalars(tb_dir: Path) -> Dict[str, List]:
    if not tb_dir.exists():
        raise FileNotFoundError(f"No TensorBoard directory at {tb_dir}")
    ea = event_accumulator.EventAccumulator(str(tb_dir))
    ea.Reload()
    out = {}
    for key in SCALARS:
        try:
            out[key] = ea.Scalars(key)
        except KeyError:
            out[key] = []
    return out


def plot_runs(runs: List[str], ckpt_dir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(len(SCALARS), 1, figsize=(8, 10), sharex=True)
    for run in runs:
        tb_dir = ckpt_dir / run / "tb"
        scalars = load_scalars(tb_dir)
        for i, key in enumerate(SCALARS):
            xs = [s.step for s in scalars[key]]
            ys = [s.value for s in scalars[key]]
            if not xs:
                continue
            axes[i].plot(xs, ys, label=run, linewidth=1.4)
            axes[i].set_ylabel(key.split("/")[-1])
    axes[-1].set_xlabel("Global step")
    axes[0].legend()
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run names under results/checkpoints/<run>/tb/")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("results/plots/training_curves.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_runs(args.runs, args.ckpt_dir, args.out)
    print(f"Saved curves to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
