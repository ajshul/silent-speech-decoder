#!/usr/bin/env python
"""
Summarize scalar metrics from a sweep directory.

Reads TensorBoard event files under each experiment folder and prints the final
val/train losses for quick comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from tensorboard.backend.event_processing import event_accumulator


def load_scalars(tb_dir: Path, tags: List[str]) -> Dict[str, float]:
    acc = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    metrics: Dict[str, float] = {}
    for tag in tags:
        if tag not in acc.Tags().get("scalars", []):
            continue
        events = acc.Scalars(tag)
        if events:
            metrics[tag] = events[-1].value
    return metrics


def summarize_sweep(root: Path, tags: List[str]) -> List[Tuple[str, Dict[str, float]]]:
    summaries: List[Tuple[str, Dict[str, float]]] = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        tb_dir = exp_dir / "tb"
        if not tb_dir.exists():
            continue
        metrics = load_scalars(tb_dir, tags)
        summaries.append((exp_dir.name, metrics))
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sweep metrics from TensorBoard logs.")
    parser.add_argument("sweep_dir", type=Path, help="Path to the sweep timestamp directory (e.g., results/sweeps/overfit/<timestamp>).")
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["val/total_loss", "val/ctc_loss", "val/distill_loss", "train/total_loss"],
        help="Scalar tags to extract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = summarize_sweep(args.sweep_dir, args.tags)
    if not summaries:
        print("No TensorBoard logs found.")
        return

    print(f"Sweep: {args.sweep_dir}")
    for name, metrics in summaries:
        tag_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"- {name}: {tag_str}")


if __name__ == "__main__":
    main()
