"""
Plot EMG frame length distributions and effective lengths after subsampling.

Usage:
  python tools/viz_lengths.py --subsample 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_lengths(meta_paths: List[Path]) -> List[int]:
    lengths = []
    for p in meta_paths:
        try:
            meta = json.loads(p.read_text())
            lengths.append(int(meta["frames"]))
        except Exception:
            continue
    return lengths


def plot_lengths(lengths: List[int], subsample: int, out_path: Path) -> None:
    lengths = np.array(lengths)
    eff = np.ceil(lengths / max(1, subsample))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(lengths, bins=50, color="steelblue", alpha=0.8)
    axes[0].set_title("Raw EMG frames (10ms hop)")
    axes[0].set_xlabel("frames")
    axes[0].set_ylabel("count")
    axes[1].hist(eff, bins=50, color="darkorange", alpha=0.8)
    axes[1].set_title(f"Frames after subsample_factor={subsample}")
    axes[1].set_xlabel("frames")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=Path("results/features/emg"))
    parser.add_argument("--subsample", type=int, default=2, help="Encoder subsample_factor for effective length view.")
    parser.add_argument("--out", type=Path, default=Path("results/plots/emg_lengths.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_paths = list(Path(args.features_root).rglob("*.json"))
    if not meta_paths:
        raise FileNotFoundError("No EMG metadata JSON files found; run preprocessing with metadata enabled.")
    lengths = load_lengths(meta_paths)
    plot_lengths(lengths, args.subsample, args.out)
    print(f"Saved length histograms to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
