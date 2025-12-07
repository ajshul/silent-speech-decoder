"""
Visualize results produced by tools/run_overnight_experiments.py.

Usage:
  python tools/viz_overnight_summary.py --summary results/plots/overnight_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _maybe_import_seaborn():
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        return None
    return sns


def scatter(df: pd.DataFrame, out: Path) -> None:
    sns = _maybe_import_seaborn()
    fig, ax = plt.subplots(figsize=(7, 6))
    if sns:
        sns.scatterplot(
            data=df,
            x="cer",
            y="wer",
            hue="phase",
            style="decoder",
            size="beam_width",
            palette={"voiced": "tab:blue", "silent": "tab:orange"},
            ax=ax,
        )
    else:
        colors = {"voiced": "tab:blue", "silent": "tab:orange"}
        for _, row in df.iterrows():
            ax.scatter(row["cer"], row["wer"], c=colors.get(row["phase"], "gray"), label=row["phase"], alpha=0.8)
    for _, row in df.iterrows():
        ax.text(row["cer"], row["wer"], row["eval_run"], fontsize=7)
    ax.set_xlabel("CER")
    ax.set_ylabel("WER")
    ax.set_title("WER vs CER across runs/decoders")
    ax.grid(True, alpha=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def best_bars(df: pd.DataFrame, out: Path) -> None:
    best = df.sort_values("cer").groupby(["phase", "run"]).first().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    runs = [f"{r['phase']}/{r['run']}" for _, r in best.iterrows()]
    ax.bar(runs, best["cer"], color="slateblue")
    ax.set_ylabel("Best CER")
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.set_title("Best CER per base run")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def silent_beam_curves(df: pd.DataFrame, out: Path) -> None:
    beam = df[(df["phase"] == "silent") & (df["decoder"].str.startswith("beam"))].copy()
    if beam.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for run, sub in beam.groupby("run"):
        sub = sub.sort_values("beam_width")
        ax.plot(sub["beam_width"], sub["cer"], marker="o", label=run)
    ax.set_xlabel("Beam width")
    ax.set_ylabel("CER")
    ax.set_title("Silent CER vs beam width")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/plots/overnight_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text())
    df = pd.DataFrame(data)
    scatter(df, args.out_dir / "overnight_scatter.png")
    best_bars(df, args.out_dir / "overnight_best_cer.png")
    silent_beam_curves(df, args.out_dir / "overnight_silent_beam.png")
    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
