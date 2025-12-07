"""
Analyze prediction errors per run: confusion heatmaps + sample diffs.

Usage:
  python tools/viz_predictions.py --run mps_fast_plus
  python tools/viz_predictions.py --run mps_fast_plus_cer --top-n 15
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _maybe_import_seaborn():
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        return None
    return sns


def load_preds(run: str, eval_dir: Path) -> pd.DataFrame:
    path = eval_dir / run / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions for run {run}")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    df = pd.DataFrame(rows)
    df["run"] = run
    return df


def confusion_counts(ref: str, hyp: str) -> Counter:
    counter: Counter = Counter()
    sm = SequenceMatcher(a=ref, b=hyp)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "replace":
            for r_ch, h_ch in zip(ref[a0:a1], hyp[b0:b1]):
                counter[(r_ch, h_ch)] += 1
        elif op == "delete":
            for r_ch in ref[a0:a1]:
                counter[(r_ch, "<del>")] += 1
        elif op == "insert":
            for h_ch in hyp[b0:b1]:
                counter[("<ins>", h_ch)] += 1
    return counter


def build_confusion(df: pd.DataFrame) -> pd.DataFrame:
    agg: Counter = Counter()
    for _, row in df.iterrows():
        agg.update(confusion_counts(row["ref"], row["hyp"]))
    if not agg:
        return pd.DataFrame()
    data = []
    for (ref_ch, hyp_ch), count in agg.items():
        data.append({"ref": ref_ch, "hyp": hyp_ch, "count": count})
    mat = pd.DataFrame(data)
    pivot = mat.pivot(index="ref", columns="hyp", values="count").fillna(0).sort_index()
    return pivot


def plot_confusion(pivot: pd.DataFrame, run: str, out_path: Path) -> None:
    sns = _maybe_import_seaborn()
    fig, ax = plt.subplots(figsize=(10, 8))
    if sns:
        sns.heatmap(pivot, cmap="magma", ax=ax, cbar_kws={"label": "count"})
    else:
        im = ax.imshow(pivot.values, cmap="magma")
        ax.figure.colorbar(im, ax=ax, label="count")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
    ax.set_title(f"Character confusions: {run}")
    ax.set_xlabel("Hypothesis char")
    ax.set_ylabel("Reference char")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def top_diffs(df: pd.DataFrame, n: int) -> List[Dict[str, str]]:
    df = df.copy()
    df["ref_len"] = df["ref"].str.len()
    df = df.sort_values("ref_len", ascending=False).head(n)
    return df[["utterance_id", "ref", "hyp"]].to_dict(orient="records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run name under results/eval/<run>/predictions.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots/predictions"))
    parser.add_argument("--top-n", type=int, default=10, help="Number of longest-utterance diffs to save.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_preds(args.run, args.eval_dir)
    pivot = build_confusion(df)
    if pivot.empty:
        print("No confusion data found.")
    else:
        plot_confusion(pivot, args.run, args.out_dir / f"{args.run}_confusion.png")
    diffs = top_diffs(df, args.top_n)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diff_path = args.out_dir / f"{args.run}_sample_diffs.jsonl"
    with diff_path.open("w") as f:
        for rec in diffs:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote confusion heatmap and {diff_path}")


if __name__ == "__main__":
    sys.exit(main())
