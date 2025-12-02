"""Generate a text corpus from the train subset for LM building."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.data.text_normalizer import normalize_transcript


def load_transcripts(
    index_path: Path, splits: Sequence[str], subsets: Sequence[str]
) -> Iterable[str]:
    df = pd.read_parquet(index_path)
    df = df[df["split"].isin(splits)]
    df = df[df["subset"].isin(subsets)]
    for t in df["transcript"].dropna():
        norm = normalize_transcript(str(t))
        if norm:
            yield norm


def write_corpus(transcripts: Iterable[str], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w") as f:
        for line in transcripts:
            f.write(line + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a text corpus from the train subset for LM training.")
    parser.add_argument("--index", type=Path, required=True, help="Path to dataset index (Parquet).")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the corpus file.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["voiced_parallel_data"],
        help="Splits to include (default: voiced_parallel_data).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train"],
        help="Subsets to include (default: train).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcripts = load_transcripts(args.index, splits=args.splits, subsets=args.subsets)
    count = write_corpus(transcripts, args.out)
    print(f"Wrote {count} lines to {args.out}")


if __name__ == "__main__":
    main()
