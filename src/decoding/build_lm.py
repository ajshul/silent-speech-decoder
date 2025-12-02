"""Build a character-level KenLM ARPA file from training transcripts without leakage.

Defaults:
  - Uses only `subset == "train"` rows from the index to avoid val/test leakage.
  - Writes a text corpus and trains an n-gram LM via `lmplz` (KenLM) into results/lm/.

Requires KenLM binaries on PATH (`lmplz`, optionally `build_binary`). On macOS, install via
  brew install kenlm
or build from source; on Linux, apt-get install libkenlm-dev or similar then ensure binaries are on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _load_index(path: Path) -> pd.DataFrame:
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    # fall back to jsonl
    return pd.read_json(path, lines=True)


def _filter_texts(df: pd.DataFrame, subsets: Iterable[str]) -> List[str]:
    if "transcript" not in df.columns:
        raise ValueError("Index is missing 'transcript' column.")
    if "subset" in df.columns:
        df = df[df["subset"].isin(subsets)]
    texts = []
    for t in df["transcript"]:
        if isinstance(t, str):
            s = t.strip()
            if s:
                texts.append(s.lower())
    return texts


def build_kenlm(corpus_path: Path, arpa_path: Path, order: int, memory: str) -> None:
    if shutil.which("lmplz") is None:
        raise SystemExit("lmplz not found. Install KenLM binaries and ensure they are on PATH.")
    arpa_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "lmplz",
            "-o",
            str(order),
            "--discount_fallback",
            "-S",
            memory,
            "-T",
            tmp,
            "-text",
            str(corpus_path),
            "-arpa",
            str(arpa_path),
        ]
        subprocess.run(cmd, check=True)


def build_binary(arpa_path: Path, binary_path: Path) -> None:
    if shutil.which("build_binary") is None:
        raise SystemExit("build_binary not found. Install KenLM binaries and ensure they are on PATH.")
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "build_binary",
        "-s",
        "-q",
        "8",
        "trie",
        str(arpa_path),
        str(binary_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Path to dataset index (Parquet or JSONL).")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train"],
        help="Which subsets to include (default: train only to avoid leakage).",
    )
    parser.add_argument("--order", type=int, default=4, help="KenLM n-gram order.")
    parser.add_argument(
        "--memory",
        default="1G",
        help="Memory limit passed to lmplz (-S). Reduce on low-memory systems.",
    )
    parser.add_argument(
        "--text-out",
        type=Path,
        default=Path("results/lm/char_corpus.txt"),
        help="Where to write the training corpus (one transcript per line).",
    )
    parser.add_argument(
        "--arpa-out",
        type=Path,
        default=Path("results/lm/char.arpa"),
        help="Where to write the ARPA LM.",
    )
    parser.add_argument(
        "--binary-out",
        type=Path,
        help="Optional KenLM binary LM output path (run build_binary). If unset, skip binary build.",
    )
    parser.add_argument("--limit", type=int, help="Use at most this many transcripts (for quick tests).")
    parser.add_argument("--no-train", action="store_true", help="Write corpus only; skip LM training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_index(args.index)
    texts = _filter_texts(df, subsets=args.subsets)
    if not texts:
        raise SystemExit("No transcripts found after filtering; check index and subset selection.")
    if args.limit:
        texts = texts[: args.limit]

    args.text_out.parent.mkdir(parents=True, exist_ok=True)
    args.text_out.write_text("\n".join(texts))
    print(f"Wrote corpus with {len(texts)} lines to {args.text_out}")

    if args.no_train:
        return

    build_kenlm(args.text_out, args.arpa_out, order=args.order, memory=args.memory)
    print(f"Wrote ARPA LM to {args.arpa_out}")

    if args.binary_out:
        build_binary(args.arpa_out, args.binary_out)
        print(f"Wrote binary LM to {args.binary_out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
