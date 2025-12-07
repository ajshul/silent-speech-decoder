"""Build a small character-level KenLM from normalized transcripts."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence

from src.data.index_dataset import load_index
from src.data.text_normalizer import normalize_transcript

logger = logging.getLogger(__name__)


def _filter_df(index_path: Path, splits: Sequence[str], subsets: Sequence[str] | None) -> List[str]:
    df = load_index(index_path)
    df = df[df["split"].isin(splits)].reset_index(drop=True)
    if subsets and "subset" in df.columns:
        df = df[df["subset"].isin(subsets)].reset_index(drop=True)
    df["transcript_norm"] = df["transcript"].apply(normalize_transcript)
    transcripts = [t for t in df["transcript_norm"].tolist() if t]
    return transcripts


def _write_corpus(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in lines:
            f.write(line + "\n")


def _train_kenlm(corpus_path: Path, arpa_path: Path, order: int) -> None:
    if shutil.which("lmplz") is None:
        raise FileNotFoundError("KenLM binary 'lmplz' not found in PATH.")
    arpa_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["lmplz", "-o", str(order), "--text", str(corpus_path), "--arpa", str(arpa_path)]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a character n-gram LM from transcripts.")
    parser.add_argument("--index", type=Path, default=Path("results/index.parquet"), help="Path to dataset index.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["voiced_parallel_data"],
        help="Splits to include when building the corpus.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train", "val"],
        help="Subsets to include (ignored if index lacks a subset column).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=5,
        help="KenLM n-gram order (recommended: 4-6 for this corpus size).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lm/char_5gram.arpa"),
        help="Path to save the ARPA LM.",
    )
    parser.add_argument(
        "--skip-kenlm",
        action="store_true",
        help="Only write the normalized corpus; do not invoke KenLM binaries.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    transcripts = _filter_df(args.index, args.splits, args.subsets)
    if not transcripts:
        raise ValueError("No transcripts found after filtering; check splits/subsets and index path.")

    corpus_path = args.output.with_suffix(".txt")
    _write_corpus(transcripts, corpus_path)
    logger.info("Wrote %d normalized lines to %s", len(transcripts), corpus_path)

    if args.skip_kenlm:
        logger.info("Skipping KenLM build (--skip-kenlm set).")
        return

    try:
        _train_kenlm(corpus_path, args.output, args.order)
    except FileNotFoundError as exc:
        logger.error("%s Install kenlm binaries or rerun with --skip-kenlm to only produce the corpus.", exc)
        return
    logger.info("KenLM ARPA saved to %s", args.output)


if __name__ == "__main__":
    main()
