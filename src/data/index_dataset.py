"""Dataset indexing utilities and CLI for the EMG-to-text project.

The indexer walks the expected EMG dataset layout (see ``docs/DATA_LAYOUT.md``),
filters out unusable entries, and writes a manifest (Parquet or JSONL) with
relative paths. A ``--stats`` flag summarizes counts per split and, optionally,
average durations computed from EMG arrays.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from logical split names to their on-disk subfolders.
SPLIT_PATHS: Dict[str, str] = {
    "voiced_parallel_data": "voiced_parallel_data",
    "silent_parallel_data": "silent_parallel_data",
    "closed_vocab_voiced": "closed_vocab/voiced",
    "closed_vocab_silent": "closed_vocab/silent",
    "nonparallel_data": "nonparallel_data",
}

DEFAULT_SPLITS = [
    "voiced_parallel_data",
    "silent_parallel_data",
    "closed_vocab_voiced",
    "closed_vocab_silent",
]

EMG_SAMPLE_RATE = 1000  # Hz


@dataclass
class IndexEntry:
    """Single utterance record in the dataset index."""

    utterance_id: str
    split: str
    subset: str
    speaker: str
    stem: str
    emg_path: str
    audio_path: Optional[str]
    transcript: str
    sentence_index: int
    book: str
    has_audio: bool
    metadata_json: str


def _resolve_split_path(root: Path, split: str) -> Path:
    if split not in SPLIT_PATHS:
        raise ValueError(f"Unknown split '{split}'. Known splits: {list(SPLIT_PATHS)}")
    return root / SPLIT_PATHS[split]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_audio_path(base_dir: Path, stem: str) -> Optional[Path]:
    """Return the preferred audio path (clean > raw) if present."""
    candidates = [
        base_dir / f"{stem}_audio_clean.flac",
        base_dir / f"{stem}_audio.flac",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def assign_subset(split: str, utterance_id: str) -> str:
    """Deterministic train/val/test assignment via MD5 hash for parallel data."""
    if split == "voiced_parallel_data" or split == "silent_parallel_data":
        # Apply consistent 80/10/10 hashing to both voiced and silent parallel data
        h = int(hashlib.md5(utterance_id.encode("utf-8")).hexdigest(), 16) % 100
        if h < 80:
            return "train"
        if h < 90:
            return "val"
        return "test"
    if split.startswith("closed_vocab"):
        return "closed_vocab"
    return "unused"


def _build_entry(
    info_path: Path, root: Path, split: str
) -> Optional[IndexEntry]:
    metadata = _load_json(info_path)
    transcript = (metadata.get("text") or "").strip()
    sentence_index = metadata.get("sentence_index", -1)

    if sentence_index is None or sentence_index < 0 or not transcript:
        return None

    stem = info_path.stem.removesuffix("_info")
    emg_path = info_path.with_name(f"{stem}_emg.npy")
    if not emg_path.exists():
        logger.warning("Missing EMG file for %s", info_path)
        return None

    audio_path = _find_audio_path(info_path.parent, stem)
    speaker = info_path.parent.name
    utterance_id = f"{split}/{speaker}/{stem}"
    subset = assign_subset(split, utterance_id)

    return IndexEntry(
        utterance_id=utterance_id,
        split=split,
        subset=subset,
        speaker=speaker,
        stem=stem,
        emg_path=str(emg_path.relative_to(root)),
        audio_path=str(audio_path.relative_to(root)) if audio_path else None,
        transcript=transcript,
        sentence_index=int(sentence_index),
        book=metadata.get("book", ""),
        has_audio=audio_path is not None,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def build_index(root: Path, splits: Iterable[str]) -> pd.DataFrame:
    """Construct an index DataFrame for the requested splits."""
    root = root.expanduser().resolve()
    entries: List[IndexEntry] = []

    for split in splits:
        split_path = _resolve_split_path(root, split)
        if not split_path.exists():
            logger.warning("Split path missing: %s", split_path)
            continue
        for info_path in sorted(split_path.rglob("*_info.json")):
            entry = _build_entry(info_path, root, split)
            if entry:
                entries.append(entry)

    if not entries:
        logger.error("No entries were indexed. Check dataset paths and filters.")
        return pd.DataFrame()

    df = pd.DataFrame([asdict(e) for e in entries])
    df = df.sort_values(["split", "utterance_id"]).reset_index(drop=True)
    return df


def save_index(df: pd.DataFrame, out_path: Path) -> None:
    """Persist the index as Parquet or JSONL based on extension."""
    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif suffix in {".jsonl", ".json"}:
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format for {out_path}")

    logger.info("Wrote %d entries to %s", len(df), out_path)


def load_index(index_path: Path) -> pd.DataFrame:
    """Load an index file from disk."""
    index_path = index_path.expanduser()
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    suffix = index_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(index_path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(index_path, orient="records", lines=True)
    raise ValueError(f"Unsupported index format: {index_path}")


def summarize_index(
    df: pd.DataFrame, root: Optional[Path] = None, include_durations: bool = False
) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics per split."""
    summary: Dict[str, Dict[str, float]] = {}
    root = root.expanduser().resolve() if root else None

    for split, group in df.groupby("split"):
        stats = {
            "count": int(len(group)),
            "with_audio": int(group["has_audio"].sum()),
            "subset_counts": group["subset"].value_counts().to_dict(),
        }
        if include_durations and root is not None:
            durations: List[float] = []
            for rel_path in group["emg_path"]:
                emg_path = root / rel_path
                if not emg_path.exists():
                    logger.warning("EMG file missing while computing stats: %s", emg_path)
                    continue
                samples = np.load(emg_path, mmap_mode="r").shape[0]
                durations.append(samples / EMG_SAMPLE_RATE)
            if durations:
                stats["mean_duration_sec"] = float(np.mean(durations))
                stats["total_hours"] = float(np.sum(durations) / 3600.0)
        summary[split] = stats
    return summary


def _print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    lines = []
    for split, stats in sorted(summary.items()):
        line = f"{split}: {stats['count']} utterances ({stats['with_audio']} with audio)"
        if "mean_duration_sec" in stats:
            line += (
                f", mean duration {stats['mean_duration_sec']:.2f}s,"
                f" total {stats['total_hours']:.2f}h"
            )
        lines.append(line)
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="Path to dataset root (required when building a new index).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Where to write the index (Parquet or JSONL).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="Existing index file to load for stats only.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help=f"Splits to include. Choices: {list(SPLIT_PATHS)}",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics after indexing (or for --index).",
    )
    parser.add_argument(
        "--durations",
        action="store_true",
        help="When used with --stats, compute mean/total durations from EMG.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing index file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    df: Optional[pd.DataFrame] = None

    if args.out:
        if not args.root:
            raise SystemExit("--root is required when writing an index.")
        out_path = args.out.expanduser()
        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"{out_path} exists; pass --overwrite to replace it.")
        df = build_index(args.root, args.splits)
        if df.empty:
            raise SystemExit("Indexing produced zero entries.")
        save_index(df, out_path)

    if args.stats:
        if df is None:
            if not args.index:
                raise SystemExit("Provide --index or --out when using --stats.")
            df = load_index(args.index)
        summary = summarize_index(df, root=args.root, include_durations=args.durations)
        _print_summary(summary)

    if args.out is None and not args.stats:
        raise SystemExit("No action requested. Use --out to write an index or --stats.")


if __name__ == "__main__":
    main()
