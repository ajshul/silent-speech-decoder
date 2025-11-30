import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.index_dataset import assign_subset, build_index, summarize_index


def _write_sample(
    root: Path,
    split: str,
    speaker: str,
    stem: str,
    sentence_index: int,
    transcript: str,
    *,
    has_clean_audio: bool = True,
    has_raw_audio: bool = True,
    emg_len: int = 100,
) -> None:
    split_dir = root / split / speaker
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"{stem}_emg.npy", np.zeros((emg_len, 8), dtype=np.float32))
    if has_clean_audio:
        (split_dir / f"{stem}_audio_clean.flac").touch()
    if has_raw_audio:
        (split_dir / f"{stem}_audio.flac").touch()
    metadata = {
        "book": "test_book.txt",
        "sentence_index": sentence_index,
        "text": transcript,
        "chunks": [],
    }
    (split_dir / f"{stem}_info.json").write_text(json.dumps(metadata))


def test_build_index_filters_and_prefers_clean(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_sample(
        root,
        "voiced_parallel_data",
        "speaker1",
        "001",
        sentence_index=0,
        transcript="hello world",
        has_clean_audio=True,
        has_raw_audio=True,
    )
    _write_sample(
        root,
        "voiced_parallel_data",
        "speaker1",
        "002",
        sentence_index=1,
        transcript="fallback audio",
        has_clean_audio=False,
        has_raw_audio=True,
    )
    _write_sample(
        root,
        "voiced_parallel_data",
        "speaker1",
        "003",
        sentence_index=-1,
        transcript="should skip",
    )
    _write_sample(
        root,
        "silent_parallel_data",
        "speaker2",
        "010",
        sentence_index=5,
        transcript="silent example",
        has_clean_audio=False,
        has_raw_audio=False,
    )

    df = build_index(root, splits=["voiced_parallel_data", "silent_parallel_data"])
    assert len(df) == 3

    clean_row = df[df["stem"] == "001"].iloc[0]
    assert clean_row["audio_path"] == "voiced_parallel_data/speaker1/001_audio_clean.flac"

    raw_row = df[df["stem"] == "002"].iloc[0]
    assert raw_row["audio_path"] == "voiced_parallel_data/speaker1/002_audio.flac"

    silent_row = df[df["stem"] == "010"].iloc[0]
    assert pd.isna(silent_row["audio_path"])
    assert not bool(silent_row["has_audio"])
    assert silent_row["subset"] == "eval_silent"

    # Deterministic subset assignment for voiced.
    for _, r in df[df["split"] == "voiced_parallel_data"].iterrows():
        assert r["subset"] == assign_subset(r["split"], r["utterance_id"])


def test_summarize_index_reports_counts_and_durations(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_sample(
        root,
        "voiced_parallel_data",
        "speaker1",
        "100",
        sentence_index=2,
        transcript="duration check",
        emg_len=2000,
    )
    df = build_index(root, splits=["voiced_parallel_data"])

    summary = summarize_index(df, root=root, include_durations=True)
    voiced_stats = summary["voiced_parallel_data"]

    assert voiced_stats["count"] == 1
    assert voiced_stats["with_audio"] == 1
    assert abs(voiced_stats["mean_duration_sec"] - 2.0) < 1e-6
    assert voiced_stats["subset_counts"][assign_subset("voiced_parallel_data", "voiced_parallel_data/speaker1/100")] == 1
