import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from src.data.preprocessing import (
    EMGConfig,
    TeacherConfig,
    process_emg_row,
    process_teacher_row,
)


def _make_index(root: Path, emg_path: Path, audio_path: Path | None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "utterance_id": "split/spk/000",
                "split": "split",
                "speaker": "spk",
                "stem": "000",
                "emg_path": str(emg_path.relative_to(root)),
                "audio_path": str(audio_path.relative_to(root)) if audio_path else None,
                "transcript": "hello",
                "sentence_index": 0,
                "book": "",
                "has_audio": audio_path is not None,
                "metadata_json": "{}",
            }
        ]
    )


def test_emg_preprocessing_writes_features_and_metadata(tmp_path: Path) -> None:
    root = tmp_path / "data"
    emg_dir = root / "split" / "spk"
    emg_dir.mkdir(parents=True)

    emg_array = np.random.randn(1000, 2).astype(np.float32)
    emg_path = emg_dir / "000_emg.npy"
    np.save(emg_path, emg_array)

    df = _make_index(root, emg_path, audio_path=None)
    row = next(df.itertuples(index=False))

    out_dir = tmp_path / "out" / "emg"
    cfg = EMGConfig(n_fft=64, hop_length=16, n_mels=8)
    wrote = process_emg_row(row, root, out_dir, cfg, overwrite=True)
    assert wrote

    feature_path = out_dir / "split/spk/000.npy"
    meta_path = out_dir / "split/spk/000.json"
    assert feature_path.exists()
    assert meta_path.exists()

    features = np.load(feature_path)
    assert features.ndim == 3  # (frames, channels, n_mels)
    assert features.shape[1] == 2
    assert features.shape[2] == cfg.n_mels

    meta = json.loads(meta_path.read_text())
    assert meta["frames"] == features.shape[0]
    assert meta["n_mels"] == cfg.n_mels
    assert "mean" in meta and "std" in meta


class _DummyProcessor:
    def __call__(self, waveform, sampling_rate, return_tensors, padding):
        return {"input_values": torch.tensor(waveform, dtype=torch.float32)[None, :]}


class _DummyModel(torch.nn.Module):
    def forward(self, input_values, output_hidden_states=False):
        hidden = input_values.unsqueeze(-1)  # (1, T, 1)
        return type("obj", (), {"hidden_states": [hidden, hidden + 1.0]})


def test_teacher_preprocessing_with_stub_model(tmp_path: Path) -> None:
    root = tmp_path / "data"
    audio_dir = root / "split" / "spk"
    audio_dir.mkdir(parents=True)

    waveform = torch.randn(1, 320)
    audio_path = audio_dir / "000_audio.flac"
    torchaudio.save(audio_path, waveform, 16000, format="FLAC")

    emg_path = audio_dir / "000_emg.npy"
    np.save(emg_path, np.zeros((10, 2), dtype=np.float32))

    df = _make_index(root, emg_path, audio_path=audio_path)
    row = next(df.itertuples(index=False))

    out_dir = tmp_path / "out" / "teacher"
    cfg = TeacherConfig(model_name="dummy", layer=1, device="cpu", sample_rate=16000)

    wrote = process_teacher_row(
        row,
        root,
        out_dir,
        cfg,
        processor=_DummyProcessor(),
        model=_DummyModel(),
        overwrite=True,
    )
    assert wrote

    feature_path = out_dir / "split/spk/000.npy"
    meta_path = out_dir / "split/spk/000.json"
    assert feature_path.exists()
    assert meta_path.exists()

    feats = np.load(feature_path)
    assert feats.shape[1] == 1  # dim from dummy model
    meta = json.loads(meta_path.read_text())
    assert meta["layer"] == cfg.layer
