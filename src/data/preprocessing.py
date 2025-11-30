"""Feature preprocessing for EMG and teacher audio.

Usage examples:
  python -m src.data.preprocessing --mode emg --index results/index.parquet --root data/emg_data --out results/features/emg
  python -m src.data.preprocessing --mode teacher --index results/index.parquet --root data/emg_data --out results/features/teacher

See docs/EXECUTION_PLAN.md and docs/TODO.md for the full workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

from src.data.index_dataset import load_index

logger = logging.getLogger(__name__)


@dataclass
class EMGConfig:
    sample_rate: int = 1000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    fmin: int = 0
    fmax: Optional[int] = None
    normalize: str = "per_file"  # choices: per_file, none


@dataclass
class TeacherConfig:
    model_name: str = "microsoft/wavlm-base-plus"
    layer: int = 9
    sample_rate: int = 16000
    device: str = "cpu"


def _ensure_out_path(base_out: Path, utterance_id: str) -> Tuple[Path, Path]:
    """Return (feature_path, metadata_path) creating parent dirs if needed."""
    feature_path = base_out / f"{utterance_id}.npy"
    meta_path = base_out / f"{utterance_id}.json"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    return feature_path, meta_path


def _normalize(x: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, float]]:
    if mode == "none":
        return x, {}
    mean = float(x.mean())
    std = float(x.std() + 1e-8)
    return (x - mean) / std, {"mean": mean, "std": std}


def _compute_logmel(emg: np.ndarray, cfg: EMGConfig) -> np.ndarray:
    """Compute log-mel per channel and return shape (frames, channels, n_mels)."""
    channels = []
    mel_basis = librosa.filters.mel(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    for c in range(emg.shape[1]):
        stft = librosa.stft(
            emg[:, c], n_fft=cfg.n_fft, hop_length=cfg.hop_length, center=False
        )
        power = np.abs(stft) ** 2
        mel = np.dot(mel_basis, power)
        logmel = librosa.power_to_db(np.maximum(mel, 1e-10), ref=1.0)
        channels.append(logmel.T)  # (frames, n_mels)
    return np.stack(channels, axis=1).astype(np.float32)  # (frames, channels, n_mels)


def process_emg_row(row, root: Path, out_dir: Path, cfg: EMGConfig, overwrite: bool) -> bool:
    feature_path, meta_path = _ensure_out_path(out_dir, row.utterance_id)
    if feature_path.exists() and not overwrite:
        return False

    emg_path = root / row.emg_path
    if not emg_path.exists():
        logger.warning("Missing EMG file: %s", emg_path)
        return False

    emg = np.load(emg_path)
    if emg.ndim != 2:
        logger.warning("Unexpected EMG shape %s for %s", emg.shape, emg_path)
        return False
    if emg.dtype != np.float32:
        emg = emg.astype(np.float32)

    features = _compute_logmel(emg, cfg)
    features, stats = _normalize(features, cfg.normalize)

    np.save(feature_path, features)
    meta = {
        "utterance_id": row.utterance_id,
        "frames": int(features.shape[0]),
        "channels": int(features.shape[1]),
        "n_mels": int(features.shape[2]),
        "sample_rate": cfg.sample_rate,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "fmin": cfg.fmin,
        "fmax": cfg.fmax,
        "normalize": cfg.normalize,
        **stats,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return True


def _load_teacher(cfg: TeacherConfig) -> Tuple[Wav2Vec2FeatureExtractor, WavLMModel]:
    processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model_name)
    model = WavLMModel.from_pretrained(cfg.model_name)
    model.to(cfg.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return processor, model


def _prepare_audio(audio_path: Path, target_sr: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform[:1]  # keep mono
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)


def process_teacher_row(
    row,
    root: Path,
    out_dir: Path,
    cfg: TeacherConfig,
    processor: Wav2Vec2FeatureExtractor,
    model: WavLMModel,
    overwrite: bool,
) -> bool:
    feature_path, meta_path = _ensure_out_path(out_dir, row.utterance_id)
    if feature_path.exists() and not overwrite:
        return False

    if row.audio_path is None or not isinstance(row.audio_path, str):
        logger.debug("Skipping (no audio) %s", row.utterance_id)
        return False

    audio_path = root / row.audio_path
    if not audio_path.exists():
        logger.warning("Missing audio for %s: %s", row.utterance_id, audio_path)
        return False

    waveform = _prepare_audio(audio_path, cfg.sample_rate)
    inputs = processor(
        waveform.numpy(),
        sampling_rate=cfg.sample_rate,
        return_tensors="pt",
        padding="longest",
    )
    input_values = inputs["input_values"].to(cfg.device)
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if cfg.layer >= len(hidden_states):
        raise ValueError(
            f"Requested layer {cfg.layer} but model returned {len(hidden_states)} hidden states"
        )
    feats = hidden_states[cfg.layer].squeeze(0).cpu().numpy().astype(np.float32)

    np.save(feature_path, feats)
    meta = {
        "utterance_id": row.utterance_id,
        "frames": int(feats.shape[0]),
        "dim": int(feats.shape[1]),
        "layer": cfg.layer,
        "model_name": cfg.model_name,
        "sample_rate": cfg.sample_rate,
        "frame_stride_sec": 0.02,  # WavLM conv stride
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return True


def _iter_rows(df):
    return df.itertuples(index=False)


def _process_mode(
    mode: str,
    df,
    root: Path,
    out_dir: Path,
    emg_cfg: EMGConfig,
    teacher_cfg: TeacherConfig,
    overwrite: bool,
    limit: Optional[int],
) -> None:
    processed = 0
    if mode == "emg":
        for row in tqdm(_iter_rows(df), desc="EMG", total=len(df)):
            changed = process_emg_row(row, root, out_dir, emg_cfg, overwrite)
            if changed:
                processed += 1
            if limit and processed >= limit:
                break
        logger.info("EMG processed: %d", processed)
        return

    processor = None
    model = None
    if mode == "teacher":
        processor, model = _load_teacher(teacher_cfg)
    for row in tqdm(_iter_rows(df), desc="Teacher", total=len(df)):
        changed = process_teacher_row(
            row, root, out_dir, teacher_cfg, processor, model, overwrite
        )
        if changed:
            processed += 1
        if limit and processed >= limit:
            break
    logger.info("Teacher processed: %d", processed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["emg", "teacher"],
        required=True,
        help="Which feature pipeline to run.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="Path to dataset index (Parquet or JSONL).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/emg_data"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the selected mode.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing feature files."
    )
    parser.add_argument("--limit", type=int, help="Process at most this many items.")

    # EMG options
    parser.add_argument("--emg-sample-rate", type=int, default=1000)
    parser.add_argument("--emg-n-fft", type=int, default=400)
    parser.add_argument("--emg-hop-length", type=int, default=160)
    parser.add_argument("--emg-n-mels", type=int, default=80)
    parser.add_argument(
        "--emg-normalize",
        choices=["per_file", "none"],
        default="per_file",
        help="Normalization strategy for EMG features.",
    )

    # Teacher options
    parser.add_argument(
        "--teacher-model",
        default="microsoft/wavlm-base-plus",
        help="Hugging Face model name for WavLM.",
    )
    parser.add_argument("--teacher-layer", type=int, default=9)
    parser.add_argument("--teacher-sample-rate", type=int, default=16000)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device for teacher model (mps/cpu).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    df = load_index(args.index)
    root = args.root.expanduser().resolve()
    out_dir = args.out.expanduser()

    emg_cfg = EMGConfig(
        sample_rate=args.emg_sample_rate,
        n_fft=args.emg_n_fft,
        hop_length=args.emg_hop_length,
        n_mels=args.emg_n_mels,
        normalize=args.emg_normalize,
    )
    teacher_cfg = TeacherConfig(
        model_name=args.teacher_model,
        layer=args.teacher_layer,
        sample_rate=args.teacher_sample_rate,
        device=args.device,
    )

    _process_mode(
        args.mode,
        df,
        root,
        out_dir,
        emg_cfg,
        teacher_cfg,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

