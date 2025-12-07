"""
Plot student-teacher alignment (framewise MSE) for a single utterance and run.

Usage:
  python tools/viz_alignment.py --run mps_fast_plus --utterance-id voiced_parallel_data/5-10/100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.models.emg_encoder import EMGConformerEncoder, EncoderConfig  # noqa: E402
from src.models.heads import ProjectionHead  # noqa: E402


def load_checkpoint(run: str, ckpt_dir: Path, device: torch.device):
    ckpt_path = ckpt_dir / run / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    cfg = payload["config"]["model"]["encoder"]
    encoder = EMGConformerEncoder(
        EncoderConfig(
            input_dim=cfg["input_dim"],
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            ffn_dim=cfg["ffn_dim"],
            depthwise_conv_kernel_size=cfg["depthwise_conv_kernel_size"],
            dropout=cfg.get("dropout", 0.1),
            subsample_factor=cfg.get("subsample_factor", 2),
        )
    )
    projection = ProjectionHead(cfg["d_model"], payload["config"]["model"]["projection_dim"])
    encoder.load_state_dict(payload["encoder"])
    projection.load_state_dict(payload["projection"])
    encoder.to(device).eval()
    projection.to(device).eval()
    return encoder, projection, payload["config"]


def load_features(features_root: Path, utterance_id: str) -> tuple[np.ndarray, np.ndarray]:
    emg_path = features_root / "emg" / f"{utterance_id}.npy"
    teacher_path = features_root / "teacher" / f"{utterance_id}.npy"
    if not emg_path.exists() or not teacher_path.exists():
        raise FileNotFoundError(f"Missing features for {utterance_id}")
    emg = np.load(emg_path)
    teacher = np.load(teacher_path)
    return emg, teacher


def plot_alignment(run: str, utterance_id: str, emg: np.ndarray, teacher: np.ndarray, encoder, projection, device: torch.device, out_path: Path) -> None:
    emg_flat = torch.from_numpy(emg.reshape(emg.shape[0], -1)).unsqueeze(0).float().to(device)
    lengths = torch.tensor([emg_flat.shape[1]], device=device)
    with torch.no_grad():
        enc_out, enc_len = encoder(emg_flat, lengths)
        student = projection(enc_out)  # (1, T', D)
    student = student.squeeze(0).cpu()

    # Align teacher to student time steps
    teacher_t = torch.from_numpy(teacher).unsqueeze(0)  # (1, T, D)
    teacher_up = F.interpolate(
        teacher_t.transpose(1, 2),
        size=student.shape[0],
        mode="linear",
        align_corners=False,
    ).transpose(1, 2).squeeze(0)
    mse = (student - teacher_up).pow(2).mean(dim=-1).numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    axes[0].imshow(emg.mean(axis=1).T, aspect="auto", origin="lower")
    axes[0].set_title(f"EMG (channel-avg) | {utterance_id}")
    axes[0].set_ylabel("mel")

    axes[1].imshow(teacher_up.T, aspect="auto", origin="lower")
    axes[1].set_title("Teacher (aligned to student length)")
    axes[1].set_ylabel("dim")

    axes[2].plot(mse, label="framewise MSE")
    axes[2].set_title(f"{run} | subsample={encoder.subsample_factor}")
    axes[2].set_xlabel("time (student frames)")
    axes[2].set_ylabel("MSE")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run name under results/checkpoints/")
    parser.add_argument("--utterance-id", required=True, help="Utterance ID e.g., voiced_parallel_data/5-10/100")
    parser.add_argument("--features-root", type=Path, default=Path("results/features"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--out", type=Path, help="Output image path.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu/mps/cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    encoder, projection, _ = load_checkpoint(args.run, args.ckpt_dir, device)
    emg, teacher = load_features(args.features_root, args.utterance_id)
    out_path = args.out or Path("results/plots/alignment") / f"{args.run}_{args.utterance_id.replace('/', '_')}.png"
    plot_alignment(args.run, args.utterance_id, emg, teacher, encoder, projection, device, out_path)
    print(f"Saved alignment plot to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
