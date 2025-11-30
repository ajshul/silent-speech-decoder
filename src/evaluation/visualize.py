"""Visualization utilities for qualitative inspection of cached features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_features(features_root: Path, utterance_id: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    emg_path = features_root / "emg" / f"{utterance_id}.npy"
    teacher_path = features_root / "teacher" / f"{utterance_id}.npy"
    if not emg_path.exists():
        raise FileNotFoundError(emg_path)
    emg = np.load(emg_path)  # (T, C, M)
    teacher = np.load(teacher_path) if teacher_path.exists() else None
    return emg, teacher


def plot_emg(emg: np.ndarray, out_path: Path, title: str = "EMG log-mel") -> None:
    """
    Plot EMG log-mel features channel-wise.
    emg: (T, C, M)
    """
    t, c, m = emg.shape
    fig, axes = plt.subplots(c, 1, figsize=(10, 2 + c), sharex=True)
    if c == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(emg[:, i, :].T, aspect="auto", origin="lower")
        ax.set_ylabel(f"ch{i}")
        if i == 0:
            ax.set_title(title)
    axes[-1].set_xlabel("Frames")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_emg_vs_teacher(emg: np.ndarray, teacher: np.ndarray, out_path: Path) -> None:
    """Overlay EMG (channel-averaged) and teacher embeddings side-by-side."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    axes[0].imshow(emg.mean(axis=1).T, aspect="auto", origin="lower")
    axes[0].set_title("EMG log-mel (channel-averaged)")
    axes[0].set_ylabel("Mel bins")

    axes[1].imshow(teacher.T, aspect="auto", origin="lower")
    axes[1].set_title("Teacher embeddings")
    axes[1].set_ylabel("Dim")
    for ax in axes:
        ax.set_xlabel("Frames")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_umap(teacher: np.ndarray, out_path: Path) -> None:
    """Project teacher embeddings to 2D via UMAP (if available) or PCA fallback."""
    try:
        import umap  # type: ignore

        reducer = umap.UMAP()
        coords = reducer.fit_transform(teacher)
    except Exception:
        # PCA fallback
        teacher = teacher - teacher.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(teacher, full_matrices=False)
        coords = u[:, :2] * s[:2]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=np.linspace(0, 1, len(coords)), cmap="viridis", s=6)
    ax.set_title("Teacher embedding projection (UMAP/PCA)")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize cached EMG/teacher features.")
    parser.add_argument("--features-root", type=Path, required=True, help="results/features root.")
    parser.add_argument("--utterance-id", type=str, required=True, help="Utterance ID (e.g., split/spk/000).")
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots"), help="Output directory.")
    parser.add_argument("--umap", action="store_true", help="If set, plot UMAP/PCA of teacher embeddings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emg, teacher = load_features(args.features_root, args.utterance_id)
    plot_emg(emg, args.out_dir / f"{args.utterance_id.replace('/', '_')}_emg.png")
    if teacher is not None:
        plot_emg_vs_teacher(emg, teacher, args.out_dir / f"{args.utterance_id.replace('/', '_')}_emg_teacher.png")
        if args.umap:
            plot_umap(teacher, args.out_dir / f"{args.utterance_id.replace('/', '_')}_teacher_umap.png")


if __name__ == "__main__":
    main()
