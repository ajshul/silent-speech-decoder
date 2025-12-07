"""
Train/evaluate a small slate of voiced and silent configs with consistent settings,
then summarize results for visualization. Designed to be kicked off overnight.

What it does:
- Trains 3 voiced configs (including viz_voiced_fast_plus) for up to 30 epochs (patience=5).
- Evaluates each voiced checkpoint with greedy, beam, and beam+LM (if LM exists).
- Picks the voiced checkpoint with lowest CER across decoders.
- Fine-tunes 3 silent configs from that best voiced checkpoint.
- Evaluates each silent checkpoint with greedy, three beam widths, and three LM alpha/beta pairs (if LM exists).
- Writes a summary JSON/CSV to results/plots/overnight_summary.{json,csv}.

Prereqs:
- Cached features in results/features/{emg,teacher}
- KenLM char LM at results/lm/char_5gram.arpa (LM evals are skipped if missing)

Usage:
  python tools/run_overnight_experiments.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent


@dataclass
class EvalResult:
    phase: str  # voiced or silent
    run: str
    eval_run: str
    decoder: str
    beam_width: Optional[int]
    alpha: Optional[float]
    beta: Optional[float]
    wer: float
    cer: float
    init_from: Optional[str] = None


def run_cmd(cmd: List[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def train(config: Path, run_dir: Optional[Path] = None, init: Optional[Path] = None) -> None:
    cmd = ["python", "-m", "src.training.train", "--config", str(config)]
    if run_dir:
        cmd += ["--run-dir", str(run_dir)]
    if init:
        cmd += ["--init-checkpoint", str(init)]
    run_cmd(cmd)


def evaluate(
    checkpoint: Path,
    eval_run: str,
    decoder: str,
    splits: List[str],
    subsets: List[str],
    beam_width: Optional[int] = None,
    lm_path: Optional[Path] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> Dict[str, float]:
    out_dir = REPO / "results" / "eval" / eval_run
    cmd = [
        "python",
        "-m",
        "src.evaluation.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--splits",
        *splits,
        "--subsets",
        *subsets,
        "--decoder",
        decoder,
        "--run-name",
        eval_run,
        "--output",
        str(out_dir),
    ]
    if decoder == "beam":
        if beam_width:
            cmd += ["--beam-width", str(beam_width)]
        if lm_path:
            cmd += ["--lm-path", str(lm_path)]
        if alpha is not None:
            cmd += ["--alpha", str(alpha)]
        if beta is not None:
            cmd += ["--beta", str(beta)]
    run_cmd(cmd)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    return metrics


def pick_best_voiced(results: List[EvalResult]) -> EvalResult:
    voiced = [r for r in results if r.phase == "voiced"]
    if not voiced:
        raise RuntimeError("No voiced results to select from.")
    return min(voiced, key=lambda r: r.cer)


def main() -> None:
    lm_path = REPO / "results" / "lm" / "char_5gram.arpa"
    lm_available = lm_path.exists()
    if not lm_available:
        print(f"[warn] LM not found at {lm_path}; LM evals will be skipped.")

    voiced_cfgs = [
        "configs/experiments/viz_voiced_fast_plus.yaml",
        "configs/experiments/viz_voiced_ctc_heavy_noaug.yaml",
        "configs/experiments/viz_voiced_channel_dropout_linear.yaml",
    ]
    silent_cfgs = [
        "configs/experiments/viz_silent_sub4_baseline.yaml",
        "configs/experiments/viz_silent_sub2_light_aug.yaml",
        "configs/experiments/viz_silent_channel_dropout.yaml",
    ]

    results: List[EvalResult] = []

    # 1) Train voiced configs.
    for cfg in voiced_cfgs:
        train(Path(cfg))

    # 2) Evaluate voiced runs.
    for cfg in voiced_cfgs:
        run_name = Path(cfg).stem
        ckpt = REPO / "results" / "checkpoints" / run_name / "best.pt"
        if not ckpt.exists():
            print(f"[warn] Missing checkpoint for {run_name}, skipping eval.")
            continue
        # greedy
        m = evaluate(ckpt, f"{run_name}_greedy", "greedy", ["voiced_parallel_data"], ["val"])
        results.append(EvalResult("voiced", run_name, f"{run_name}_greedy", "greedy", None, None, None, m["wer"], m["cer"]))
        # beam
        m = evaluate(ckpt, f"{run_name}_beam50", "beam", ["voiced_parallel_data"], ["val"], beam_width=50)
        results.append(EvalResult("voiced", run_name, f"{run_name}_beam50", "beam", 50, None, None, m["wer"], m["cer"]))
        # beam + LM
        if lm_available:
            m = evaluate(
                ckpt,
                f"{run_name}_beam50_lm",
                "beam",
                ["voiced_parallel_data"],
                ["val"],
                beam_width=50,
                lm_path=lm_path,
                alpha=0.5,
                beta=0.0,
            )
            results.append(EvalResult("voiced", run_name, f"{run_name}_beam50_lm", "beam_lm", 50, 0.5, 0.0, m["wer"], m["cer"]))

    # 3) Pick best voiced by CER.
    best_voiced = pick_best_voiced(results)
    best_ckpt = REPO / "results" / "checkpoints" / best_voiced.run / "best.pt"
    print(f"[info] Best voiced checkpoint: {best_voiced.run} ({best_voiced.cer:.4f} CER) at {best_ckpt}")

    # 4) Train silent configs initialized from best voiced.
    for cfg in silent_cfgs:
        run_name = Path(cfg).stem
        run_dir = REPO / "results" / "checkpoints" / f"{run_name}_from_{best_voiced.run}"
        train(Path(cfg), run_dir=run_dir, init=best_ckpt)

    # 5) Evaluate silent runs.
    beam_widths = [20, 50, 100]
    lm_grid = [(0.45, 0.0), (0.55, 0.0), (0.50, 0.1)]
    for cfg in silent_cfgs:
        base_name = Path(cfg).stem
        run_dir = REPO / "results" / "checkpoints" / f"{base_name}_from_{best_voiced.run}"
        ckpt = run_dir / "best.pt"
        if not ckpt.exists():
            print(f"[warn] Missing checkpoint for {base_name}, skipping silent eval.")
            continue
        # greedy
        m = evaluate(ckpt, f"{base_name}_greedy", "greedy", ["silent_parallel_data"], ["eval_silent"])
        results.append(EvalResult("silent", base_name, f"{base_name}_greedy", "greedy", None, None, None, m["wer"], m["cer"], init_from=best_voiced.run))
        # beam widths
        for bw in beam_widths:
            tag = f"{base_name}_beam{bw}"
            m = evaluate(ckpt, tag, "beam", ["silent_parallel_data"], ["eval_silent"], beam_width=bw)
            results.append(EvalResult("silent", base_name, tag, "beam", bw, None, None, m["wer"], m["cer"], init_from=best_voiced.run))
        # LM grid
        if lm_available:
            for alpha, beta in lm_grid:
                tag = f"{base_name}_lm_bw50_a{alpha}_b{beta}"
                m = evaluate(
                    ckpt,
                    tag,
                    "beam",
                    ["silent_parallel_data"],
                    ["eval_silent"],
                    beam_width=50,
                    lm_path=lm_path,
                    alpha=alpha,
                    beta=beta,
                )
                results.append(EvalResult("silent", base_name, tag, "beam_lm", 50, alpha, beta, m["wer"], m["cer"], init_from=best_voiced.run))

    # 6) Persist summary.
    out_dir = REPO / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        {
            "phase": r.phase,
            "run": r.run,
            "eval_run": r.eval_run,
            "decoder": r.decoder,
            "beam_width": r.beam_width,
            "alpha": r.alpha,
            "beta": r.beta,
            "wer": r.wer,
            "cer": r.cer,
            "init_from": r.init_from,
        }
        for r in results
    ]
    (out_dir / "overnight_summary.json").write_text(json.dumps(summary, indent=2))
    import pandas as pd  # type: ignore

    pd.DataFrame(summary).to_csv(out_dir / "overnight_summary.csv", index=False)
    print(f"[done] Wrote {out_dir/'overnight_summary.json'} and .csv")


if __name__ == "__main__":
    sys.exit(main())
