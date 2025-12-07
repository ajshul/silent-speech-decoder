"""
Two-stage experiment runner:
Stage 1 (small models): fast hyperparam/decoding triage on reduced-capacity configs.
Stage 2 (full models): deeper sweeps on full configs using insights from Stage 1.

Outputs a consolidated summary JSON/CSV for visualization.

Usage:
  python tools/run_two_stage_experiments.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd  # type: ignore

REPO = Path(__file__).resolve().parent.parent


@dataclass
class EvalResult:
    stage: str
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


def train(config: Path, run_dir: Optional[Path] = None, init: Optional[Path] = None, max_batches: Optional[int] = None) -> None:
    cmd = ["python", "-m", "src.training.train", "--config", str(config)]
    if run_dir:
        cmd += ["--run-dir", str(run_dir)]
    if init:
        cmd += ["--init-checkpoint", str(init)]
    if max_batches is not None:
        cmd += ["--overfit-batches", str(max_batches)]
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


def pick_best(results: List[EvalResult], phase: str, stage: str) -> EvalResult:
    pool = [r for r in results if r.phase == phase and r.stage == stage]
    if not pool:
        raise RuntimeError(f"No results to select for phase={phase} stage={stage}")
    return min(pool, key=lambda r: r.cer)

def maybe_limit_stage1_train() -> Dict[str, int]:
    """
    Return a dict of kwargs to pass to train() to limit Stage 1 runtime.
    We cap number of batches to shorten CPU CTCLoss exposure.
    """
    return {"max_batches": 20}

def main() -> None:
    lm_path = REPO / "results" / "lm" / "char_5gram.arpa"
    lm_available = lm_path.exists()
    if not lm_available:
        print(f"[warn] LM not found at {lm_path}; LM evals that need LM will be skipped.")

    small_voiced_cfgs = [
        "configs/experiments/viz_small_voiced_192_basic.yaml",
        "configs/experiments/viz_small_voiced_192_specaug.yaml",
        "configs/experiments/viz_small_voiced_224_linear.yaml",
    ]
    small_silent_cfgs = [
        "configs/experiments/viz_small_silent_sub4.yaml",
        "configs/experiments/viz_small_silent_sub4_aug.yaml",
        "configs/experiments/viz_small_silent_sub2.yaml",
    ]
    # Full configs will be chosen after Stage 1 based on best small runs.
    full_voiced_cfgs: List[str] = []
    full_silent_cfgs: List[str] = []

    results: List[EvalResult] = []

    # Stage 1: small voiced
    stage1_limit = maybe_limit_stage1_train()
    for cfg in small_voiced_cfgs:
        train(REPO / cfg, max_batches=stage1_limit.get("max_batches"))
    for cfg in small_voiced_cfgs:
        run_name = Path(cfg).stem
        ckpt = REPO / "results" / "checkpoints" / run_name / "best.pt"
        if not ckpt.exists():
            print(f"[warn] Missing checkpoint for {run_name}, skipping eval.")
            continue
        m = evaluate(ckpt, f"{run_name}_greedy", "greedy", ["voiced_parallel_data"], ["val"])
        results.append(EvalResult("small", "voiced", run_name, f"{run_name}_greedy", "greedy", None, None, None, m["wer"], m["cer"]))
        m = evaluate(ckpt, f"{run_name}_beam10", "beam", ["voiced_parallel_data"], ["val"], beam_width=10)
        results.append(EvalResult("small", "voiced", run_name, f"{run_name}_beam10", "beam", 10, None, None, m["wer"], m["cer"]))
        if lm_available:
            m = evaluate(ckpt, f"{run_name}_beam10_lm", "beam", ["voiced_parallel_data"], ["val"], beam_width=10, lm_path=lm_path, alpha=0.5, beta=0.0)
            results.append(EvalResult("small", "voiced", run_name, f"{run_name}_beam10_lm", "beam_lm", 10, 0.5, 0.0, m["wer"], m["cer"]))

    best_small_voiced = pick_best(results, phase="voiced", stage="small")
    best_small_ckpt = REPO / "results" / "checkpoints" / best_small_voiced.run / "best.pt"
    print(f"[info] Stage1 best voiced: {best_small_voiced.run} (CER {best_small_voiced.cer:.4f})")

    # Stage 1: small silent (init from best small voiced)
    for cfg in small_silent_cfgs:
        run_name = Path(cfg).stem
        run_dir = REPO / "results" / "checkpoints" / f"{run_name}_from_{best_small_voiced.run}"
        train(REPO / cfg, run_dir=run_dir, init=best_small_ckpt, max_batches=stage1_limit.get("max_batches"))
    for cfg in small_silent_cfgs:
        base_name = Path(cfg).stem
        run_dir = REPO / "results" / "checkpoints" / f"{base_name}_from_{best_small_voiced.run}"
        ckpt = run_dir / "best.pt"
        if not ckpt.exists():
            print(f"[warn] Missing checkpoint for {base_name}, skipping silent eval.")
            continue
        m = evaluate(ckpt, f"{base_name}_greedy", "greedy", ["silent_parallel_data"], ["eval_silent"])
        results.append(EvalResult("small", "silent", base_name, f"{base_name}_greedy", "greedy", None, None, None, m["wer"], m["cer"], init_from=best_small_voiced.run))
        m = evaluate(ckpt, f"{base_name}_beam10", "beam", ["silent_parallel_data"], ["eval_silent"], beam_width=10)
        results.append(EvalResult("small", "silent", base_name, f"{base_name}_beam10", "beam", 10, None, None, m["wer"], m["cer"], init_from=best_small_voiced.run))
        if lm_available:
            m = evaluate(ckpt, f"{base_name}_beam10_lm", "beam", ["silent_parallel_data"], ["eval_silent"], beam_width=10, lm_path=lm_path, alpha=0.5, beta=0.0)
            results.append(EvalResult("small", "silent", base_name, f"{base_name}_beam10_lm", "beam_lm", 10, 0.5, 0.0, m["wer"], m["cer"], init_from=best_small_voiced.run))

    # Select full voiced configs based on best small voiced insights.
    full_voiced_cfgs = ["configs/experiments/viz_voiced_fast_plus.yaml"]  # always include baseline anchor
    if "basic" in best_small_voiced.run:
        full_voiced_cfgs.append("configs/experiments/viz_voiced_ctc_heavy_noaug.yaml")
    if any(tag in best_small_voiced.run for tag in ["specaug", "linear"]):
        full_voiced_cfgs.append("configs/experiments/viz_voiced_channel_dropout_linear.yaml")
    # Fallback to full set if heuristics didn’t add variety.
    if len(full_voiced_cfgs) == 1:
        full_voiced_cfgs.extend([
            "configs/experiments/viz_voiced_ctc_heavy_noaug.yaml",
            "configs/experiments/viz_voiced_channel_dropout_linear.yaml",
        ])
    # Deduplicate while preserving order.
    seen = set()
    full_voiced_cfgs = [cfg for cfg in full_voiced_cfgs if not (cfg in seen or seen.add(cfg))]

    # Stage 2: full voiced
    for cfg in full_voiced_cfgs:
        train(REPO / cfg)
    for cfg in full_voiced_cfgs:
        run_name = Path(cfg).stem
        ckpt = REPO / "results" / "checkpoints" / run_name / "best.pt"
        if not ckpt.exists():
            print(f"[warn] Missing checkpoint for {run_name}, skipping eval.")
            continue
        m = evaluate(ckpt, f"{run_name}_greedy", "greedy", ["voiced_parallel_data"], ["val"])
        results.append(EvalResult("full", "voiced", run_name, f"{run_name}_greedy", "greedy", None, None, None, m["wer"], m["cer"]))
        m = evaluate(ckpt, f"{run_name}_beam50", "beam", ["voiced_parallel_data"], ["val"], beam_width=50)
        results.append(EvalResult("full", "voiced", run_name, f"{run_name}_beam50", "beam", 50, None, None, m["wer"], m["cer"]))
        if lm_available:
            m = evaluate(ckpt, f"{run_name}_beam50_lm", "beam", ["voiced_parallel_data"], ["val"], beam_width=50, lm_path=lm_path, alpha=0.5, beta=0.0)
            results.append(EvalResult("full", "voiced", run_name, f"{run_name}_beam50_lm", "beam_lm", 50, 0.5, 0.0, m["wer"], m["cer"]))

    best_full_voiced = pick_best(results, phase="voiced", stage="full")
    best_full_ckpt = REPO / "results" / "checkpoints" / best_full_voiced.run / "best.pt"
    print(f"[info] Stage2 best voiced: {best_full_voiced.run} (CER {best_full_voiced.cer:.4f})")

    # Choose full silent configs based on best small silent.
    best_small_silent = pick_best(results, phase="silent", stage="small")
    full_silent_cfgs = ["configs/mps_silent_finetune.yaml"]  # always include baseline
    if "sub4" in best_small_silent.run:
        full_silent_cfgs.append("configs/experiments/viz_silent_sub4_light_aug.yaml")
    else:
        full_silent_cfgs.append("configs/experiments/viz_silent_sub2_light_aug.yaml")
    full_silent_cfgs.append("configs/experiments/viz_silent_channel_dropout.yaml")
    seen = set()
    full_silent_cfgs = [cfg for cfg in full_silent_cfgs if not (cfg in seen or seen.add(cfg))]

    # Stage 2: full silent (init from best full voiced)
    beam_widths = [20, 50, 100]
    lm_grid = [(0.45, 0.0), (0.55, 0.0), (0.50, 0.1)]

    for cfg in full_silent_cfgs:
        run_name = Path(cfg).stem
        run_dir = REPO / "results" / "checkpoints" / f"{run_name}_from_{best_full_voiced.run}"
        train(REPO / cfg, run_dir=run_dir, init=best_full_ckpt)
    for cfg in full_silent_cfgs:
        base_name = Path(cfg).stem
        run_dir = REPO / "results" / "checkpoints" / f"{base_name}_from_{best_full_voiced.run}"
        ckpt = run_dir / "best.pt"
        if not ckpt.exists():
            print(f"[warn] Missing checkpoint for {base_name}, skipping silent eval.")
            continue
        m = evaluate(ckpt, f"{base_name}_greedy", "greedy", ["silent_parallel_data"], ["eval_silent"])
        results.append(EvalResult("full", "silent", base_name, f"{base_name}_greedy", "greedy", None, None, None, m["wer"], m["cer"], init_from=best_full_voiced.run))
        for bw in beam_widths:
            tag = f"{base_name}_beam{bw}"
            m = evaluate(ckpt, tag, "beam", ["silent_parallel_data"], ["eval_silent"], beam_width=bw)
            results.append(EvalResult("full", "silent", base_name, tag, "beam", bw, None, None, m["wer"], m["cer"], init_from=best_full_voiced.run))
        if lm_available:
            for alpha, beta in lm_grid:
                tag = f"{base_name}_lm_bw50_a{alpha}_b{beta}"
                m = evaluate(ckpt, tag, "beam", ["silent_parallel_data"], ["eval_silent"], beam_width=50, lm_path=lm_path, alpha=alpha, beta=beta)
                results.append(EvalResult("full", "silent", base_name, tag, "beam_lm", 50, alpha, beta, m["wer"], m["cer"], init_from=best_full_voiced.run))

    # Save summary
    out_dir = REPO / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        {
            "stage": r.stage,
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
    (out_dir / "two_stage_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(summary).to_csv(out_dir / "two_stage_summary.csv", index=False)
    print(f"[done] Wrote {out_dir/'two_stage_summary.json'} and .csv")


if __name__ == "__main__":
    sys.exit(main())
