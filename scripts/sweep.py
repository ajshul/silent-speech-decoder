#!/usr/bin/env python
"""
Lightweight experiment runner for small hyperparameter sweeps.

Takes a YAML spec describing a base config and per-experiment overrides,
writes resolved configs, and optionally launches training runs sequentially.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts (override wins)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_yaml(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def build_command(
    cfg_path: Path,
    run_dir: Path,
    overfit_batches: int | None,
    dry_run: bool,
) -> List[str]:
    cmd = ["python", "-m", "src.training.train", "--config", str(cfg_path), "--run-dir", str(run_dir)]
    if overfit_batches:
        cmd += ["--overfit-batches", str(overfit_batches)]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_experiments(spec_path: Path, execute: bool) -> None:
    spec = load_yaml(spec_path)
    base_cfg_path = Path(spec["base_config"])
    base_cfg = load_yaml(base_cfg_path)

    output_dir = Path(spec.get("output_dir", "results/sweeps")) / datetime.now().strftime("%Y%m%d_%H%M%S")
    common_overrides = spec.get("common_overrides", {})
    default_overfit = spec.get("overfit_batches")
    default_dry_run = spec.get("dry_run", False)
    experiments = spec.get("experiments", [])

    base_run_name = base_cfg.get("logging", {}).get("run_name", base_cfg_path.stem)

    for exp in experiments:
        name = exp["name"]
        overrides = exp.get("overrides", {})
        exp_cfg = deep_merge(base_cfg, common_overrides)
        exp_cfg = deep_merge(exp_cfg, overrides)

        # Ensure run_name reflects the experiment.
        exp_run_name = f"{base_run_name}__{name}"
        exp_cfg.setdefault("logging", {})
        exp_cfg["logging"]["run_name"] = exp_run_name

        overfit_batches = exp.get("overfit_batches", default_overfit)
        dry_run = exp.get("dry_run", default_dry_run)

        exp_dir = output_dir / name
        cfg_path = exp_dir / "config.yaml"
        write_yaml(exp_cfg, cfg_path)

        cmd = build_command(cfg_path, exp_dir, overfit_batches=overfit_batches, dry_run=dry_run)
        print(f"[sweep] {name}: {' '.join(cmd)}")
        if not execute:
            continue

        env = os.environ.copy()
        if env.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small hyperparameter sweeps.")
    parser.add_argument("--spec", type=Path, required=True, help="YAML spec listing experiments.")
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help="Only write resolved configs, do not launch training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(spec_path=args.spec, execute=not args.no_exec)


if __name__ == "__main__":
    main()
