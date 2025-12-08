from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

VOICED_BASE_CONFIG = Path("configs/mps_fast_plus.yaml")
SILENT_BASE_CONFIG = Path("configs/mps_silent_finetune_plus.yaml")


def _load_yaml(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _deep_update(base: Dict, overrides: Dict) -> Dict:
    out = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _attach_metadata(cfg: Dict, name: str, stage: str, dataset: str, tags: List[str], description: str, probe_batches: Optional[int]) -> Dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("logging", {})["run_name"] = name
    cfg["experiment"] = {
        "stage": stage,
        "dataset": dataset,
        "tags": tags,
        "description": description,
        "probe_batches": probe_batches,
    }
    return cfg


@dataclass
class DecoderSetting:
    name: str
    method: str = "greedy"
    beam_width: Optional[int] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    beam_prune_logp: Optional[float] = None
    blank_bias: float = 0.0
    use_lm: bool = False
    lm_path: Optional[Path] = None


@dataclass
class RunSpec:
    name: str
    stage: str
    dataset: str
    config: Dict
    decoder_grid: List[DecoderSetting]
    overfit_batches: Optional[int] = None
    init_checkpoint: Optional[Path] = None
    tags: List[str] = field(default_factory=list)
    description: str = ""


PROBE_DECODERS_VOICED: List[DecoderSetting] = [
    DecoderSetting(name="greedy", method="greedy", blank_bias=0.0),
    DecoderSetting(name="beam20", method="beam", beam_width=20, alpha=0.45, beta=0.0, beam_prune_logp=-10.0),
    DecoderSetting(name="beam20_bias", method="beam", beam_width=20, alpha=0.45, beta=0.0, beam_prune_logp=-10.0, blank_bias=0.12),
]

PROBE_DECODERS_SILENT: List[DecoderSetting] = [
    DecoderSetting(name="greedy", method="greedy", blank_bias=0.0),
    DecoderSetting(name="beam20", method="beam", beam_width=20, alpha=0.45, beta=0.0, beam_prune_logp=-10.0),
    DecoderSetting(name="beam20_bias", method="beam", beam_width=20, alpha=0.45, beta=0.0, beam_prune_logp=-10.0, blank_bias=0.2),
]

FULL_DECODERS_VOICED: List[DecoderSetting] = [
    DecoderSetting(name="greedy", method="greedy"),
    DecoderSetting(name="beam50", method="beam", beam_width=50, alpha=0.45, beta=0.0, beam_prune_logp=-10.0),
    DecoderSetting(name="beam50_bias", method="beam", beam_width=50, alpha=0.45, beta=0.0, beam_prune_logp=-10.0, blank_bias=0.1),
    DecoderSetting(name="beam50_lm", method="beam", beam_width=50, alpha=0.5, beta=0.05, beam_prune_logp=-10.0, blank_bias=0.05, use_lm=True, lm_path=Path("results/lm/char_5gram.arpa")),
]

FULL_DECODERS_SILENT: List[DecoderSetting] = [
    DecoderSetting(name="greedy", method="greedy"),
    DecoderSetting(name="beam20_bias", method="beam", beam_width=20, alpha=0.45, beta=0.0, beam_prune_logp=-10.0, blank_bias=0.2),
    DecoderSetting(name="beam50", method="beam", beam_width=50, alpha=0.5, beta=0.0, beam_prune_logp=-12.0, blank_bias=0.1),
    DecoderSetting(name="beam100", method="beam", beam_width=100, alpha=0.55, beta=0.05, beam_prune_logp=-12.0, blank_bias=0.05),
    DecoderSetting(name="beam50_lm", method="beam", beam_width=50, alpha=0.5, beta=0.05, beam_prune_logp=-10.0, blank_bias=0.05, use_lm=True, lm_path=Path("results/lm/char_5gram.arpa")),
]


def build_voiced_probe_configs(probe_batches: int) -> List[RunSpec]:
    base = _load_yaml(VOICED_BASE_CONFIG)
    base = _deep_update(
        base,
        {
            "optim": {
                "max_epochs": 6,
                "early_stopping": {"patience": 2, "min_delta": 0.0},
            },
        },
    )
    variants = [
        {
            "name": "probe_voiced_hold_lightaug",
            "tags": ["specaug_light", "warmup_hold"],
            "description": "Baseline-sized student with warmup-hold and light SpecAugment to gauge stability.",
            "overrides": {
                "augmentation": {
                    "specaugment": {"p": 0.22, "time_masks": 2, "freq_masks": 2, "time_mask_width": 0.06, "freq_mask_width": 8}
                },
                "loss": {"lambda_ctc": 0.65, "lambda_distill": 0.35, "distill_warmup_epochs": 1},
                "optim": {"scheduler": {"name": "warmup_hold", "warmup_steps": 360}},
            },
        },
        {
            "name": "probe_voiced_ctc_noaug",
            "tags": ["ctc_heavy", "no_aug"],
            "description": "CTC-leaning mix with SpecAugment off to see if regularization hurts early convergence.",
            "overrides": {
                "augmentation": {"specaugment": {"p": 0.0}},
                "loss": {"lambda_ctc": 0.8, "lambda_distill": 0.2, "distill_warmup_epochs": 0},
                "optim": {"scheduler": {"name": "warmup_hold", "warmup_steps": 260}},
            },
        },
        {
            "name": "probe_voiced_cosine_stronger_aug",
            "tags": ["cosine", "specaug_strong"],
            "description": "Cosine schedule with heavier SpecAugment to test robustness under aggressive masking.",
            "overrides": {
                "augmentation": {
                    "specaugment": {"p": 0.45, "time_masks": 3, "freq_masks": 3, "time_mask_width": 0.08, "freq_mask_width": 10}
                },
                "loss": {"lambda_ctc": 0.6, "lambda_distill": 0.4, "distill_warmup_epochs": 2},
                "optim": {"scheduler": {"name": "cosine", "t_max": 1500, "eta_min": 3e-5}},
                "model": {"encoder": {"dropout": 0.14}},
            },
        },
        {
            "name": "probe_voiced_channel_dropout",
            "tags": ["channel_dropout", "specaug_mid"],
            "description": "Moderate SpecAugment plus channel dropout to test cross-channel robustness.",
            "overrides": {
                "augmentation": {
                    "specaugment": {"p": 0.28, "time_masks": 2, "freq_masks": 2, "time_mask_width": 0.06, "freq_mask_width": 8},
                    "channel_dropout": {"p": 0.15, "max_channels": 2},
                },
                "loss": {"lambda_ctc": 0.62, "lambda_distill": 0.38, "distill_warmup_epochs": 2},
                "optim": {"scheduler": {"name": "warmup_hold", "warmup_steps": 340}},
            },
        },
        {
            "name": "probe_voiced_linear_distill",
            "tags": ["linear", "distill_heavy"],
            "description": "Linear warmup/decay with heavier distillation and light augmentation to test alignment benefits.",
            "overrides": {
                "augmentation": {
                    "specaugment": {"p": 0.18, "time_masks": 2, "freq_masks": 2, "time_mask_width": 0.05, "freq_mask_width": 8}
                },
                "loss": {"lambda_ctc": 0.55, "lambda_distill": 0.45, "distill_warmup_epochs": 3},
                "optim": {"scheduler": {"name": "linear", "warmup_steps": 420, "total_steps": 2000}},
            },
        },
    ]

    runs: List[RunSpec] = []
    for variant in variants:
        cfg = _deep_update(base, variant["overrides"])
        cfg = _attach_metadata(
            cfg,
            name=variant["name"],
            stage="stage1",
            dataset="voiced",
            tags=variant["tags"],
            description=variant["description"],
            probe_batches=probe_batches,
        )
        runs.append(
            RunSpec(
                name=variant["name"],
                stage="stage1",
                dataset="voiced",
                config=cfg,
                decoder_grid=PROBE_DECODERS_VOICED,
                overfit_batches=probe_batches,
                tags=variant["tags"],
                description=variant["description"],
            )
        )
    return runs


def build_silent_probe_configs(probe_batches: int, init_checkpoint: Optional[Path]) -> List[RunSpec]:
    base = _load_yaml(SILENT_BASE_CONFIG)
    base = _deep_update(
        base,
        {
            "optim": {
                "max_epochs": 6,
                "early_stopping": {"patience": 2, "min_delta": 0.0},
            },
            "data": {"include_teacher": False, "teacher_strict": False},
        },
    )
    variants = [
        {
            "name": "probe_silent_sub2_light",
            "tags": ["sub2", "specaug_light"],
            "description": "Silent fine-tune at sub2 with the light baseline augmentation.",
            "overrides": {
                "model": {"encoder": {"subsample_factor": 2}},
                "augmentation": {"specaugment": {"p": 0.08, "time_masks": 1, "freq_masks": 1, "time_mask_width": 0.05, "freq_mask_width": 6}},
                "optim": {"scheduler": {"name": "warmup_hold", "warmup_steps": 360}},
            },
        },
        {
            "name": "probe_silent_sub4_fast",
            "tags": ["sub4", "speed"],
            "description": "Faster CTCLoss path via subsample 4 with light SpecAugment; checks for accuracy drop.",
            "overrides": {
                "model": {"encoder": {"subsample_factor": 4}},
                "optim": {"batch_size": 5},
                "augmentation": {"specaugment": {"p": 0.05, "time_masks": 1, "freq_masks": 1, "time_mask_width": 0.05, "freq_mask_width": 6}},
                "optim": {"scheduler": {"name": "warmup_hold", "warmup_steps": 360}},
            },
        },
        {
            "name": "probe_silent_specaug_mid",
            "tags": ["sub2", "specaug_mid"],
            "description": "Sub2 with mid-strength SpecAugment to test if silent EMG benefits from stronger masking.",
            "overrides": {
                "model": {"encoder": {"subsample_factor": 2}},
                "augmentation": {"specaugment": {"p": 0.16, "time_masks": 2, "freq_masks": 2, "time_mask_width": 0.08, "freq_mask_width": 8}},
                "optim": {"scheduler": {"name": "warmup_hold", "warmup_steps": 360}},
            },
        },
        {
            "name": "probe_silent_channel_dropout",
            "tags": ["sub2", "channel_dropout"],
            "description": "Sub2 with channel dropout to encourage robustness to missing electrodes.",
            "overrides": {
                "model": {"encoder": {"subsample_factor": 2}},
                "augmentation": {
                    "specaugment": {"p": 0.1, "time_masks": 1, "freq_masks": 1, "time_mask_width": 0.05, "freq_mask_width": 6},
                    "channel_dropout": {"p": 0.12, "max_channels": 2},
                },
                "optim": {"scheduler": {"name": "warmup_hold", "warmup_steps": 360}},
            },
        },
    ]

    runs: List[RunSpec] = []
    for variant in variants:
        cfg = _deep_update(base, variant["overrides"])
        cfg = _attach_metadata(
            cfg,
            name=variant["name"],
            stage="stage1",
            dataset="silent",
            tags=variant["tags"],
            description=variant["description"],
            probe_batches=probe_batches,
        )
        runs.append(
            RunSpec(
                name=variant["name"],
                stage="stage1",
                dataset="silent",
                config=cfg,
                decoder_grid=PROBE_DECODERS_SILENT,
                overfit_batches=probe_batches,
                init_checkpoint=init_checkpoint,
                tags=variant["tags"],
                description=variant["description"],
            )
        )
    return runs


def build_voiced_stage2_configs(best_probe: Dict, include_baseline: bool = True) -> List[RunSpec]:
    """
    Build Stage 2 voiced run specs using the best Stage 1 probe metrics.
    `best_probe` is expected to contain keys extracted from a summary record.
    """
    base_cfg = _load_yaml(VOICED_BASE_CONFIG)
    runs: List[RunSpec] = []
    if include_baseline:
        baseline_cfg = _attach_metadata(
            base_cfg,
            name="stage2_voiced_baseline",
            stage="stage2",
            dataset="voiced",
            tags=["baseline"],
            description="Baseline voiced run (anchor) without Stage 1 changes.",
            probe_batches=None,
        )
        runs.append(
            RunSpec(
                name="stage2_voiced_baseline",
                stage="stage2",
                dataset="voiced",
                config=baseline_cfg,
                decoder_grid=FULL_DECODERS_VOICED,
                tags=["baseline"],
                description="Baseline voiced run (anchor) without Stage 1 changes.",
            )
        )

    adapted_cfg = copy.deepcopy(base_cfg)
    adapted_cfg = _deep_update(
        adapted_cfg,
        {
            "augmentation": {
                "specaugment": {
                    "p": max(best_probe.get("specaugment_p", 0.25), 0.15),
                    "time_masks": best_probe.get("specaugment_time_masks", 2),
                    "freq_masks": best_probe.get("specaugment_freq_masks", 2),
                    "time_mask_width": best_probe.get("specaugment_time_width", 0.06),
                    "freq_mask_width": best_probe.get("specaugment_freq_width", 8),
                }
            },
            "loss": {
                "lambda_ctc": best_probe.get("lambda_ctc", 0.65),
                "lambda_distill": best_probe.get("lambda_distill", 0.35),
                "distill_warmup_epochs": best_probe.get("distill_warmup_epochs", 2),
            },
            "optim": {
                "scheduler": best_probe.get("scheduler_cfg", {"name": best_probe.get("scheduler", "warmup_hold"), "warmup_steps": 600}),
                "max_epochs": 50,
                "early_stopping": {"patience": 5, "min_delta": 0.0},
            },
            "model": {"encoder": {"dropout": best_probe.get("dropout", 0.12)}},
        },
    )
    if best_probe.get("channel_dropout_p", 0.0) > 0.0:
        adapted_cfg = _deep_update(
            adapted_cfg,
            {
                "augmentation": {
                    "channel_dropout": {
                        "p": best_probe.get("channel_dropout_p", 0.1),
                        "max_channels": best_probe.get("channel_dropout_max", 2),
                    }
                }
            },
        )
    # Set default decoding to the observed best probe decoder.
    if best_probe.get("decoder_type"):
        adapted_cfg = _deep_update(
            adapted_cfg,
            {
                "decoding": {
                    "type": best_probe.get("decoder_type", "beam"),
                    "beam_width": best_probe.get("beam_width", 50),
                    "alpha": best_probe.get("alpha", 0.45),
                    "beta": best_probe.get("beta", 0.0),
                    "beam_prune_logp": best_probe.get("beam_prune_logp", -10.0),
                    "lm_path": best_probe.get("lm_path"),
                }
            },
        )
        if best_probe.get("blank_bias") is not None:
            adapted_cfg = _deep_update(adapted_cfg, {"decoding": {"blank_bias": best_probe.get("blank_bias")}})

    adapted_cfg = _attach_metadata(
        adapted_cfg,
        name="stage2_voiced_adapted",
        stage="stage2",
        dataset="voiced",
        tags=["stage1_guided"],
        description="Stage 2 voiced config derived from best Stage 1 probe.",
        probe_batches=None,
    )
    runs.append(
        RunSpec(
            name="stage2_voiced_adapted",
            stage="stage2",
            dataset="voiced",
            config=adapted_cfg,
            decoder_grid=FULL_DECODERS_VOICED,
            tags=["stage1_guided"],
            description="Stage 2 voiced config derived from best Stage 1 probe.",
        )
    )
    return runs


def build_silent_stage2_configs(
    best_probe: Dict,
    init_checkpoint: Path,
    include_baseline: bool = True,
) -> List[RunSpec]:
    base_cfg = _load_yaml(SILENT_BASE_CONFIG)
    runs: List[RunSpec] = []
    if include_baseline:
        run_name = "stage2_silent_baseline"
        init_for_baseline = init_checkpoint
        baseline_cfg = _attach_metadata(
            base_cfg,
            name=run_name,
            stage="stage2",
            dataset="silent",
            tags=["baseline"],
            description="Baseline silent fine-tune (anchor) from best voiced.",
            probe_batches=None,
        )
        runs.append(
            RunSpec(
                name=run_name,
                stage="stage2",
                dataset="silent",
                config=baseline_cfg,
                decoder_grid=FULL_DECODERS_SILENT,
                init_checkpoint=init_for_baseline,
                tags=["baseline"],
                description="Baseline silent fine-tune (anchor) from best voiced.",
            )
        )

    adapted_cfg = copy.deepcopy(base_cfg)
    adapted_cfg = _deep_update(
        adapted_cfg,
        {
            "model": {"encoder": {"subsample_factor": best_probe.get("subsample_factor", 2)}},
            "augmentation": {
                "specaugment": {
                    "p": best_probe.get("specaugment_p", base_cfg.get("augmentation", {}).get("specaugment", {}).get("p", 0.05)),
                    "time_masks": best_probe.get("specaugment_time_masks", 1),
                    "freq_masks": best_probe.get("specaugment_freq_masks", 1),
                    "time_mask_width": best_probe.get("specaugment_time_width", 0.05),
                    "freq_mask_width": best_probe.get("specaugment_freq_width", 6),
                }
            },
            "optim": {"max_epochs": 32, "early_stopping": {"patience": 5, "min_delta": 0.0}},
        },
    )
    if best_probe.get("channel_dropout_p", 0.0) > 0.0:
        adapted_cfg = _deep_update(
            adapted_cfg,
            {
                "augmentation": {
                    "channel_dropout": {
                        "p": best_probe.get("channel_dropout_p", 0.1),
                        "max_channels": best_probe.get("channel_dropout_max", 2),
                    }
                }
            },
        )
    if best_probe.get("decoder_type"):
        adapted_cfg = _deep_update(
            adapted_cfg,
            {
                "decoding": {
                    "type": best_probe.get("decoder_type", "beam"),
                    "beam_width": best_probe.get("beam_width", 50),
                    "alpha": best_probe.get("alpha", 0.5),
                    "beta": best_probe.get("beta", 0.0),
                    "beam_prune_logp": best_probe.get("beam_prune_logp", -10.0),
                    "lm_path": best_probe.get("lm_path"),
                }
            },
        )
        if best_probe.get("blank_bias") is not None:
            adapted_cfg = _deep_update(adapted_cfg, {"decoding": {"blank_bias": best_probe.get("blank_bias")}})

    adapted_cfg = _attach_metadata(
        adapted_cfg,
        name="stage2_silent_adapted",
        stage="stage2",
        dataset="silent",
        tags=["stage1_guided"],
        description="Silent fine-tune derived from best Stage 1 silent probe.",
        probe_batches=None,
    )
    runs.append(
        RunSpec(
            name="stage2_silent_adapted",
            stage="stage2",
            dataset="silent",
            config=adapted_cfg,
            decoder_grid=FULL_DECODERS_SILENT,
            init_checkpoint=init_checkpoint,
            tags=["stage1_guided"],
            description="Silent fine-tune derived from best Stage 1 silent probe.",
        )
    )
    return runs
