# Silent Speech EMG → Text Decoder

Lightweight PyTorch pipeline for turning throat EMG signals into text using a WavLM teacher and a CTC-trained EMG encoder.

## Quick start
- Create env: `conda env create -f environment.yml && conda activate silent-speech`
- Index dataset: `python -m src.data.index_dataset --root data/emg_data --out results/index.parquet`
- Precompute features:  
  - EMG: `python -m src.data.preprocessing --mode emg --index results/index.parquet --out results/features/emg`  
  - Teacher (voiced only): `python -m src.data.preprocessing --mode teacher --index results/index.parquet --out results/features/teacher`
- Train voiced baseline: `python -m src.training.train --config configs/mps_fast_plus.yaml`
- Fine-tune for silent: `python -m src.training.train --config configs/mps_silent_finetune_plus.yaml --init-checkpoint results/checkpoints/<voiced_run>/best.pt`
- Evaluate: `python -m src.evaluation.evaluate --checkpoint results/checkpoints/<run>/best.pt --splits silent_parallel_data --subsets test --decoder beam --beam-width 50`
- Automated pipeline (probes → stage2 → decoder sweeps): `python -m src.experiments.orchestrate`

## Repository layout
- `configs/` training configs and vocab
- `src/` data processing, models, training, decoding, experiments
- `tests/` small unit tests for data, models, and training/eval
- `results/` gitignored outputs (checkpoints, evals, features)
- `data/` gitignored dataset root (`data/emg_data/` expected)

## More detail
See `blog_post.pdf` for a narrative walkthrough of the system and methodology.
