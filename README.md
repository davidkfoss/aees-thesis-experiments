# AEES Thesis Experiments

This repository contains the experiment runners, archived result artifacts, and result-processing scripts used for the thesis on **Adaptive Episodic Exploration Scheduling (AEES)** — a small per-axis bandit that adapts learning-rate multiplier and gradient-noise std per training episode on top of a standard optimizer backbone.

The reusable AEES implementation is **not** in this repository. It is published on PyPI as [`pulseopt`](https://pypi.org/project/pulseopt/) and maintained at [`davidkfoss/pulseopt`](https://github.com/davidkfoss/pulseopt). This repository pulls it in as a normal dependency.

## Setup

Python 3.11 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

`pip install -e .` declares this directory's dependencies, including `pulseopt>=0.1.5`; the project itself ships no reusable Python package.

## Running an experiment

Each runner exposes its full flag set under `--help`:

```bash
python experiments/task_cifar100.py --help
python experiments/task_sst2.py --help
python experiments/task_agnews.py --help
```

A run writes a `RunResult` JSON under `results/` via `experiments/utils/results.py`. The runners cache HuggingFace datasets/models under `.hf_cache/` and pre-tokenized datasets under `data/`; both directories are gitignored.

Key flags exposed by all runners, with the full list available through `--help`:

- `--lr-candidates`, `--noise-candidates` — comma-separated candidate values; single-candidate axes are treated as fixed constants and skip controller creation.
- `--structured-control-mode {independent,conditional}`
- `--context-mode {none,trend,trend_phase}` (`trend_phase` requires `--total-training-steps`)
- `--episode-length`, `--reward-instability-lambda`, `--reward-clip-{min,max}`
- `--lr-scheduler {none,cosine,linear,warmup_linear}`, `--scheduler-t-max`, `--warmup-epochs`

CIFAR-specific flags include `--label-noise-type {none,symmetric,asymmetric}`, `--label-noise-rate`, and `--control-mode {baseline,adaptive,random}`. SST-2 and AG News use `--method {AdamW,AdaptiveScheduler,RandomScheduler}`.

### NLP caching and local-only runs

The SST-2 and AG News runners use HuggingFace datasets, tokenizers, and pretrained
DistilBERT weights. Dataset/model files are cached under `.hf_cache/`, while
pre-tokenized datasets can be saved under `data/`.

For runtime measurements, downloads and tokenization should be completed before the
measured training run. The recommended pattern is:

1. Run `--pretokenize-only` with `--cache-dir` and `--tokenized-dataset-dir` to
   download/cache the dataset and tokenizer files and save the tokenized dataset.
2. Run a short non-local smoke job with the same `--cache-dir` and
   `--tokenized-dataset-dir` to download/cache the pretrained model weights and verify
   that the runner works end-to-end.
3. Run the actual measured jobs with the same `--cache-dir`,
   `--tokenized-dataset-dir`, and `--local-files-only`.

Example for AG News:

```bash
uv run python experiments/task_agnews.py \
  --pretokenize-only \
  --epochs 1 \
  --batch-size 16 \
  --max-length 128 \
  --lr 5e-5 \
  --weight-decay 0.01 \
  --cache-dir .hf_cache \
  --tokenized-dataset-dir data/agnews_tokenized
```

```bash
uv run python experiments/task_agnews.py \
  --method AdamW \
  --epochs 1 \
  --batch-size 16 \
  --max-length 128 \
  --lr 5e-5 \
  --weight-decay 0.01 \
  --lr-scheduler none \
  --cache-dir .hf_cache \
  --tokenized-dataset-dir data/agnews_tokenized \
  --seed 0 \
  --output results/agnews_smoke_seed0.json
```

```bash
uv run python experiments/task_agnews.py \
  --method AdamW \
  --epochs 1 \
  --batch-size 16 \
  --max-length 128 \
  --lr 5e-5 \
  --weight-decay 0.01 \
  --lr-scheduler none \
  --cache-dir .hf_cache \
  --tokenized-dataset-dir data/agnews_tokenized \
  --local-files-only \
  --seed 0 \
  --output results/agnews_local_smoke_seed0.json
```

For SST-2, use the same sequence with `experiments/task_sst2.py` and
`--tokenized-dataset-dir data/sst2_tokenized`.

This sequence separates one-time dataset download, tokenization, and model-weight
download from runtime-sensitive training runs. It also makes repeated experiments faster
and avoids unnecessary repeated requests to the HuggingFace Hub. The measured
compute-overhead experiments should use the cached HuggingFace files via
`--cache-dir`, the saved tokenized datasets via `--tokenized-dataset-dir`, and
`--local-files-only` to avoid network access during the run.

### Example commands

Representative clean CIFAR-100 AdamW baseline:

```bash
python experiments/task_cifar100.py \
  --control-mode baseline \
  --optimizer AdamW \
  --seed 0
```

Representative noisy CIFAR-100 AEES-LR run:

```bash
python experiments/task_cifar100.py \
  --control-mode adaptive \
  --optimizer AdamW \
  --label-noise-type symmetric \
  --label-noise-rate 0.4 \
  --episode-length 200 \
  --lr-candidates 0.5,1.0,2.0 \
  --noise-candidates 0.0 \
  --seed 0
```

Representative SST-2 warmup-linear AdamW baseline:

```bash
python experiments/task_sst2.py \
  --method AdamW \
  --lr-scheduler warmup_linear \
  --seed 0
```

Representative AG News noise-only AEES run:

```bash
python experiments/task_agnews.py \
  --method AdaptiveScheduler \
  --lr-scheduler warmup_linear \
  --lr-candidates 1.0 \
  --noise-candidates 0.0,0.005,0.01 \
  --label-noise-rate 0.2 \
  --episode-length 200 \
  --seed 0
```

These commands are representative single-run examples. The thesis tables are generated from archived result artifacts rather than by launching experiments from this README.

## Tables and plots

`scripts/tables/` and `scripts/plots/` regenerate the thesis tables and figures by reading archived run-result JSON files. These scripts are read-only consumers: they do not launch training runs. `scripts/report_*.py` produce summary CSVs over a results directory.

Example help commands:

```bash
python scripts/tables/make_cifar_clean_tables.py --help
python scripts/plots/cifar_plots.py --help
```

The archived result files are the authoritative source for the numerical tables reported in the thesis. Re-running training with the same seeds should reproduce the same qualitative behavior and similar aggregate results, but exact trajectory-level or bitwise reproduction across machines is not guaranteed.

## Tests

Tests covering the experiment-side `RunResult` runtime-metric derivation:

```bash
pytest tests/test_runtime_metrics.py
```

Library-side tests for controllers, episode management, rewards, and optimizer wrapping live with the `pulseopt` source repository.

## Layout

- `experiments/task_*.py` — three thesis runners: CIFAR-100, SST-2, and AG News.
- `experiments/utils/{flops,metrics,results,run_plan}.py` — experiment-side helpers for FLOP accounting, `RunResult` serialization, JSON IO, and run-plan/manifest utilities.
- `scripts/tables/`, `scripts/plots/`, `scripts/report_*.py` — reporting scripts over archived result JSON files.
- `tests/test_runtime_metrics.py` — runtime-metric derivation tests.
- `data/`, `.hf_cache/` — local dataset/model/tokenization caches, gitignored.
- `results/` — local run outputs and archived compact result artifacts used by the thesis; large temporary outputs, checkpoints, caches, and raw logs are not tracked.

## Reproducibility notes

- Fixed run seeds and deterministic label-noise construction are used.
- Seed handling, label-noise protocols, scheduler settings, reward shaping, and gradient-noise generator construction are defined by `pulseopt` and the experiment runners.
- Newer result files log hardware/software metadata under `runtime_metrics.hardware`: GPU name(s) and memory, Python version, PyTorch and CUDA versions, cuDNN version, plus the runner's `num_workers` and `pin_memory` settings.
- Some older archived result files predate that capture and may not contain a complete hardware/software metadata block.
- Exact epoch-level or bitwise reproduction across machines is not guaranteed. GPU architecture, CUDA/cuDNN kernels, PyTorch/torchvision versions, and DataLoader/runtime behavior can introduce small trajectory differences.
- For NLP experiments, tokenized datasets and HuggingFace model/tokenizer files should be cached before runtime-sensitive experiments. `--pretokenize-only` prepares tokenized datasets, and `--local-files-only` prevents new downloads once required files are cached.
- The `pulseopt` dependency is pinned through `pyproject.toml`; changing the pinned version changes the AEES implementation backing the runners.
- Archived result files are treated as the authoritative source for the thesis tables and figures; the reporting scripts in `scripts/` consume them read-only.
- Datasets and model/tokenizer caches are pulled into `.hf_cache/` and `data/` on first run. These directories are gitignored.
- Checkpoints, raw logs, and bulk temporary outputs are not intended to be tracked in git.

## Library reference

The reusable `pulseopt` package contains the public AEES API, quick-start examples, and design notes:

- PyPI: <https://pypi.org/project/pulseopt/>
- GitHub: <https://github.com/davidkfoss/pulseopt>

## License

MIT — see [LICENSE](LICENSE).
