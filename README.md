# AEES Thesis Experiments

This repository contains the experiment runners, configurations, and result-processing scripts used for the thesis on **Adaptive Episodic Exploration Scheduling (AEES)** — a small per-axis bandit that adapts learning-rate multiplier and gradient-noise std per training "episode" on top of a standard optimizer backbone.

The reusable AEES implementation is **not** in this repository. It is published on PyPI as [`pulseopt`](https://pypi.org/project/pulseopt/) and maintained at [`davidkfoss/pulseopt`](https://github.com/davidkfoss/pulseopt). This repository pulls it in as a normal dependency.

## Setup

Python 3.11 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

`pip install -e .` declares this directory's dependencies (including `pulseopt>=0.1.5`); the project itself ships nothing.

## Running an experiment

Each runner exposes its full flag set under `--help`:

```bash
python experiments/task_cifar100.py --help
python experiments/task_sst2.py --help
python experiments/task_agnews.py --help
```

A run writes a `RunResult` JSON under `results/` via `experiments/utils/results.py`. The runners cache HuggingFace datasets/models under `.hf_cache/` and pre-tokenized datasets under `data/`; both directories are gitignored.

Key flags exposed by all runners (see `--help` for the full list):

- `--lr-candidates`, `--noise-candidates` — comma-separated candidate values; single-candidate axes are treated as fixed constants and skip controller creation.
- `--structured-control-mode {independent,conditional}`
- `--context-mode {none,trend,trend_phase}` (`trend_phase` requires `--total-training-steps`)
- `--episode-length`, `--reward-instability-lambda`, `--reward-clip-{min,max}`
- `--lr-scheduler {none,cosine,linear,warmup_linear}`, `--scheduler-t-max`, `--warmup-epochs`

CIFAR-specific: `--label-noise-type {none,symmetric,asymmetric}`, `--label-noise-rate`, and `--control-mode {baseline,adaptive,random}`. SST-2 / AG News use `--method {AdamW,AdaptiveScheduler,RandomScheduler}`.

## Tables and plots

`scripts/tables/` and `scripts/plots/` regenerate the thesis tables and figures by reading run-result JSON from `results/`. They are read-only consumers — they do not run experiments. `scripts/report_*.py` produce summary CSVs over a results directory.

```bash
python scripts/tables/make_cifar_clean_tables.py --help
python scripts/plots/cifar_plots.py --help
```

## Tests

Tests covering the experiment-side `RunResult` runtime-metric derivation:

```bash
pytest tests/test_runtime_metrics.py
```

Library-side tests (controllers, episode manager, reward, scheduler) live with the `pulseopt` source.

## Layout

- `experiments/task_*.py` — three thesis runners (CIFAR-100, SST-2, AG News).
- `experiments/utils/{flops,metrics,results,run_plan}.py` — experiment-side helpers (FLOP accounting, `RunResult` dataclass, JSON IO, run-plan/manifest helpers).
- `scripts/tables/`, `scripts/plots/`, `scripts/report_*.py` — reporting over `results/`.
- `tests/test_runtime_metrics.py` — runtime-metric derivation tests.
- `data/`, `results/` — local dataset cache and run outputs (gitignored except `.gitkeep`).

## Reproducibility notes

- Fixed run seeds and deterministic label-noise construction are used. Seed handling, label-noise protocols, scheduler settings, reward shaping, and gradient-noise generator construction are defined by `pulseopt` and the experiment runners.
- Newer result files log hardware/software metadata under `runtime_metrics.hardware`: GPU name(s) and memory, Python version, PyTorch and CUDA versions, cuDNN version, plus the runner's `num_workers` and `pin_memory` settings.
- Some older archived result files predate that capture and may not contain a complete hardware/software metadata block.
- Exact epoch-level or bitwise reproduction across machines is not guaranteed: GPU architecture, CUDA/cuDNN kernels, PyTorch/torchvision versions, and DataLoader/runtime behavior can introduce small trajectory differences. The `pulseopt` dependency is pinned through `pyproject.toml`; bumping it changes the library version backing the runners.
- Archived result files under `results/` are the authoritative source for the thesis tables and figures; the reporting scripts in `scripts/` consume them read-only.
- Datasets are pulled into `.hf_cache/` / `data/` on first run; both are gitignored. No checkpoints, raw logs, or bulk JSON outputs are tracked in git.

## Library reference

The reusable `pulseopt` package contains the public AEES API, quick-start examples, and design notes:

- PyPI: <https://pypi.org/project/pulseopt/>
- GitHub: <https://github.com/davidkfoss/pulseopt>

## License

MIT — see [LICENSE](LICENSE).
