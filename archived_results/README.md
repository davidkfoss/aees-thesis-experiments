# Archived Results

This directory is the expected location for archived experiment outputs used to regenerate thesis and ICONIP 2026 paper-submission tables and figures.

The archived result bundle is available as a GitHub release asset:

<https://github.com/davidkfoss/aees-thesis-experiments/releases/tag/v0.1.1>

The bundle is not tracked in Git because it contains large per-run output files. To regenerate tables and figures from archived results, place or extract it so that the directory structure matches:

```text
archived_results/
  checkpointing/
  cifar_clean/
  cifar_noisy/
  clean_agnews/
  compute_overhead/
  instability_penalty_ablation/
  nlp_noise_ablation/
  noisy_agnews/
  sst2/
  trend_context_ablation/
```

## Contents

- `checkpointing/`: peak-validation checkpoint rerun outputs; source for the checkpoint-diagnostics tables and the peak-checkpoint diagnostics figure.
- `cifar_clean/`: clean CIFAR-100 experiment outputs.
- `cifar_noisy/`: noisy CIFAR-100 runs, fixed-multiplier ablations, and related diagnostics.
- `clean_agnews/`: clean AG News fine-tuning outputs.
- `compute_overhead/`: timing outputs used for wall-clock overhead tables.
- `instability_penalty_ablation/`: supplementary/appendix ablation outputs for instability-penalty variants.
- `nlp_noise_ablation/`: AG News noise-axis falsification outputs.
- `noisy_agnews/`: noisy AG News main and axis-ablation outputs.
- `sst2/`: SST-2 fine-tuning outputs used in the thesis.
- `trend_context_ablation/`: supplementary/appendix ablation outputs for trend-context variants.

## Thesis and conference-paper subset

The thesis uses the broader archive, including clean CIFAR-100, clean AG News, SST-2, and supplementary ablation results.

The ICONIP 2026 paper submission uses a scoped subset of this archive, primarily:

- `checkpointing/`
- `cifar_ablation/`
- `cifar_noisy/`
- `noisy_agnews/`
- `nlp_noise_ablation/`
- `compute_overhead/`

## Regeneration workflow

Both the figure scripts (`scripts/plots/`) and table scripts (`scripts/tables/`) are configured with default paths. In normal use they run directly with no arguments: they read from `archived_results/` and write regenerated outputs to `reproduced_artifacts/figures/...` and `reproduced_artifacts/tables/...` respectively.

```bash
uv run python -m scripts.plots.cifar.plot_cifar_noisy_curves
uv run python scripts/tables/make_nlp_noise_ablation_table.py
```

To regenerate everything at once, use the bundled runners (from the repo root):

```bash
make plots      # or: ./scripts/plots/run_all.sh
make tables     # or: ./scripts/tables/run_all.sh
make artifacts  # both
```

A few figure scripts produce one figure per variant and accept an optional selector — `cifar.plot_cifar_noisy_curves` takes `--noise-setting`, and `controllers.plot_arm_selection` / `controllers.plot_reward_per_arm` take `--setting`. Omit the selector to emit all variants. Pass `--runs-root`/`--out-dir` to any script to override its default input/output paths; check a script's `--help` for details.

When a figure script cannot find its inputs, it may exit non-zero and write `<out-dir>/<figure-name>.MISSING.md` listing the exact paths it looked for. Check that file to diagnose a failed run.

Regenerated outputs should be compared against the curated artifacts in `generated_artifacts/` before replacing any committed files.

## Notes

The archive is intended to support regeneration of aggregate tables and figures. It is not intended to provide model checkpoints or bitwise replay of every training trajectory.

Large generated files, model checkpoints, dataset caches, temporary logs, and unused exploratory runs are excluded where possible.
