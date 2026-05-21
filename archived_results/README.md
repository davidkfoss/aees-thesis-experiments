# Archived Results

This directory is the expected location for archived experiment outputs used to regenerate thesis and ICONIP 2026 paper-submission tables and figures.

The full archived result bundle is not tracked in Git because it contains large per-run output files. To regenerate tables and figures from archived results, place or extract the archived bundle so that the directory structure matches:

```text
archived_results/
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

- `cifar_noisy/`
- `noisy_agnews/`
- `nlp_noise_ablation/`
- `compute_overhead/`

## Regeneration workflow

Figure-generation scripts generally require explicit input and output paths. Use folders under `archived_results/` as the input root and write regenerated figures to `reproduced_artifacts/figures/...`.

Example figure command:

```bash
uv run python -m scripts.plots.cifar.plot_cifar_noisy_curves \
  --runs-root archived_results/cifar_noisy \
  --out-dir reproduced_artifacts/figures/cifar \
  --noise-setting sym40
```

Table-generation scripts are configured with default paths. In normal use, they can be run directly with `uv run python ...`. By default, table scripts read from `archived_results/` and write regenerated outputs to `reproduced_artifacts/tables/...`.

Example table command:

```bash
uv run python scripts/tables/make_nlp_noise_ablation_table.py
```

Regenerated outputs should be compared against the curated artifacts in `generated_artifacts/` before replacing any committed files.

## Notes

The archive is intended to support regeneration of aggregate tables and figures. It is not intended to provide model checkpoints or bitwise replay of every training trajectory.

Large generated files, model checkpoints, dataset caches, temporary logs, and unused exploratory runs are excluded where possible.
