# Archived Results

This directory is the expected location for archived experiment outputs used to regenerate thesis and ICONIP 2026 paper-submission tables and figures.

The archived result bundle is available as a GitHub release asset:

<https://github.com/davidkfoss/aees-thesis-experiments/releases/tag/v0.1.0>

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

- `cifar_noisy/`
- `noisy_agnews/`
- `nlp_noise_ablation/`
- `compute_overhead/`
- `checkpointing/`

## Regeneration workflow

Figure-generation scripts require explicit input and output paths (`--runs-root` and `--out-dir`). Most read from the matching subfolder of `archived_results/` and write to `reproduced_artifacts/figures/...`, but the correct `--runs-root` is **not uniform**, and several scripts take additional required flags. Always check a script's `--help` for its expected layout. The non-obvious cases are:

| Figure script                                                    | `--runs-root`                                                                                            | Required extra flags                                                         |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `cifar.plot_cifar_noisy_curves`                                  | `archived_results/cifar_noisy`                                                                           | `--noise-setting {asym20,sym20,sym40}` (run once per setting)                |
| `controllers.plot_arm_selection`                                 | `archived_results/cifar_noisy` _or_ `archived_results/noisy_agnews` (depends on setting)                 | `--setting {agnews_noisy,cifar_sym40,cifar_sym40_cosine,cifar_sym40_paired}` |
| `controllers.plot_reward_per_arm`                                | `archived_results/cifar_noisy` _or_ `archived_results/noisy_agnews` (depends on setting)                 | `--setting {cifar_sym40,agnews_noisy}`                                       |
| `controllers.plot_update_norm_trajectory`                        | `archived_results` (spans `cifar_clean/`, `sst2/`, `noisy_agnews/`)                                      | —                                                                            |
| `nlp_ablation.plot_agnews_noise_ablation_barplot` / `..._curves` | `archived_results` (the whole tree; walked recursively across `noisy_agnews/` and `nlp_noise_ablation/`) | —                                                                            |
| `diagnostics.plot_peak_checkpoint_diagnostics`                   | `archived_results/checkpointing`                                                                         | —                                                                            |

Example figure command:

```bash
uv run python -m scripts.plots.cifar.plot_cifar_noisy_curves \
  --runs-root archived_results/cifar_noisy \
  --out-dir reproduced_artifacts/figures/cifar \
  --noise-setting sym40
```

When a figure script cannot find its inputs, it may exit non-zero and write `<out-dir>/<figure-name>.MISSING.md` listing the exact paths it looked for. Check that file to diagnose a failed run.

Table-generation scripts are configured with default paths. In normal use, they can be run directly with `uv run python ...`. By default, table scripts read from `archived_results/` and write regenerated outputs to `reproduced_artifacts/tables/...`.

Example table command:

```bash
uv run python scripts/tables/make_nlp_noise_ablation_table.py
```

Regenerated outputs should be compared against the curated artifacts in `generated_artifacts/` before replacing any committed files.

## Notes

The archive is intended to support regeneration of aggregate tables and figures. It is not intended to provide model checkpoints or bitwise replay of every training trajectory.

Large generated files, model checkpoints, dataset caches, temporary logs, and unused exploratory runs are excluded where possible.
