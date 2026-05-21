# Reproduced Artifacts

This directory is intended for locally regenerated tables, summary CSV files, and figure files.

It is separate from `generated_artifacts/`, which contains the curated generated outputs used in the thesis and the ICONIP 2026 paper submission. Use this directory when rerunning table or figure generation scripts so that the committed artifacts are not overwritten accidentally.

## Typical workflow

1. Place or extract the archived result bundle under `archived_results/`.
2. Regenerate figures by running the figure scripts with explicit `--runs-root` and `--out-dir` arguments.
3. Regenerate tables by running the table-generation scripts directly with `uv run python ...`; their defaults read from `archived_results/` and write to `reproduced_artifacts/tables/...`.
4. Compare regenerated files against the committed files in `generated_artifacts/`.
5. Only copy files into `generated_artifacts/` if you intentionally want to update the curated artifacts.

Example figure command:

```bash
uv run python -m scripts.plots.cifar.plot_cifar_noisy_curves \
  --runs-root archived_results/cifar_noisy \
  --out-dir reproduced_artifacts/figures/cifar \
  --noise-setting sym40
```

Example table command:

```bash
uv run python scripts/tables/make_nlp_noise_ablation_table.py
```

## Directory convention

Suggested layout:

```text
reproduced_artifacts/
  figures/
    cifar/
    cifar_ablation/
    controllers/
    diagnostics/
    method/
    nlp/
    nlp_ablation/
  tables/
    ablations/
    checkpointing/
    cifar_ablation/
    cifar_clean/
    cifar_noisy/
    compute_overhead/
    nlp_tables/
```

This mirrors the structure of `generated_artifacts/` so that regenerated outputs can be compared easily. Two scripts deviate from the mirror:

- `make_nlp_noise_ablation_table.py` writes `tables/nlp_noise_ablation_table.tex` at the top of `tables/`; the curated copy of that table lives under `tables/nlp_tables/` (it is also regenerated there by `make_nlp_latex_tables.py`).
- `plot_wallclock_overhead` writes to `figures/compute/`, which has no curated counterpart under `generated_artifacts/figures/`.

Table scripts also emit a few combined files (e.g. `*_tables_all.tex`, `all_nlp_tables.tex`) that are not curated in `generated_artifacts/`; these are expected extras, not regressions.

## Notes

Files in this directory are not canonical. The canonical numerical source is the archived result bundle under `archived_results/`, and the curated generated outputs used in the thesis and ICONIP 2026 paper submission are stored under `generated_artifacts/`.

PDF and PNG files may differ slightly across systems because of plotting backends, fonts, or library versions. For numerical checks, compare generated summary CSV or text files where available.
