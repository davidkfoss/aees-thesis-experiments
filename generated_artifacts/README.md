# Generated Artifacts

This directory contains generated LaTeX tables, summary CSV files, and figure files used in the thesis and the ICONIP 2026 paper submission.

These artifacts are generated from archived experiment results and are included for convenience and inspection. The canonical numerical source is the archived result bundle expected under `archived_results/`; generated artifacts should be regenerated from scripts rather than edited manually.

## Structure

```text
generated_artifacts/
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

## Thesis and conference formatting

Some tables appear in both the thesis and the ICONIP 2026 paper submission with different LaTeX formatting, captions, column labels, or row grouping. These are presentation differences only. When a table reports the same experiment in both documents, the numerical values are generated from the same archived result artifacts.

The thesis versions are generally more detailed and may include ranked emphasis, expanded captions, or broader experimental context. The conference-paper versions are compressed for page limits and may report a scoped subset of the same results.

## Summary CSV files

Some table-generation scripts also emit summary CSV files. These CSVs contain the same aggregate values used to render the LaTeX tables and are kept as machine-readable companion artifacts. Not every table has a CSV companion because not all scripts emit one.

## Regeneration workflow

Figure scripts write to a user-specified output directory and should normally be run with `--out-dir reproduced_artifacts/figures/...`. The required `--runs-root` differs per figure script (see `archived_results/README.md` for the per-script table); check each script's `--help`.

`figures/method/aees_overview.{pdf,png}` is built from TikZ by `scripts/plots/method/build_aees_overview.sh` (compiling `aees_overview.tex`), not by the Python figure scripts.

Table scripts are configured with defaults that read from `archived_results/` and write to `reproduced_artifacts/tables/...`; they can normally be run directly with `uv run python ...`.

Files in this directory are the curated generated artifacts used in the thesis and ICONIP 2026 paper submission. Regenerated outputs should be compared against these files before replacing them.
