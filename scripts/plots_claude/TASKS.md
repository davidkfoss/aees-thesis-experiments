# TASKS.md — Thesis figure generation

This file describes 10 figure-generation scripts to add to the experiments repo. Each script reads archived run data, computes the aggregated quantities described, and writes a `.pdf` + `.png` pair to a fixed target directory. The figures are then consumed by the thesis LaTeX source (separate repo) via `\thesisfigure{figures/<category>/<name>.pdf}`.

Read the "Operating principles" and "Common conventions" sections **first**. They apply to every task.

---

## Operating principles

1. **Do not invent data.** If a required log field is missing or a config doesn't exist, stop and report what's missing. Do not synthesize, interpolate, or estimate values that are not in the archived results.
2. **Mean ± std across seeds.** For every aggregate over runs, aggregate over seeds with mean and population std (or 1-σ band). Do not draw confidence intervals — seed counts are too low (typically n=5).
3. **Reuse existing aggregation code where it exists.** The repo already has table-generation scripts that read the same archived run files for the published tables. Reuse those readers and config loaders. Don't write a parallel data layer.
4. **Placeholder paths in `<angle-brackets>` must be filled in before running.** The user will replace them with concrete paths. Keep them descriptive in the script (named CLI args or top-of-file constants), not hardcoded inline.
5. **Fail loudly on degenerate plots.** If a plot would have only one method, only one seed, or no data after filtering, raise — don't emit a blank PDF.

---

## Common conventions

### Output layout

All figures go to `results/plots/<category>/<name>.{pdf,png}` under the experiments-repo root. Both extensions every time. PDF for thesis inclusion, PNG for previews and PR review.

The eventual thesis-side paths are `figures/<category>/<name>.pdf` — i.e. the thesis repo expects `<category>/<name>.pdf` to match what this repo emits. Do not change `<name>` from what each task specifies; the LaTeX source already references those names.

### Categories (used as `<category>` below)

- `method` — schematic diagrams, no data
- `cifar` — CIFAR-100 results
- `nlp` — Transformer fine-tuning results
- `controllers` — bandit-controller diagnostics
- `diagnostics` — memorization / checkpoint diagnostics
- `compute` — wall-clock overhead

### File / script layout

One script per task at `scripts/plots/<category>/plot_<short_name>.py`. Each script:

- Exposes a `python -m scripts.plots_claude.<category>.plot_<short_name> --runs-root <path> --out-dir <path>` CLI.
- Accepts a `--config-filter` style argument when the figure aggregates a specific subset.
- Writes only the two files specified — does not produce intermediate artifacts in the output directory.
- Is deterministic given the same input data (no random sub-sampling, no `np.random` calls without a fixed seed).

Add a top-level `scripts/plots/__init__.py` and `scripts/plots/_style.py`. Put all matplotlib styling there:

```python
# scripts/plots/_style.py
PALETTE = {
    "flat":              "#666666",
    "cosine":            "#1f77b4",
    "linear":            "#17becf",
    "warmup_linear":     "#1f77b4",
    "aees":              "#d62728",
    "aees_lr":           "#d62728",
    "aees_noise":        "#9467bd",
    "aees_dual":         "#8c564b",
    "cosine_aees":       "#ff7f0e",
    "warmup_linear_aees":"#ff7f0e",
}
```

Style rules:

- `figure.figsize`: 6.5 × 4.0 in for single-panel; 6.5 × 3.5 per panel for multi-panel (use shared y-axis when comparable).
- Fonts: `text.usetex=False`, `font.family="serif"`, `font.size=10`, `axes.labelsize=10`, `legend.fontsize=8`.
- Save with `bbox_inches="tight"`, `dpi=200` for PNG, vector for PDF.
- Always annotate axis units. y-axes for accuracy in **percent** (e.g. `72.1`, not `0.721`).
- Legends placed outside the axes (`bbox_to_anchor=(1.02, 1)`) when the panel is busy; otherwise inside, lower-right or upper-left, whichever covers no data.
- Grid: light, dashed, behind data (`ax.set_axisbelow(True)`).

### Method / variant naming (use in legends and tick labels)

| Internal key         | Display label          |
| -------------------- | ---------------------- |
| `cosine`             | "Cosine"               |
| `linear`             | "Linear"               |
| `warmup_linear`      | "Warmup-linear"        |
| `aees`               | "AEES"                 |
| `aees_lr`            | "AEES-LR"              |
| `aees_noise`         | "AEES-Noise"           |
| `aees_dual`          | "AEES-Dual"            |
| `cosine_aees`        | "Cosine + AEES"        |
| `warmup_linear_aees` | "Warmup-linear + AEES" |

Use these strings verbatim in legends so the thesis prose and the figures match.

For reference of a result file for cifar and agnews, see the example result files for structure, in the `scripts/plots/` folder.

### Data sources (descriptive placeholders — user will fill in)

Some of the folders containing results in the result folder have subfolders. So it may be reccommended to recursively list out all files in the folder to not depend on exact subfolder convention in a particular result/<task_setting>/ folder

All paths below are relative to the experiments-repo root.

| Setting                       | Path                                                                                               | Layout                                      | Variants present                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Clean CIFAR-100               | `results/cifar_clean/*.json`                                                                       | flat                                        | `cifar_clean_{adamw,aees}_{none,cosine,linear}_seed<N>.json` + `cifar_clean_fixed_lr{0p5,2}_n0_none_seed<N>.json` (fixed-LR ablation). 8 variants × 5 seeds = 40 files.                                                                                                                                                                                                                                                                          |
| Noisy CIFAR-100               | `results/cifar_noisy/cifar100_{asym20,sym20,sym40}_seed<N>/<variant>`                              | nested, **files have no `.json` extension** | Variants per (noise, seed) folder: `adamw_baseline`, `adamw_cosine`, `adamw_aees_ep200_lr05102`, `adamw_cosine_aees_ep200_lr05102`, `adamw_fixed_lr{05,10,20}`, `sgd_baseline_lr01`, `sgd_cosine_lr01`, `sgd_aees_ep200_lr05102_base01`, `sgd_cosine_aees_ep200_lr05102_base01`, `sgd_fixed_lr{05,10,20}_base01`. The `asym20` seeds additionally include `sgd_baseline_lr002` and `sgd_baseline_lr005`.                                         |
| Clean AG News                 | `results/clean_agnews/*.json` + `results/clean_agnews/seed_<N>/*.json`                             | mixed flat / nested                         | Top-level: `agnews_{adamw,aees}_warmup_linear_5ep_*_seed<N>.json` (the warmup-linear baseline and AEES-Dual+WL). Inside `seed_<N>/`: `agnews_{adamw,aees}_none_5ep_*_seed<N>.json` (the flat-AdamW baseline and AEES-Dual no-scheduler).                                                                                                                                                                                                         |
| Noisy AG News                 | `results/noisy_agnews/seed_<N>/*.json`                                                             | nested                                      | 6 variants per seed, all `agnews_noise20_*_seed<N>.json`: `adamw_none`, `adamw_warmup_linear`, `aees_none` (Dual, no scheduler), `aees_warmup_linear` (Dual + WL — the flagship), `aees_lronly_warmup_linear`, `aees_noiseonly_warmup_linear`.                                                                                                                                                                                                   |
| SST-2                         | `results/sst2/*.json` + `results/sst2/seed_<N>/*.json`                                             | mixed flat / nested                         | Top-level: `sst2_adamw_none_seed<N>.json`, `sst2_adamw_warmup_linear_5ep_lr5e5_seed<N>.json`, `sst2_aees_warmup_linear_5ep_small_ep200_trend_seed<N>.json` (Dual + WL), `sst2_main_small_ep200_context_trend_seed<N>.json` (Dual, no scheduler). Inside `seed_<N>/`: `sst2_aees_{lronly,noiseonly}_warmup_linear_5ep_ep200_trend_seed<N>.json` (the axis ablations).                                                                             |
| Checkpoint diagnostics        | `results/checkpointing/cifar100_{asym20,sym40}/cifar100_<noise>_seed<N>_<variant>.json`            | flat per noise level                        | Same schema as `cifar_noisy/`; these are separate runs that produced the peak-checkpoint corrupted-subset numbers in `tables/checkpointing_main.tex`. **They do not contain per-epoch diagnostics** — the corrupted-subset values are computed once at the peak-validation checkpoint and aggregated outside the JSON.                                                                                                                           |
| Compute overhead              | `results/compute_overhead/<task>_<optim>_<variant>_<scheduler>_seed0[_run<k>].json`                | flat                                        | Three timing repeats per (task, optimizer, variant, scheduler): the seed-0 file plus `_run2` and `_run3` suffixes. Used for Task 10. Variant slugs use `baseline`, `aees_lr_only`, `aees_noise_only`, `aees_both`. Tasks: `agnews_clean`, `cifar_clean`, `sst2_clean`. **No CIFAR-noisy or AG-News-noisy timing runs** — the table uses these clean-setting times as the canonical overhead for the matching optimizer/architecture combination. |
| Ablations (already in thesis) | `results/instability_penalty_ablation/seed_<N>/*.json` and `results/trend_context_ablation/*.json` | mixed                                       | Out of scope for the 10 figure tasks; included here for completeness.                                                                                                                                                                                                                                                                                                                                                                            |

---

## Result file schema

Each run is a single JSON file (not JSONL). The example files `scripts/plots/example_cifar_adamw_cosine.json` (CIFAR, 200 epochs, AEES-LR run) and `scripts/plots/example_agnews_adamw_warmup_linear.json` (noisy AG News, 5 epochs, baseline) are canonical for the schema below.

### Top-level keys you'll read

| Key                        | Type           | What it holds                                                                            |
| -------------------------- | -------------- | ---------------------------------------------------------------------------------------- |
| `config`                   | object         | All hyperparameters. Use to identify the variant — see "Run identification" below.       |
| `val_accuracies`           | list[float]    | Per-epoch validation accuracy, length = `total_epochs`. **In [0, 1], not percent.**      |
| `train_accuracies`         | list[float]    | Per-epoch training accuracy. In [0, 1].                                                  |
| `train_losses`             | list[float]    | Per-epoch mean training loss.                                                            |
| `best_val_accuracy`        | float          | Peak validation accuracy. In [0, 1].                                                     |
| `final_val_accuracy`       | float          | Final-epoch validation accuracy. In [0, 1].                                              |
| `total_epochs`             | int            | Training horizon (CIFAR=200, AG News=5, SST-2 varies).                                   |
| `total_steps`              | int            | Total optimizer steps.                                                                   |
| `epoch_wall_clock_seconds` | list[float]    | Per-epoch wall-clock time.                                                               |
| `epoch_tflops`             | list[float]    | Per-epoch model TFLOPs.                                                                  |
| `wall_clock_seconds`       | float          | Whole-run wall-clock.                                                                    |
| `seed`                     | int            | Run seed.                                                                                |
| `task_name`                | string         | "cifar100", "agnews", or "sst2".                                                         |
| `method_name`              | string         | Optimizer / scheduler label ("AdamW", "AdaptiveScheduler", etc.).                        |
| `controller_logs`          | object \| null | **`null` for baseline / non-AEES runs.** Present and populated for AEES runs. See below. |
| `episode_logs`             | object \| null | Same as above — present only for AEES runs.                                              |

### `config` fields that identify the variant

| Field                                                   | Used for                                                                      |
| ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `task_name`                                             | "cifar100", "agnews", "sst2"                                                  |
| `method`                                                | base optimizer key (e.g. "AdamW", "SGD")                                      |
| `lr_scheduler`                                          | "none", "cosine", "linear", "warmup_linear"                                   |
| `lr_candidates`                                         | list[float]. `len > 1` ⇒ adapting LR.                                         |
| `noise_candidates`                                      | list[float]. `len > 1` ⇒ adapting σ.                                          |
| `noise_std`                                             | base σ when noise axis is fixed (ignore unless `len(noise_candidates) == 1`). |
| `label_noise_type`                                      | "symmetric" or "asymmetric" (CIFAR & AG News noisy).                          |
| `label_noise_rate`                                      | 0.2, 0.4, etc.                                                                |
| `episode_length`                                        | episode size in steps.                                                        |
| `structured_control_mode`                               | "independent" for the main results.                                           |
| `fixed_mode_name`                                       | If non-null, the run is a fixed-multiplier ablation (not AEES).               |
| `base_lr`, `epochs`, `batch_size`, `model_name`, `seed` | metadata for sanity checks                                                    |

### `controller_logs` (AEES runs only)

```
controller_logs:
  lr_controller_logs:
    arm_values: list[float]                   # candidate LR multipliers
    axis_name: "lr"
    controller_type: "DiscountedUCBController" | "FixedModeController"
    controller_updates_history: list[int]     # episode indices, 1-based
    effective_counts_history: list[dict]      # per episode: {arm_value_str: discounted_count}
    value_estimates_history: list[dict]       # per episode: {arm_value_str: estimated_mean_reward}
    warmup_counts_history: list[dict]         # per episode: {arm_value_str: warmup_pulls}
  noise_controller_logs:
    ...same shape, with arm_values = candidate σ. Often "FixedModeController" with one arm on CIFAR.
  structured_control_mode: "independent"
  context_mode: "none"
```

Arm-value keys in the per-episode dicts are stringified floats (`"0.5"`, `"1.0"`, `"2.0"`, etc.) — match them against `arm_values`.

### `episode_logs` (AEES runs only)

Aligned-length lists, one entry per completed episode. Length = total episodes (`total_steps / episode_length`, give or take a tail). Keys you'll touch:

| Field                                                 | Type         | What it is                                                                                                                                                                                                 |
| ----------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `episode_rewards`                                     | list[float]  | **Final clipped reward used by the controller** for each episode. Use this for reward plots (Task 7).                                                                                                      |
| `reward_base`                                         | list[float]  | Unclipped log-EMA improvement (before penalty/clip). Use only if you want to see saturation.                                                                                                               |
| `reward_final_clipped`                                | list[float]  | Same as `episode_rewards`.                                                                                                                                                                                 |
| `reward_instability`, `reward_penalty`                | list[float]  | Inactive in the main runs (λ_inst=0); ignore unless you're plotting the penalty ablation.                                                                                                                  |
| `selected_lr_values`                                  | list[float]  | LR multiplier selected for each episode.                                                                                                                                                                   |
| `selected_noise_values`                               | list[float]  | σ selected for each episode.                                                                                                                                                                               |
| `selected_lr_indices`, `selected_noise_indices`       | list[int]    | Same selections as indices into `arm_values`.                                                                                                                                                              |
| `selected_candidate_names`, `selected_combined_names` | list[string] | Human-readable arm labels ("lr1_n0", etc.). Useful for sanity checks.                                                                                                                                      |
| `episode_start_steps`, `episode_end_steps`            | list[int]    | Optimizer-step range for each episode. Use to map episodes → epochs (via `total_steps / total_epochs` ≈ steps per epoch).                                                                                  |
| `episode_ema_loss_starts`, `episode_ema_loss_ends`    | list[float]  | The EMA-loss endpoints the reward was computed from.                                                                                                                                                       |
| `episode_start_losses`, `episode_end_losses`          | list[float]  | Raw losses at the same boundaries.                                                                                                                                                                         |
| `mean_update_norms`                                   | list[float]  | **Mean L2 norm of the parameter update applied during each episode.** This is post-optimizer (after AdamW's per-coordinate rescaling), so it's a proxy for "step size" rather than raw `‖g‖`. See Task 11. |

### Run identification (decision tree, given a single JSON `r`)

1. If `r["controller_logs"] is None`: this is a baseline / non-AEES run.
   - If `r["config"]["lr_scheduler"] == "none"`: **flat** baseline.
   - Else: **scheduler-only** baseline (`cosine`, `linear`, `warmup_linear`).
   - If `r["config"]["fixed_mode_name"]` is non-null: **fixed-multiplier ablation**, used only in the appendix ablation plots (not in Tier 1 figures).
2. Else (controller_logs present):
   - LR active iff `len(r["config"]["lr_candidates"]) > 1`.
   - Noise active iff `len(r["config"]["noise_candidates"]) > 1`.
   - LR-only ⇒ **AEES-LR**, Noise-only ⇒ **AEES-Noise**, both ⇒ **AEES-Dual**.
   - Scheduler combination is taken from `r["config"]["lr_scheduler"]` — if it's `"none"` the run is the plain "AEES" variant, otherwise it's "Scheduler + AEES".

Use this decision tree in `scripts/plots/_common.py` (one shared loader) so every task agrees on naming.

### Path discovery

Walk each `results/<setting>/` recursively and parse anything that looks like JSON. **Do not filter on the `.json` extension** — files under `results/cifar_noisy/cifar100_*_seed<N>/` are extensionless. A safe pattern:

```python
for p in pathlib.Path(root).rglob("*"):
    if not p.is_file():
        continue
    if p.suffix == ".log":
        continue
    try:
        with open(p) as f:
            r = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        continue
    yield p, r
```

Don't depend on a fixed depth — `clean_agnews`/`sst2` mix flat files with `seed_<n>/` subfolders, and `cifar_noisy`/`noisy_agnews` always nest under `cifar100_<noise>_seed<n>/` or `seed_<n>/`. The filename usually encodes the variant (e.g. `agnews_noise20_adamw_warmup_linear_5ep_lr5e5_seed2.json`, or just `adamw_aees_ep200_lr05102` for CIFAR-noisy), but rely on the `config` block for ground truth — not filenames.

### Important quirks

- **Accuracies are in [0, 1] everywhere** in the JSON. Multiply by 100 when rendering (the table conventions are percent).
- **AG News is only 5 epochs, SST-2 also short.** A "validation trajectory" plot is therefore 5 points. Render with line + markers (not a smooth curve); don't try to draw shaded bands narrower than the marker. CIFAR is 200 epochs and renders normally.
- **Episodes vs. epochs.** Episode indices in `controller_logs` and `episode_logs` align via `episode_end_steps`. To convert episode→epoch, use `epoch = episode_end_steps[i] * total_epochs / total_steps` and round.
- **Baselines have no controller/episode logs.** Trajectory and overhead plots that include baselines must branch on `controller_logs is None`.
- **`epoch_wall_clock_seconds[0]` includes JIT/setup time** for many runs; the published table excluded the first epoch. Match that convention.
- **Numbers in the published thesis tables are means over 5 seeds**, computed by the existing table-generation scripts. Use the same helper if it exists; otherwise reproduce mean ± population std across whatever seed-files exist (`r["seed"]`).

---

## Tasks

### Task 1 — AEES method overview diagram

**Output:** `results/plots/method/aees_overview.{pdf,png}`
**Script:** `scripts/plots/method/plot_aees_overview.py`
**Priority:** P1 (referenced in `chapters/03_method.tex` via `\thesisfigure`).
**Input data:** None — pure schematic.

**Spec.** A block diagram with three labeled regions:

1. **Training loop** (left): boxes for "Mini-batch", "Forward / backward", "Base optimizer (AdamW or SGD+M)", "Parameter update". Arrow flow top to bottom.
2. **Episode controller** (right): "Episodic Bandit Controller" per active axis (show two stacked bandit boxes labeled "LR-multiplier controller" and "Gradient-noise controller"). Inputs to each: selected arm → episode config; outputs from each: arm reward (from training loss EMA).
3. **Bridge** (center): "Selected arm values" flow from controllers into the training loop ("Apply learning-rate multiplier", "Apply gradient noise"); after each episode of $E$ training steps, "Compute reward $r_e$ from EMA-loss trace" flows from training back to controllers.

Implementation options, in order of preference: TikZ via `matplotlib` + `matplotlib.patches` and `FancyArrowPatch`, or as a `tikzpicture` saved by `tikzplotlib` if it's already in use. Avoid heavy diagramming dependencies (no `graphviz`, no `pyplot.text` walls).

**Acceptance:**

- Two-tone, readable at half-textwidth.
- All eight labels exactly match: `Mini-batch`, `Forward / backward`, `Base optimizer`, `Parameter update`, `Episodic Bandit Controller (LR-multiplier)`, `Episodic Bandit Controller (Gradient noise)`, `Selected arm values`, `Reward $r_e$ from EMA loss`.
- No legend (it's a diagram). No axes.

---

### Task 2 — Noisy CIFAR-100 (40% symmetric) validation trajectories

**Output:** `results/plots/cifar/cifar_symmetric40_curves.{pdf,png}`
**Script:** `scripts/plots/cifar/plot_cifar_symmetric40_curves.py`
**Priority:** P1 (already referenced from `sections/results/cifar_noisy.tex:36`).
**Input data:** `results/cifar_noisy/cifar100_sym40_seed<0-4>/<variant>` (extensionless). Read `val_accuracies` from each. The full filename list per seed is in the Data sources table above.

**Runs to load:**

| Panel | Optimizer | Variant filenames                                                                                               |
| ----- | --------- | --------------------------------------------------------------------------------------------------------------- |
| Left  | AdamW     | `adamw_baseline`, `adamw_cosine`, `adamw_aees_ep200_lr05102`, `adamw_cosine_aees_ep200_lr05102`                 |
| Right | SGD+M     | `sgd_baseline_lr01`, `sgd_cosine_lr01`, `sgd_aees_ep200_lr05102_base01`, `sgd_cosine_aees_ep200_lr05102_base01` |

All four methods × 5 seeds × full epoch budget under **40% symmetric label noise**.

**Spec.** Two panels side-by-side, shared y-axis.

- x: epoch (1-indexed). y: validation accuracy (%).
- Each method: solid line for mean across seeds, shaded band for ±1 std.
- Mark **peak validation accuracy** with a hollow circle at `(peak_epoch_mean, peak_acc_mean)`.
- Mark **final validation accuracy** with a filled square at `(T, final_acc_mean)`.
- Title each panel with the optimizer name. Single legend below both panels.
- y-axis range: auto, but clamp lower bound at 30% so the late-stage drops are visible without compressing the upper region.

**Acceptance:**

- AdamW panel: AEES line peaks visibly higher than flat then drops to ~40% at final epoch; cosine variants stay flatter.
- SGD+M panel: same qualitative shape.
- Peak/final markers visually distinguishable.

**Gotcha.** If runs were logged with epoch indices starting at 0 elsewhere, align to 1-indexed here to match the table conventions.

---

### Task 3 — Noisy AG News validation trajectories (flagship)

**Output:** `results/plots/nlp/agnews_noisy_curves.{pdf,png}`
**Script:** `scripts/plots/nlp/plot_agnews_noisy_curves.py`
**Priority:** P1 (already referenced from `sections/results/nlp_noisy_agnews.tex:16`).
**Input data:** `noisy_agnews/seed_<1-4>/` under the 20% symmetric noisy-AG-News configuration. (Tip: All are 20% symmetric noise in this experiment result.)

**Runs to load:** `adamw`, `adamw_warmup_linear`, `aees_none`, `aees_noiseonly`, `aees_warmup_linear` (= dual + warmup-linear).

**Spec.** Single panel.

- x: epoch. y: validation accuracy (%).
- Mean ± std band per method, 5 methods total.
- Peak and final markers as in Task 2.
- y-range chosen so the **late-training divergence between baselines (which drop to ~90.8%) and AEES variants (which hold at ~93.5%) is clearly visible**. This is the flagship result; the visual must show it.
- Legend inside the panel, lower-right.

**Acceptance:**

- The `adamw` and `adamw_warmup_linear` baselines should drop visibly between peak and final.
- The `aees_warmup_linear` (= AEES-Dual + WL) line should be flattest (drop ≈ 0.05 pp).
- Caption-ready: the visible gap at the final epoch should be ~2.7 pp.

**Gotcha — short trajectories.** AG News runs are only 5 epochs (`total_epochs = 5`). Use line + circle markers, not smooth curves; do not draw the ±std band thinner than the marker; turn on minor x-tick labels at every epoch.

---

### Task 4 — Clean CIFAR-100 peak vs. final bar plot

**Output:** `results/plots/cifar/clean_cifar_barplot.{pdf,png}`
**Script:** `scripts/plots/cifar/plot_clean_cifar_barplot.py`
**Priority:** P2 (already referenced from `sections/results/cifar_clean.tex:8`).
**Input data:** `results/cifar_clean/` under the **clean** CIFAR setting, AdamW only.

**Runs to load:** `adamw`, `adamw_cosine`, `adamw_linear`, `aees`, `aees_cosine`, `aees_linear`. Six configurations total.

**Spec.** Grouped bar chart.

- x: method (six groups, ordered as listed above).
- Each group: two bars side-by-side — **solid** bar for peak, **hatched (`////`)** bar for final.
- Bar heights = mean across seeds; thin black caps = ±1 std.
- y-axis: validation accuracy (%), clamped to [68, 74].
- Annotate each bar's mean above the cap to 1 decimal place.
- Single legend ("Peak", "Final") top-right.

**Acceptance:**

- AEES-alone bar pair shows visibly larger gap (peak much higher than final) than the scheduled variants.
- Cosine and Linear (and their AEES combinations) show tightly matched peak/final pairs.

---

### Task 5 — SST-2 validation trajectories

**Output:** `results/plots/nlp/sst2_curves.{pdf,png}`
**Script:** `scripts/plots/nlp/plot_sst2_curves.py`
**Priority:** P2 (referenced from `sections/results/nlp_sst2.tex:16`).
**Input data:** `results/sst2/`.

**Runs to load:** `adamw_none`, `adamw_warmup_linear`, `main_small (is aees)`, `aees_warmup_linear`.

**Spec.** Single panel, same template as Task 3.

- The result magnitude is smaller; the figure's job is to show that the _qualitative_ pattern (modest gains, noise-axis dominance, smaller-but-real best-to-final drop in baselines) reproduces.
- y-range: auto, but ensure differences of ~0.5 pp are visually legible (don't let the y-axis span more than ~4 pp around the data).

**Acceptance:**

- AEES variants peak slightly above warmup-linear and remain stable; flat baseline drops visibly.

**Gotcha — short trajectories.** SST-2 runs are also short (a handful of epochs). Same rendering note as Task 3: markers + line, no over-thin std bands.

---

### Task 6 — Selected-arm trajectory over training

**Output:** Two files for two settings:

- `results/plots/controllers/arm_selection_cifar_sym40.{pdf,png}` — LR-multiplier axis on noisy CIFAR 40% sym (AdamW).
- `results/plots/controllers/arm_selection_agnews_noisy.{pdf,png}` — σ axis on noisy AG News (AEES-Dual + warmup-linear).

**Script:** `scripts/plots/controllers/plot_arm_selection.py` (one script, `--setting` flag picks which output).
**Priority:** P2 (would populate the currently-stub `appendix_controller_traces.tex`).
**Input data:**

- CIFAR setting: `results/cifar_noisy/cifar100_sym40_seed<0-4>/adamw_aees_ep200_lr05102` (extensionless). Read `episode_logs.selected_lr_values` (length = number of episodes).
- AG News setting: `results/noisy_agnews/seed_<0-4>/agnews_noise20_aees_warmup_linear_5ep_coarse_highnoise_ep200_seed<N>.json` (Dual + WL). Read `episode_logs.selected_noise_values`.

Both settings: candidate arm values come from `controller_logs.lr_controller_logs.arm_values` (or `noise_controller_logs.arm_values`); use these as the canonical ordered axis. **Skip any baseline file** where `controller_logs is None`.

**Spec.** Stacked area chart per setting.

- x: episode index (1-indexed). y: proportion of seeds choosing each arm at that episode, summing to 1.0.
- One stacked band per candidate arm value. Order arms low-to-high (e.g. LR-multiplier: 0.5, 1.0, 2.0 from bottom to top; σ: 0.0, 0.005, 0.01).
- Legend labels: `m=0.5`, `m=1.0`, `m=2.0` for LR-axis; `σ=0`, `σ=0.005`, `σ=0.01` for noise-axis.
- Use the `PALETTE` constants for `aees_lr` / `aees_noise` as the _base_ color and lighten/darken per arm.
- Add a secondary x-axis (top) with epoch index — convert via `episode_end_steps[i] * total_epochs / total_steps` from one representative seed.

**Acceptance:**

- Bands should not look uniform across training — that would mean the controller never settled on anything. If they _do_ look uniform, that's the actual result, but flag it back so the prose interpretation in `appendix_controller_traces.tex` reflects reality.
- For the σ axis on noisy AG News: if a clear early-high-σ / late-low-σ pattern (or its reverse) appears, callout that fact in the script's printed summary — it's a key signal for the axis-dependence discussion.

**Gotcha.** Different seeds may have slightly different total episode counts (last partial episode). Truncate to the shortest seed's episode count before stacking.

---

### Task 7 — Per-arm reward distribution

**Output:** Two files:

- `results/plots/controllers/reward_per_arm_cifar_sym40.{pdf,png}` — LR axis, noisy CIFAR 40% sym AdamW.
- `results/plots/controllers/reward_per_arm_agnews_noisy.{pdf,png}` — both axes, noisy AG News (AEES-Dual + warmup-linear).

**Script:** `scripts/plots/controllers/plot_reward_per_arm.py` (`--setting` flag).
**Priority:** P2.
**Input data:** Same source runs as Task 6. For each completed episode, pair `episode_logs.episode_rewards[i]` (clipped log-EMA reward used by the controller) with `episode_logs.selected_lr_values[i]` and/or `episode_logs.selected_noise_values[i]`. Pool across all seeds for that setting. Group by selected arm value.

**Spec.** Violin plot.

- x: arm value (one violin per arm). y: clipped log-EMA episode reward, pooled across all completed episodes × all seeds.
- For the noisy AG News figure, side-by-side panels (LR axis | σ axis), shared y-axis.
- Overlay a small horizontal tick for the median.
- Reference line at $r_e = 0$ (the "no improvement" boundary) — dashed, light gray.

**Acceptance:**

- The shape of the violin makes it visually obvious which arms tend to produce positive vs. negative rewards.
- On noisy AG News, the σ axis should show a clearly preferred non-zero value if the headline result holds.

---

### Task 8 — Selected σ vs. epoch validation accuracy (noisy AG News)

**Output:** `results/plots/controllers/sigma_vs_val_acc_agnews_noisy.{pdf,png}`
**Script:** `scripts/plots/controllers/plot_sigma_vs_val_acc.py`
**Priority:** P3 (interpretive, supports the axis-dependence discussion).
**Input data:** Noisy AG News AEES-Noise and AEES-Dual runs (any variant with `len(config.noise_candidates) > 1`). For each (seed, epoch) pair, compute the mean of `episode_logs.selected_noise_values[i]` over the episodes whose `episode_end_steps[i]` fall within that epoch's step range (epoch step range = `[k * total_steps/total_epochs, (k+1) * total_steps/total_epochs)`), and pair with `val_accuracies[epoch]`.

**Spec.** Single panel.

- 2D histogram (`plt.hexbin` with `gridsize=20`) of (mean σ in epoch, validation accuracy at epoch end).
- Color: viridis, count-based.
- x: mean σ in epoch (continuous, since σ is averaged over episodes). y: validation accuracy (%).
- Overlay a thin line for the per-σ-bin mean validation accuracy.

**Gotcha.** With only 5 AG News epochs and ~3 σ candidates, the hexbin will be very sparse. If the scatter is too thin to read, fall back to a strip plot (`sns.stripplot`) with one column per σ candidate and val-acc on y. Print the per-σ aggregate to the summary file regardless.

**Acceptance:**

- If a monotone or single-peak relationship between σ and validation accuracy appears, this becomes a directly-citable visual for the axis-dependence section.
- If the relationship is flat, that's important to know too — flag it.

---

### Task 9 — Corrupted-subset accuracy curves under asymmetric CIFAR noise — **ON HOLD**

**Status:** ON HOLD. Per-epoch corrupted-subset accuracy is not in the archived runs. The `results/checkpointing/cifar100_asym20/` and `results/checkpointing/cifar100_sym40/` files use the standard result schema (`val_accuracies`, `controller_logs`, `episode_logs`) — no per-epoch `corr_vs_noisy` / `corr_vs_clean` / `clean_vs_clean` arrays. The numbers in `tables/checkpointing_main.tex` were computed at the peak-validation checkpoint only. The curve form therefore requires a fresh run of the asymmetric-CIFAR configurations with diagnostics logged every epoch (or every k epochs). Do not start this task until the user explicitly re-enables it.

**Configs to load:** `cosine`, `aees`, `cosine_aees`. Three lines per metric.

**Spec.** Three panels stacked vertically, shared x-axis.

- Top panel: `Corr. vs noisy` (corrupted-subset accuracy against the _imposed_ corrupted labels). Each method: mean across seeds, ±std band.
- Middle panel: `Corr. vs clean` (corrupted-subset accuracy against the _original clean_ labels — the key memorization-vs-generalization signal).
- Bottom panel: `Clean vs clean` (uncorrupted-subset training accuracy — sanity check).
- x: epoch. y: accuracy (%) in [0, 100].
- Mark each method's peak-validation epoch with a vertical dashed line in its color.

**Acceptance:**

- Cosine's `Corr. vs noisy` should rise to ~100% by the end of training; `Corr. vs clean` should drop to ~0%.
- AEES and Cosine+AEES should reach their peak-validation epochs much earlier (vertical dashed lines on the left side of the plot).
- At those early peak epochs, `Corr. vs clean` for AEES variants should be visibly higher than for cosine at the same epoch.

**Gotcha.** Diagnostics may have been logged only every k epochs. If so, plot what's there (step / scatter-with-line is fine) and note the resolution in the script's summary print.

---

### Task 10 — Wall-clock overhead bar chart

**Output:** `results/plots/compute/wallclock_overhead.{pdf,png}`
**Script:** `scripts/plots/compute/plot_wallclock_overhead.py`
**Priority:** P3 (visual companion to `tables/compute_overhead_detailed_table.tex`).
**Input data:** `results/compute_overhead/*.json`. Each file is one of three timing repeats per (task, optimizer, variant, scheduler) cell — base file plus `_run2` and `_run3` siblings (use the existing helper if it groups them; otherwise group on the filename stem with the trailing `_run<k>` stripped).

For each cell:

- Compute mean epoch time as `mean(r["runtime_metrics"]["epoch_wall_clock_seconds"][1:])` (drop epoch 0 to exclude JIT/setup); fall back to `r["epoch_wall_clock_seconds"][1:]` if `runtime_metrics` is missing. `runtime_metrics["mean_epoch_wall_clock_seconds"]` is also pre-computed and usable.
- Average across the timing repeats for that cell.
- Compute overhead (%) vs. the matching baseline cell within the same (task, optimizer): the baseline is the row where the variant key is `baseline` and scheduler key matches. For example, `cifar_clean_adamw_aees_lr_only_cosine` is compared against `cifar_clean_adamw_baseline_cosine` for the "with-scheduler" overhead, or against `cifar_clean_adamw_baseline_none` for the "no-scheduler" overhead — the published table uses the no-scheduler-baseline anchoring, so do that.

Cells available (from filename slugs): tasks `{cifar_clean, agnews_clean, sst2_clean}` × optimizers `{adamw}` for NLP, plus `{adamw, sgd}` for CIFAR; variants `{baseline, aees_lr_only, aees_noise_only, aees_both}`; schedulers `{none, cosine, warmup_linear}`. **Note:** there are no CIFAR-noisy or AG-News-noisy timing files; the chart should use the matching clean-setting cell (this matches the published table).

**Spec.** Grouped horizontal bar chart.

- y-axis groups: four (CIFAR-SGD+M, CIFAR-AdamW, SST-2-AdamW, AG-News-AdamW).
- Within each group: bars for the scheduler baseline (`cosine` or `warmup_linear`), `aees_lr`, `aees_noise`, `aees_dual` (skip `aees_noise` and `aees_dual` for CIFAR results to match the published configuration — flag and confirm).
- x: overhead (%) over no-scheduler baseline.
- Annotate each bar with its value to 1 dp.
- Reference line at 0% (solid black).

**Acceptance:**

- CIFAR-SGD+M bars should sit near 0%.
- NLP bars should clearly show the +19% to +39% range.
- The asymmetry between CIFAR-SGD+M and NLP is visually obvious.

---

### Task 11 — Mean update-norm trajectory (optional, hypothesis-A diagnostic)

**Output:** `results/plots/controllers/update_norm_trajectory.{pdf,png}`
**Script:** `scripts/plots/controllers/plot_update_norm_trajectory.py`
**Priority:** P3 (interpretive only — supports the three hypotheses in `sections/discussion/axis_dependence.tex` about why σ behaves differently in CIFAR vs. Transformer fine-tuning).
**Input data:** `episode_logs.mean_update_norms` (per-episode mean L2 norm of the parameter update _after_ optimizer rescaling) for representative AEES runs in three regimes:

1. CIFAR-100 / AdamW / AEES-LR / clean — `results/cifar_clean/cifar_clean_aees_none_seed<0-4>.json` (5 seeds). Anchors the "from-scratch ResNet" magnitude.
2. SST-2 / AdamW / AEES-Dual + warmup-linear — `results/sst2/sst2_aees_warmup_linear_5ep_small_ep200_trend_seed<0-4>.json`. Pretrained Transformer.
3. AG News (noisy 20% sym) / AdamW / AEES-Dual + warmup-linear — `results/noisy_agnews/seed_<0-4>/agnews_noise20_aees_warmup_linear_5ep_coarse_highnoise_ep200_seed<N>.json`. Pretrained Transformer under noise.

**Spec.** Three small subplots stacked vertically, shared x-axis (training step from `episode_end_steps`), independent y-axes (log scale).

- One line per seed (light, semi-transparent), bold mean curve overlay.
- Annotate the median update norm in late training (last 20% of episodes) as a horizontal reference line, with the value labeled.
- Caption notes that this is the _update_ norm after AdamW rescaling, not the raw gradient norm.

**Acceptance / what to look for.**

- If the late-training update-norm in CIFAR-AdamW is several orders of magnitude larger than in DistilBERT fine-tuning, that supports hypothesis A1 in `axis_dependence.tex` (the same nominal σ corresponds to very different relative perturbations across regimes).
- If the magnitudes are within a small constant factor, hypothesis A1 is weakened and you should flag that back so the discussion paragraph is reworded accordingly.

**Gotcha.** This is a proxy, not a direct gradient-norm comparison. AdamW's coordinate-wise rescaling means the update norm is bounded near `‖m · η · v⁻¹/²‖` rather than `η · ‖g‖`. The qualitative cross-regime ratio is still informative; the _absolute_ numbers are not directly the σ-to-signal ratio I wrote into the discussion.

---

## After running

For each task, the script should print (a) the resolved input paths it used, (b) the seed counts found per config, (c) one-line summary statistics of the output (e.g. peak/final per method for trajectory plots). Save this to `results/plots/<category>/<name>.summary.txt` alongside the PDF/PNG.

If any task can't be completed (missing data, missing fields, missing configs), **don't emit a partial figure**. Write a stub `results/plots/<category>/<name>.MISSING.md` describing what was missing and stop. The thesis-side `\thesisfigure{}` macro will render its placeholder until the file exists.

## Style consistency check

Before submitting, run all 10 figures and visually confirm:

- Same font family and size across PDFs.
- Same legend labels and color assignments where the same method appears in multiple figures (e.g. `aees_noise` is the same shade of purple in Task 3, Task 6, Task 7, Task 8).
- Same axis-label conventions ("Validation accuracy (%)", "Epoch", "Episode", "Overhead (%)").

## What to flag back

When this is done, flag any of the following in the PR description:

- Configs that the registry didn't recognize (mismatch between thesis tables and run-data layout).
- Diagnostics fields that are not logged for every relevant run (likely for Task 9 — partial coverage is normal, but the user needs to know which epochs/seeds were available).
- Any case where the visual contradicts what the published table says — that's a real signal, not a figure bug; do **not** quietly fix it.
