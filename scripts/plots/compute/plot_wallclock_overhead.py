"""Task 10 — Wall-clock overhead bar chart.

Grouped horizontal bar chart showing per-epoch wall-clock overhead (%) of each
AEES variant (and the matching scheduler-only baseline) against the
NO-scheduler baseline within the same (task, optimizer). Companion to
``tables/compute_overhead_detailed_table.tex``.

Outputs (on success):
    results/plots/compute/wallclock_overhead.pdf
    results/plots/compute/wallclock_overhead.png
    results/plots/compute/wallclock_overhead.summary.txt

On failure:
    results/plots/compute/wallclock_overhead.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from scripts.plots._common import (
    RunInfo,
    episode_to_epoch,
    epoch_step_range,
    identify,
    load_set,
    walk_runs,
)
from scripts.plots._style import (
    LABEL,
    PALETTE,
    arm_palette,
    make_figure,
    mark_peak_final,
    mean_std_band,
    save_figure,
    short_trajectory_kwargs,
    write_missing,
    write_summary,
)


NAME = "wallclock_overhead"


# ---------------------------------------------------------------------------
# Filename-slug parsing. Compute-overhead JSONs are named:
#   <task_slug>_<optim_slug>_<variant_slug>_<sched_slug>_seed0[ _run<k> ].json
# where task_slug ∈ {cifar_clean, agnews_clean, sst2_clean},
# optim_slug ∈ {adamw, sgd}, variant_slug ∈ {baseline, aees_lr_only,
# aees_noise_only, aees_both}, sched_slug ∈ {none, cosine, warmup_linear}.
# ---------------------------------------------------------------------------

TASK_SLUGS = ("cifar_clean", "agnews_clean", "sst2_clean")
OPTIM_SLUGS = ("adamw", "sgd")
VARIANT_SLUGS = ("baseline", "aees_lr_only", "aees_noise_only", "aees_both")
SCHED_SLUGS = ("none", "cosine", "warmup_linear")

# Group ordering top -> bottom on the y-axis.
GROUPS: list[tuple[str, str, str]] = [
    # (task_slug, optim_slug, display label)
    ("cifar_clean", "sgd",   "CIFAR-100 (SGD+M)"),
    ("cifar_clean", "adamw", "CIFAR-100 (AdamW)"),
    ("sst2_clean",  "adamw", "SST-2 (AdamW)"),
    ("agnews_clean", "adamw", "AG News (AdamW)"),
]

# Per (task, optim), which scheduler is the "scheduler baseline" bar?
SCHEDULER_BASELINE_BY_TASK: dict[str, str] = {
    "cifar_clean":  "cosine",
    "agnews_clean": "warmup_linear",
    "sst2_clean":   "warmup_linear",
}

# Variants drawn within a group, top-to-bottom inside the group:
# (variant_slug or "_sched_baseline" sentinel, palette key, display label).
NLP_BARS: list[tuple[str, str, str]] = [
    ("_sched_baseline", "_sched_color", "_sched_label"),
    ("aees_lr_only",     "aees_lr",     "AEES-LR"),
    ("aees_noise_only",  "aees_noise",  "AEES-Noise"),
    ("aees_both",        "aees_dual",   "AEES-Dual"),
]

# CIFAR drops aees_noise_only and aees_both (matches published config).
CIFAR_BARS: list[tuple[str, str, str]] = [
    ("_sched_baseline", "_sched_color", "_sched_label"),
    ("aees_lr_only",     "aees_lr",     "AEES-LR"),
]


_STEM_RE = re.compile(r"_run\d+$")


def _strip_run_suffix(stem: str) -> str:
    """Map both 'foo_seed0' and 'foo_seed0_run2' -> the seed0 base key."""
    return _STEM_RE.sub("", stem)


def _parse_filename(stem: str) -> tuple[str, str, str, str] | None:
    """Parse '<task>_<optim>_<variant>_<sched>_seed<int>' -> (task, optim, variant, sched).

    Returns None if the stem doesn't match the compute-overhead naming.
    """
    base = _strip_run_suffix(stem)
    # Find a "_seed<int>" anchor at the end.
    m = re.match(r"^(?P<core>.+?)_seed(?P<seed>\d+)$", base)
    if m is None:
        return None
    core = m.group("core")
    task = next((t for t in TASK_SLUGS if core.startswith(t + "_")), None)
    if task is None:
        return None
    remainder = core[len(task) + 1:]
    optim = next(
        (o for o in OPTIM_SLUGS if remainder.startswith(o + "_")), None)
    if optim is None:
        return None
    remainder = remainder[len(optim) + 1:]
    sched = next((s for s in SCHED_SLUGS if remainder.endswith(
        "_" + s) or remainder == s), None)
    if sched is None:
        return None
    if remainder == sched:
        variant = ""
    else:
        variant = remainder[: -(len(sched) + 1)]
    if variant not in VARIANT_SLUGS:
        return None
    return task, optim, variant, sched


def _mean_epoch_seconds(record: dict[str, Any], path: pathlib.Path) -> float:
    """Mean per-epoch wall-clock seconds, dropping epoch 0 to exclude JIT/setup."""
    rm = record.get("runtime_metrics")
    ewcs = None
    if isinstance(rm, dict):
        ewcs = rm.get("epoch_wall_clock_seconds")
    if ewcs is None:
        ewcs = record.get("epoch_wall_clock_seconds")
    if not isinstance(ewcs, list) or len(ewcs) == 0:
        raise ValueError(f"{path}: no epoch_wall_clock_seconds array found")
    if len(ewcs) < 2:
        raise ValueError(
            f"{path}: only {len(ewcs)} epoch(s) recorded; "
            "need >=2 to drop epoch-0 JIT/setup"
        )
    arr = [float(v) for v in ewcs[1:]]
    return float(np.mean(arr))


def _collect_cells(
    runs_root: pathlib.Path,
) -> tuple[dict[tuple[str, str, str, str], list[tuple[pathlib.Path, float]]], list[str]]:
    """Discover all compute_overhead files and group by (task, optim, variant, sched).

    Returns ({cell: [(path, mean_epoch_s), ...]}, list of warnings).
    """
    cells: dict[tuple[str, str, str, str],
                list[tuple[pathlib.Path, float]]] = defaultdict(list)
    warnings: list[str] = []
    for path, record in walk_runs(runs_root):
        parsed = _parse_filename(path.stem)
        if parsed is None:
            warnings.append(f"unrecognized filename, skipped: {path.name}")
            continue
        try:
            mean_s = _mean_epoch_seconds(record, path)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        cells[parsed].append((path, mean_s))
    return cells, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        required=True,
        help="Directory containing compute_overhead/*.json files.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where the PDF/PNG/summary (or MISSING.md) get written.",
    )
    args = parser.parse_args(argv)

    runs_root = pathlib.Path(args.runs_root).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()

    if not runs_root.exists():
        write_missing(out_dir, NAME, f"Runs root does not exist: {runs_root}")
        return 1

    cells, walk_warnings = _collect_cells(runs_root)

    # Decide required cells:
    #   - For each group: no-scheduler baseline (variant=baseline, sched=none).
    #   - Scheduler baseline (variant=baseline, sched=group-specific).
    #   - Variants per group as listed above.
    required: list[tuple[str, str, str, str]] = []
    for task, optim, _label in GROUPS:
        sched_baseline = SCHEDULER_BASELINE_BY_TASK[task]
        required.append((task, optim, "baseline", "none"))
        required.append((task, optim, "baseline", sched_baseline))
        bars = CIFAR_BARS if task == "cifar_clean" else NLP_BARS
        for var_slug, _pal, _disp in bars:
            if var_slug == "_sched_baseline":
                continue
            required.append((task, optim, var_slug, sched_baseline))

    missing_cells: list[tuple[str, str, str, str]] = []
    short_cells: list[tuple[tuple[str, str, str, str], int]] = []
    for cell in required:
        runs = cells.get(cell, [])
        if not runs:
            missing_cells.append(cell)
        elif len(runs) < 3:
            short_cells.append((cell, len(runs)))

    if missing_cells or short_cells:
        lines = [f"Resolved --runs-root: {runs_root}", ""]
        if missing_cells:
            lines.append("Missing required cell(s) (no JSONs matched):")
            for c in missing_cells:
                lines.append(
                    f"  - task={c[0]} optim={c[1]} variant={c[2]} sched={c[3]}")
            lines.append("")
        if short_cells:
            lines.append("Cells with fewer than 3 repeats:")
            for c, n in short_cells:
                lines.append(
                    f"  - task={c[0]} optim={c[1]} variant={c[2]} sched={c[3]} -> {n}/3"
                )
            lines.append("")
        if walk_warnings:
            lines.append("Walk warnings:")
            lines.extend(f"  - {w}" for w in walk_warnings)
        write_missing(out_dir, NAME, "\n".join(lines))
        return 1

    # ---- Aggregate per cell ------------------------------------------------
    # cell_summary[cell] = {"paths": [...], "means": [...], "avg": float}
    cell_summary: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for cell in required:
        runs = sorted(cells[cell], key=lambda x: x[0].name)
        paths = [p for p, _ in runs]
        means = [m for _, m in runs]
        avg = float(np.mean(means))
        cell_summary[cell] = {"paths": paths, "means": means, "avg": avg}

    # ---- Overhead vs. no-scheduler baseline within each (task, optim) -----
    bar_records: list[dict[str, Any]] = []  # in y-axis order
    summary_blocks: list[str] = []

    for task, optim, group_label in GROUPS:
        sched_baseline = SCHEDULER_BASELINE_BY_TASK[task]
        sched_baseline_display = LABEL[sched_baseline]
        no_sched_cell = (task, optim, "baseline", "none")
        baseline_time = cell_summary[no_sched_cell]["avg"]

        bars = CIFAR_BARS if task == "cifar_clean" else NLP_BARS
        for var_slug, pal_key, disp in bars:
            if var_slug == "_sched_baseline":
                cell = (task, optim, "baseline", sched_baseline)
                display = sched_baseline_display
                color = PALETTE[sched_baseline]
            else:
                cell = (task, optim, var_slug, sched_baseline)
                display = disp
                color = PALETTE[pal_key]
            cell_time = cell_summary[cell]["avg"]
            overhead_pct = 100.0 * (cell_time - baseline_time) / baseline_time
            bar_records.append(
                {
                    "group_label": group_label,
                    "task": task,
                    "optim": optim,
                    "cell": cell,
                    "display": display,
                    "color": color,
                    "overhead_pct": overhead_pct,
                    "cell_time": cell_time,
                    "baseline_time": baseline_time,
                }
            )

        # Per-cell summary lines for this group.
        summary_blocks.append(
            f"--- {group_label}  (task={task}, optim={optim}) ---")
        no_sched = cell_summary[no_sched_cell]
        summary_blocks.append(
            f"  no-scheduler baseline (variant=baseline, sched=none):"
        )
        for p, m in zip(no_sched["paths"], no_sched["means"]):
            summary_blocks.append(f"    {p.name}: mean_epoch_s = {m:.4f}")
        summary_blocks.append(
            f"    avg across {len(no_sched['paths'])} repeats = {no_sched['avg']:.4f} s"
        )
        for var_slug, pal_key, disp in bars:
            if var_slug == "_sched_baseline":
                cell = (task, optim, "baseline", sched_baseline)
                label_for_summary = f"scheduler baseline ({sched_baseline})"
            else:
                cell = (task, optim, var_slug, sched_baseline)
                label_for_summary = f"{disp} ({var_slug}, sched={sched_baseline})"
            info = cell_summary[cell]
            overhead_pct = 100.0 * \
                (info["avg"] - baseline_time) / baseline_time
            summary_blocks.append(f"  {label_for_summary}:")
            for p, m in zip(info["paths"], info["means"]):
                summary_blocks.append(f"    {p.name}: mean_epoch_s = {m:.4f}")
            summary_blocks.append(
                f"    avg across {len(info['paths'])} repeats = {info['avg']:.4f} s"
                f"   ->  overhead vs. no-sched baseline = {overhead_pct:+.1f}%"
            )
        summary_blocks.append("")

    # ---- Acceptance sanity check ------------------------------------------
    # CIFAR-SGD+M bars should sit near 0%, NLP bars in +19..+39% range.
    cifar_sgd_overheads = [
        r["overhead_pct"]
        for r in bar_records
        if r["task"] == "cifar_clean" and r["optim"] == "sgd"
    ]
    nlp_overheads = [
        r["overhead_pct"]
        for r in bar_records
        if r["task"] in ("sst2_clean", "agnews_clean")
    ]
    sanity_msgs: list[str] = []
    if cifar_sgd_overheads:
        max_abs = max(abs(v) for v in cifar_sgd_overheads)
        sanity_msgs.append(
            f"CIFAR-SGD+M overheads: "
            f"{[round(v, 2) for v in cifar_sgd_overheads]} "
            f"(max |x| = {max_abs:.2f}%)"
        )
    if nlp_overheads:
        sanity_msgs.append(
            f"NLP overheads: "
            f"{[round(v, 2) for v in nlp_overheads]} "
            f"(min={min(nlp_overheads):.2f}%, max={max(nlp_overheads):.2f}%)"
        )

    # ---- Figure ------------------------------------------------------------
    # 12 bars total + 3 group gaps: ~12 axis units. Custom figsize gives the
    # vertical room for one y-tick (variant label) per bar plus group labels
    # on the left margin without crowding the value annotations.
    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    bar_height = 0.7
    group_gap = 0.9  # extra blank space between groups
    inner_gap = 0.10  # tight stacking within a group

    # Walk groups top-to-bottom: highest y-value at the top in matplotlib's
    # default "y goes up" axes, so we assign decreasing y to subsequent items.
    y_per_bar: list[float] = []
    group_y_centers: list[float] = []

    y_cursor = 0.0
    for task, optim, group_label in GROUPS:
        bars = CIFAR_BARS if task == "cifar_clean" else NLP_BARS
        group_ys: list[float] = []
        for _ in bars:
            y_per_bar.append(y_cursor)
            group_ys.append(y_cursor)
            y_cursor -= bar_height + inner_gap
        group_y_centers.append(float(np.mean(group_ys)))
        y_cursor -= group_gap  # extra gap before next group

    # Draw bars.
    for rec, y in zip(bar_records, y_per_bar):
        ax.barh(
            y,
            rec["overhead_pct"],
            height=bar_height,
            color=rec["color"],
            edgecolor="black",
            linewidth=0.6,
        )
        # Value annotation just past the bar tip. Anchor to the right of
        # max(x, 0) so near-zero negative bars (CIFAR-SGD+M) don't collide
        # with the y-tick variant labels on the left of the axis.
        x = rec["overhead_pct"]
        text = f"{x:+.1f}%"
        text_x = max(x, 0.0) + 0.4
        ax.text(text_x, y, text, ha="left", va="center", fontsize=8.0)

    # Reference line at 0%.
    ax.axvline(0.0, color="black", linewidth=0.8)

    # y-axis: one tick per bar with the variant display name. Group labels
    # (task + optimizer) live on the far-left margin via a blended-transform
    # text() call so they don't compete with the per-bar value annotations.
    ax.set_yticks(y_per_bar)
    ax.set_yticklabels([rec["display"] for rec in bar_records], fontsize=8.5)
    ax.invert_yaxis()  # top-to-bottom in spec ordering

    trans = ax.get_yaxis_transform()
    for (_t, _o, group_label), gy in zip(GROUPS, group_y_centers):
        ax.text(
            -0.36,  # axes-x outside the plot area (left of the tick labels)
            gy,
            group_label,
            transform=trans,
            ha="left",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color="#222222",
        )

    # x-range with headroom for value annotations.
    min_overhead = min(r["overhead_pct"] for r in bar_records)
    max_overhead = max(r["overhead_pct"] for r in bar_records)
    xmin = min(0.0, min_overhead) - 2.0
    xmax = max_overhead + 6.0
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel(
        "Per-epoch wall-clock overhead vs. no-scheduler baseline (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)

    # Legend: bar-type swatches.
    legend_handles = [
        Patch(facecolor=PALETTE["cosine"], edgecolor="black", linewidth=0.6,
              label="Cosine (CIFAR sched baseline)"),
        Patch(facecolor=PALETTE["warmup_linear"], edgecolor="black", linewidth=0.6,
              label="Warmup-linear (NLP sched baseline)"),
        Patch(facecolor=PALETTE["aees_lr"], edgecolor="black", linewidth=0.6,
              label="AEES-LR"),
        Patch(facecolor=PALETTE["aees_noise"], edgecolor="black", linewidth=0.6,
              label="AEES-Noise"),
        Patch(facecolor=PALETTE["aees_dual"], edgecolor="black", linewidth=0.6,
              label="AEES-Dual"),
    ]
    # Cosine and warmup_linear share the same hex; the first two patches will
    # collapse visually. Keep only one combined entry for clarity.
    if PALETTE["cosine"] == PALETTE["warmup_linear"]:
        legend_handles = [
            Patch(
                facecolor=PALETTE["cosine"],
                edgecolor="black",
                linewidth=0.6,
                label="Cosine / Warmup-linear",
            ),
            Patch(
                facecolor=PALETTE["aees_lr"], edgecolor="black", linewidth=0.6,
                label="AEES-LR",
            ),
            Patch(
                facecolor=PALETTE["aees_noise"], edgecolor="black", linewidth=0.6,
                label="AEES-Noise",
            ),
            Patch(
                facecolor=PALETTE["aees_dual"], edgecolor="black", linewidth=0.6,
                label="AEES-Dual",
            ),
        ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    # Leave room on the left for the bold group labels (transform x = -0.36).
    fig.subplots_adjust(left=0.30, right=0.97, top=0.97, bottom=0.08)

    pdf, png = save_figure(fig, out_dir, NAME)

    # ---- Summary file -----------------------------------------------------
    lines: list[str] = []
    lines.append(f"Resolved --runs-root: {runs_root}")
    lines.append(f"Output PDF: {pdf}")
    lines.append(f"Output PNG: {png}")
    lines.append("")
    lines.append("Method: per-cell mean epoch wall-clock seconds = mean of "
                 "runtime_metrics.epoch_wall_clock_seconds[1:] "
                 "(drop epoch 0 to exclude JIT/setup), averaged over 3 timing repeats.")
    lines.append("Overhead (%) = 100 * (cell_time - no_sched_baseline_time) / "
                 "no_sched_baseline_time, within the same (task, optimizer).")
    lines.append("")
    lines.append("NOTE: CIFAR cells skip aees_noise_only and aees_both per the "
                 "published configuration (CIFAR pipelines use only the LR axis).")
    lines.append("")
    lines.extend(summary_blocks)

    lines.append("Sanity-check probes:")
    for msg in sanity_msgs:
        lines.append(f"  - {msg}")
    if walk_warnings:
        lines.append("")
        lines.append("Walk warnings (non-fatal):")
        lines.extend(f"  - {w}" for w in walk_warnings)

    write_summary(out_dir, NAME, lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
