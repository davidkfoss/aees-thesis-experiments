"""Clean CIFAR-100 peak vs. final bar plot.

Grouped bar chart comparing peak (best val) vs. final-epoch validation accuracy
for six AdamW variants on clean CIFAR-100. Solid bar = peak, hatched bar = final.

Outputs are written under the directory passed via ``--out-dir``.

Typical reproduction command:
    uv run python -m scripts.plots.cifar.plot_clean_cifar_barplot \
        --runs-root archived_results/cifar_clean \
        --out-dir reproduced_artifacts/figures/cifar

Outputs on success:
    <out-dir>/clean_cifar_barplot.pdf
    <out-dir>/clean_cifar_barplot.png
    <out-dir>/clean_cifar_barplot.summary.txt

On failure:
    <out-dir>/clean_cifar_barplot.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from matplotlib.patches import Patch

from scripts.plots._common import (
    walk_runs,
    identify,
    load_set,
    RunInfo,
    episode_to_epoch,
    epoch_step_range,
)
from scripts.plots._style import (
    PALETTE,
    LABEL,
    save_figure,
    write_summary,
    write_missing,
    mean_std_band,
    mark_peak_final,
    short_trajectory_kwargs,
    arm_palette,
    make_figure,
)


NAME = "clean_cifar_barplot"

# Order on x-axis (left to right).
ORDERED_VARIANTS: list[str] = [
    "flat",
    "cosine",
    "linear",
    "aees_lr",
    "cosine_aees",
    "linear_aees",
]

# This figure overrides the AEES-LR display label to plain "AEES" to match
# the prose in sections/results/cifar_clean.tex.
LOCAL_LABEL_OVERRIDES: dict[str, str] = {
    "aees_lr": "AEES",
}


def _predicate(info: RunInfo, _record: dict) -> bool:
    return (
        info.task == "cifar100"
        and info.noise_setting == "clean"
        and info.optimizer == "AdamW"
        and info.variant_key in set(ORDERED_VARIANTS)
    )


def _display_label(variant_key: str) -> str:
    return LOCAL_LABEL_OVERRIDES.get(variant_key, LABEL[variant_key])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        required=True,
        help="Directory containing the clean CIFAR-100 result JSONs.",
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

    grouped = load_set(runs_root, predicate=_predicate)

    missing_variants = [
        v for v in ORDERED_VARIANTS if v not in grouped or not grouped[v]]
    if missing_variants:
        write_missing(
            out_dir,
            NAME,
            "No runs found for variant(s): " + ", ".join(missing_variants)
            + f"\n\nResolved runs root: {runs_root}",
        )
        return 1

    # Per-variant aggregates.
    cells: dict[str, dict[str, object]] = {}
    for variant in ORDERED_VARIANTS:
        runs = grouped[variant]
        peaks: list[float] = []
        finals: list[float] = []
        seeds: list[int] = []
        for info, record in runs:
            best = record.get("best_val_accuracy")
            final = record.get("final_val_accuracy")
            if best is None or final is None:
                write_missing(
                    out_dir,
                    NAME,
                    f"Variant {variant} run {info.path} is missing "
                    f"best_val_accuracy or final_val_accuracy.",
                )
                return 1
            peaks.append(float(best))
            finals.append(float(final))
            seeds.append(info.seed if info.seed is not None else -1)
        order = np.argsort(seeds)
        seeds_sorted = [seeds[i] for i in order]
        peaks_arr = np.asarray([peaks[i] for i in order], dtype=float)
        finals_arr = np.asarray([finals[i] for i in order], dtype=float)

        # Aggregate on the raw fractions, then scale to percent once — this
        # matches the table-generator convention (mean-of-fractions ×100,
        # sample SD) and avoids last-ULP drift between figure and table.
        cells[variant] = {
            "seeds": seeds_sorted,
            "peaks": peaks_arr,
            "finals": finals_arr,
            "peak_mean": float(peaks_arr.mean()) * 100.0,
            "peak_std": float(peaks_arr.std(ddof=1)) * 100.0,
            "final_mean": float(finals_arr.mean()) * 100.0,
            "final_std": float(finals_arr.std(ddof=1)) * 100.0,
        }

    # ---- Figure ---------------------------------------------------------
    fig, axes = make_figure(n_panels=1)
    ax = axes[0]

    n = len(ORDERED_VARIANTS)
    x = np.arange(n)
    bar_w = 0.38

    peak_means = np.array([cells[v]["peak_mean"] for v in ORDERED_VARIANTS])
    peak_stds = np.array([cells[v]["peak_std"] for v in ORDERED_VARIANTS])
    final_means = np.array([cells[v]["final_mean"] for v in ORDERED_VARIANTS])
    final_stds = np.array([cells[v]["final_std"] for v in ORDERED_VARIANTS])
    colors = [PALETTE[v] for v in ORDERED_VARIANTS]

    peak_bars = ax.bar(
        x - bar_w / 2,
        peak_means,
        bar_w,
        yerr=peak_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        ecolor="black",
        error_kw={"linewidth": 0.8},
    )
    final_bars = ax.bar(
        x + bar_w / 2,
        final_means,
        bar_w,
        yerr=final_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        hatch="////",
        capsize=3,
        ecolor="black",
        error_kw={"linewidth": 0.8},
    )

    # Annotate each bar's mean to 1 decimal place. Cap the visual offset so
    # bars with very large SD (AEES Final, whose ±1 SD whisker reaches ~66%)
    # don't push the label far above the bar top, which is otherwise misleading.
    label_offset_cap = 0.45
    for xi, mean, std in zip(x - bar_w / 2, peak_means, peak_stds):
        ax.text(
            xi,
            mean + min(std, label_offset_cap) + 0.15,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    for xi, mean, std in zip(x + bar_w / 2, final_means, final_stds):
        ax.text(
            xi,
            mean + min(std, label_offset_cap) + 0.15,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_display_label(v) for v in ORDERED_VARIANTS],
        rotation=15,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Validation accuracy (%)")
    # y-axis lowered just enough to fit AEES Final's ±1 SD whisker (~66%);
    # the 5pp of extra range below 68% keeps the main cluster (70-73%)
    # readable. Caption flags the seed-level outlier behind that wide SD.
    ax.set_ylim(65, 74)
    ax.grid(axis="x", visible=False)

    legend_handles = [
        Patch(facecolor="#888888", edgecolor="black",
              linewidth=0.6, label="Peak"),
        Patch(
            facecolor="#888888",
            edgecolor="black",
            linewidth=0.6,
            hatch="////",
            label="Final",
        ),
    ]
    # Place legend at upper-left so the rightmost bar's peak annotation has
    # clear vertical headroom (the upper-right region is needed for the value
    # label above the rightmost peak bar).
    ax.legend(handles=legend_handles, loc="upper left")

    fig.tight_layout()

    # ---- Sanity check: AEES-alone should have the widest peak/final gap. -
    gaps = {v: cells[v]["peak_mean"] - cells[v]["final_mean"]
            for v in ORDERED_VARIANTS}
    max_gap_variant = max(gaps, key=gaps.get)
    if max_gap_variant != "aees_lr":
        # Don't silently emit a misleading figure; the acceptance criterion fails.
        write_missing(
            out_dir,
            NAME,
            "Sanity check failed: expected aees_lr to have the widest peak-to-final"
            f" gap, but got {max_gap_variant} (gaps: {gaps}).",
        )
        return 1

    pdf, png = save_figure(fig, out_dir, NAME)

    # ---- Summary file --------------------------------------------------
    lines: list[str] = []
    lines.append(f"Resolved --runs-root: {runs_root}")
    lines.append(
        f"Variants plotted (left to right): {', '.join(ORDERED_VARIANTS)}")
    lines.append("")
    for variant in ORDERED_VARIANTS:
        c = cells[variant]
        seeds = c["seeds"]
        gap = c["peak_mean"] - c["final_mean"]
        lines.append(
            f"[{variant}] (display='{_display_label(variant)}') "
            f"n={len(seeds)} seeds={seeds}"
        )
        lines.append(
            f"  peak  mean={c['peak_mean']:.2f}%  std={c['peak_std']:.2f}%"
        )
        lines.append(
            f"  final mean={c['final_mean']:.2f}%  std={c['final_std']:.2f}%"
        )
        lines.append(f"  peak-to-final gap = {gap:.2f} pp")
        lines.append("")
    lines.append("Outputs:")
    lines.append(f"  {pdf}")
    lines.append(f"  {png}")
    write_summary(out_dir, NAME, lines)

    return 0


if __name__ == "__main__":
    sys.exit(main())
