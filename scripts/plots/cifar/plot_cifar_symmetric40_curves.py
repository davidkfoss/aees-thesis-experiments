"""CIFAR-100 (40% symmetric label noise) validation trajectories.

Two side-by-side panels with shared y-axis showing per-epoch validation
accuracy (mean +/- std over 5 seeds) for four schedules:
    Flat, Cosine, AEES, Cosine + AEES
on AdamW (left) and SGD+M (right).

Each method gets a hollow-circle marker at its mean peak and a filled-square
marker at its mean final epoch.

CLI:
    python -m scripts.plots.cifar.plot_cifar_symmetric40_curves \
        --runs-root results/cifar_noisy --out-dir results/plots/cifar

Outputs (on success):
    cifar_symmetric40_curves.pdf
    cifar_symmetric40_curves.png
    cifar_symmetric40_curves.summary.txt

Outputs (on failure):
    cifar_symmetric40_curves.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from collections import defaultdict
from typing import Any

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D

from scripts.plots._common import RunInfo, load_set
from scripts.plots._style import (
    LABEL,
    PALETTE,
    make_figure,
    save_figure,
    write_missing,
    write_summary,
)


# ---------------------------------------------------------------------------
# Spec: methods (in legend order) and panels (in column order).
# ---------------------------------------------------------------------------

METHODS = ["flat", "cosine", "aees_lr", "cosine_aees"]
# In the CIFAR-noisy chapter prose the LR-only AEES variant is referred to as
# just "AEES"; keep the visual continuity with chapter 5 by overriding LABEL
# for "aees_lr" within this figure only.
DISPLAY_LABEL = {
    "flat": LABEL["flat"],
    "cosine": LABEL["cosine"],
    "aees_lr": "AEES",
    "cosine_aees": LABEL["cosine_aees"],
}

PANELS = [("AdamW", "AdamW"), ("SGD", "SGD+M")]

NAME = "cifar_symmetric40_curves"

# Rolling-mean window applied per seed before averaging across seeds. Window=5
# matches the spec and tames the per-epoch noise (especially the SGD+M
# panel's red AEES curve) without erasing the peak.
SMOOTH_WINDOW = 5


def _predicate(info: RunInfo, _record: dict[str, Any]) -> bool:
    if info.noise_setting != "sym40":
        return False
    if info.optimizer not in ("AdamW", "SGD"):
        return False
    if info.variant_key not in METHODS:
        return False
    # The "flat" bucket also pulls in adamw_fixed_lr* / sgd_fixed_lr* runs
    # (controller_logs is None, lr_scheduler="none") because their configs
    # are byte-identical to the baseline. Disambiguate by file name.
    if info.variant_key == "flat":
        if info.path is None:
            return False
        if "fixed" in info.path.name:
            return False
        if "baseline" not in info.path.name:
            return False
    return True


def _bucket(
    runs: dict[str, list[tuple[RunInfo, dict[str, Any]]]],
) -> dict[tuple[str, str], list[tuple[RunInfo, dict[str, Any]]]]:
    """Group surviving runs by (optimizer, variant_key)."""
    out: dict[tuple[str, str],
              list[tuple[RunInfo, dict[str, Any]]]] = defaultdict(list)
    for variant_key, items in runs.items():
        for info, record in items:
            out[(info.optimizer, variant_key)].append((info, record))
    return out


def _stack_val_accs(
    items: list[tuple[RunInfo, dict[str, Any]]]
) -> tuple[np.ndarray, list[int], int]:
    """Return (matrix [n_seeds, T], sorted seed list, T).

    Sorts seeds for deterministic ordering. Validates that all runs share T.
    """
    items_sorted = sorted(items, key=lambda iv: (
        iv[0].seed if iv[0].seed is not None else -1))
    seeds = [info.seed for info, _ in items_sorted]
    val_lists = [r["val_accuracies"] for _, r in items_sorted]
    lengths = {len(v) for v in val_lists}
    if len(lengths) != 1:
        raise ValueError(
            f"val_accuracies length mismatch across seeds: {sorted(lengths)} for seeds {seeds}"
        )
    (T,) = lengths
    mat = np.asarray(val_lists, dtype=float)  # [n_seeds, T]
    return mat, seeds, T


def _peak_final_stats(mat: np.ndarray) -> tuple[float, float, float]:
    """Return (peak_epoch_mean, peak_acc_mean_pct, final_acc_mean_pct).

    peak_epoch is 1-indexed (argmax + 1 per seed, then mean, rounded).
    Accuracies are returned as percentages.
    """
    peak_idxs = np.argmax(mat, axis=1) + 1  # 1-indexed
    peak_epoch_mean = float(round(float(np.mean(peak_idxs))))
    peak_acc_mean = float(np.mean(np.max(mat, axis=1))) * 100.0
    final_acc_mean = float(np.mean(mat[:, -1])) * 100.0
    return peak_epoch_mean, peak_acc_mean, final_acc_mean


def _rolling_mean_axis1(mat: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean along axis=1. ``window <= 1`` returns input."""
    if window <= 1:
        return mat
    n_rows, T = mat.shape
    out = np.empty_like(mat)
    half = window // 2
    for i in range(T):
        lo = max(0, i - half)
        hi = min(T, i + half + 1)
        out[:, i] = mat[:, lo:hi].mean(axis=1)
    return out


def _build_figure(
    cells: dict[tuple[str, str], tuple[np.ndarray, list[int], int]],
):
    fig, axes = make_figure(n_panels=2, panel_height=3.5)

    legend_handles: list = []
    legend_labels: list[str] = []
    seen_methods: set[str] = set()

    # Pass 1: draw lines + bands. Pass 2 draws peak/final markers on top
    # so the halo never gets clipped by a later line segment.
    # Last tuple field is `is_flat` so the second pass can render the Flat
    # baseline's markers with the same de-emphasis as its line.
    marker_jobs: list[tuple[Any, float, float, int, float, str, bool]] = []

    for ax, (opt_key, opt_title) in zip(axes, PANELS):
        for variant_key in METHODS:
            cell = cells.get((opt_key, variant_key))
            if cell is None:
                continue
            mat, _seeds, T = cell
            x = np.arange(1, T + 1)
            mat_smooth = _rolling_mean_axis1(mat, SMOOTH_WINDOW)
            mean_smooth = mat_smooth.mean(axis=0) * 100.0
            # ±1 SD band across seeds, computed from the smoothed per-seed
            # curves so the band's width reflects between-seed variability
            # at the same smoothing scale as the mean line.
            std_smooth = mat_smooth.std(axis=0, ddof=1) * 100.0  # sample std
            color = PALETTE[variant_key]
            label = DISPLAY_LABEL[variant_key]

            is_flat = variant_key == "flat"
            if is_flat:
                # De-emphasized baseline: faint band, thin line. Markers are
                # still drawn (with reduced prominence) so the reader can
                # cross-reference Flat's peak/final positions against the
                # scheduled methods; see _build_figure pass 2.
                ax.fill_between(
                    x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                    color=color, alpha=0.08, linewidth=0, zorder=1,
                )
                line, = ax.plot(
                    x, mean_smooth,
                    color=color, alpha=0.55, linewidth=1.1, zorder=2,
                    label=label,
                )
            else:
                # ±1 SD band underneath the bold mean line.
                ax.fill_between(
                    x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                    color=color, alpha=0.16, linewidth=0, zorder=1,
                )
                line, = ax.plot(
                    x, mean_smooth,
                    color=color, linewidth=1.8, zorder=3,
                    label=label,
                )

            peak_epoch, peak_acc, final_acc = _peak_final_stats(mat)
            marker_jobs.append(
                (ax, peak_epoch, peak_acc, T, final_acc, color, is_flat)
            )

            if variant_key not in seen_methods:
                legend_handles.append(line)
                legend_labels.append(label)
                seen_methods.add(variant_key)

        ax.set_title(opt_title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, max(T for _m, _s, T in cells.values()))

    # Pass 2: peak/final markers with white halo, drawn on top of every line.
    # Flat markers are drawn first (lower zorder) and with reduced alpha so
    # they sit subordinately behind any overlapping scheduled-method marker
    # (in this figure, Flat and Cosine peak at essentially the same point on
    # both optimizers, so Cosine's peak ring will hide Flat's — addressed in
    # the caption).
    halo = pe.withStroke(linewidth=3.0, foreground="white")
    # Stable two-pass order: Flat first, then scheduled methods.
    marker_jobs.sort(key=lambda j: 0 if j[6] else 1)
    for ax, peak_epoch, peak_acc, T, final_acc, color, is_flat in marker_jobs:
        marker_alpha = 0.55 if is_flat else 1.0
        peak_z = 4 if is_flat else 6
        final_z = 4 if is_flat else 5
        ax.plot(
            peak_epoch, peak_acc, marker="o",
            markerfacecolor="none", markeredgecolor=color,
            markersize=8, markeredgewidth=1.8,
            linestyle="none", zorder=peak_z, alpha=marker_alpha,
            path_effects=[halo],
        )
        ax.plot(
            T, final_acc, marker="s",
            color=color, markeredgecolor="black", markeredgewidth=0.6,
            markersize=6, linestyle="none", zorder=final_z, alpha=marker_alpha,
            path_effects=[halo],
        )

    # Identical y-limits across both panels. The figure was built with
    # sharey=True (see make_figure(n_panels=2)), so we set on axes[0]; matplotlib
    # propagates to axes[1]. Lower bound clamped at 30% so the late-stage
    # drops are visible.
    cur_bot, cur_top = axes[0].get_ylim()
    for ax in axes:
        ax.set_ylim(30.0, cur_top)

    axes[0].set_ylabel("Validation accuracy (%)")

    # Explain the peak/final markers in the legend so the reader doesn't
    # have to infer them from the shapes alone.
    marker_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="none", markeredgecolor="black",
            markersize=7, markeredgewidth=1.4,
            label="Mean peak",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none",
            color="black", markeredgecolor="black",
            markeredgewidth=0.5, markersize=5,
            label="Mean final",
        ),
    ]
    all_handles = legend_handles + marker_handles
    all_labels = legend_labels + ["Mean peak", "Mean final"]

    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=len(all_labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.08)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=pathlib.Path,
        required=True,
        help="Root directory containing cifar100_sym40_seed* subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="Directory in which to write the cifar_symmetric40_curves outputs.",
    )
    args = parser.parse_args(argv)

    try:
        runs = load_set(args.runs_root, _predicate)
        bucketed = _bucket(runs)

        # Validate: every (optimizer, variant) cell must have at least one seed.
        missing_cells: list[str] = []
        for opt_key, _opt_title in PANELS:
            for variant_key in METHODS:
                if not bucketed.get((opt_key, variant_key)):
                    missing_cells.append(f"{opt_key}/{variant_key}")
        if missing_cells:
            reason = (
                "Missing runs for the following (optimizer, variant_key) cells:\n\n"
                + "\n".join(f"- {c}" for c in missing_cells)
                + f"\n\nSearched under: {args.runs_root}\n"
            )
            write_missing(args.out_dir, NAME, reason)
            return 1

        # Stack & validate per cell.
        cells: dict[tuple[str, str], tuple[np.ndarray, list[int], int]] = {}
        for (opt_key, variant_key), items in bucketed.items():
            mat, seeds, T = _stack_val_accs(items)
            cells[(opt_key, variant_key)] = (mat, seeds, T)

        # Sanity: avoid blank PDF by checking we have non-trivial data.
        Ts = {T for _m, _s, T in cells.values()}
        if not Ts or max(Ts) < 2:
            raise ValueError(f"degenerate trajectory length(s): {sorted(Ts)}")

        fig = _build_figure(cells)
    except Exception as exc:
        reason = (
            "Failed to build cifar_symmetric40_curves.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, NAME, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, NAME)

    summary_lines: list[str] = []
    summary_lines.append(f"runs-root: {args.runs_root.resolve()}")
    summary_lines.append("")
    summary_lines.append("Per-cell seed counts and statistics:")
    for opt_key, opt_title in PANELS:
        for variant_key in METHODS:
            mat, seeds, _T = cells[(opt_key, variant_key)]
            peak_epoch, peak_acc, final_acc = _peak_final_stats(mat)
            summary_lines.append(
                f"  {opt_title:5s} / {variant_key:12s} ({DISPLAY_LABEL[variant_key]}):"
                f" n={len(seeds)} seeds={seeds}"
                f" peak_epoch_mean={peak_epoch:.2f}"
                f" peak_acc_mean={peak_acc:.2f}%"
                f" final_acc_mean={final_acc:.2f}%"
            )
    summary_lines.append("")
    summary_lines.append("outputs:")
    summary_lines.append(f"  - {pdf_path}")
    summary_lines.append(f"  - {png_path}")

    summary_path = write_summary(args.out_dir, NAME, summary_lines)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
