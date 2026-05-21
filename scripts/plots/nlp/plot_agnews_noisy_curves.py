"""Validation-accuracy trajectories on noisy AG News with 20% symmetric noise.

Single-panel figure: epoch (1..5) on x-axis and validation accuracy (%) on the
y-axis, with a mean +/- SD band per method across five seeds. All runs use
AdamW, so the optimizer is implied by the task name.

Methods:
    1. Flat                            - flat baseline
    2. Warmup-linear                   - warmup-linear baseline
    3. AEES-Dual                       - LR + noise, scheduler=none
    4. AEES-Noise + Warmup-linear      - noise-only AEES, with warmup-linear
    5. AEES-Dual + Warmup-linear       - LR + noise AEES, with warmup-linear

Three AEES variants share variant_key="warmup_linear_aees"; identify() cannot
distinguish LR-only, noise-only, and dual-axis scheduler variants from
variant_key alone. This script therefore groups runs using the finer key
(variant_key, lr_active, noise_active).

Typical reproduction command:
    uv run python -m scripts.plots.nlp.plot_agnews_noisy_curves \\
        --runs-root archived_results/noisy_agnews \\
        --out-dir reproduced_artifacts/figures/nlp

Outputs on success:
    <out-dir>/agnews_noisy_curves.pdf
    <out-dir>/agnews_noisy_curves.png
    <out-dir>/agnews_noisy_curves.summary.txt

On failure:
    <out-dir>/agnews_noisy_curves.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from scripts.plots._common import RunInfo, episode_to_epoch, epoch_step_range, identify, load_set, walk_runs
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


N_EXPECTED_SEEDS = 5
TOTAL_EPOCHS = 5


# ---------------------------------------------------------------------------
# Plot specs: one entry per line in the figure. The "selector" decides which
# (RunInfo, record) tuples belong to this line.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineSpec:
    key: str            # short id used in error messages / summary lines
    label: str          # legend text
    color: str          # hex from PALETTE
    variant_key: str    # match info.variant_key
    scheduler: str      # match info.scheduler ("none" or "warmup_linear")
    # required len(lr_candidates)>1 (None = don't filter)
    lr_active: bool | None
    # required len(noise_candidates)>1 (None = don't filter)
    noise_active: bool | None


# Display-label overrides: in this figure (and throughout the NLP-noisy
# section of the thesis) the warmup-linear schedule is referred to simply
# as "Linear", and combined methods use scheduler-first ordering
# ("Linear + AEES-Dual") for consistency with the CIFAR figures.
_LINEAR_LABEL = "Linear"

PLOT_LINES: list[LineSpec] = [
    LineSpec(
        key="adamw_flat",
        label=LABEL["flat"],
        color=PALETTE["flat"],
        variant_key="flat",
        scheduler="none",
        lr_active=False,
        noise_active=False,
    ),
    LineSpec(
        key="adamw_wl",
        label=_LINEAR_LABEL,
        color=PALETTE["warmup_linear"],
        variant_key="warmup_linear",
        scheduler="warmup_linear",
        lr_active=False,
        noise_active=False,
    ),
    LineSpec(
        key="aees_dual",
        label=LABEL["aees_dual"],
        color=PALETTE["aees_dual"],
        variant_key="aees_dual",
        scheduler="none",
        lr_active=True,
        noise_active=True,
    ),
    LineSpec(
        key="aees_noise_wl",
        label=f"{_LINEAR_LABEL} + {LABEL['aees_noise']}",
        color=PALETTE["aees_noise"],
        variant_key="warmup_linear_aees",
        scheduler="warmup_linear",
        lr_active=False,
        noise_active=True,
    ),
    LineSpec(
        key="aees_dual_wl",
        label=f"{_LINEAR_LABEL} + {LABEL['aees_dual']}",
        color=PALETTE["warmup_linear_aees"],
        variant_key="warmup_linear_aees",
        scheduler="warmup_linear",
        lr_active=True,
        noise_active=True,
    ),
]


# ---------------------------------------------------------------------------
# Data discovery and validation.
# ---------------------------------------------------------------------------


def _is_lr_active(record: dict) -> bool:
    return len(record.get("config", {}).get("lr_candidates") or []) > 1


def _is_noise_active(record: dict) -> bool:
    return len(record.get("config", {}).get("noise_candidates") or []) > 1


def _matches(spec: LineSpec, info: RunInfo, record: dict) -> bool:
    if info.variant_key != spec.variant_key:
        return False
    if info.scheduler != spec.scheduler:
        return False
    if spec.lr_active is not None and _is_lr_active(record) != spec.lr_active:
        return False
    if spec.noise_active is not None and _is_noise_active(record) != spec.noise_active:
        return False
    return True


def _collect_runs(
    runs_root: pathlib.Path,
) -> tuple[dict[str, list[tuple[RunInfo, dict]]], list[str]]:
    """Walk runs_root and bucket files into the five plot lines.

    Returns (per_line_runs, errors). errors is empty when every line has
    exactly N_EXPECTED_SEEDS distinct seeds with the right shape.
    """
    per_line: dict[str, list[tuple[RunInfo, dict]]] = {
        spec.key: [] for spec in PLOT_LINES}

    for path, record in walk_runs(runs_root):
        info = identify(record, path)
        if info.task != "agnews":
            continue
        if info.noise_setting != "sym20":
            continue
        for spec in PLOT_LINES:
            if _matches(spec, info, record):
                per_line[spec.key].append((info, record))
                break

    errors: list[str] = []
    for spec in PLOT_LINES:
        runs = per_line[spec.key]
        seeds = sorted(
            {info.seed for info, _ in runs if info.seed is not None})
        if len(runs) != N_EXPECTED_SEEDS:
            errors.append(
                f"{spec.key} ({spec.label}): found {len(runs)} run(s), expected "
                f"{N_EXPECTED_SEEDS}; seeds={seeds}"
            )
            continue
        if len(seeds) != N_EXPECTED_SEEDS:
            errors.append(
                f"{spec.key} ({spec.label}): {N_EXPECTED_SEEDS} runs but "
                f"seeds={seeds} (need {N_EXPECTED_SEEDS} distinct seeds)"
            )
            continue
        for info, record in runs:
            vals = record.get("val_accuracies")
            if not isinstance(vals, list) or len(vals) != TOTAL_EPOCHS:
                errors.append(
                    f"{spec.key} ({spec.label}) seed={info.seed}: "
                    f"val_accuracies length={len(vals) if isinstance(vals, list) else 'missing'}, "
                    f"expected {TOTAL_EPOCHS}"
                )
                break

    return per_line, errors


# ---------------------------------------------------------------------------
# Aggregation: stack per-seed val_accuracies, compute mean/std/peak/final.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineStats:
    spec: LineSpec
    seeds: list[int]
    mean: np.ndarray            # shape (TOTAL_EPOCHS,) in [0,1]
    std: np.ndarray             # shape (TOTAL_EPOCHS,) in [0,1]
    mean_peak_epoch: float      # 1-indexed
    mean_peak_acc: float        # in [0,1]
    mean_final_acc: float       # in [0,1]


def _aggregate(spec: LineSpec, runs: list[tuple[RunInfo, dict]]) -> LineStats:
    runs_sorted = sorted(
        runs, key=lambda pair: pair[0].seed if pair[0].seed is not None else -1)
    seeds = [info.seed for info, _ in runs_sorted]
    matrix = np.array(
        [record["val_accuracies"] for _, record in runs_sorted], dtype=float
    )
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)  # sample std
    # Peak per seed: 1-indexed argmax + accuracy at that epoch.
    peak_epochs = matrix.argmax(axis=1) + 1
    peak_accs = matrix.max(axis=1)
    final_accs = matrix[:, -1]
    return LineStats(
        spec=spec,
        seeds=seeds,
        mean=mean,
        std=std,
        mean_peak_epoch=float(np.mean(peak_epochs)),
        mean_peak_acc=float(np.mean(peak_accs)),
        mean_final_acc=float(np.mean(final_accs)),
    )


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def _build_figure(stats: list[LineStats]):
    fig, axes = make_figure(n_panels=1)
    ax = axes[0]

    epochs = np.arange(1, TOTAL_EPOCHS + 1)
    line_kwargs = short_trajectory_kwargs()

    for st in stats:
        # mean_std_band gives us the band + base line; we then redraw with
        # marker styling so the dots/line are honored without losing the band.
        ax.fill_between(
            epochs,
            (st.mean - st.std) * 100.0,
            (st.mean + st.std) * 100.0,
            color=st.spec.color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            epochs,
            st.mean * 100.0,
            color=st.spec.color,
            label=st.spec.label,
            **line_kwargs,
        )
        # Use the mean-of-per-seed-peaks for both x and y. On a 5-epoch axis
        # this naturally spreads the peak rings across fractional epochs
        # (e.g., 4.4 vs 4.6) so they don't stack on top of each other or on
        # the final-epoch squares. Consistent with cifar_symmetric40_curves;
        # the caption acknowledges that markers can float above the mean
        # line (Jensen's inequality for the max operator).
        peak_xy = (st.mean_peak_epoch, st.mean_peak_acc * 100.0)
        final_xy = (float(TOTAL_EPOCHS), st.mean_final_acc * 100.0)
        mark_peak_final(ax, peak_xy, final_xy, st.spec.color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xticks(list(range(1, TOTAL_EPOCHS + 1)))
    ax.set_xlim(0.85, TOTAL_EPOCHS + 0.15)

    # Y-range: fit to data with comfortable headroom so the ~2.7pp gap between
    # baselines (~90.8%) and AEES variants (~93.5%) at the final epoch reads
    # clearly. Compute from data with a small margin and clamp to a reasonable
    # window.
    all_means = np.concatenate([st.mean for st in stats]) * 100.0
    all_stds = np.concatenate([st.std for st in stats]) * 100.0
    lo = float(np.min(all_means - all_stds)) - 0.4
    hi = float(np.max(all_means + all_stds)) + 0.4
    ax.set_ylim(lo, hi)

    # With 5 entries and a tight 5-epoch plot, every in-axes legend position
    # collides with some curve. Move the legend below the plot in a single
    # horizontal row, matching the cifar_symmetric40_curves convention.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    return fig


# ---------------------------------------------------------------------------
# Summary file.
# ---------------------------------------------------------------------------


def _summary_lines(
    runs_root: pathlib.Path,
    stats: list[LineStats],
    pdf_path: pathlib.Path,
    png_path: pathlib.Path,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"runs-root: {runs_root}")
    lines.append("")
    lines.append("Per-line statistics (5 seeds each, mean across seeds):")
    for st in stats:
        drop = (st.mean_peak_acc - st.mean_final_acc) * 100.0
        lines.append(f"- {st.spec.label}")
        lines.append(f"    key: {st.spec.key}")
        lines.append(f"    n_seeds: {len(st.seeds)}")
        lines.append(f"    seeds: {st.seeds}")
        lines.append(f"    mean_peak_epoch: {st.mean_peak_epoch:.2f}")
        lines.append(
            f"    mean_peak_val_acc_pct: {st.mean_peak_acc * 100.0:.2f}")
        lines.append(
            f"    mean_final_val_acc_pct: {st.mean_final_acc * 100.0:.2f}")
        lines.append(f"    peak_to_final_drop_pp: {drop:.2f}")
    lines.append("")
    lines.append("outputs:")
    lines.append(f"  - {pdf_path}")
    lines.append(f"  - {png_path}")
    return lines


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=pathlib.Path,
        required=True,
        help="Directory holding seed_<N>/ subdirs of noisy AG News results.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="Directory in which to write agnews_noisy_curves.{pdf,png,summary.txt}.",
    )
    args = parser.parse_args(argv)
    name = "agnews_noisy_curves"

    try:
        if not args.runs_root.exists():
            write_missing(
                args.out_dir,
                name,
                f"--runs-root does not exist: {args.runs_root}",
            )
            return 1

        per_line, errors = _collect_runs(args.runs_root)
        if errors:
            reason_lines = [
                "Required runs are missing or malformed; refusing to emit a partial PDF.",
                "",
                f"--runs-root: {args.runs_root}",
                "",
                "Issues:",
                *[f"- {e}" for e in errors],
            ]
            write_missing(args.out_dir, name, "\n".join(reason_lines))
            return 1

        stats = [_aggregate(spec, per_line[spec.key]) for spec in PLOT_LINES]

        # Sanity: refuse to emit a degenerate (constant or near-constant) plot.
        all_means = np.concatenate([st.mean for st in stats])
        if float(np.ptp(all_means)) < 1e-6:
            write_missing(
                args.out_dir,
                name,
                "Aggregated means are constant across all lines and epochs; "
                "this looks like a data-loading bug. Refusing to emit blank PDF.",
            )
            return 1

        fig = _build_figure(stats)
    except Exception as exc:  # pragma: no cover - defensive
        reason = (
            "Failed to build agnews_noisy_curves.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, name, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, name)
    summary_path = write_summary(
        args.out_dir, name, _summary_lines(
            args.runs_root, stats, pdf_path, png_path)
    )

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
