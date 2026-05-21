"""Noisy AG News sigma-axis falsification: validation trajectories.

Visualizes that, on AG News with 20% symmetric training-label noise, the
nonzero-noise conditions produce similar late-epoch validation accuracy to the
AEES controller-driven variants, while both sigma=0 baselines, Flat AdamW and
AdamW + Warmup-linear, show the characteristic late-epoch drop.

Conditions:
    1. Flat AdamW, sigma=0
    2. AdamW + Warmup-linear, sigma=0
    3. Fixed sigma=0.005 + Warmup-linear
    4. Fixed sigma=0.01 + Warmup-linear
    5. Random sigma in {0, 0.005, 0.01} + Warmup-linear
    6. AEES-Noise + Warmup-linear
    7. AEES-Dual + Warmup-linear

The figure shows validation accuracy over five epochs, with mean +/- sample SD
bands across five seeds.

Typical reproduction command:
    uv run python -m scripts.plots.nlp_ablation.plot_agnews_noise_ablation_curves \\
        --runs-root archived_results \\
        --out-dir reproduced_artifacts/figures/nlp_ablation

Reads from:
    <runs-root>/noisy_agnews/seed_<N>/agnews_noise20_adamw_warmup_linear_...
    <runs-root>/noisy_agnews/seed_<N>/agnews_noise20_aees_warmup_linear_...
    <runs-root>/nlp_noise_ablation/agnews_noise20_adamw_none_...
    <runs-root>/nlp_noise_ablation/agn20_fixed005_seed*.json
    <runs-root>/nlp_noise_ablation/agn20_fixed01_seed*.json
    <runs-root>/nlp_noise_ablation/agn20_ada_rnd_seed*.json
    <runs-root>/nlp_noise_ablation/agnews_noise20_aees_noiseonly_warmup_linear_*

Outputs on success:
    <out-dir>/agnews_noise_ablation_curves.pdf
    <out-dir>/agnews_noise_ablation_curves.png
    <out-dir>/agnews_noise_ablation_curves.summary.txt

On failure:
    <out-dir>/agnews_noise_ablation_curves.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from dataclasses import dataclass
from typing import Callable

import numpy as np

from scripts.plots._common import walk_runs
from scripts.plots._style import (
    PALETTE,
    make_figure,
    mark_peak_final,
    save_figure,
    short_trajectory_kwargs,
    write_missing,
    write_summary,
)


NAME = "agnews_noise_ablation_curves"
N_EXPECTED_SEEDS = 5
TOTAL_EPOCHS = 5


# ---------------------------------------------------------------------------
# Per-line condition specs. Each line has a predicate that selects records
# from anywhere under --runs-root. The predicate matches on
# (method_name, lr_scheduler, lr_candidates, noise_candidates) — these are
# always present in the config and uniquely identify each condition.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineSpec:
    key: str
    label: str
    color: str
    linestyle: str
    predicate: Callable[[dict, pathlib.Path], bool]


def _cfg(r: dict) -> dict:
    c = r.get("config", {})
    return c if isinstance(c, dict) else {}


def _is_noisy_agnews(r: dict) -> bool:
    c = _cfg(r)
    return (
        c.get("task_name") == "agnews"
        and c.get("label_noise_type") == "symmetric"
        and float(c.get("label_noise_rate") or 0.0) == 0.2
    )


def _close_list(a, b, atol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))


def _make_predicate(
    *,
    method_name: str,
    lr_scheduler: str,
    lr_candidates,
    noise_candidates,
) -> Callable[[dict, pathlib.Path], bool]:
    def predicate(r: dict, _path: pathlib.Path) -> bool:
        if not _is_noisy_agnews(r):
            return False
        if r.get("method_name") != method_name:
            return False
        c = _cfg(r)
        if c.get("lr_scheduler") != lr_scheduler:
            return False
        if not _close_list(c.get("lr_candidates"), lr_candidates):
            return False
        return _close_list(c.get("noise_candidates"), noise_candidates)
    return predicate


# All remaining conditions share the warmup-linear scheduler, so the "+ WL"
# suffix is dropped and the σ=0 warmup-linear row is renamed "Baseline (σ=0)".
# The Flat (σ=0, no scheduler) variant is omitted from this ablation; the
# Flat-vs-Linear comparison lives in agnews_noisy_curves instead.
PLOT_LINES: list[LineSpec] = [
    LineSpec(
        key="adamw_wl",
        label="Baseline (σ=0)",
        color=PALETTE["warmup_linear"],
        linestyle="--",
        predicate=_make_predicate(
            method_name="AdamW",
            lr_scheduler="warmup_linear",
            lr_candidates=[1.0],
            noise_candidates=[0.0],
        ),
    ),
    LineSpec(
        key="fixed_005",
        label="Fixed σ=0.005",
        color="#2ca02c",  # green
        linestyle="-",
        predicate=_make_predicate(
            method_name="RandomScheduler",
            lr_scheduler="warmup_linear",
            lr_candidates=[1.0],
            noise_candidates=[0.005],
        ),
    ),
    LineSpec(
        key="fixed_01",
        label="Fixed σ=0.01",
        color="#bcbd22",  # olive
        linestyle="-",
        predicate=_make_predicate(
            method_name="RandomScheduler",
            lr_scheduler="warmup_linear",
            lr_candidates=[1.0],
            noise_candidates=[0.01],
        ),
    ),
    LineSpec(
        key="random_sigma",
        label="Random σ",
        color="#e377c2",  # pink
        linestyle="-",
        predicate=_make_predicate(
            method_name="RandomScheduler",
            lr_scheduler="warmup_linear",
            lr_candidates=[1.0],
            noise_candidates=[0.0, 0.005, 0.01],
        ),
    ),
    LineSpec(
        key="aees_noise_wl",
        label="AEES-Noise",
        color=PALETTE["aees_noise"],
        linestyle="-",
        predicate=_make_predicate(
            method_name="AdaptiveScheduler",
            lr_scheduler="warmup_linear",
            lr_candidates=[1.0],
            noise_candidates=[0.0, 0.005, 0.01],
        ),
    ),
    LineSpec(
        key="aees_dual_wl",
        label="AEES-Dual",
        color=PALETTE["warmup_linear_aees"],
        linestyle="-",
        predicate=_make_predicate(
            method_name="AdaptiveScheduler",
            lr_scheduler="warmup_linear",
            lr_candidates=[0.5, 1.0, 2.0],
            noise_candidates=[0.0, 0.005, 0.01],
        ),
    ),
]


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineStats:
    spec: LineSpec
    seeds: list[int]
    paths: list[pathlib.Path]
    mean: np.ndarray            # shape (TOTAL_EPOCHS,) in [0,1]
    std: np.ndarray             # shape (TOTAL_EPOCHS,) in [0,1]
    mean_peak_epoch: float      # 1-indexed
    mean_peak_acc: float        # in [0,1]
    std_peak_acc: float         # sample SD of per-seed peaks, in [0,1]
    mean_final_acc: float       # in [0,1]


def _seed(r: dict) -> int | None:
    seed = r.get("seed")
    if seed is None:
        seed = _cfg(r).get("seed")
    return None if seed is None else int(seed)


def _collect(
    runs_root: pathlib.Path,
) -> tuple[dict[str, list[tuple[int, pathlib.Path, dict]]], list[str]]:
    """Bucket files into the 7 line specs. Returns (per_key, errors).

    Files matching the same (line, seed) under multiple directories — e.g.
    the same run mirrored into both `results/nlp_noise_ablation/` and
    `results/noisy_agnews/` — are deduped: first occurrence (walk_runs
    sorts paths lexicographically) wins.
    """
    by_key_seed: dict[str, dict[int, tuple[pathlib.Path, dict]]] = {
        spec.key: {} for spec in PLOT_LINES
    }
    for path, record in walk_runs(runs_root):
        seed = _seed(record)
        if seed is None:
            continue
        for spec in PLOT_LINES:
            if spec.predicate(record, path):
                if seed not in by_key_seed[spec.key]:
                    by_key_seed[spec.key][seed] = (path, record)
                break

    per_key: dict[str, list[tuple[int, pathlib.Path, dict]]] = {
        spec.key: [(seed, p, r)
                   for seed, (p, r) in sorted(by_key_seed[spec.key].items())]
        for spec in PLOT_LINES
    }

    errors: list[str] = []
    for spec in PLOT_LINES:
        runs = per_key[spec.key]
        seeds = sorted({s for s, _, _ in runs})
        if len(seeds) != N_EXPECTED_SEEDS:
            errors.append(
                f"{spec.key} ({spec.label}): found {len(runs)} run(s) "
                f"across seeds={seeds}, expected {N_EXPECTED_SEEDS} distinct seeds"
            )
            continue
        for _, path, record in runs:
            vals = record.get("val_accuracies")
            if not isinstance(vals, list) or len(vals) != TOTAL_EPOCHS:
                errors.append(
                    f"{spec.key} ({spec.label}) file={path}: "
                    f"val_accuracies length={len(vals) if isinstance(vals, list) else 'missing'}, "
                    f"expected {TOTAL_EPOCHS}"
                )
                break
    return per_key, errors


def _aggregate(spec: LineSpec, runs: list[tuple[int, pathlib.Path, dict]]) -> LineStats:
    runs_sorted = sorted(runs, key=lambda triple: triple[0])
    seeds = [s for s, _, _ in runs_sorted]
    paths = [p for _, p, _ in runs_sorted]
    matrix = np.array(
        [r["val_accuracies"] for _, _, r in runs_sorted], dtype=float
    )
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    peak_epochs = matrix.argmax(axis=1) + 1
    peak_accs = matrix.max(axis=1)
    final_accs = matrix[:, -1]
    return LineStats(
        spec=spec,
        seeds=seeds,
        paths=paths,
        mean=mean,
        std=std,
        mean_peak_epoch=float(np.mean(peak_epochs)),
        mean_peak_acc=float(np.mean(peak_accs)),
        std_peak_acc=float(peak_accs.std(ddof=1)),
        mean_final_acc=float(np.mean(final_accs)),
    )


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def _build_figure(stats: list[LineStats]):
    fig, axes = make_figure(n_panels=1)
    ax = axes[0]

    epochs = np.arange(1, TOTAL_EPOCHS + 1)
    base_kwargs = short_trajectory_kwargs()

    for st in stats:
        ax.fill_between(
            epochs,
            (st.mean - st.std) * 100.0,
            (st.mean + st.std) * 100.0,
            color=st.spec.color,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            epochs,
            st.mean * 100.0,
            color=st.spec.color,
            linestyle=st.spec.linestyle,
            label=st.spec.label,
            **base_kwargs,
        )
        peak_xy = (st.mean_peak_epoch, st.mean_peak_acc * 100.0)
        final_xy = (float(TOTAL_EPOCHS), st.mean_final_acc * 100.0)
        mark_peak_final(ax, peak_xy, final_xy, st.spec.color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xticks(list(range(1, TOTAL_EPOCHS + 1)))
    ax.set_xlim(0.85, TOTAL_EPOCHS + 0.15)

    # Y-range: clamp to show the σ>0 cluster (~93.0-93.7) and σ=0 drop (~90.1-90.8)
    # together, with comfortable headroom.
    all_means = np.concatenate([st.mean for st in stats]) * 100.0
    all_stds = np.concatenate([st.std for st in stats]) * 100.0
    lo = float(np.min(all_means - all_stds)) - 0.5
    hi = float(np.max(all_means + all_stds)) + 0.5
    ax.set_ylim(lo, hi)

    # Place the legend below the plot. With 6 entries fitting in one row
    # this keeps the data area free of the legend, which on the previous
    # "lower right" placement was sitting on top of the Baseline σ=0 drop.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=7.5,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    return fig


# ---------------------------------------------------------------------------
# Summary text.
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
    lines.append("Per-condition statistics (5 seeds each, mean across seeds):")
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
    lines.append("σ>0 cluster spread (max-min final-epoch accuracy, %):")
    sigma_pos = [st for st in stats if st.spec.key != "adamw_wl"]
    finals_pos = [st.mean_final_acc * 100.0 for st in sigma_pos]
    spread = max(finals_pos) - min(finals_pos)
    lines.append(
        f"  {spread:.3f} pp across {[st.spec.key for st in sigma_pos]}")

    lines.append("")
    lines.append("σ=0 baseline drop (peak-to-final, pp):")
    for st in stats:
        if st.spec.key == "adamw_wl":
            drop = (st.mean_peak_acc - st.mean_final_acc) * 100.0
            lines.append(f"  {st.spec.key}: {drop:.2f} pp")

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
        "--runs-root", type=pathlib.Path, required=True,
        help="Parent directory; walked recursively for noisy AG News runs.",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, required=True,
        help=f"Output directory for {NAME}.{{pdf,png,summary.txt}}",
    )
    args = parser.parse_args(argv)

    try:
        if not args.runs_root.exists():
            write_missing(
                args.out_dir, NAME,
                f"--runs-root does not exist: {args.runs_root}",
            )
            return 1

        per_key, errors = _collect(args.runs_root)
        if errors:
            reason_lines = [
                "Required runs are missing or malformed; refusing to emit a partial PDF.",
                "",
                f"--runs-root: {args.runs_root}",
                "",
                "Issues:",
                *[f"- {e}" for e in errors],
            ]
            write_missing(args.out_dir, NAME, "\n".join(reason_lines))
            return 1

        stats = [_aggregate(spec, per_key[spec.key]) for spec in PLOT_LINES]

        all_means = np.concatenate([st.mean for st in stats])
        if float(np.ptp(all_means)) < 1e-6:
            write_missing(
                args.out_dir, NAME,
                "Aggregated means are constant across lines; data-loading bug.",
            )
            return 1

        fig = _build_figure(stats)
    except Exception as exc:
        reason = (
            f"Failed to build {NAME}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, NAME, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, NAME)
    summary_path = write_summary(
        args.out_dir, NAME, _summary_lines(
            args.runs_root, stats, pdf_path, png_path)
    )

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
