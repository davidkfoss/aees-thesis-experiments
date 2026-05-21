"""SST-2 validation-accuracy trajectories.

Single-panel figure: epoch, 1-indexed, vs. validation accuracy (%) for four
methods: flat AdamW, AdamW + Warmup-linear, AEES-Dual without a scheduler, and
AEES-Dual combined with Warmup-linear. The figure reports mean +/- SD bands over
five seeds, with peak/final markers per line.

Typical reproduction command:
    uv run python -m scripts.plots.nlp.plot_sst2_curves \\
        --runs-root archived_results/sst2 \\
        --out-dir reproduced_artifacts/figures/nlp

Outputs on success:
    <out-dir>/sst2_curves.pdf
    <out-dir>/sst2_curves.png
    <out-dir>/sst2_curves.summary.txt

On failure:
    <out-dir>/sst2_curves.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from scripts.plots._common import (  # noqa: F401  (episode/epoch helpers kept per task contract)
    RunInfo,
    episode_to_epoch,
    epoch_step_range,
    identify,
    load_set,
    walk_runs,
)
from scripts.plots._style import (  # noqa: F401  (arm_palette kept per task contract)
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


NAME = "sst2_curves"

# Variant keys we want, in plotting order (legend top-to-bottom).
VARIANTS = ("flat", "warmup_linear", "aees_dual", "warmup_linear_aees")

# Display labels: optimizer is implied by the task (SST-2 is AdamW-only), so
# drop the "AdamW" prefix from baselines. Throughout the NLP-noisy section of
# the thesis the warmup-linear schedule is referred to simply as "Linear", and
# combined methods use scheduler-first ordering ("Linear + AEES-Dual") for
# consistency with the CIFAR figures.
_LINEAR_LABEL = "Linear"
DISPLAY: dict[str, str] = {
    "flat":               LABEL["flat"],
    "warmup_linear":      _LINEAR_LABEL,
    "aees_dual":          LABEL["aees_dual"],
    "warmup_linear_aees": f"{_LINEAR_LABEL} + {LABEL['aees_dual']}",
}

EXPECTED_SEEDS = (0, 1, 2, 3, 4)


def _sst2_predicate(info: RunInfo, record: dict) -> bool:
    """Keep only clean SST-2 runs in our four target variants.

    For warmup_linear_aees we additionally require both lr and noise axes to be
    active, so the seed_<N>/ lronly + noiseonly ablations are not pulled in.
    """
    if info.task != "sst2":
        return False
    if info.noise_setting != "clean":
        return False
    if info.variant_key not in VARIANTS:
        return False
    if info.variant_key == "warmup_linear_aees":
        cfg = record.get("config", {})
        lr_cands = cfg.get("lr_candidates") or []
        noise_cands = cfg.get("noise_candidates") or []
        if len(lr_cands) <= 1 or len(noise_cands) <= 1:
            return False
    return True


def _stack_val_accuracies(records: list[dict]) -> tuple[np.ndarray, int]:
    """Return a (n_seeds, total_epochs) array of val accuracies, and total_epochs.

    Raises ValueError if the runs disagree on total_epochs or if any
    val_accuracies list has the wrong length.
    """
    lengths = {len(r["val_accuracies"]) for r in records}
    if len(lengths) != 1:
        raise ValueError(
            f"Inconsistent val_accuracies lengths across seeds: {sorted(lengths)}"
        )
    total_epochs = lengths.pop()
    epochs_field = {r.get("total_epochs") for r in records}
    if epochs_field != {total_epochs}:
        raise ValueError(
            f"total_epochs ({sorted(epochs_field)}) disagrees with len(val_accuracies)={total_epochs}"
        )
    arr = np.array([r["val_accuracies"] for r in records], dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("val_accuracies contains non-finite values")
    return arr, total_epochs


def build(runs_root: pathlib.Path, out_dir: pathlib.Path) -> None:
    grouped = load_set(runs_root, _sst2_predicate)

    # Bail out cleanly if any variant is missing data.
    missing: list[str] = []
    for v in VARIANTS:
        if v not in grouped or len(grouped[v]) == 0:
            missing.append(v)
    if missing:
        write_missing(
            out_dir,
            NAME,
            reason=(
                "No runs found under "
                f"`{runs_root}` for variants: {missing}. Expected 5 seeds per "
                "variant in: " + ", ".join(VARIANTS) + "."
            ),
        )
        return

    # Sort each group by seed for deterministic ordering.
    for v in VARIANTS:
        grouped[v].sort(key=lambda ir: (
            ir[0].seed if ir[0].seed is not None else -1))

    fig, axes = make_figure(n_panels=1)
    ax = axes[0]

    summary_lines: list[str] = [
        f"runs_root: {runs_root}",
        f"variants: {', '.join(VARIANTS)}",
        "",
    ]

    all_means_min = []
    all_means_max = []
    line_kw = short_trajectory_kwargs()

    for variant in VARIANTS:
        pairs = grouped[variant]
        infos = [p[0] for p in pairs]
        records = [p[1] for p in pairs]
        seeds = sorted(int(i.seed) for i in infos if i.seed is not None)
        if len(seeds) < 2:
            raise RuntimeError(
                f"Variant {variant!r} has only {len(seeds)} seed(s); need >= 2 for std."
            )

        accs, total_epochs = _stack_val_accuracies(records)

        # Aggregate on the raw fractions and scale to percent only at the end,
        # matching the table-generator convention (mean-of-fractions ×100,
        # sample SD); avoids last-ULP drift between figure and table values.
        mean = accs.mean(axis=0) * 100.0
        std = accs.std(axis=0, ddof=1) * 100.0  # sample std
        epochs_x = np.arange(1, total_epochs + 1)

        color = PALETTE[variant]
        label = DISPLAY.get(variant, LABEL.get(variant, variant))

        mean_std_band(ax, epochs_x, mean, std, color=color, label=label)
        # Overplot marker-anchored line on top so each epoch has a marker.
        ax.plot(epochs_x, mean, color=color, **line_kw)

        all_means_min.append(float(mean.min()))
        all_means_max.append(float(mean.max()))

        # Per-seed peak epoch (1-indexed) for the summary and marker. Take the
        # per-seed peak/final on fractions, average, then ×100 (same order as
        # the band and the table).
        peak_idx = np.argmax(accs, axis=1)
        per_seed_peak_epoch = (peak_idx + 1).astype(float)
        per_seed_peak_frac = accs[np.arange(accs.shape[0]), peak_idx]
        per_seed_final_frac = accs[:, -1]

        mean_peak_epoch = float(per_seed_peak_epoch.mean())
        mean_peak_val = float(per_seed_peak_frac.mean()) * 100.0
        mean_final_val = float(per_seed_final_frac.mean()) * 100.0
        peak_to_final_drop = mean_peak_val - mean_final_val

        # Peak marker at mean-of-per-seed-peaks (consistent with the other
        # NLP curve plots: agnews_noisy_curves, agnews_noise_ablation_curves).
        # Floats above the mean line by Jensen's inequality — see caption note.
        peak_xy = (mean_peak_epoch, mean_peak_val)
        final_xy = (float(epochs_x[-1]), mean_final_val)
        mark_peak_final(ax, peak_xy, final_xy, color=color)

        summary_lines.append(f"[{variant}] {label}")
        summary_lines.append(f"  seeds: n={len(seeds)} list={seeds}")
        summary_lines.append(f"  total_epochs: {total_epochs}")
        summary_lines.append(f"  mean peak epoch: {mean_peak_epoch:.2f}")
        summary_lines.append(f"  mean peak val acc (%): {mean_peak_val:.2f}")
        summary_lines.append(f"  mean final val acc (%): {mean_final_val:.2f}")
        summary_lines.append(
            f"  peak-to-final drop (pp): {peak_to_final_drop:.2f}")
        summary_lines.append("")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")

    # Integer epoch ticks (SST-2 is 5 epochs).
    # Use the longest variant's epoch range to set ticks.
    max_epochs = max(
        len(r["val_accuracies"]) for v in VARIANTS for _, r in grouped[v]
    )
    ax.set_xticks(np.arange(1, max_epochs + 1))
    ax.set_xlim(1 - 0.1, max_epochs + 0.1)

    # Tight y-range so ~0.5 pp differences are legible.
    data_min = min(all_means_min)
    data_max = max(all_means_max)
    if data_max - data_min < 1e-6:
        raise RuntimeError(
            f"Degenerate plot: all method means collapse to {data_min:.4f}%"
        )
    ax.set_ylim(data_min - 0.5, data_max + 0.5)

    # Move the legend below the plot to keep the data area clean — the Flat
    # curve drops into the lower-right region where the legend would otherwise
    # sit. Matches the agnews_noisy_curves convention for 5-epoch NLP plots.
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
    fig.subplots_adjust(bottom=0.22)

    pdf, png = save_figure(fig, out_dir, NAME)
    summary_lines.append("outputs:")
    summary_lines.append(f"  {pdf}")
    summary_lines.append(f"  {png}")
    write_summary(out_dir, NAME, summary_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        required=True,
        type=pathlib.Path,
        help="Directory containing SST-2 result JSONs (e.g. results/sst2).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=pathlib.Path,
        help="Output directory for the figure (e.g. results/plots/nlp).",
    )
    args = parser.parse_args()
    build(args.runs_root, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
