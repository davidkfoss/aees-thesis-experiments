"""Plot SST-2 validation trajectories for the thesis NLP results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
from typing import Iterable

import numpy as np

from scripts.plots_gpt._common import (  # noqa: F401 - full shared API imported per task contract.
    epoch_step_range,
    episode_to_epoch,
    identify,
    load_set,
    walk_runs,
)
from scripts.plots_gpt._style import (  # noqa: F401 - full shared API imported per task contract.
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
REQUIRED_VARIANTS = {
    "flat": "adamw_flat",
    "warmup_linear": "adamw_warmup_linear",
    "aees_dual": "adamw_aees_dual_none",
    "warmup_linear_aees": "adamw_aees_dual_warmup_linear",
}
PLOT_ORDER = ["flat", "warmup_linear", "aees_dual", "warmup_linear_aees"]


@dataclass(frozen=True)
class VariantStats:
    key: str
    seeds: list[int]
    epochs: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    peak_epoch_mean: float
    peak_acc_mean: float
    final_acc_mean: float
    drop_pp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=pathlib.Path,
                        default=pathlib.Path("results/sst2"))
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path("results/plots/nlp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        stats = collect_stats(args.runs_root)
        fig = draw(stats)
        pdf, png = save_figure(fig, args.out_dir, NAME)
        write_summary(args.out_dir, NAME, summary_lines(
            args.runs_root, stats, pdf, png))
        remove_missing(args.out_dir)
    except Exception as exc:  # noqa: BLE001 - CLI should emit the required failure artifact.
        remove_outputs(args.out_dir)
        write_missing(args.out_dir, NAME, str(exc))
        raise


def collect_stats(runs_root: pathlib.Path) -> dict[str, VariantStats]:
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")

    grouped = load_set(runs_root, is_required_sst2_run)
    missing = [key for key in PLOT_ORDER if key not in grouped]
    if missing:
        raise ValueError(
            f"missing required SST-2 variants: {', '.join(missing)}")

    stats: dict[str, VariantStats] = {}
    for key in PLOT_ORDER:
        per_seed = grouped[key]
        if len(per_seed) < 2:
            raise ValueError(
                f"{key} has fewer than two seeds; refusing degenerate plot")

        seed_series: list[np.ndarray] = []
        peak_epochs: list[int] = []
        peak_accs: list[float] = []
        final_accs: list[float] = []
        total_epochs: int | None = None

        for seed, entries in sorted(per_seed.items()):
            if len(entries) != 1:
                paths = ", ".join(str(path) for path, _, _ in entries)
                raise ValueError(
                    f"{key} seed {seed} has {len(entries)} matching runs: {paths}")
            _, record, _ = entries[0]
            vals = record.get("val_accuracies")
            if not vals:
                raise ValueError(
                    f"{key} seed {seed} is missing val_accuracies")
            series = np.asarray(vals, dtype=float) * 100.0
            if np.any(~np.isfinite(series)):
                raise ValueError(
                    f"{key} seed {seed} has non-finite val_accuracies")

            epochs = int(record.get("total_epochs") or len(series))
            if epochs != len(series):
                raise ValueError(
                    f"{key} seed {seed}: total_epochs != len(val_accuracies)")
            if total_epochs is None:
                total_epochs = epochs
            elif total_epochs != epochs:
                raise ValueError(
                    f"{key} has inconsistent epoch counts across seeds")

            peak_index = int(np.argmax(series))
            seed_series.append(series)
            peak_epochs.append(peak_index + 1)
            peak_accs.append(float(series[peak_index]))
            final_accs.append(float(series[-1]))

        matrix = np.vstack(seed_series)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=0)
        peak_acc_mean = float(np.mean(peak_accs))
        final_acc_mean = float(np.mean(final_accs))
        stats[key] = VariantStats(
            key=key,
            seeds=sorted(per_seed),
            epochs=np.arange(1, int(total_epochs or 0) + 1),
            mean=mean,
            std=std,
            peak_epoch_mean=float(np.mean(peak_epochs)),
            peak_acc_mean=peak_acc_mean,
            final_acc_mean=final_acc_mean,
            drop_pp=peak_acc_mean - final_acc_mean,
        )

    return stats


def is_required_sst2_run(path: pathlib.Path, record: dict, info) -> bool:
    del path
    if info.task != "sst2" or info.optimizer != "adamw" or info.noise_setting != "clean":
        return False
    return REQUIRED_VARIANTS.get(info.variant_key) == info.variant_id


def draw(stats: dict[str, VariantStats]):
    fig, axes = make_figure()
    ax = axes[0]
    line_kwargs = short_trajectory_kwargs()

    for key in PLOT_ORDER:
        stat = stats[key]
        color = PALETTE[key]
        line = mean_std_band(ax, stat.epochs, stat.mean,
                             stat.std, color, label=LABEL[key])
        line.set(**line_kwargs)
        peak_index = int(np.argmax(stat.mean))
        mark_peak_final(
            ax,
            (float(stat.epochs[peak_index]), float(stat.mean[peak_index])),
            (float(stat.epochs[-1]), float(stat.mean[-1])),
            color,
        )

    ax.set_title("SST-2 validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xticks(stats[PLOT_ORDER[0]].epochs)
    ax.set_ylim(*readable_ylim(stats.values()))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    return fig


def readable_ylim(stats: Iterable[VariantStats]) -> tuple[float, float]:
    means = np.concatenate([stat.mean for stat in stats])
    low = float(np.min(means))
    high = float(np.max(means))
    center = (low + high) / 2.0
    span = max(high - low + 0.8, 2.0)
    span = min(span, 4.0)
    y0 = center - span / 2.0
    y1 = center + span / 2.0
    return round(y0 * 2.0) / 2.0, round(y1 * 2.0) / 2.0


def summary_lines(
    runs_root: pathlib.Path,
    stats: dict[str, VariantStats],
    pdf: pathlib.Path,
    png: pathlib.Path,
) -> list[str]:
    lines = [
        f"runs_root: {runs_root}",
        f"variants: {', '.join(PLOT_ORDER)}",
        "aggregation: mean +/- population std across seeds; no smoothing",
        "",
    ]
    for key in PLOT_ORDER:
        stat = stats[key]
        lines.extend(
            [
                f"[{key}] {LABEL[key]}",
                f"  required variant_id: {REQUIRED_VARIANTS[key]}",
                f"  seeds: n={len(stat.seeds)} list={stat.seeds}",
                f"  total_epochs: {len(stat.epochs)}",
                f"  mean peak epoch: {stat.peak_epoch_mean:.2f}",
                f"  mean peak val acc (%): {stat.peak_acc_mean:.2f}",
                f"  mean final val acc (%): {stat.final_acc_mean:.2f}",
                f"  peak-to-final drop (pp): {stat.drop_pp:.2f}",
                "",
            ]
        )
    lines.extend(["outputs:", f"  {pdf}", f"  {png}"])
    return lines


def remove_outputs(out_dir: pathlib.Path) -> None:
    for suffix in (".pdf", ".png", ".summary.txt"):
        path = out_dir / f"{NAME}{suffix}"
        if path.exists():
            path.unlink()


def remove_missing(out_dir: pathlib.Path) -> None:
    path = out_dir / f"{NAME}.MISSING.md"
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
