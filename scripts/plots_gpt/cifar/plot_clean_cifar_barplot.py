"""Clean CIFAR-100 peak-vs-final validation accuracy bar plot."""

from __future__ import annotations
from scripts.plots_gpt._style import (
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
from scripts.plots_gpt._common import (
    epoch_step_range,
    episode_to_epoch,
    identify,
    load_set,
    walk_runs,
)
from matplotlib.patches import Patch
import numpy as np

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


NAME = "clean_cifar_barplot"
VARIANT_ORDER = ("flat", "cosine", "linear", "aees",
                 "cosine_aees", "linear_aees")
DISPLAY_LABEL = {
    "flat": "AdamW",
    "cosine": "AdamW\n+ Cosine",
    "linear": "AdamW\n+ Linear",
    "aees": "AEES",
    "cosine_aees": "AEES\n+ Cosine",
    "linear_aees": "AEES\n+ Linear",
}


@dataclass(frozen=True)
class Aggregate:
    variant: str
    seeds: tuple[int, ...]
    peak_values: np.ndarray
    final_values: np.ndarray

    @property
    def peak_mean(self) -> float:
        return float(np.mean(self.peak_values))

    @property
    def peak_std(self) -> float:
        return float(np.std(self.peak_values, ddof=0))

    @property
    def final_mean(self) -> float:
        return float(np.mean(self.final_values))

    @property
    def final_std(self) -> float:
        return float(np.std(self.final_values, ddof=0))


def parse_args() -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=pathlib.Path,
        default=repo_root / "results" / "cifar_clean",
        help="Root containing clean CIFAR-100 run JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=repo_root / "results" / "plots" / "cifar",
        help="Directory for clean_cifar_barplot.{pdf,png,summary.txt}.",
    )
    parser.add_argument(
        "--config-filter",
        default="adamw-clean-main",
        help="Compatibility knob; only the AdamW clean main configurations are plotted.",
    )
    return parser.parse_args()


def is_requested_run(path: pathlib.Path, record: dict, info) -> bool:
    del path
    return (
        info.task == "cifar100"
        and info.optimizer == "adamw"
        and info.noise_setting == "clean"
        and not info.is_fixed
        and info.variant_key in VARIANT_ORDER
        and int(record.get("total_epochs", 0)) > 0
    )


def aggregate_runs(runs_root: pathlib.Path) -> list[Aggregate]:
    grouped = load_set(runs_root, is_requested_run)
    missing_variants = [
        variant for variant in VARIANT_ORDER if variant not in grouped]
    if missing_variants:
        raise ValueError("missing variants: " + ", ".join(missing_variants))

    aggregates: list[Aggregate] = []
    for variant in VARIANT_ORDER:
        by_seed = grouped[variant]
        if len(by_seed) <= 1:
            raise ValueError(
                f"{variant} has only {len(by_seed)} seed(s); refusing degenerate plot")

        peak_values: list[float] = []
        final_values: list[float] = []
        seeds: list[int] = []
        for seed in sorted(by_seed):
            entries = by_seed[seed]
            if len(entries) != 1:
                paths = ", ".join(str(path) for path, _, _ in entries)
                raise ValueError(
                    f"{variant} seed {seed} has {len(entries)} runs: {paths}")
            path, record, _ = entries[0]
            if "best_val_accuracy" not in record or "final_val_accuracy" not in record:
                raise ValueError(
                    f"{path} is missing best_val_accuracy or final_val_accuracy")
            peak = float(record["best_val_accuracy"]) * 100.0
            final = float(record["final_val_accuracy"]) * 100.0
            if not (0.0 <= peak <= 100.0 and 0.0 <= final <= 100.0):
                raise ValueError(f"{path} has accuracy outside [0, 1]")
            seeds.append(seed)
            peak_values.append(peak)
            final_values.append(final)

        aggregates.append(
            Aggregate(
                variant=variant,
                seeds=tuple(seeds),
                peak_values=np.array(peak_values, dtype=float),
                final_values=np.array(final_values, dtype=float),
            )
        )

    return aggregates


def draw_plot(aggregates: list[Aggregate]):
    fig, (ax,) = make_figure()

    x = np.arange(len(aggregates), dtype=float)
    width = 0.34
    peak_means = np.array([agg.peak_mean for agg in aggregates])
    final_means = np.array([agg.final_mean for agg in aggregates])
    peak_stds = np.array([agg.peak_std for agg in aggregates])
    final_stds = np.array([agg.final_std for agg in aggregates])
    colors = [PALETTE[agg.variant] for agg in aggregates]

    peak_bars = ax.bar(
        x - width / 2,
        peak_means,
        width,
        yerr=peak_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.65,
        error_kw={"ecolor": "black", "elinewidth": 0.8,
                  "capsize": 3, "capthick": 0.8},
        label="Peak",
    )
    final_bars = ax.bar(
        x + width / 2,
        final_means,
        width,
        yerr=final_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.65,
        hatch="////",
        alpha=0.9,
        error_kw={"ecolor": "black", "elinewidth": 0.8,
                  "capsize": 3, "capthick": 0.8},
        label="Final",
    )

    for bars, means, stds in ((peak_bars, peak_means, peak_stds), (final_bars, final_means, final_stds)):
        for bar, mean, std in zip(bars, means, stds, strict=True):
            y = min(mean + std + 0.14, 73.72)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
            )

    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(68.0, 74.0)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABEL[agg.variant] for agg in aggregates])
    ax.set_xlim(-0.65, len(aggregates) - 0.35)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[
            Patch(facecolor="#bbbbbb", edgecolor="black", label="Peak"),
            Patch(facecolor="#bbbbbb", edgecolor="black",
                  hatch="////", label="Final"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
        columnspacing=1.2,
        handlelength=2.2,
    )

    return fig


def build_summary(runs_root: pathlib.Path, out_dir: pathlib.Path, aggregates: list[Aggregate]) -> list[str]:
    lines = [
        f"Resolved --runs-root: {runs_root.resolve()}",
        f"Variants plotted (left to right): {', '.join(agg.variant for agg in aggregates)}",
        "",
    ]
    for agg in aggregates:
        display = DISPLAY_LABEL[agg.variant].replace("\n", " ")
        lines.extend(
            [
                f"[{agg.variant}] (display='{display}', style-label='{LABEL[agg.variant]}') "
                f"n={len(agg.seeds)} seeds={list(agg.seeds)}",
                f"  peak  mean={agg.peak_mean:.2f}%  std={agg.peak_std:.2f}%",
                f"  final mean={agg.final_mean:.2f}%  std={agg.final_std:.2f}%",
                f"  peak-to-final gap = {agg.peak_mean - agg.final_mean:.2f} pp",
                "",
            ]
        )
    lines.extend(
        [
            "Outputs:",
            f"  {out_dir.resolve() / (NAME + '.pdf')}",
            f"  {out_dir.resolve() / (NAME + '.png')}",
        ]
    )
    return lines


def remove_outputs(out_dir: pathlib.Path, *, keep_missing: bool) -> None:
    suffixes = [".pdf", ".png", ".summary.txt"] if keep_missing else [
        ".MISSING.md"]
    for suffix in suffixes:
        path = out_dir / f"{NAME}{suffix}"
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    args = parse_args()
    try:
        aggregates = aggregate_runs(args.runs_root)
        fig = draw_plot(aggregates)
        pdf, png = save_figure(fig, args.out_dir, NAME)
        summary = write_summary(args.out_dir, NAME, build_summary(
            args.runs_root, args.out_dir, aggregates))
        remove_outputs(args.out_dir, keep_missing=False)
        print(f"Wrote {pdf}")
        print(f"Wrote {png}")
        print(f"Wrote {summary}")
    except Exception as exc:  # noqa: BLE001 - script must convert failures to .MISSING.md.
        remove_outputs(args.out_dir, keep_missing=True)
        missing = write_missing(args.out_dir, NAME, str(exc))
        print(f"Wrote {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
