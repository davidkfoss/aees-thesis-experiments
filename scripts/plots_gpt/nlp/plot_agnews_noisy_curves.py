"""Plot noisy AG News validation trajectories for the thesis."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import pathlib
from typing import Callable

import numpy as np
from matplotlib.ticker import MultipleLocator

from scripts.plots_gpt._common import (
    RunInfo,
    epoch_step_range,
    episode_to_epoch,
    identify,
    load_set,
    walk_runs,
)
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


NAME = "agnews_noisy_curves"
EXPECTED_SEEDS = 5


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    plot_key: str
    order: int
    accepts: Callable[[RunInfo], bool]


METHODS = [
    MethodSpec(
        "adamw",
        "flat",
        0,
        lambda info: (
            info.task == "agnews"
            and info.optimizer == "adamw"
            and not info.is_aees
            and info.scheduler == "none"
        ),
    ),
    MethodSpec(
        "adamw_warmup_linear",
        "warmup_linear",
        1,
        lambda info: (
            info.task == "agnews"
            and info.optimizer == "adamw"
            and not info.is_aees
            and info.scheduler == "warmup_linear"
        ),
    ),
    MethodSpec(
        "aees_none",
        "aees_dual",
        2,
        lambda info: (
            info.task == "agnews"
            and info.optimizer == "adamw"
            and info.is_aees
            and info.scheduler == "none"
            and info.lr_active
            and info.noise_active
        ),
    ),
    MethodSpec(
        "aees_noiseonly",
        "aees_noise",
        3,
        lambda info: (
            info.task == "agnews"
            and info.optimizer == "adamw"
            and info.is_aees
            and info.scheduler == "warmup_linear"
            and not info.lr_active
            and info.noise_active
        ),
    ),
    MethodSpec(
        "aees_warmup_linear",
        "warmup_linear_aees",
        4,
        lambda info: (
            info.task == "agnews"
            and info.optimizer == "adamw"
            and info.is_aees
            and info.scheduler == "warmup_linear"
            and info.lr_active
            and info.noise_active
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="results/noisy_agnews")
    parser.add_argument("--out-dir", default="results/plots/nlp")
    parser.add_argument(
        "--config-filter",
        default="sym20",
        help="Noise setting to plot; Task 3 expects sym20.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = pathlib.Path(args.runs_root)
    out_dir = pathlib.Path(args.out_dir)

    try:
        _plot(runs_root, out_dir, args.config_filter)
    except Exception as exc:  # noqa: BLE001 - turn plot failures into MISSING.md.
        _remove_outputs(out_dir, keep_missing=True)
        write_missing(out_dir, NAME, str(exc))
        return 1
    return 0


def _plot(runs_root: pathlib.Path, out_dir: pathlib.Path, config_filter: str) -> None:
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")

    by_method: dict[str, dict[int, tuple[pathlib.Path, dict, RunInfo]]] = {
        method.slug: {} for method in METHODS
    }
    ignored: list[str] = []

    for path, record in walk_runs(runs_root):
        info = identify(record)
        if info.noise_setting != config_filter:
            ignored.append(f"{path}: noise_setting={info.noise_setting}")
            continue
        matches = [method for method in METHODS if method.accepts(info)]
        if not matches:
            ignored.append(f"{path}: {info.variant_id}")
            continue
        method = matches[0]
        if info.seed is None:
            raise ValueError(f"{path} has no seed")
        if info.seed in by_method[method.slug]:
            prev, _, _ = by_method[method.slug][info.seed]
            raise ValueError(
                f"duplicate run for {method.slug} seed {info.seed}: {prev} and {path}"
            )
        _validate_record(path, record)
        by_method[method.slug][info.seed] = (path, record, info)

    missing = {
        method.slug: sorted(set(range(EXPECTED_SEEDS)) -
                            set(by_method[method.slug]))
        for method in METHODS
        if len(by_method[method.slug]) != EXPECTED_SEEDS
    }
    if missing:
        details = ", ".join(
            f"{slug}: missing seeds {seeds}" for slug, seeds in missing.items())
        raise ValueError(
            f"required noisy AG News runs are incomplete ({details})")

    epochs = np.arange(1, 6, dtype=float)
    fig, axes = make_figure()
    ax = axes[0]
    summary_lines = [
        "Noisy AG News validation trajectories (20% symmetric label noise)",
        "Aggregation: mean +/- population std across five seeds; accuracies rendered as percent.",
        "Smoothing: none; AG News trajectories have five epoch-level points.",
        "",
    ]

    final_means: dict[str, float] = {}
    peak_means: dict[str, float] = {}
    plotted_means: list[np.ndarray] = []
    plotted_stds: list[np.ndarray] = []

    for method in sorted(METHODS, key=lambda item: item.order):
        records = [by_method[method.slug][seed][1]
                   for seed in sorted(by_method[method.slug])]
        values = np.array([record["val_accuracies"]
                          for record in records], dtype=float) * 100.0
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        plotted_means.append(mean)
        plotted_stds.append(std)

        color = PALETTE[method.plot_key]
        line = mean_std_band(ax, epochs, mean, std, color,
                             label=LABEL[method.plot_key])
        line.set(**short_trajectory_kwargs())
        line.set_markerfacecolor(color)
        line.set_markeredgecolor("white")
        line.set_markeredgewidth(0.55)

        peak_idx = int(np.argmax(mean))
        mark_peak_final(
            ax,
            (float(epochs[peak_idx]), float(mean[peak_idx])),
            (float(epochs[-1]), float(mean[-1])),
            color,
        )

        seed_list = ",".join(str(seed)
                             for seed in sorted(by_method[method.slug]))
        peak_means[method.slug] = float(mean[peak_idx])
        final_means[method.slug] = float(mean[-1])
        summary_lines.append(
            f"{method.slug} ({LABEL[method.plot_key]}): seeds={seed_list}; "
            f"peak={mean[peak_idx]:.3f}% at epoch {peak_idx + 1}; "
            f"final={mean[-1]:.3f}%; peak-final drop={mean[peak_idx] - mean[-1]:.3f} pp; "
            f"final std={std[-1]:.3f} pp"
        )

    lower, upper = _y_limits(plotted_means, plotted_stds)
    ax.set_ylim(lower, upper)
    ax.set_xlim(0.82, 5.18)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Noisy AG News validation accuracy")
    ax.set_xticks(epochs)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="x", which="minor", labelbottom=True)
    ax.legend(loc="lower right", frameon=True,
              framealpha=0.88, facecolor="white")

    final_gap = final_means["aees_warmup_linear"] - \
        final_means["adamw_warmup_linear"]
    flat_gap = final_means["aees_warmup_linear"] - final_means["adamw"]
    flatness = peak_means["aees_warmup_linear"] - \
        final_means["aees_warmup_linear"]
    summary_lines.extend(
        [
            "",
            f"Final AEES-Dual+WL vs Warmup-linear baseline gap: {final_gap:.3f} pp.",
            f"Final AEES-Dual+WL vs flat AdamW gap: {flat_gap:.3f} pp.",
            f"AEES-Dual+WL peak-final drop: {flatness:.3f} pp.",
            "",
            "Ignored JSON-decodable runs outside the requested five methods:",
            *(ignored if ignored else ["none"]),
        ]
    )

    _remove_outputs(out_dir, keep_missing=False)
    save_figure(fig, out_dir, NAME)
    write_summary(out_dir, NAME, summary_lines)


def _validate_record(path: pathlib.Path, record: dict) -> None:
    values = record.get("val_accuracies")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError(f"{path} must contain five val_accuracies")
    if int(record.get("total_epochs") or 0) != 5:
        raise ValueError(f"{path} must have total_epochs=5")
    if any((not isinstance(value, (int, float))) for value in values):
        raise ValueError(f"{path} has non-numeric val_accuracies")


def _y_limits(means: list[np.ndarray], stds: list[np.ndarray]) -> tuple[float, float]:
    low = min(float((mean - std).min()) for mean, std in zip(means, stds))
    high = max(float((mean + std).max()) for mean, std in zip(means, stds))
    lower = min(89.2, np.floor((low - 0.15) * 2.0) / 2.0)
    upper = max(93.9, np.ceil((high + 0.12) * 2.0) / 2.0)
    return lower, upper


def _remove_outputs(out_dir: pathlib.Path, *, keep_missing: bool) -> None:
    suffixes = [".pdf", ".png", ".summary.txt"]
    if not keep_missing:
        suffixes.append(".MISSING.md")
    for suffix in suffixes:
        path = out_dir / f"{NAME}{suffix}"
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
