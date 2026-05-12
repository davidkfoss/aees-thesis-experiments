"""Noisy CIFAR-100 symmetric-40 validation trajectories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.plots_gpt._common import (
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


NAME = "cifar_symmetric40_curves"
VARIANT_ORDER = ("flat", "cosine", "aees", "cosine_aees")
PANEL_SPECS = (
    ("AdamW", "adamw"),
    ("SGD+M", "sgd"),
)
REQUIRED_FILENAMES = {
    "adamw": {
        "flat": "adamw_baseline",
        "cosine": "adamw_cosine",
        "aees": "adamw_aees_ep200_lr05102",
        "cosine_aees": "adamw_cosine_aees_ep200_lr05102",
    },
    "sgd": {
        "flat": "sgd_baseline_lr01",
        "cosine": "sgd_cosine_lr01",
        "aees": "sgd_aees_ep200_lr05102_base01",
        "cosine_aees": "sgd_cosine_aees_ep200_lr05102_base01",
    },
}


@dataclass(frozen=True)
class Trajectory:
    variant_key: str
    seeds: tuple[int, ...]
    epochs: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    peak_epoch: int
    peak_acc: float
    final_epoch: int
    final_acc: float


def _remove_stale_outputs(out_dir: Path, *, keep_missing: bool) -> None:
    suffixes = (".pdf", ".png", ".summary.txt")
    for suffix in suffixes:
        path = out_dir / f"{NAME}{suffix}"
        if path.exists():
            path.unlink()
    missing = out_dir / f"{NAME}.MISSING.md"
    if missing.exists() and not keep_missing:
        missing.unlink()


def _select_required_runs(runs_root: Path, optimizer: str) -> dict[str, dict[int, list]]:
    expected = REQUIRED_FILENAMES[optimizer]
    return load_set(
        runs_root,
        lambda path, record, info: (
            info.task == "cifar100"
            and info.noise_setting == "sym40"
            and info.optimizer == optimizer
            and not info.is_fixed
            and info.variant_key in VARIANT_ORDER
            and path.name == expected.get(info.variant_key)
            and bool(record.get("val_accuracies"))
        ),
    )


def _aggregate(grouped: dict[str, dict[int, list]], optimizer: str) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    missing: list[str] = []

    for variant_key in VARIANT_ORDER:
        seed_runs = grouped.get(variant_key, {})
        seeds = tuple(sorted(seed_runs))
        if len(seeds) != 5:
            missing.append(
                f"{optimizer}/{variant_key}: expected 5 seeds, found {len(seeds)} ({list(seeds)})"
            )
            continue

        arrays: list[np.ndarray] = []
        for seed in seeds:
            runs = seed_runs[seed]
            if len(runs) != 1:
                paths = ", ".join(str(path) for path, _record, _info in runs)
                missing.append(
                    f"{optimizer}/{variant_key}/seed{seed}: expected 1 run, found {len(runs)} [{paths}]"
                )
                continue
            _path, record, _info = runs[0]
            values = np.asarray(record["val_accuracies"], dtype=float)
            total_epochs = int(record.get("total_epochs") or len(values))
            if values.ndim != 1 or len(values) != total_epochs:
                missing.append(
                    f"{optimizer}/{variant_key}/seed{seed}: val_accuracies length "
                    f"{len(values)} does not match total_epochs {total_epochs}"
                )
                continue
            arrays.append(values * 100.0)

        if len(arrays) != 5:
            continue

        lengths = {len(array) for array in arrays}
        if len(lengths) != 1:
            missing.append(
                f"{optimizer}/{variant_key}: inconsistent epoch counts {sorted(lengths)}")
            continue

        stacked = np.vstack(arrays)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        epochs = np.arange(1, len(mean) + 1)
        peak_index = int(np.argmax(mean))
        trajectories.append(
            Trajectory(
                variant_key=variant_key,
                seeds=seeds,
                epochs=epochs,
                mean=mean,
                std=std,
                peak_epoch=int(epochs[peak_index]),
                peak_acc=float(mean[peak_index]),
                final_epoch=int(epochs[-1]),
                final_acc=float(mean[-1]),
            )
        )

    if missing:
        raise ValueError("\n".join(missing))
    if len(trajectories) != len(VARIANT_ORDER):
        raise ValueError(
            f"{optimizer}: expected {len(VARIANT_ORDER)} trajectories, found {len(trajectories)}"
        )
    return trajectories


def build_figure(runs_root: Path):
    panel_data = {
        optimizer: _aggregate(_select_required_runs(
            runs_root, optimizer), optimizer)
        for _title, optimizer in PANEL_SPECS
    }

    fig, axes = make_figure(n_panels=2, panel_height=3.5)
    legend_handles = []
    legend_labels = []
    y_min = 100.0
    y_max = 0.0

    for ax, (title, optimizer) in zip(axes, PANEL_SPECS, strict=True):
        for trajectory in panel_data[optimizer]:
            color = PALETTE[trajectory.variant_key]
            handle = mean_std_band(
                ax,
                trajectory.epochs,
                trajectory.mean,
                trajectory.std,
                color=color,
                label=LABEL[trajectory.variant_key],
            )
            mark_peak_final(
                ax,
                (trajectory.peak_epoch, trajectory.peak_acc),
                (trajectory.final_epoch, trajectory.final_acc),
                color,
            )
            if optimizer == PANEL_SPECS[0][1]:
                legend_handles.append(handle)
                legend_labels.append(LABEL[trajectory.variant_key])
            y_min = min(y_min, float(np.min(trajectory.mean - trajectory.std)))
            y_max = max(y_max, float(np.max(trajectory.mean + trajectory.std)))

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, 200)
        ax.set_xticks([1, 50, 100, 150, 200])

    axes[0].set_ylabel("Validation accuracy (%)")
    lower = max(30.0, np.floor((y_min - 1.0) / 5.0) * 5.0)
    upper = min(100.0, np.ceil((y_max + 1.0) / 5.0) * 5.0)
    axes[0].set_ylim(lower, upper)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.04),
        columnspacing=1.4,
        handlelength=2.2,
    )
    fig.subplots_adjust(bottom=0.24, wspace=0.12)
    return fig, panel_data


def _summary_lines(panel_data: dict[str, list[Trajectory]]) -> list[str]:
    lines = [
        "Noisy CIFAR-100, 40% symmetric label noise validation trajectories.",
        "Aggregation: raw per-epoch validation accuracy means with population std across 5 seeds; no smoothing.",
        "Peak marker: hollow circle at the epoch maximizing the mean trajectory. Final marker: filled square at epoch 200.",
    ]
    for title, optimizer in PANEL_SPECS:
        lines.append(f"{title}:")
        for trajectory in panel_data[optimizer]:
            lines.append(
                f"  {LABEL[trajectory.variant_key]} seeds={list(trajectory.seeds)} "
                f"peak={trajectory.peak_acc:.2f}%@epoch{trajectory.peak_epoch} "
                f"final={trajectory.final_acc:.2f}%"
            )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path,
                        default=Path("results/cifar_noisy"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/plots/cifar"))
    args = parser.parse_args()

    try:
        fig, panel_data = build_figure(args.runs_root)
    except Exception as exc:  # noqa: BLE001 - scripts should emit a clear MISSING file.
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _remove_stale_outputs(args.out_dir, keep_missing=True)
        write_missing(args.out_dir, NAME, str(exc))
        return 1

    _remove_stale_outputs(args.out_dir, keep_missing=False)
    save_figure(fig, args.out_dir, NAME)
    write_summary(args.out_dir, NAME, _summary_lines(panel_data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
