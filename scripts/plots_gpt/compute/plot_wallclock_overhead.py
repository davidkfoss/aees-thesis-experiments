"""Task 10: wall-clock overhead grouped horizontal bar chart."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

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


NAME = "wallclock_overhead"

GROUPS = [
    ("CIFAR-SGD+M", "cifar_clean", "sgd", "cosine"),
    ("CIFAR-AdamW", "cifar_clean", "adamw", "cosine"),
    ("SST-2-AdamW", "sst2_clean", "adamw", "warmup_linear"),
    ("AG-News-AdamW", "agnews_clean", "adamw", "warmup_linear"),
]

DISPLAY_BARS = [
    ("scheduler", "Scheduler baseline"),
    ("aees_lr_only", "AEES-LR"),
    ("aees_noise_only", "AEES-Noise"),
    ("aees_both", "AEES-Dual"),
]

VARIANT_TO_KEY = {
    "scheduler": None,
    "aees_lr_only": "aees_lr",
    "aees_noise_only": "aees_noise",
    "aees_both": "aees_dual",
}

SKIP_FOR_CIFAR = {"aees_noise_only", "aees_both"}


@dataclass(frozen=True)
class Repeat:
    path: Path
    mean_epoch_seconds: float


@dataclass(frozen=True)
class Cell:
    task: str
    optimizer: str
    variant: str
    scheduler: str
    repeats: tuple[Repeat, ...]

    @property
    def mean_seconds(self) -> float:
        return mean(r.mean_epoch_seconds for r in self.repeats)

    @property
    def std_seconds(self) -> float:
        values = [r.mean_epoch_seconds for r in self.repeats]
        return pstdev(values) if len(values) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path,
                        default=Path("results/compute_overhead"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/plots/compute"))
    return parser.parse_args()


def task_slug(path: Path) -> str | None:
    name = path.name
    for slug in ("cifar_clean", "agnews_clean", "sst2_clean"):
        if name.startswith(f"{slug}_"):
            return slug
    return None


def variant_slug(record: dict) -> str:
    info = identify(record)
    if not info.is_aees:
        return "baseline"
    if info.lr_active and info.noise_active:
        return "aees_both"
    if info.lr_active:
        return "aees_lr_only"
    if info.noise_active:
        return "aees_noise_only"
    raise ValueError("AEES record has no active LR or noise axis")


def epoch_wall_clock(record: dict) -> list[float]:
    runtime = record.get("runtime_metrics") or {}
    values = runtime.get("epoch_wall_clock_seconds")
    if values is None:
        values = record.get("epoch_wall_clock_seconds")
    if not isinstance(values, list) or len(values) <= 1:
        raise ValueError(
            "epoch_wall_clock_seconds must contain at least two epochs")
    return [float(v) for v in values]


def repeat_mean_epoch_seconds(record: dict) -> float:
    values = epoch_wall_clock(record)
    return mean(values[1:])


def load_cells(root: Path) -> dict[tuple[str, str, str, str], Cell]:
    grouped: dict[tuple[str, str, str, str], list[Repeat]] = defaultdict(list)
    missing_identity: list[str] = []

    for path, record in walk_runs(root):
        slug = task_slug(path)
        if slug is None:
            missing_identity.append(str(path))
            continue
        info = identify(record)
        variant = variant_slug(record)
        key = (slug, info.optimizer, variant, info.scheduler)
        grouped[key].append(
            Repeat(path=path, mean_epoch_seconds=repeat_mean_epoch_seconds(record)))

    cells = {
        key: Cell(
            task=key[0],
            optimizer=key[1],
            variant=key[2],
            scheduler=key[3],
            repeats=tuple(sorted(repeats, key=lambda r: r.path.name)),
        )
        for key, repeats in grouped.items()
    }

    if missing_identity:
        filenames = "\n".join(f"- {name}" for name in missing_identity)
        raise ValueError(
            f"Could not identify task slug for files:\n{filenames}")
    return cells


def require_cell(
    cells: dict[tuple[str, str, str, str], Cell],
    task: str,
    optimizer: str,
    variant: str,
    scheduler: str,
) -> Cell:
    key = (task, optimizer, variant, scheduler)
    if key not in cells:
        raise KeyError(
            f"Missing timing cell: task={task}, optimizer={optimizer}, variant={variant}, scheduler={scheduler}")
    cell = cells[key]
    if len(cell.repeats) != 3:
        names = ", ".join(r.path.name for r in cell.repeats)
        raise ValueError(
            f"Expected 3 timing repeats for {key}, found {len(cell.repeats)}: {names}")
    return cell


def overhead_percent(cell: Cell, baseline: Cell) -> float:
    return 100.0 * (cell.mean_seconds - baseline.mean_seconds) / baseline.mean_seconds


def bar_color(variant: str, scheduler: str) -> str:
    key = scheduler if variant == "scheduler" else VARIANT_TO_KEY[variant]
    if key is None:
        raise ValueError(f"No color key for {variant}")
    return PALETTE[key]


def bar_label(variant: str, scheduler: str) -> str:
    if variant == "scheduler":
        return LABEL[scheduler]
    key = VARIANT_TO_KEY[variant]
    if key is None:
        raise ValueError(f"No label key for {variant}")
    return LABEL[key]


def fmt_overhead(value: float) -> str:
    rounded = 0.0 if abs(value) < 0.05 else value
    return f"{rounded:+.1f}%"


def build_rows(cells: dict[tuple[str, str, str, str], Cell]):
    rows = []
    skipped = []
    for group_label, task, optimizer, scheduler in GROUPS:
        baseline = require_cell(cells, task, optimizer, "baseline", "none")
        scheduler_cell = require_cell(
            cells, task, optimizer, "baseline", scheduler)

        candidates = [("scheduler", scheduler_cell)]
        for variant in ("aees_lr_only", "aees_noise_only", "aees_both"):
            cell = require_cell(cells, task, optimizer, variant, scheduler)
            if task == "cifar_clean" and variant in SKIP_FOR_CIFAR:
                skipped.append((group_label, task, optimizer, variant,
                               scheduler, cell, overhead_percent(cell, baseline)))
            else:
                candidates.append((variant, cell))

        rows.append((group_label, task, optimizer,
                    scheduler, baseline, candidates))
    return rows, skipped


def draw(rows) -> object:
    fig, (ax,) = make_figure()
    full_offsets = {
        "scheduler": -0.27,
        "aees_lr_only": -0.09,
        "aees_noise_only": 0.09,
        "aees_both": 0.27,
    }
    height = 0.15
    group_gap = 1.0
    legend_handles = {}
    max_x = 0.0
    min_x = 0.0

    for group_index, (group_label, _task, _optimizer, scheduler, baseline, candidates) in enumerate(rows):
        y_center = (len(rows) - 1 - group_index) * group_gap
        offsets = full_offsets
        if len(candidates) == 2:
            offsets = {candidates[0][0]: -0.12, candidates[1][0]: 0.12}
        for variant, cell in candidates:
            value = overhead_percent(cell, baseline)
            max_x = max(max_x, value)
            min_x = min(min_x, value)
            color = bar_color(variant, scheduler)
            label = bar_label(variant, scheduler)
            bar = ax.barh(
                y_center + offsets[variant],
                value,
                height=height,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=label if label not in legend_handles else None,
            )
            legend_handles.setdefault(label, bar[0])
            if abs(value) < 1.0:
                label_x = 0.9
                ha = "left"
            else:
                label_x = value + (0.65 if value >= 0 else -0.65)
                ha = "left" if value >= 0 else "right"
            ax.text(label_x, y_center +
                    offsets[variant], fmt_overhead(value), va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks([(len(rows) - 1 - i) * group_gap for i in range(len(rows))])
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("Overhead vs. no-scheduler baseline (%)")
    ax.set_title("Wall-clock overhead by setting")
    ax.grid(axis="x", linestyle="--", alpha=0.55)
    ax.grid(axis="y", visible=False)

    x_left = min(-2.0, min_x - 2.0)
    x_right = max(42.0, max_x + 7.5)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(-0.65, (len(rows) - 1) * group_gap + 0.65)
    ax.legend(
        [legend_handles[label] for label in legend_handles],
        list(legend_handles),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )
    fig.tight_layout()
    return fig


def repeat_lines(cell: Cell) -> list[str]:
    lines = []
    for repeat in cell.repeats:
        lines.append(
            f"    {repeat.path.name}: mean_epoch_s = {repeat.mean_epoch_seconds:.4f}")
    lines.append(
        f"    avg across {len(cell.repeats)} repeats = {cell.mean_seconds:.4f} s"
        f" (std = {cell.std_seconds:.4f} s)"
    )
    return lines


def summary_lines(runs_root: Path, out_dir: Path, rows, skipped) -> list[str]:
    lines = [
        f"Resolved --runs-root: {runs_root.resolve()}",
        f"Output PDF: {(out_dir / (NAME + '.pdf')).resolve()}",
        f"Output PNG: {(out_dir / (NAME + '.png')).resolve()}",
        "",
        "Method: per-repeat mean epoch wall-clock seconds = mean(runtime_metrics.epoch_wall_clock_seconds[1:]),",
        "falling back to top-level epoch_wall_clock_seconds[1:] when runtime_metrics is absent.",
        "Epoch 0 is dropped to match the published table convention and avoid setup/JIT time.",
        "Overhead (%) = 100 * (cell_time - no_scheduler_baseline_time) / no_scheduler_baseline_time,",
        "within the same clean-setting (task, optimizer) cell.",
        "",
        "Flag: CIFAR AEES-Noise and AEES-Dual timing cells exist but are omitted from the plot per",
        "the published CIFAR configuration, which uses the LR axis only. Their values are listed below.",
    ]

    for group_label, task, optimizer, scheduler, baseline, candidates in rows:
        lines.extend(
            ["", f"--- {group_label} (task={task}, optim={optimizer}) ---"])
        lines.append("  no-scheduler baseline (variant=baseline, sched=none):")
        lines.extend(repeat_lines(baseline))
        for variant, cell in candidates:
            name = bar_label(variant, scheduler)
            lines.append(
                f"  {name} (variant={cell.variant}, sched={cell.scheduler}):")
            lines.extend(repeat_lines(cell))
            lines.append(
                f"    overhead vs. no-sched baseline = {fmt_overhead(overhead_percent(cell, baseline))}")

    lines.extend(["", "Skipped CIFAR noise/dual bars:"])
    for group_label, task, optimizer, variant, scheduler, cell, overhead in skipped:
        lines.append(
            f"  {group_label}: {variant}, sched={scheduler}, repeats={len(cell.repeats)}, "
            f"mean_epoch_s={cell.mean_seconds:.4f}, overhead={fmt_overhead(overhead)}"
        )

    cifar_sgd = [
        overhead_percent(cell, baseline)
        for group_label, _task, _optimizer, _scheduler, baseline, candidates in rows
        if group_label == "CIFAR-SGD+M"
        for _variant, cell in candidates
    ]
    nlp = [
        overhead_percent(cell, baseline)
        for group_label, _task, _optimizer, _scheduler, baseline, candidates in rows
        if group_label in {"SST-2-AdamW", "AG-News-AdamW"}
        for variant, cell in candidates
        if variant != "scheduler"
    ]
    lines.extend(
        [
            "",
            "Sanity-check probes:",
            f"  CIFAR-SGD+M overheads: {[round(v, 2) for v in cifar_sgd]} "
            f"(max |x| = {max(abs(v) for v in cifar_sgd):.2f}%)",
            f"  NLP AEES overheads: {[round(v, 2) for v in nlp]} "
            f"(min={min(nlp):.2f}%, max={max(nlp):.2f}%)",
        ]
    )
    return lines


def clear_outputs(out_dir: Path) -> None:
    for suffix in (".pdf", ".png", ".summary.txt", ".MISSING.md"):
        path = out_dir / f"{NAME}{suffix}"
        if path.exists():
            path.unlink()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    try:
        cells = load_cells(args.runs_root)
        rows, skipped = build_rows(cells)
        if not rows:
            raise ValueError("No rows to plot")
        clear_outputs(out_dir)
        fig = draw(rows)
        save_figure(fig, out_dir, NAME)
        write_summary(out_dir, NAME, summary_lines(
            args.runs_root, out_dir, rows, skipped))
    except Exception as exc:  # noqa: BLE001 - plotting task should emit a single MISSING file.
        clear_outputs(out_dir)
        write_missing(out_dir, NAME, str(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
