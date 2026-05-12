"""Plot AEES controller arm-selection trajectories."""

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


EXPECTED_SEEDS = tuple(range(5))


@dataclass(frozen=True)
class SettingSpec:
    key: str
    name: str
    setting_dir: str
    title: str
    axis_label: str
    controller_key: str
    selected_key: str
    base_palette_key: str
    expected_filename: str


SETTINGS = {
    "cifar_sym40": SettingSpec(
        key="cifar_sym40",
        name="arm_selection_cifar_sym40",
        setting_dir="cifar_noisy",
        title="Noisy CIFAR-100 symmetric 40%: LR-multiplier controller",
        axis_label="LR multiplier arm",
        controller_key="lr_controller_logs",
        selected_key="selected_lr_values",
        base_palette_key="aees_lr",
        expected_filename="adamw_aees_ep200_lr05102",
    ),
    "agnews_noisy": SettingSpec(
        key="agnews_noisy",
        name="arm_selection_agnews_noisy",
        setting_dir="noisy_agnews",
        title="Noisy AG News: gradient-noise controller",
        axis_label="Gradient-noise arm",
        controller_key="noise_controller_logs",
        selected_key="selected_noise_values",
        base_palette_key="aees_noise",
        expected_filename="agnews_noise20_aees_warmup_linear_5ep_coarse_highnoise_ep200_seed",
    ),
}


@dataclass(frozen=True)
class ArmSelectionData:
    spec: SettingSpec
    arms: tuple[float, ...]
    seeds: tuple[int, ...]
    episode_count: int
    proportions: np.ndarray
    display_window: int
    epoch_values: np.ndarray
    records: tuple[dict, ...]
    paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setting",
        choices=("cifar_sym40", "agnews_noisy", "all"),
        default="all",
        help="Which controller trace to render.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/plots/controllers"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = tuple(SETTINGS) if args.setting == "all" else (args.setting,)
    failures = 0

    for key in selected:
        spec = SETTINGS[key]
        try:
            data = load_arm_selection(args.runs_root, spec)
            fig = build_figure(data)
        except Exception as exc:  # noqa: BLE001 - plot scripts report via MISSING files.
            failures += 1
            _remove_outputs(args.out_dir, spec.name, keep_missing=True)
            write_missing(args.out_dir, spec.name, str(exc))
            continue

        _remove_outputs(args.out_dir, spec.name, keep_missing=False)
        save_figure(fig, args.out_dir, spec.name)
        write_summary(args.out_dir, spec.name, summary_lines(data))

    return 1 if failures else 0


def load_arm_selection(runs_root: Path, spec: SettingSpec) -> ArmSelectionData:
    root = _setting_root(runs_root, spec.setting_dir)
    if not root.exists():
        raise FileNotFoundError(f"runs root does not exist: {root}")

    grouped = load_set(root, lambda path, record,
                       info: _matches_setting(path, record, info, spec))
    seed_runs = _flatten_expected_seed_runs(grouped, spec)

    records = tuple(seed_runs[seed][1] for seed in EXPECTED_SEEDS)
    paths = tuple(seed_runs[seed][0] for seed in EXPECTED_SEEDS)
    arms = _canonical_arms(records, spec)
    selected = _selected_values(records, spec)
    episode_count = min(len(values) for values in selected)
    if episode_count <= 1:
        raise ValueError(
            f"{spec.name}: need at least two selected-arm episodes")

    selected = [values[:episode_count] for values in selected]
    proportions = _arm_proportions(selected, arms)
    epoch_values = _episode_epoch_values(records[0], episode_count)

    return ArmSelectionData(
        spec=spec,
        arms=tuple(arms),
        seeds=EXPECTED_SEEDS,
        episode_count=episode_count,
        proportions=proportions,
        display_window=_display_window(episode_count),
        epoch_values=epoch_values,
        records=records,
        paths=paths,
    )


def build_figure(data: ArmSelectionData):
    fig, axes = make_figure()
    ax = axes[0]

    x = np.arange(1, data.episode_count + 1)
    colors = arm_palette(data.spec.base_palette_key, len(data.arms))
    labels = [_arm_label(data.spec, arm) for arm in data.arms]
    displayed = _centered_rolling_mean(data.proportions, data.display_window)

    ax.stackplot(
        x,
        displayed.T,
        labels=labels,
        colors=colors,
        alpha=0.94,
        linewidth=0.3,
        edgecolor="white",
    )

    ax.set_xlim(1, data.episode_count)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Proportion of seeds (rolling mean)")
    ax.set_title(data.spec.title)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(
        title=data.spec.axis_label,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    _add_epoch_axis(ax, data)

    fig.subplots_adjust(right=0.78, top=0.82)
    return fig


def summary_lines(data: ArmSelectionData) -> list[str]:
    lines = [
        data.spec.title,
        "Aggregation: raw per-episode arm proportions across five seeds.",
        f"Displayed stack: centered rolling mean with window={data.display_window} episodes; summaries below use raw proportions.",
        f"Seeds: {list(data.seeds)}.",
        f"Episode count: truncated to shortest seed ({data.episode_count} episodes).",
        f"Canonical arm order: {', '.join(_arm_label(data.spec, arm) for arm in data.arms)}.",
        "",
        "Overall arm usage:",
    ]

    overall = data.proportions.mean(axis=0)
    early, middle, late = _third_window_means(data.proportions)
    ranges = data.proportions.max(axis=0) - data.proportions.min(axis=0)

    for idx, arm in enumerate(data.arms):
        label = _arm_label(data.spec, arm)
        lines.append(
            f"{label}: overall={overall[idx]:.3f}; "
            f"early={early[idx]:.3f}; middle={middle[idx]:.3f}; late={late[idx]:.3f}; "
            f"episode-range={ranges[idx]:.3f}"
        )

    first_peak = _arm_label(data.spec, data.arms[int(np.argmax(early))])
    late_peak = _arm_label(data.spec, data.arms[int(np.argmax(late))])
    lines.extend(
        [
            "",
            f"Dominant early-third arm: {first_peak}.",
            f"Dominant late-third arm: {late_peak}.",
            *_interpretation_flags(data, early, late, ranges),
            "",
            "Input files:",
            *(str(path) for path in data.paths),
        ]
    )
    return lines


def _matches_setting(path: Path, record: dict, info, spec: SettingSpec) -> bool:
    if record.get("controller_logs") is None:
        return False
    if info.seed not in EXPECTED_SEEDS or not info.is_aees:
        return False
    if spec.key == "cifar_sym40":
        return (
            info.task == "cifar100"
            and info.optimizer == "adamw"
            and info.noise_setting == "sym40"
            and info.scheduler == "none"
            and info.lr_active
            and not info.noise_active
            and path.name == spec.expected_filename
        )
    if spec.key == "agnews_noisy":
        return (
            info.task == "agnews"
            and info.optimizer == "adamw"
            and info.noise_setting == "sym20"
            and info.scheduler == "warmup_linear"
            and info.lr_active
            and info.noise_active
            and path.name
            == f"{spec.expected_filename}{info.seed}.json"
        )
    raise ValueError(f"unknown setting: {spec.key}")


def _flatten_expected_seed_runs(
    grouped: dict[str, dict[int, list[tuple[Path, dict, object]]]], spec: SettingSpec
) -> dict[int, tuple[Path, dict, object]]:
    runs: dict[int, list[tuple[Path, dict, object]]] = {}
    for _variant_key, by_seed in grouped.items():
        for seed, entries in by_seed.items():
            runs.setdefault(seed, []).extend(entries)

    missing = [seed for seed in EXPECTED_SEEDS if seed not in runs]
    duplicates = {seed: entries for seed,
                  entries in runs.items() if len(entries) != 1}
    if missing or duplicates:
        parts: list[str] = []
        if missing:
            parts.append(f"missing seeds {missing}")
        for seed, entries in sorted(duplicates.items()):
            paths = ", ".join(str(path) for path, _record, _info in entries)
            parts.append(f"seed {seed} has {len(entries)} runs: {paths}")
        raise ValueError(
            f"{spec.name}: expected exactly one run per seed; " + "; ".join(parts))

    return {seed: runs[seed][0] for seed in EXPECTED_SEEDS}


def _canonical_arms(records: tuple[dict, ...], spec: SettingSpec) -> list[float]:
    arm_lists: list[list[float]] = []
    for record in records:
        controller = (record.get("controller_logs") or {}
                      ).get(spec.controller_key) or {}
        arms = controller.get("arm_values")
        if not isinstance(arms, list) or not arms:
            raise ValueError(
                f"{spec.name}: missing {spec.controller_key}.arm_values")
        arm_lists.append(sorted(float(value) for value in arms))

    first = arm_lists[0]
    for arms in arm_lists[1:]:
        if len(arms) != len(first) or any(not np.isclose(a, b) for a, b in zip(arms, first)):
            raise ValueError(
                f"{spec.name}: inconsistent controller arm values {arm_lists}")
    return first


def _selected_values(records: tuple[dict, ...], spec: SettingSpec) -> list[np.ndarray]:
    selected: list[np.ndarray] = []
    for record in records:
        episode_logs = record.get("episode_logs") or {}
        values = episode_logs.get(spec.selected_key)
        end_steps = episode_logs.get("episode_end_steps")
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"{spec.name}: missing episode_logs.{spec.selected_key}")
        if not isinstance(end_steps, list) or len(end_steps) < len(values):
            raise ValueError(
                f"{spec.name}: missing or short episode_end_steps")
        selected.append(np.asarray(values, dtype=float))
    return selected


def _arm_proportions(selected: list[np.ndarray], arms: list[float]) -> np.ndarray:
    episode_count = len(selected[0])
    proportions = np.zeros((episode_count, len(arms)), dtype=float)
    for episode_idx in range(episode_count):
        choices = np.asarray([seed_values[episode_idx]
                             for seed_values in selected], dtype=float)
        for arm_idx, arm in enumerate(arms):
            proportions[episode_idx, arm_idx] = float(
                np.isclose(choices, arm).mean())
    return proportions


def _display_window(episode_count: int) -> int:
    window = max(5, int(round(episode_count * 0.05)))
    return window if window % 2 else window + 1


def _centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    half = window // 2
    smoothed = np.zeros_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        smoothed[index] = values[start:end].mean(axis=0)
    row_sums = smoothed.sum(axis=1, keepdims=True)
    return np.divide(smoothed, row_sums, out=np.zeros_like(smoothed), where=row_sums > 0)


def _episode_epoch_values(record: dict, episode_count: int) -> np.ndarray:
    episode_logs = record.get("episode_logs") or {}
    end_steps = np.asarray(episode_logs.get("episode_end_steps")[
                           :episode_count], dtype=float)
    total_steps = float(record.get("total_steps") or 0)
    total_epochs = float(record.get("total_epochs") or 0)
    if total_steps <= 0 or total_epochs <= 0:
        raise ValueError(
            "record must define positive total_steps and total_epochs")
    return end_steps * total_epochs / total_steps


def _add_epoch_axis(ax, data: ArmSelectionData) -> None:
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    x = np.arange(1, data.episode_count + 1, dtype=float)
    max_epoch = float(data.epoch_values[-1])
    if max_epoch > 20:
        target_epochs = np.array([1, 50, 100, 150, 200], dtype=float)
        target_epochs = target_epochs[target_epochs <= max_epoch + 0.5]
    else:
        target_epochs = np.arange(1, int(round(max_epoch)) + 1, dtype=float)
    tick_positions = np.interp(target_epochs, data.epoch_values, x)
    top.set_xticks(tick_positions)
    top.set_xticklabels([str(int(round(value))) for value in target_epochs])
    top.set_xlabel("Epoch")
    top.grid(False)
    top.tick_params(axis="x", direction="out", pad=3)


def _third_window_means(proportions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = np.array_split(proportions, 3, axis=0)
    # type: ignore[return-value]
    return tuple(split.mean(axis=0) for split in splits)


def _interpretation_flags(
    data: ArmSelectionData, early: np.ndarray, late: np.ndarray, ranges: np.ndarray
) -> list[str]:
    lines: list[str] = ["", "Interpretation flags:"]
    if _looks_time_uniform(early, late, ranges):
        lines.append(
            "FLAG: Arm proportions are close to time-uniform at the coarse early-vs-late level; "
            "the appendix prose should avoid claiming strong controller settling."
        )
    else:
        lines.append(
            "Arm proportions change over training at the coarse early-vs-late level; "
            "the controller does not look uniform."
        )

    if data.spec.key == "agnews_noisy":
        low_idx = 0
        high_idx = len(data.arms) - 1
        low_shift = late[low_idx] - early[low_idx]
        high_shift = late[high_idx] - early[high_idx]
        threshold = 0.10
        if high_shift <= -threshold and low_shift >= threshold:
            lines.append(
                "FLAG: clear early-high-sigma / late-low-sigma pattern on noisy AG News."
            )
        elif high_shift >= threshold and low_shift <= -threshold:
            lines.append(
                "FLAG: clear early-low-sigma / late-high-sigma pattern on noisy AG News."
            )
        else:
            lines.append(
                "FLAG: no clear early/late sigma-axis reversal or settling pattern on noisy AG News."
            )
    return lines


def _looks_time_uniform(early: np.ndarray, late: np.ndarray, ranges: np.ndarray) -> bool:
    return float(np.max(np.abs(late - early))) < 0.10 and float(np.max(ranges)) <= 1.0


def _arm_label(spec: SettingSpec, arm: float) -> str:
    if spec.key == "cifar_sym40":
        return f"m={arm:.1f}"
    if np.isclose(arm, 0.0):
        return "sigma=0".replace("sigma", "σ")
    return f"σ={arm:g}"


def _setting_root(runs_root: Path, setting_dir: str) -> Path:
    if runs_root.name == setting_dir:
        return runs_root
    return runs_root / setting_dir


def _remove_outputs(out_dir: Path, name: str, *, keep_missing: bool) -> None:
    suffixes = [".pdf", ".png", ".summary.txt"]
    if not keep_missing:
        suffixes.append(".MISSING.md")
    for suffix in suffixes:
        path = out_dir / f"{name}{suffix}"
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
