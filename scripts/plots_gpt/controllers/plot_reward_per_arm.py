"""Plot AEES per-arm episode reward distributions."""

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
class AxisSpec:
    key: str
    title: str
    controller_key: str
    selected_key: str
    base_palette_key: str
    xlabel: str


@dataclass(frozen=True)
class SettingSpec:
    key: str
    name: str
    setting_dir: str
    title: str
    expected_filename: str
    axes: tuple[AxisSpec, ...]


SETTINGS = {
    "cifar_sym40": SettingSpec(
        key="cifar_sym40",
        name="reward_per_arm_cifar_sym40",
        setting_dir="cifar_noisy",
        title="Noisy CIFAR-100 symmetric 40%: reward by LR arm",
        expected_filename="adamw_aees_ep200_lr05102",
        axes=(
            AxisSpec(
                key="lr",
                title="LR multiplier",
                controller_key="lr_controller_logs",
                selected_key="selected_lr_values",
                base_palette_key="aees_lr",
                xlabel="Selected LR multiplier arm",
            ),
        ),
    ),
    "agnews_noisy": SettingSpec(
        key="agnews_noisy",
        name="reward_per_arm_agnews_noisy",
        setting_dir="noisy_agnews",
        title="Noisy AG News: reward by selected arm",
        expected_filename="agnews_noise20_aees_warmup_linear_5ep_coarse_highnoise_ep200_seed",
        axes=(
            AxisSpec(
                key="lr",
                title="LR multiplier axis",
                controller_key="lr_controller_logs",
                selected_key="selected_lr_values",
                base_palette_key="aees_lr",
                xlabel="Selected LR multiplier arm",
            ),
            AxisSpec(
                key="noise",
                title="Gradient-noise axis",
                controller_key="noise_controller_logs",
                selected_key="selected_noise_values",
                base_palette_key="aees_noise",
                xlabel="Selected σ arm",
            ),
        ),
    ),
}


@dataclass(frozen=True)
class AxisRewardData:
    spec: AxisSpec
    arms: tuple[float, ...]
    rewards_by_arm: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class RewardPlotData:
    spec: SettingSpec
    seeds: tuple[int, ...]
    axes: tuple[AxisRewardData, ...]
    paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setting",
        choices=("cifar_sym40", "agnews_noisy", "all"),
        default="all",
        help="Which per-arm reward distribution to render.",
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
            data = load_reward_data(args.runs_root, spec)
            fig = build_figure(data)
            _remove_outputs(args.out_dir, spec.name, keep_missing=False)
            save_figure(fig, args.out_dir, spec.name)
            write_summary(args.out_dir, spec.name, summary_lines(data))
        except Exception as exc:  # noqa: BLE001 - plot scripts report via MISSING files.
            failures += 1
            _remove_outputs(args.out_dir, spec.name, keep_missing=True)
            write_missing(args.out_dir, spec.name, str(exc))

    return 1 if failures else 0


def load_reward_data(runs_root: Path, spec: SettingSpec) -> RewardPlotData:
    root = _setting_root(runs_root, spec.setting_dir)
    if not root.exists():
        raise FileNotFoundError(f"runs root does not exist: {root}")

    grouped = load_set(root, lambda path, record,
                       info: _matches_setting(path, record, info, spec))
    seed_runs = _flatten_expected_seed_runs(grouped, spec)
    records = tuple(seed_runs[seed][1] for seed in EXPECTED_SEEDS)
    paths = tuple(seed_runs[seed][0] for seed in EXPECTED_SEEDS)

    axes = tuple(_collect_axis_rewards(records, axis_spec, spec)
                 for axis_spec in spec.axes)
    return RewardPlotData(spec=spec, seeds=EXPECTED_SEEDS, axes=axes, paths=paths)


def build_figure(data: RewardPlotData):
    fig, axes = make_figure(n_panels=len(data.axes), panel_height=3.5)

    all_rewards = np.concatenate(
        [values for axis_data in data.axes for values in axis_data.rewards_by_arm]
    )
    lower, upper = _reward_limits(all_rewards)

    for panel_idx, (ax, axis_data) in enumerate(zip(axes, data.axes)):
        _draw_axis(ax, axis_data)
        ax.set_ylim(lower, upper)
        ax.set_title(axis_data.spec.title)
        ax.set_xlabel(axis_data.spec.xlabel)
        if panel_idx == 0:
            ax.set_ylabel("Clipped log-EMA episode reward $r_e$")
        else:
            ax.set_ylabel("")

    if len(data.axes) == 1:
        axes[0].set_title(data.spec.title)
    fig.subplots_adjust(wspace=0.12 if len(data.axes) > 1 else 0.20)
    return fig


def summary_lines(data: RewardPlotData) -> list[str]:
    lines = [
        data.spec.title,
        "Aggregation: raw completed episode rewards pooled across five seeds; no smoothing.",
        "Reward source: episode_logs.episode_rewards paired with selected arm values.",
        f"Seeds: {list(data.seeds)}.",
        "",
    ]

    for axis_data in data.axes:
        lines.append(f"{axis_data.spec.title}:")
        medians = []
        positive_rates = []
        counts = []
        for arm, rewards in zip(axis_data.arms, axis_data.rewards_by_arm):
            median = float(np.median(rewards))
            positive_rate = float(np.mean(rewards > 0.0))
            medians.append(median)
            positive_rates.append(positive_rate)
            counts.append(len(rewards))
            lines.append(
                f"{_arm_label(axis_data.spec, arm)}: n={len(rewards)}; "
                f"median={median:.4f}; mean={float(np.mean(rewards)):.4f}; "
                f"positive-rate={positive_rate:.3f}; "
                f"q25={float(np.quantile(rewards, 0.25)):.4f}; "
                f"q75={float(np.quantile(rewards, 0.75)):.4f}"
            )

        preferred_idx = int(np.argmax(medians))
        lines.append(
            f"Highest median reward arm: {_arm_label(axis_data.spec, axis_data.arms[preferred_idx])} "
            f"(median={medians[preferred_idx]:.4f})."
        )
        lines.extend(_axis_flags(data.spec, axis_data,
                     medians, positive_rates, counts))
        lines.append("")

    lines.extend(["Input files:", *(str(path) for path in data.paths)])
    return lines


def _draw_axis(ax, axis_data: AxisRewardData) -> None:
    positions = np.arange(1, len(axis_data.arms) + 1, dtype=float)
    colors = arm_palette(axis_data.spec.base_palette_key, len(axis_data.arms))
    violins = ax.violinplot(
        axis_data.rewards_by_arm,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.86)
        body.set_linewidth(0.7)

    medians = [float(np.median(values)) for values in axis_data.rewards_by_arm]
    for x_pos, median, color in zip(positions, medians, colors):
        ax.hlines(
            median,
            x_pos - 0.24,
            x_pos + 0.24,
            color="#111111",
            linewidth=1.6,
            zorder=4,
        )
        ax.plot(
            x_pos,
            median,
            marker="o",
            color=color,
            markeredgecolor="#111111",
            markeredgewidth=0.5,
            markersize=3.2,
            zorder=5,
        )

    ax.axhline(0.0, color="#9a9a9a", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels([_arm_label(axis_data.spec, arm)
                       for arm in axis_data.arms])
    ax.set_xlim(0.45, len(axis_data.arms) + 0.55)
    ax.grid(axis="x", visible=False)


def _collect_axis_rewards(
    records: tuple[dict, ...], axis_spec: AxisSpec, setting_spec: SettingSpec
) -> AxisRewardData:
    arms = _canonical_arms(records, axis_spec, setting_spec)
    rewards_by_arm: list[list[float]] = [[] for _arm in arms]

    for record in records:
        episode_logs = record.get("episode_logs") or {}
        rewards = episode_logs.get("episode_rewards")
        selections = episode_logs.get(axis_spec.selected_key)
        if not isinstance(rewards, list) or not rewards:
            raise ValueError(
                f"{setting_spec.name}: missing episode_logs.episode_rewards")
        if not isinstance(selections, list) or not selections:
            raise ValueError(
                f"{setting_spec.name}: missing episode_logs.{axis_spec.selected_key}")
        if len(rewards) != len(selections):
            raise ValueError(
                f"{setting_spec.name}: reward/selection length mismatch for {axis_spec.key} "
                f"({len(rewards)} rewards vs {len(selections)} selections)"
            )

        for reward_raw, selected_raw in zip(rewards, selections):
            reward = float(reward_raw)
            selected = float(selected_raw)
            if not np.isfinite(reward) or not np.isfinite(selected):
                raise ValueError(
                    f"{setting_spec.name}: non-finite reward or selected arm")
            arm_idx = _arm_index(selected, arms)
            rewards_by_arm[arm_idx].append(reward)

    arrays = tuple(np.asarray(values, dtype=float)
                   for values in rewards_by_arm)
    empty = [arm for arm, values in zip(arms, arrays) if len(values) == 0]
    if empty:
        labels = ", ".join(_arm_label(axis_spec, arm) for arm in empty)
        raise ValueError(
            f"{setting_spec.name}: no selected episodes for arm(s): {labels}")
    return AxisRewardData(
        spec=axis_spec,
        arms=tuple(arms),
        rewards_by_arm=arrays,
    )


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
            and path.name == f"{spec.expected_filename}{info.seed}.json"
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


def _canonical_arms(
    records: tuple[dict, ...], axis_spec: AxisSpec, setting_spec: SettingSpec
) -> list[float]:
    arm_lists: list[list[float]] = []
    for record in records:
        controller = (record.get("controller_logs") or {}).get(
            axis_spec.controller_key) or {}
        arms = controller.get("arm_values")
        if not isinstance(arms, list) or not arms:
            raise ValueError(
                f"{setting_spec.name}: missing {axis_spec.controller_key}.arm_values"
            )
        arm_lists.append(sorted(float(value) for value in arms))

    first = arm_lists[0]
    for arms in arm_lists[1:]:
        if len(arms) != len(first) or any(not np.isclose(a, b) for a, b in zip(arms, first)):
            raise ValueError(
                f"{setting_spec.name}: inconsistent controller arm values {arm_lists}")
    return first


def _axis_flags(
    setting_spec: SettingSpec,
    axis_data: AxisRewardData,
    medians: list[float],
    positive_rates: list[float],
    counts: list[int],
) -> list[str]:
    lines: list[str] = []
    if max(medians) <= 0.0:
        lines.append(
            "FLAG: all arm medians are non-positive; prose should avoid saying this axis "
            "usually improves reward."
        )
    if min(medians) < 0.0 < max(medians):
        lines.append("FLAG: arms split across the r_e=0 boundary.")
    if min(counts) < 10:
        lines.append(
            "FLAG: at least one arm has fewer than 10 selected episodes.")

    if setting_spec.key == "agnews_noisy" and axis_data.spec.key == "noise":
        nonzero_indices = [idx for idx, arm in enumerate(
            axis_data.arms) if not np.isclose(arm, 0.0)]
        if nonzero_indices:
            best_nonzero = max(nonzero_indices, key=lambda idx: medians[idx])
            zero_idx = _arm_index(0.0, list(axis_data.arms))
            gap = medians[best_nonzero] - medians[zero_idx]
            if gap > 0.02 and positive_rates[best_nonzero] >= positive_rates[zero_idx]:
                lines.append(
                    "FLAG: noisy AG News sigma axis prefers a non-zero arm by median reward."
                )
            else:
                lines.append(
                    "FLAG: noisy AG News sigma-axis non-zero preference is weak or absent."
                )
    return lines


def _reward_limits(rewards: np.ndarray) -> tuple[float, float]:
    lower = float(np.min(rewards))
    upper = float(np.max(rewards))
    if np.isclose(lower, upper):
        return lower - 0.1, upper + 0.1
    pad = 0.08 * (upper - lower)
    return lower - pad, upper + pad


def _arm_index(value: float, arms: list[float]) -> int:
    for idx, arm in enumerate(arms):
        if np.isclose(value, arm):
            return idx
    raise ValueError(
        f"selected arm {value:g} is not in controller arm values {arms}")


def _arm_label(axis_spec: AxisSpec, arm: float) -> str:
    if axis_spec.key == "lr":
        return f"m={arm:.1f}"
    if np.isclose(arm, 0.0):
        return "σ=0"
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
