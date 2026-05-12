"""Plot AEES mean update-norm trajectories for the hypothesis-A diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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


NAME = "update_norm_trajectory"
EXPECTED_SEEDS = tuple(range(5))


@dataclass(frozen=True)
class RegimeSpec:
    slug: str
    title: str
    root_name: str
    filename_prefix: str
    task: str
    optimizer: str
    variant_key: str
    scheduler: str
    lr_active: bool
    noise_active: bool
    color_key: str
    display_variant: str


@dataclass(frozen=True)
class RegimeData:
    spec: RegimeSpec
    seeds: tuple[int, ...]
    steps: np.ndarray
    curves: np.ndarray
    mean: np.ndarray
    late_median: float
    per_seed_late_medians: tuple[float, ...]
    paths: tuple[Path, ...]


REGIMES = (
    RegimeSpec(
        slug="cifar_clean_aees_lr",
        title="CIFAR-100 clean / AdamW / AEES-LR",
        root_name="cifar_clean",
        filename_prefix="cifar_clean_aees_none_seed",
        task="cifar100",
        optimizer="adamw",
        variant_key="aees",
        scheduler="none",
        lr_active=True,
        noise_active=False,
        color_key="aees_lr",
        display_variant="AEES-LR",
    ),
    RegimeSpec(
        slug="sst2_aees_dual_wl",
        title="SST-2 / AdamW / AEES-Dual + warmup-linear",
        root_name="sst2",
        filename_prefix="sst2_aees_warmup_linear_5ep_small_ep200_trend_seed",
        task="sst2",
        optimizer="adamw",
        variant_key="warmup_linear_aees",
        scheduler="warmup_linear",
        lr_active=True,
        noise_active=True,
        color_key="warmup_linear_aees",
        display_variant="AEES-Dual + warmup-linear",
    ),
    RegimeSpec(
        slug="agnews_noisy_aees_dual_wl",
        title="AG News noisy / AdamW / AEES-Dual + warmup-linear",
        root_name="noisy_agnews",
        filename_prefix="agnews_noise20_aees_warmup_linear_5ep_coarse_highnoise_ep200_seed",
        task="agnews",
        optimizer="adamw",
        variant_key="warmup_linear_aees",
        scheduler="warmup_linear",
        lr_active=True,
        noise_active=True,
        color_key="warmup_linear_aees",
        display_variant="AEES-Dual + warmup-linear",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default=".",
        help="Repository root or results root containing cifar_clean/, sst2/, and noisy_agnews/.",
    )
    parser.add_argument("--out-dir", default="results/plots/controllers")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)

    try:
        _plot(runs_root, out_dir)
    except Exception as exc:  # noqa: BLE001 - plot tasks emit MISSING.md instead of partials.
        _remove_outputs(out_dir, keep_missing=True)
        write_missing(out_dir, NAME, str(exc))
        return 1
    return 0


def _plot(runs_root: Path, out_dir: Path) -> None:
    results_root = _results_root(runs_root)
    data = [_load_regime(results_root, spec) for spec in REGIMES]
    if len(data) != len(REGIMES):
        raise ValueError(f"expected {len(REGIMES)} regimes, found {len(data)}")

    fig, axes = plt.subplots(
        len(data),
        1,
        sharex=True,
        figsize=(6.5, 8.2),
        constrained_layout=False,
    )
    if len(data) == 1:
        axes = [axes]

    proxy_line = None
    mean_line = None
    for ax, regime in zip(axes, data, strict=True):
        color = PALETTE[regime.spec.color_key]
        for curve in regime.curves:
            (proxy_line,) = ax.plot(
                regime.steps,
                curve,
                color=color,
                linewidth=0.75,
                alpha=0.23,
                solid_capstyle="round",
                label="Seed trace" if proxy_line is None else None,
            )
        (mean_line,) = ax.plot(
            regime.steps,
            regime.mean,
            color=color,
            linewidth=2.15,
            label="Mean across seeds" if mean_line is None else None,
            zorder=4,
        )

        ax.axhline(
            regime.late_median,
            color="#333333",
            linestyle=(0, (4, 3)),
            linewidth=0.9,
            alpha=0.72,
            zorder=1,
        )
        ax.annotate(
            f"late median = {_format_norm(regime.late_median)}",
            xy=(0.985, regime.late_median),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#dddddd",
                "alpha": 0.88,
            },
            zorder=8,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Mean update norm")
        ax.set_title(regime.spec.title, loc="left", pad=4)
        ax.margins(x=0.015)
        ax.grid(True, which="major", axis="both")
        ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.28)

    axes[-1].set_xlabel("Training step")
    axes[-1].xaxis.set_major_formatter(FuncFormatter(_format_step_tick))
    fig.legend(
        handles=[proxy_line, mean_line],
        labels=["Seed trace", "Mean across seeds"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
        framealpha=0.9,
        facecolor="white",
    )
    fig.text(
        0.5,
        0.012,
        "Mean update norm is logged after AdamW rescaling; it is a step-size proxy, not a raw gradient norm.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#333333",
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.93,
                        bottom=0.095, hspace=0.32)

    _remove_outputs(out_dir, keep_missing=False)
    save_figure(fig, out_dir, NAME)
    write_summary(out_dir, NAME, _summary_lines(data))


def _results_root(runs_root: Path) -> Path:
    if (runs_root / "results").is_dir():
        return runs_root / "results"
    return runs_root


def _load_regime(results_root: Path, spec: RegimeSpec) -> RegimeData:
    root = results_root / spec.root_name
    if not root.exists():
        raise FileNotFoundError(
            f"{spec.slug}: runs root does not exist: {root}")

    selected: dict[int, tuple[Path, dict]] = {}
    rejected_target_like: list[str] = []
    for path, record in walk_runs(root):
        if not path.name.startswith(spec.filename_prefix):
            continue
        try:
            info = identify(record)
        except Exception as exc:  # noqa: BLE001 - include bad target-like records in diagnostics.
            rejected_target_like.append(f"{path}: could not identify ({exc})")
            continue
        if not _matches_spec(info, spec):
            rejected_target_like.append(
                f"{path}: identified as task={info.task}, optimizer={info.optimizer}, "
                f"variant={info.variant_key}, scheduler={info.scheduler}, "
                f"lr_active={info.lr_active}, noise_active={info.noise_active}"
            )
            continue
        if info.seed is None:
            raise ValueError(f"{spec.slug}: {path} has no seed")
        if info.seed in selected:
            previous, _ = selected[info.seed]
            raise ValueError(
                f"{spec.slug}: duplicate seed {info.seed}: {previous} and {path}"
            )
        _validate_episode_logs(path, record)
        selected[info.seed] = (path, record)

    missing = sorted(set(EXPECTED_SEEDS) - set(selected))
    extras = sorted(set(selected) - set(EXPECTED_SEEDS))
    if missing or extras or rejected_target_like:
        details = []
        if missing:
            details.append(f"missing seeds {missing}")
        if extras:
            details.append(f"unexpected seeds {extras}")
        if rejected_target_like:
            details.append("rejected target-like files: " +
                           "; ".join(rejected_target_like))
        raise ValueError(f"{spec.slug}: " + "; ".join(details))

    seeds = tuple(sorted(selected))
    paths = tuple(selected[seed][0] for seed in seeds)
    records = [selected[seed][1] for seed in seeds]
    steps = np.asarray((records[0]["episode_logs"] or {})[
                       "episode_end_steps"], dtype=float)
    curves = np.asarray(
        [(record["episode_logs"] or {})["mean_update_norms"]
         for record in records],
        dtype=float,
    )

    for seed, record in zip(seeds, records, strict=True):
        seed_steps = np.asarray((record["episode_logs"] or {})[
                                "episode_end_steps"], dtype=float)
        if seed_steps.shape != steps.shape or not np.array_equal(seed_steps, steps):
            raise ValueError(
                f"{spec.slug}: episode_end_steps differ for seed {seed}")

    late_start = int(np.floor(0.8 * curves.shape[1]))
    if late_start >= curves.shape[1]:
        raise ValueError(
            f"{spec.slug}: not enough episodes to compute last-20% median")
    late = curves[:, late_start:]
    per_seed_late = tuple(float(np.median(row)) for row in late)

    return RegimeData(
        spec=spec,
        seeds=seeds,
        steps=steps,
        curves=curves,
        mean=curves.mean(axis=0),
        late_median=float(np.median(late)),
        per_seed_late_medians=per_seed_late,
        paths=paths,
    )


def _matches_spec(info, spec: RegimeSpec) -> bool:
    return (
        info.task == spec.task
        and info.optimizer == spec.optimizer
        and info.variant_key == spec.variant_key
        and info.scheduler == spec.scheduler
        and info.lr_active == spec.lr_active
        and info.noise_active == spec.noise_active
        and info.is_aees
        and not info.is_fixed
    )


def _validate_episode_logs(path: Path, record: dict) -> None:
    episode_logs = record.get("episode_logs")
    if not isinstance(episode_logs, dict):
        raise ValueError(f"{path}: missing episode_logs")
    steps = episode_logs.get("episode_end_steps")
    norms = episode_logs.get("mean_update_norms")
    if not isinstance(steps, list) or not isinstance(norms, list):
        raise ValueError(
            f"{path}: missing episode_end_steps or mean_update_norms")
    if len(steps) < 2 or len(norms) < 2:
        raise ValueError(f"{path}: too few episode points")
    if len(steps) != len(norms):
        raise ValueError(
            f"{path}: episode_end_steps length {len(steps)} != mean_update_norms length {len(norms)}"
        )
    steps_array = np.asarray(steps, dtype=float)
    norms_array = np.asarray(norms, dtype=float)
    if not np.all(np.isfinite(steps_array)) or not np.all(np.isfinite(norms_array)):
        raise ValueError(
            f"{path}: non-finite episode step or update-norm values")
    if not np.all(np.diff(steps_array) > 0):
        raise ValueError(
            f"{path}: episode_end_steps must be strictly increasing")
    if not np.all(norms_array > 0):
        raise ValueError(
            f"{path}: mean_update_norms must be positive for log-scale plotting")


def _summary_lines(data: list[RegimeData]) -> list[str]:
    cifar = data[0].late_median
    nlp = [regime.late_median for regime in data[1:]]
    max_nlp = max(nlp)
    ratio_to_max_nlp = cifar / max_nlp
    ratio_to_sst2 = cifar / data[1].late_median
    ratio_to_agnews = cifar / data[2].late_median
    if ratio_to_max_nlp >= 10.0:
        flag = "supports A1: CIFAR late update norms are more than one order larger than both DistilBERT regimes."
    elif ratio_to_max_nlp <= 3.0:
        flag = "weakens A1: CIFAR late update norms are within a small factor of DistilBERT."
    else:
        flag = "mixed/moderate A1 evidence: CIFAR late update norms are larger, but not by a full order of magnitude."

    lines = [
        "Mean update-norm trajectory diagnostic",
        "Aggregation: raw per-seed episode trajectories plus mean across five seeds; no smoothing.",
        "Late-training median: pooled median over each run's final 20% of logged episodes.",
        "Important: mean_update_norms are post-optimizer update norms after AdamW rescaling, not raw gradient norms.",
        "",
    ]
    for regime in data:
        lines.extend(
            [
                f"{regime.spec.slug} ({regime.spec.title})",
                f"  variant: {regime.spec.display_variant}",
                f"  seeds: {','.join(str(seed) for seed in regime.seeds)}",
                f"  episodes: {regime.curves.shape[1]}, step range: {int(regime.steps[0])}-{int(regime.steps[-1])}",
                f"  late pooled median: {regime.late_median:.8g}",
                "  per-seed late medians: "
                + ", ".join(f"seed{seed}={value:.8g}" for seed, value in zip(
                    regime.seeds, regime.per_seed_late_medians, strict=True)),
                "  files: " + ", ".join(str(path) for path in regime.paths),
                "",
            ]
        )
    lines.extend(
        [
            f"CIFAR/SST-2 late-median ratio: {ratio_to_sst2:.2f}x",
            f"CIFAR/AG-News-noisy late-median ratio: {ratio_to_agnews:.2f}x",
            f"CIFAR/max-DistilBERT late-median ratio: {ratio_to_max_nlp:.2f}x",
            f"Hypothesis flag: {flag}",
        ]
    )
    return lines


def _format_step_tick(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def _format_norm(value: float) -> str:
    if value < 0.01:
        return f"{value:.2e}"
    if value < 1.0:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _remove_outputs(out_dir: Path, *, keep_missing: bool) -> None:
    for suffix in (".pdf", ".png", ".summary.txt"):
        path = out_dir / f"{NAME}{suffix}"
        if path.exists():
            path.unlink()
    missing = out_dir / f"{NAME}.MISSING.md"
    if missing.exists() and not keep_missing:
        missing.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
