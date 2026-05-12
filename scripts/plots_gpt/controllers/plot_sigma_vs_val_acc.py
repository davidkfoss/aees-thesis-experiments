"""Selected sigma versus validation accuracy for noisy AG News AEES runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import pathlib

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


NAME = "sigma_vs_val_acc_agnews_noisy"
DEFAULT_RUNS_ROOT = pathlib.Path("results/noisy_agnews")
DEFAULT_OUT_DIR = pathlib.Path("results/plots/controllers")


@dataclass(frozen=True)
class SigmaPoint:
    """One observed seed/epoch pair for the sigma diagnostic."""

    mean_sigma: float
    modal_sigma: float
    val_accuracy_pct: float
    variant_label: str
    scheduler: str
    seed: int
    epoch: int
    n_episodes: int
    path: pathlib.Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=pathlib.Path,
                        default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--config-filter",
        default=None,
        help="Optional substring filter matched against result paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        points, run_counts, candidates = collect_points(
            args.runs_root, args.config_filter)
        if not points:
            raise ValueError(
                "No noisy AG News AEES runs with adaptive noise were found.")

        fig = draw(points, candidates)
        pdf, png = save_figure(fig, args.out_dir, NAME)
        summary = build_summary(
            args.runs_root, args.out_dir, points, run_counts, candidates, pdf, png)
        write_summary(args.out_dir, NAME, summary)
    except Exception as exc:  # noqa: BLE001 - figure tasks should emit MISSING instead of partial output.
        write_missing(args.out_dir, NAME, str(exc))
        raise


def collect_points(
    runs_root: pathlib.Path, config_filter: str | None = None
) -> tuple[list[SigmaPoint], Counter[str], list[float]]:
    """Load noisy AG News AEES-Noise/AEES-Dual runs and aggregate by epoch."""

    points: list[SigmaPoint] = []
    run_counts: Counter[str] = Counter()
    candidate_union: set[float] = set()
    problems: list[str] = []

    for path, record in walk_runs(runs_root):
        if config_filter and config_filter not in str(path):
            continue
        info = identify(record)
        config = record.get("config") or {}
        noise_candidates = [float(v) for v in (
            config.get("noise_candidates") or [])]

        if info.task != "agnews" or not info.is_aees or len(noise_candidates) <= 1:
            continue
        if not info.noise_active:
            continue

        episode_logs = record.get("episode_logs") or {}
        selected = episode_logs.get("selected_noise_values") or []
        end_steps = episode_logs.get("episode_end_steps") or []
        val_accuracies = record.get("val_accuracies") or []
        total_epochs = int(record.get("total_epochs") or 0)

        if not selected or not end_steps:
            problems.append(
                f"{path}: missing selected_noise_values or episode_end_steps")
            continue
        if len(selected) != len(end_steps):
            problems.append(
                f"{path}: selected_noise_values length {len(selected)} != "
                f"episode_end_steps length {len(end_steps)}"
            )
            continue
        if len(val_accuracies) != total_epochs:
            problems.append(
                f"{path}: val_accuracies length {len(val_accuracies)} != total_epochs {total_epochs}"
            )
            continue

        label = axis_label(info.lr_active, info.noise_active, info.scheduler)
        run_counts[label] += 1
        candidate_union.update(noise_candidates)

        for epoch_index in range(total_epochs):
            lo, hi = epoch_step_range(record, epoch_index)
            epoch_sigmas = [
                float(selected[i])
                for i, end_step in enumerate(end_steps)
                if lo <= float(end_step) < hi
            ]
            if not epoch_sigmas:
                problems.append(
                    f"{path}: no completed noise episodes in epoch {epoch_index + 1}")
                continue
            points.append(
                SigmaPoint(
                    mean_sigma=float(np.mean(epoch_sigmas)),
                    modal_sigma=modal_value(epoch_sigmas),
                    val_accuracy_pct=float(
                        val_accuracies[epoch_index]) * 100.0,
                    variant_label=label,
                    scheduler=info.scheduler,
                    seed=int(info.seed if info.seed is not None else -1),
                    epoch=epoch_index + 1,
                    n_episodes=len(epoch_sigmas),
                    path=path,
                )
            )

    if problems:
        raise ValueError(
            "Required fields were missing or inconsistent:\n" + "\n".join(problems[:20]))
    return points, run_counts, sorted(candidate_union)


def axis_label(lr_active: bool, noise_active: bool, scheduler: str) -> str:
    if lr_active and noise_active:
        base = LABEL["aees_dual"]
    elif noise_active:
        base = LABEL["aees_noise"]
    else:
        base = LABEL["aees_lr"]
    if scheduler != "none":
        return f"{base} + {LABEL.get(scheduler, scheduler)}"
    return base


def modal_value(values: list[float]) -> float:
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def draw(points: list[SigmaPoint], candidates: list[float]):
    xs = np.array([p.mean_sigma for p in points], dtype=float)
    ys = np.array([p.val_accuracy_pct for p in points], dtype=float)
    use_strip = should_use_strip(points)

    fig, (ax,) = make_figure()
    if use_strip:
        draw_strip(ax, points, candidates)
    else:
        hb = ax.hexbin(
            xs,
            ys,
            gridsize=20,
            cmap="viridis",
            mincnt=1,
            linewidths=0.12,
            edgecolors="white",
            alpha=0.95,
        )
        cbar = fig.colorbar(hb, ax=ax, pad=0.02)
        cbar.set_label("Count")
        centers, means = sigma_bin_means(xs, ys)
        if len(centers) >= 2:
            ax.plot(
                centers,
                means,
                color="#111111",
                linewidth=1.1,
                marker="o",
                markersize=3.2,
                label="Per-bin mean",
                zorder=4,
            )
            ax.legend(loc="lower right")
        ax.set_xlabel("Mean selected sigma within epoch")

    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Noisy AG News: selected sigma vs. validation accuracy")
    ax.grid(True, linestyle="--", alpha=0.45)
    pad_y = max(0.15, 0.08 * (float(np.max(ys)) - float(np.min(ys))))
    ax.set_ylim(float(np.min(ys)) - pad_y, float(np.max(ys)) + pad_y)
    return fig


def should_use_strip(points: list[SigmaPoint]) -> bool:
    """Use the TASKS.md fallback when the hexbin would be mostly singleton cells."""

    xs = np.array([p.mean_sigma for p in points], dtype=float)
    ys = np.array([p.val_accuracy_pct for p in points], dtype=float)
    if len(points) < 40:
        return True
    if float(np.max(xs)) == float(np.min(xs)) or float(np.max(ys)) == float(np.min(ys)):
        return True
    counts, _, _ = np.histogram2d(xs, ys, bins=20)
    nonempty = int(np.count_nonzero(counts))
    return nonempty / max(len(points), 1) > 0.75


def draw_strip(ax, points: list[SigmaPoint], candidates: list[float]) -> None:
    candidate_positions = {sigma: i for i, sigma in enumerate(candidates)}
    grouped: dict[float, list[float]] = defaultdict(list)
    for point in points:
        grouped[point.modal_sigma].append(point.val_accuracy_pct)

    for point in points:
        idx = candidate_positions[point.modal_sigma]
        jitter = deterministic_jitter(point)
        ax.scatter(
            idx + jitter,
            point.val_accuracy_pct,
            s=24,
            color="#4c78a8",
            alpha=0.62,
            linewidth=0,
            zorder=3,
        )

    centers = []
    means = []
    for sigma in candidates:
        vals = grouped.get(sigma, [])
        if vals:
            centers.append(candidate_positions[sigma])
            means.append(float(np.mean(vals)))
    ax.plot(
        centers,
        means,
        color="#111111",
        linewidth=1.2,
        marker="D",
        markersize=4,
        label="Candidate mean",
        zorder=5,
    )
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels([format_sigma(v) for v in candidates])
    ax.set_xlabel("Modal selected sigma within epoch")
    ax.legend(loc="lower right")


def deterministic_jitter(point: SigmaPoint) -> float:
    key = (point.seed * 37 + point.epoch * 13 +
           len(point.variant_label) * 7) % 17
    return (key - 8) * 0.010


def sigma_bin_means(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_bins = min(6, max(3, len(np.unique(xs)) // 4))
    edges = np.linspace(float(np.min(xs)), float(np.max(xs)), n_bins + 1)
    centers: list[float] = []
    means: list[float] = []
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        mask = (xs >= lo) & (xs < hi)
        if idx == n_bins - 1:
            mask = (xs >= lo) & (xs <= hi)
        if np.any(mask):
            centers.append((lo + hi) / 2.0)
            means.append(float(np.mean(ys[mask])))
    return np.array(centers), np.array(means)


def build_summary(
    runs_root: pathlib.Path,
    out_dir: pathlib.Path,
    points: list[SigmaPoint],
    run_counts: Counter[str],
    candidates: list[float],
    pdf: pathlib.Path,
    png: pathlib.Path,
) -> list[str]:
    xs = np.array([p.mean_sigma for p in points], dtype=float)
    ys = np.array([p.val_accuracy_pct for p in points], dtype=float)
    grouped: dict[float, list[float]] = defaultdict(list)
    for point in points:
        grouped[point.modal_sigma].append(point.val_accuracy_pct)

    candidate_stats = []
    for sigma in candidates:
        vals = grouped.get(sigma, [])
        if vals:
            candidate_stats.append(
                (sigma, float(np.mean(vals)), float(np.std(vals)), len(vals)))

    verdict = relationship_verdict(candidate_stats)
    counts, _, _ = np.histogram2d(xs, ys, bins=20)
    nonempty = int(np.count_nonzero(counts))
    plot_mode = "strip fallback" if should_use_strip(points) else "hexbin"

    lines = [
        f"Resolved --runs-root: {runs_root.resolve()}",
        f"Resolved --out-dir:   {out_dir.resolve()}",
        "",
        "Files included by axis/scheduler:",
    ]
    for label, count in sorted(run_counts.items()):
        lines.append(f"  {label}: n_runs={count}")
    lines.extend(
        [
            "",
            f"Total (seed, epoch) pairs: {len(points)}",
            f"sigma candidates (union across runs): {[format_sigma(v) for v in candidates]}",
            f"sigma min/max observed (epoch mean): {float(np.min(xs)):.6f} / {float(np.max(xs)):.6f}",
            f"val-acc min/max observed (%): {float(np.min(ys)):.2f} / {float(np.max(ys)):.2f}",
            f"Render mode: {plot_mode}; hexbin non-empty cells at gridsize=20 would be {nonempty}",
            "",
            "Per-sigma-candidate aggregate val-acc (grouped by modal selected sigma in epoch):",
        ]
    )
    for sigma, mean, std, n in candidate_stats:
        lines.append(
            f"  sigma={format_sigma(sigma)}: mean={mean:.2f}% std={std:.2f} pp n_epochs={n}")
    lines.extend(
        [
            "",
            verdict,
            "",
            "Outputs:",
            f"  {pdf.resolve()}",
            f"  {png.resolve()}",
        ]
    )
    return lines


def relationship_verdict(candidate_stats: list[tuple[float, float, float, int]]) -> str:
    if len(candidate_stats) < 2:
        return "Relationship verdict: insufficient sigma candidates to assess trend."
    ordered = sorted(candidate_stats)
    means = np.array([mean for _, mean, _, _ in ordered], dtype=float)
    spread = float(np.max(means) - np.min(means))
    diffs = np.diff(means)
    if spread < 0.5:
        return f"Relationship verdict: approximately flat (spread={spread:.2f} pp across candidates)."
    if np.all(diffs >= -0.1):
        return "Relationship verdict: monotone increasing with selected sigma."
    if np.all(diffs <= 0.1):
        return "Relationship verdict: monotone decreasing with selected sigma."
    peak_idx = int(np.argmax(means))
    if 0 < peak_idx < len(means) - 1:
        return f"Relationship verdict: single peak near sigma={format_sigma(ordered[peak_idx][0])}."
    return "Relationship verdict: non-monotone without a clear single interior peak."


def format_sigma(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.3f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    main()
