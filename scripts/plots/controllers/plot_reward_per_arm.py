"""Per-arm reward distribution (violin plot).

For a given setting, pool the clipped log-EMA episode reward across all seeds
and all completed episodes, group by the selected arm value, and draw a violin
per arm. Shows which arms tend to produce positive vs. negative rewards.

Settings:
    cifar_sym40       — LR axis only (single panel). Noisy CIFAR sym40 AdamW
                        AEES-LR (adamw_aees_ep200_lr05102).
    agnews_noisy      — LR axis | sigma axis (two panels, shared y).
                        AG News sym20 AEES-Dual + warmup-linear.

CLI:
    python -m scripts.plots.controllers.plot_reward_per_arm \
        --runs-root results/cifar_noisy   --out-dir results/plots/controllers \
        --setting cifar_sym40

    python -m scripts.plots.controllers.plot_reward_per_arm \
        --runs-root results/noisy_agnews  --out-dir results/plots/controllers \
        --setting agnews_noisy

Outputs (on success):
    reward_per_arm_<setting>.pdf
    reward_per_arm_<setting>.png
    reward_per_arm_<setting>.summary.txt

Outputs (on failure):
    reward_per_arm_<setting>.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from typing import Any

import numpy as np

from scripts.plots._common import RunInfo, load_set
from scripts.plots._style import (
    arm_palette,
    make_figure,
    save_figure,
    write_missing,
    write_summary,
)


# ---------------------------------------------------------------------------
# Setting registry.
# ---------------------------------------------------------------------------

SETTINGS = ("cifar_sym40", "agnews_noisy")

NAME_PREFIX = "reward_per_arm"


# ---------------------------------------------------------------------------
# Predicates per setting.
# ---------------------------------------------------------------------------


def _predicate_cifar_sym40(info: RunInfo, _record: dict[str, Any]) -> bool:
    if info.task != "cifar100":
        return False
    if info.noise_setting != "sym40":
        return False
    if info.optimizer != "AdamW":
        return False
    if info.path is None:
        return False
    if info.path.name != "adamw_aees_ep200_lr05102":
        return False
    return True


def _predicate_agnews_noisy(info: RunInfo, record: dict[str, Any]) -> bool:
    if info.task != "agnews":
        return False
    if info.noise_setting != "sym20":
        return False
    if info.path is None:
        return False
    if "aees_warmup_linear" not in info.path.name:
        return False
    cfg = record.get("config", {})
    lr_cands = cfg.get("lr_candidates") or []
    noise_cands = cfg.get("noise_candidates") or []
    if len(lr_cands) <= 1:
        return False
    if len(noise_cands) <= 1:
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers: snap a continuous-valued selected_*_values entry to the nearest
# canonical arm in arm_values. Using nearest-by-distance avoids float-key
# noise that direct dict lookup would produce.
# ---------------------------------------------------------------------------


def _snap_index(value: float, arm_values: list[float]) -> int:
    diffs = [abs(value - a) for a in arm_values]
    return int(np.argmin(diffs))


def _format_arm(v: float) -> str:
    """Compact tick label for an arm value."""
    # Render zero exactly; render small/large floats without trailing 0s.
    if v == 0:
        return "0.0"
    # Up to 4 decimal places, strip trailing zeros (but leave one decimal).
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


# ---------------------------------------------------------------------------
# Pool rewards per arm across all matching runs for a given axis.
# Returns (arm_values, list-of-list-of-rewards, seed-list, n_total_pooled).
# ---------------------------------------------------------------------------


def _per_seed_medians(
    items: list[tuple[RunInfo, dict[str, Any]]],
    axis: str,
) -> tuple[list[float], list[list[float]], list[int]]:
    """For each arm, return one median reward per seed.

    Returns ``(arm_values, per_arm_per_seed, seeds_sorted)`` where
    ``per_arm_per_seed[arm_idx]`` is a list of length ``n_seeds`` containing
    the median reward that the matching seed received whenever that arm was
    pulled. NaN if a seed never pulled the arm.

    This is the cross-seed view that ``§5.5.1`` is actually claiming on:
    "between-arm differences are small relative to the variability observed
    across seeds". The pooled view returned by ``_pool_rewards`` mixes
    within-seed (episode-to-episode) and between-seed variability, which
    is not the comparison the thesis is making.
    """
    assert axis in ("lr", "noise")
    sel_key = "selected_lr_values" if axis == "lr" else "selected_noise_values"
    ctrl_key = "lr_controller_logs" if axis == "lr" else "noise_controller_logs"

    items_sorted = sorted(
        items, key=lambda iv: iv[0].seed if iv[0].seed is not None else -1
    )
    seeds = [info.seed for info, _ in items_sorted]

    # Establish canonical arm values (same as _pool_rewards).
    canonical_arms: list[float] | None = None
    for info, record in items_sorted:
        ctrl_logs = record.get("controller_logs") or {}
        sub = ctrl_logs.get(ctrl_key) or {}
        arms = sub.get("arm_values")
        if not arms:
            raise ValueError(
                f"missing {ctrl_key}.arm_values for seed={info.seed} path={info.path}"
            )
        arms_f = [float(a) for a in arms]
        if canonical_arms is None:
            canonical_arms = arms_f
        elif arms_f != canonical_arms:
            raise ValueError(
                f"arm_values mismatch across seeds: {canonical_arms} vs {arms_f} "
                f"(seed={info.seed} path={info.path})"
            )
    assert canonical_arms is not None

    n_arms = len(canonical_arms)
    n_seeds = len(items_sorted)
    per_arm_per_seed: list[list[float]] = [
        [float("nan")] * n_seeds for _ in range(n_arms)
    ]

    for seed_idx, (info, record) in enumerate(items_sorted):
        ep_logs = record.get("episode_logs") or {}
        rewards = ep_logs.get("episode_rewards") or []
        selected = ep_logs.get(sel_key) or []
        if len(rewards) != len(selected):
            raise ValueError(
                f"episode_rewards / {sel_key} length mismatch ({len(rewards)} vs "
                f"{len(selected)}) for seed={info.seed} path={info.path}"
            )
        # Bucket this seed's rewards into arms first, then take per-arm median
        # for this seed.
        per_arm_this_seed: list[list[float]] = [[] for _ in range(n_arms)]
        for r_val, s_val in zip(rewards, selected):
            idx = _snap_index(float(s_val), canonical_arms)
            per_arm_this_seed[idx].append(float(r_val))
        for arm_idx in range(n_arms):
            if per_arm_this_seed[arm_idx]:
                per_arm_per_seed[arm_idx][seed_idx] = float(
                    np.median(per_arm_this_seed[arm_idx])
                )

    return canonical_arms, per_arm_per_seed, seeds


def _pool_rewards(
    items: list[tuple[RunInfo, dict[str, Any]]],
    axis: str,
) -> tuple[list[float], list[list[float]], list[int], int]:
    assert axis in ("lr", "noise")
    sel_key = "selected_lr_values" if axis == "lr" else "selected_noise_values"
    ctrl_key = "lr_controller_logs" if axis == "lr" else "noise_controller_logs"

    # Validate that all runs share the same canonical arm_values list. (They
    # should — same controller config across seeds — but flag a mismatch
    # loudly rather than silently producing inconsistent groupings.)
    canonical_arms: list[float] | None = None
    seeds: list[int] = []
    for info, record in items:
        ctrl_logs = record.get("controller_logs") or {}
        sub = ctrl_logs.get(ctrl_key) or {}
        arms = sub.get("arm_values")
        if not arms:
            raise ValueError(
                f"missing {ctrl_key}.arm_values for seed={info.seed} path={info.path}"
            )
        arms = [float(a) for a in arms]
        if canonical_arms is None:
            canonical_arms = arms
        elif arms != canonical_arms:
            raise ValueError(
                f"arm_values mismatch across seeds: {canonical_arms} vs {arms} "
                f"(seed={info.seed} path={info.path})"
            )
        seeds.append(info.seed)

    assert canonical_arms is not None
    n_arms = len(canonical_arms)
    per_arm: list[list[float]] = [[] for _ in range(n_arms)]
    n_total = 0

    for info, record in items:
        ep_logs = record.get("episode_logs") or {}
        rewards = ep_logs.get("episode_rewards") or []
        selected = ep_logs.get(sel_key) or []
        if len(rewards) != len(selected):
            raise ValueError(
                f"episode_rewards / {sel_key} length mismatch ({len(rewards)} vs "
                f"{len(selected)}) for seed={info.seed} path={info.path}"
            )
        for r_val, s_val in zip(rewards, selected):
            idx = _snap_index(float(s_val), canonical_arms)
            per_arm[idx].append(float(r_val))
            n_total += 1

    return canonical_arms, per_arm, sorted(seeds), n_total


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def _draw_per_seed_dots(
    ax,
    arm_values: list[float],
    per_arm_per_seed: list[list[float]],
    base_key: str,
    xlabel: str,
    ylabel: str | None,
) -> None:
    """Per-seed median reward as jittered dots, with cross-seed mean ± SD overlaid.

    Each dot is one seed's median reward for the arm; the heavy black bar is the
    mean of those per-seed medians; the vertical error bar is the across-seed SD.
    This view directly answers the §5.5.1 claim because the within-column dot
    spread (across-seed variability) and the between-column mean-bar offset
    (per-arm signal) are both visible at the same y-scale.
    """
    n_arms = len(arm_values)
    positions = list(range(n_arms))
    colors = arm_palette(base_key, n_arms)
    rng = np.random.default_rng(0)  # deterministic jitter

    for pos, seed_meds, color in zip(positions, per_arm_per_seed, colors):
        valid = [m for m in seed_meds if not np.isnan(m)]
        if not valid:
            continue
        valid_arr = np.asarray(valid, dtype=float)
        n_pts = valid_arr.size
        jitter = rng.uniform(-0.12, 0.12, size=n_pts)
        ax.scatter(
            np.full(n_pts, pos) + jitter,
            valid_arr,
            s=22,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            zorder=3,
        )

        mean = float(valid_arr.mean())
        std = float(valid_arr.std(ddof=0))
        # Mean bar (wide, bold) so the cross-seed mean is unambiguous against
        # the dot cloud.
        ax.hlines(
            mean,
            pos - 0.28, pos + 0.28,
            colors="black", linewidth=2.0, zorder=4,
        )
        # Cross-seed ±1 SD error bar.
        ax.errorbar(
            pos, mean, yerr=std,
            fmt="none", color="black",
            capsize=4, capthick=1.0, linewidth=1.0, zorder=4,
        )
        # Numeric annotation for the mean.
        ax.annotate(
            f"{mean:+.3f}",
            xy=(pos + 0.30, mean),
            xytext=(2, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=7.0, color="black", zorder=5,
        )

    # No-improvement reference line.
    ax.axhline(0, color="lightgray", linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xticks(positions)
    ax.set_xticklabels([_format_arm(v) for v in arm_values])
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, n_arms - 0.4)


def _draw_violin(
    ax,
    arm_values: list[float],
    per_arm: list[list[float]],
    base_key: str,
    xlabel: str,
    ylabel: str | None,
) -> None:
    n_arms = len(arm_values)
    positions = list(range(n_arms))

    # Replace empty arms with a single NaN so violinplot doesn't crash; the
    # corresponding body is just dropped by matplotlib in that case.
    safe_data = [d if len(d) > 0 else [float("nan")] for d in per_arm]
    parts = ax.violinplot(
        safe_data,
        positions=positions,
        showmedians=False,
        showextrema=False,
    )
    colors = arm_palette(base_key, n_arms)
    bodies = parts.get("bodies", []) if isinstance(
        parts, dict) else parts["bodies"]
    for body, color in zip(bodies, colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.75)

    # No-improvement reference line.
    ax.axhline(0, color="lightgray", linestyle="--", linewidth=0.8, zorder=0)

    # Median tick per arm, with the numeric value annotated to the right.
    # On the AG News σ axis the three medians span only ~0.07 on a ±1 scale,
    # so the U-shape is invisible without an explicit number.
    for pos, data in zip(positions, per_arm):
        if not data:
            continue
        med = float(np.median(data))
        ax.hlines(
            med,
            pos - 0.25,
            pos + 0.25,
            colors="black",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            f"{med:+.3f}",
            xy=(pos + 0.26, med),
            xytext=(2, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.0,
            color="black",
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([_format_arm(v) for v in arm_values])
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, n_arms - 0.4)


# ---------------------------------------------------------------------------
# Per-setting orchestration.
# ---------------------------------------------------------------------------


def _per_arm_stats(per_arm: list[list[float]]) -> list[dict[str, float]]:
    out = []
    for data in per_arm:
        if not data:
            out.append(
                {"n": 0, "mean": float("nan"), "median": float("nan"),
                 "std": float("nan"), "frac_pos": float("nan")}
            )
            continue
        arr = np.asarray(data, dtype=float)
        out.append(
            {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),  # population std
                "frac_pos": float((arr > 0).mean()),
            }
        )
    return out


def _summary_block(
    title: str,
    arm_values: list[float],
    per_arm: list[list[float]],
    stats: list[dict[str, float]],
) -> list[str]:
    lines = [f"  Axis: {title}", f"    arm_values: {arm_values}"]
    for arm, s in zip(arm_values, stats):
        lines.append(
            f"    arm={_format_arm(arm):>8s}"
            f"  n={s['n']:5d}"
            f"  mean={s['mean']:+.4f}"
            f"  median={s['median']:+.4f}"
            f"  std={s['std']:.4f}"
            f"  frac_pos={s['frac_pos']:.3f}"
        )
    return lines


def _run_cifar_sym40(args) -> int:
    name = f"{NAME_PREFIX}_cifar_sym40"
    try:
        runs = load_set(args.runs_root, _predicate_cifar_sym40)
        # AEES-LR for CIFAR-noisy is variant_key="aees_lr".
        items = runs.get("aees_lr", [])
        if not items:
            reason = (
                "No matching runs found.\n\n"
                "Expected runs:\n"
                "  results/cifar_noisy/cifar100_sym40_seed<0..4>/adamw_aees_ep200_lr05102\n\n"
                f"Searched under: {args.runs_root}\n"
            )
            write_missing(args.out_dir, name, reason)
            return 1
        if len(items) < 2:
            reason = (
                f"Need >=2 seeds for a meaningful pool; found {len(items)} run(s).\n\n"
                f"Searched under: {args.runs_root}\n"
            )
            write_missing(args.out_dir, name, reason)
            return 1

        arm_values, per_arm, seeds, n_total = _pool_rewards(items, axis="lr")
        stats = _per_arm_stats(per_arm)

        fig, axes = make_figure(n_panels=1)
        _draw_violin(
            axes[0],
            arm_values,
            per_arm,
            base_key="aees_lr",
            xlabel="LR multiplier",
            ylabel="Clipped log-EMA reward",
        )
        fig.tight_layout()
    except Exception as exc:
        reason = (
            f"Failed to build {name}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, name, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, name)

    summary_lines = []
    summary_lines.append(f"runs-root: {args.runs_root.resolve()}")
    summary_lines.append(f"setting: cifar_sym40")
    summary_lines.append(f"n_runs: {len(items)}  seeds: {seeds}")
    summary_lines.append(f"pooled_episodes_total: {n_total}")
    summary_lines.append("")
    summary_lines.append("files loaded:")
    for info, _ in sorted(items, key=lambda iv: iv[0].seed if iv[0].seed is not None else -1):
        summary_lines.append(f"  - {info.path}")
    summary_lines.append("")
    summary_lines.extend(_summary_block(
        "LR multiplier", arm_values, per_arm, stats))
    summary_lines.append("")
    summary_lines.append("outputs:")
    summary_lines.append(f"  - {pdf_path}")
    summary_lines.append(f"  - {png_path}")
    write_summary(args.out_dir, name, summary_lines)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


def _run_agnews_noisy(args) -> int:
    name = f"{NAME_PREFIX}_agnews_noisy"
    try:
        runs = load_set(args.runs_root, _predicate_agnews_noisy)
        # AEES-Dual + warmup-linear -> variant_key="warmup_linear_aees".
        items = runs.get("warmup_linear_aees", [])
        if not items:
            reason = (
                "No matching runs found.\n\n"
                "Expected runs:\n"
                "  results/noisy_agnews/seed_<0..4>/agnews_noise20_aees_warmup_linear_*.json\n"
                "  (with len(config.lr_candidates)>1 AND len(config.noise_candidates)>1)\n\n"
                f"Searched under: {args.runs_root}\n"
            )
            write_missing(args.out_dir, name, reason)
            return 1
        if len(items) < 2:
            reason = (
                f"Need >=2 seeds for a meaningful pool; found {len(items)} run(s).\n\n"
                f"Searched under: {args.runs_root}\n"
            )
            write_missing(args.out_dir, name, reason)
            return 1

        # Pooled view (kept for the summary file because the existing
        # text reports pooled medians and per-arm n).
        lr_arms, lr_per_arm_pool, seeds, n_total = _pool_rewards(
            items, axis="lr")
        noise_arms, noise_per_arm_pool, _seeds_noise, n_total_noise = _pool_rewards(
            items, axis="noise"
        )
        if n_total != n_total_noise:
            raise ValueError(
                f"pooled episode counts disagree across axes: lr={n_total} noise={n_total_noise}"
            )
        lr_stats = _per_arm_stats(lr_per_arm_pool)
        noise_stats = _per_arm_stats(noise_per_arm_pool)

        # Per-seed view (what the figure now displays). One median per
        # (arm, seed) pair, exposing across-seed variability — which is the
        # comparison §5.5.1 actually makes.
        _, lr_per_arm_per_seed, _ = _per_seed_medians(items, axis="lr")
        _, noise_per_arm_per_seed, _ = _per_seed_medians(items, axis="noise")

        fig, axes = make_figure(n_panels=2)
        _draw_per_seed_dots(
            axes[0],
            lr_arms,
            lr_per_arm_per_seed,
            base_key="aees_lr",
            xlabel="LR multiplier",
            ylabel="Per-seed median clipped log-EMA reward",
        )
        _draw_per_seed_dots(
            axes[1],
            noise_arms,
            noise_per_arm_per_seed,
            base_key="aees_noise",
            xlabel=r"Gradient noise $\sigma$",
            ylabel=None,
        )
        # Match y-axis across panels so visual comparison of cross-arm
        # spread vs cross-seed spread is honest between LR and σ.
        y_lo = min(ax.get_ylim()[0] for ax in axes)
        y_hi = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(y_lo, y_hi)
        fig.tight_layout()
    except Exception as exc:
        reason = (
            f"Failed to build {name}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, name, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, name)

    summary_lines = []
    summary_lines.append(f"runs-root: {args.runs_root.resolve()}")
    summary_lines.append(f"setting: agnews_noisy")
    summary_lines.append(f"n_runs: {len(items)}  seeds: {seeds}")
    summary_lines.append(f"pooled_episodes_total: {n_total}")
    summary_lines.append("")
    summary_lines.append("files loaded:")
    for info, _ in sorted(items, key=lambda iv: iv[0].seed if iv[0].seed is not None else -1):
        summary_lines.append(f"  - {info.path}")
    summary_lines.append("")
    summary_lines.extend(_summary_block(
        "LR multiplier", lr_arms, lr_per_arm_pool, lr_stats))
    summary_lines.append("")
    summary_lines.extend(
        _summary_block("Gradient noise sigma", noise_arms,
                       noise_per_arm_pool, noise_stats)
    )
    summary_lines.append("")
    summary_lines.append("outputs:")
    summary_lines.append(f"  - {pdf_path}")
    summary_lines.append(f"  - {png_path}")
    write_summary(args.out_dir, name, summary_lines)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--setting",
        choices=SETTINGS,
        required=True,
        help="Which figure to emit (cifar_sym40 or agnews_noisy).",
    )
    args = parser.parse_args(argv)

    if args.setting == "cifar_sym40":
        return _run_cifar_sym40(args)
    if args.setting == "agnews_noisy":
        return _run_agnews_noisy(args)
    parser.error(f"unknown --setting {args.setting!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
