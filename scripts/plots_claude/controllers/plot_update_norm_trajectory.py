"""Mean update-norm trajectory across three AEES regimes (hypothesis-A diagnostic).

Three stacked panels, shared training-step x-axis, independent log y-axes:

  1. CIFAR-100 / AdamW / AEES-LR / clean
     results/cifar_clean/cifar_clean_aees_none_seed<0-4>.json
  2. SST-2 / AdamW / AEES-Dual + warmup-linear
     results/sst2/sst2_aees_warmup_linear_5ep_small_ep200_trend_seed<0-4>.json
  3. AG News (sym20) / AdamW / AEES-Dual + warmup-linear
     results/noisy_agnews/seed_<N>/agnews_noise20_aees_warmup_linear_5ep_coarse_highnoise_ep200_seed<N>.json

Per panel: one thin alpha-shaded line per seed plus a bold mean overlay; a
horizontal reference line marks the pooled late-training (last 20% of
episodes per seed) median update norm. Supports/weakens hypothesis A1 in
sections/discussion/axis_dependence.tex about cross-regime sigma scaling.

CLI:
    python -m scripts.plots_claude.controllers.plot_update_norm_trajectory \\
        --runs-root . --out-dir results/plots/controllers

Outputs (on success):
    update_norm_trajectory.pdf
    update_norm_trajectory.png
    update_norm_trajectory.summary.txt

Outputs (on failure):
    update_norm_trajectory.MISSING.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots_claude._common import (  # noqa: F401  - imports required by task contract
    RunInfo,
    episode_to_epoch,
    epoch_step_range,
    identify,
    load_set,
    walk_runs,
)
from scripts.plots_claude._style import (  # noqa: F401  - imports required by task contract
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


# ---------------------------------------------------------------------------
# Regime configuration. Each regime has 5 seeds; file paths are relative to
# --runs-root so the script can be re-pointed at a different checkout layout
# without code changes. Order is panel order (top -> bottom).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    key: str
    title: str
    color_key: str
    rel_paths: tuple[str, ...]


REGIMES: tuple[Regime, ...] = (
    Regime(
        key="cifar_clean_aees_lr",
        title="CIFAR-100 / AdamW / AEES-LR + Linear",
        # CIFAR clean is an LR-only AEES run.
        color_key="aees_lr",
        rel_paths=tuple(
            f"results/cifar_clean/cifar_clean_aees_linear_seed{s}.json" for s in range(5)
        ),
    ),
    Regime(
        key="sst2_aees_dual_wl",
        title="SST-2 / AdamW / AEES-Dual + Linear",
        color_key="aees_dual",
        rel_paths=tuple(
            f"results/sst2/sst2_aees_warmup_linear_5ep_small_ep200_trend_seed{s}.json"
            for s in range(5)
        ),
    ),
    Regime(
        key="agnews_sym20_aees_dual_wl",
        title="AG News (sym20) / AdamW / AEES-Dual + Linear",
        # The two NLP regimes are semantically both AEES-Dual+WL; use a
        # distinct PALETTE entry here so the single-panel layout doesn't put
        # two brown lines on top of each other.
        color_key="aees_noise",
        rel_paths=tuple(
            f"results/noisy_agnews/seed_{s}/"
            f"agnews_noise20_aees_warmup_linear_5ep_coarse_highnoise_ep200_seed{s}.json"
            for s in range(5)
        ),
    ),
)

# Fraction of episodes (from the tail) used to compute the late-training
# median reference line. 0.2 == last 20 percent.
LATE_FRACTION = 0.20

# Rolling-median smoothing window per regime (units: episodes). Roughly 5% of
# the regime's episode count -- spec'd directly in the Round 3 brief.
SMOOTH_WINDOW_BY_KEY = {
    "cifar_clean_aees_lr":       21,  # ~5% of 391 CIFAR episodes
    "sst2_aees_dual_wl":          9,  # 106 SST-2 episodes
    "agnews_sym20_aees_dual_wl": 11,  # 169 AG News episodes
}


# ---------------------------------------------------------------------------
# Data-loading helpers.
# ---------------------------------------------------------------------------


@dataclass
class SeedSeries:
    seed: int
    path: pathlib.Path
    steps: np.ndarray         # shape [E]
    update_norms: np.ndarray  # shape [E]


@dataclass
class RegimeData:
    regime: Regime
    seeds: list[SeedSeries]
    # Per-episode cross-seed aggregates, all of shape [E].
    steps_ref: np.ndarray
    median_raw: np.ndarray
    q25_raw: np.ndarray
    q75_raw: np.ndarray
    # Same series after a centered rolling-median pass of `smooth_window`.
    median_smooth: np.ndarray
    q25_smooth: np.ndarray
    q75_smooth: np.ndarray
    smooth_window: int
    # Late-training summary computed on the SMOOTHED median series.
    late_median: float


def _load_seed_series(
    runs_root: pathlib.Path, regime: Regime
) -> tuple[list[SeedSeries], list[str]]:
    """Return (seed_series_list, missing_descriptions).

    A non-empty missing list signals failure for this regime.
    """
    series: list[SeedSeries] = []
    missing: list[str] = []
    for seed_index, rel in enumerate(regime.rel_paths):
        path = (runs_root / rel).resolve()
        if not path.is_file():
            missing.append(
                f"{regime.key} seed{seed_index}: file not found at {path}")
            continue
        try:
            with open(path) as f:
                record = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            missing.append(
                f"{regime.key} seed{seed_index}: unreadable ({exc!r}) at {path}")
            continue

        episode_logs = record.get("episode_logs")
        if not isinstance(episode_logs, dict):
            missing.append(
                f"{regime.key} seed{seed_index}: missing 'episode_logs' object in {path}"
            )
            continue
        mun = episode_logs.get("mean_update_norms")
        ees = episode_logs.get("episode_end_steps")
        if mun is None or ees is None:
            missing.append(
                f"{regime.key} seed{seed_index}: missing mean_update_norms or "
                f"episode_end_steps in {path}"
            )
            continue
        if len(mun) == 0 or len(ees) == 0:
            missing.append(
                f"{regime.key} seed{seed_index}: empty episode logs in {path}")
            continue
        if len(mun) != len(ees):
            missing.append(
                f"{regime.key} seed{seed_index}: length mismatch "
                f"len(mean_update_norms)={len(mun)} vs len(episode_end_steps)={len(ees)}"
            )
            continue

        info = identify(record, path)
        seed_value = info.seed if info.seed is not None else seed_index
        series.append(
            SeedSeries(
                seed=int(seed_value),
                path=path,
                steps=np.asarray(ees, dtype=float),
                update_norms=np.asarray(mun, dtype=float),
            )
        )

    return series, missing


def _rolling_median_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median. ``window <= 1`` returns input unchanged."""
    if window <= 1:
        return arr.copy()
    n = len(arr)
    out = np.empty_like(arr)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.median(arr[lo:hi]))
    return out


def _aggregate_regime(regime: Regime, series: list[SeedSeries]) -> RegimeData:
    # Every seed in the regime shares the same episode count by the validation
    # in main(); enforce here as a guard.
    lens = {len(s.update_norms) for s in series}
    if len(lens) != 1:
        raise ValueError(
            f"regime '{regime.key}' has inconsistent episode counts {sorted(lens)}"
        )
    stacked = np.vstack([s.update_norms for s in series])  # [n_seeds, E]
    # Treat non-positive entries as missing so the log-y plot stays clean.
    stacked_pos = np.where(stacked > 0.0, stacked, np.nan)
    with np.errstate(invalid="ignore"):
        median_raw = np.nanmedian(stacked_pos, axis=0)
        q25_raw = np.nanpercentile(stacked_pos, 25, axis=0)
        q75_raw = np.nanpercentile(stacked_pos, 75, axis=0)

    if not np.any(np.isfinite(median_raw)):
        raise ValueError(
            f"regime '{regime.key}' has no positive update-norm values")

    window = SMOOTH_WINDOW_BY_KEY[regime.key]
    median_smooth = _rolling_median_1d(median_raw, window)
    q25_smooth = _rolling_median_1d(q25_raw, window)
    q75_smooth = _rolling_median_1d(q75_raw, window)

    # The reference x-axis is the first seed's episode_end_steps; all seeds
    # share episode counts so step boundaries match within rounding.
    steps_ref = series[0].steps.copy()

    # Late-training summary now computed on the SMOOTHED median series.
    n = len(median_smooth)
    cut = min(int(n * (1.0 - LATE_FRACTION)), n - 1)
    late_tail = median_smooth[cut:]
    late_tail = late_tail[np.isfinite(late_tail)]
    if late_tail.size == 0:
        raise ValueError(
            f"regime '{regime.key}' smoothed-median late tail is empty"
        )
    return RegimeData(
        regime=regime,
        seeds=series,
        steps_ref=steps_ref,
        median_raw=median_raw,
        q25_raw=q25_raw,
        q75_raw=q75_raw,
        median_smooth=median_smooth,
        q25_smooth=q25_smooth,
        q75_smooth=q75_smooth,
        smooth_window=window,
        late_median=float(np.median(late_tail)),
    )


# ---------------------------------------------------------------------------
# Figure builder.
# ---------------------------------------------------------------------------


def _format_norm(value: float) -> str:
    """Compact scientific formatting suitable for in-axis annotation."""
    if value <= 0 or not np.isfinite(value):
        return "n/a"
    return f"{value:.2e}"


def _build_figure(regime_data: list[RegimeData]):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Shaded horizontal band for the gradient-noise injection range tested
    # across the NLP experiments. SST-2 uses σ ∈ {0, 0.0025, 0.005} and
    # AG News uses σ ∈ {0, 0.005, 0.01}; the union (also = [0, 0.01]) is
    # what the band represents. On a log y-axis σ = 0 is at −∞, so we
    # extend the band's lower bound to an arbitrarily small positive value;
    # matplotlib clips it to the plot floor. Per-regime σ sets are spelled
    # out in the caption.
    sigma_hi = 0.01
    ax.axhspan(
        1e-12, sigma_hi,
        color="#999999", alpha=0.12, zorder=0,
        label=r"$\sigma$-injection range, "
              r"$\sigma \in \{0,\ 0.0025,\ 0.005,\ 0.01\}$",
    )

    # Per-regime trajectories on a normalized training-progress x-axis so
    # the three regimes (391 / 106 / 169 episodes) are directly comparable.
    for rd in regime_data:
        color = PALETTE[rd.regime.color_key]
        n_ep = len(rd.median_smooth)
        if n_ep <= 1:
            continue
        x_norm = np.arange(n_ep) / float(n_ep - 1)

        finite = (
            np.isfinite(rd.median_smooth)
            & np.isfinite(rd.q25_smooth)
            & np.isfinite(rd.q75_smooth)
        )
        if not np.any(finite):
            continue

        ax.fill_between(
            x_norm[finite],
            rd.q25_smooth[finite],
            rd.q75_smooth[finite],
            color=color, alpha=0.16, linewidth=0,
        )
        ax.plot(
            x_norm[finite],
            rd.median_smooth[finite],
            color=color, linewidth=1.8, label=rd.regime.title, zorder=3,
        )
        # Right-edge value annotation so the reader can read off the
        # late-training median without consulting the caption.
        last_x = float(x_norm[finite][-1])
        last_y = float(rd.median_smooth[finite][-1])
        ax.annotate(
            f"{last_y:.1e}",
            xy=(last_x, last_y),
            xytext=(6, 0), textcoords="offset points",
            color=color, fontsize=7.5, va="center",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Training progress (fraction of episodes)")
    ax.set_ylabel(r"Mean update norm (after AdamW rescaling)")
    ax.set_xlim(0.0, 1.0)
    # Clamp y-limits explicitly: the σ-band's lower bound is set to 1e-12
    # so it visually represents σ = 0, but without this clamp the autoscale
    # picks that up and ~half the plot becomes empty space below the data.
    ax.set_ylim(5e-4, 1.0)

    # Lower-left is the cleanest spot: NLP curves start in the upper-middle
    # at x ≈ 0 and decay toward the σ-band on the right, leaving lower-left
    # clear; CIFAR sits in the upper band throughout.
    ax.legend(loc="lower left", frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary writer.
# ---------------------------------------------------------------------------


def _ratio_str(numerator: float, denominator: float) -> str:
    if denominator <= 0 or not np.isfinite(denominator):
        return "n/a"
    return f"{numerator / denominator:.1f}x"


def _build_summary(
    runs_root: pathlib.Path,
    regime_data: list[RegimeData],
    pdf_path: pathlib.Path,
    png_path: pathlib.Path,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"runs-root: {runs_root.resolve()}")
    lines.append(
        f"late-fraction (median of smoothed median series, tail): {LATE_FRACTION:.0%}"
    )
    lines.append("")

    by_key: dict[str, RegimeData] = {rd.regime.key: rd for rd in regime_data}

    for rd in regime_data:
        lines.append(f"Regime: {rd.regime.title}")
        lines.append(f"  variant_key (color): {rd.regime.color_key}")
        lines.append(f"  seed count: {len(rd.seeds)}")
        lines.append(f"  rolling-median window (episodes): {rd.smooth_window}")
        for s in rd.seeds:
            n = len(s.update_norms)
            cut = min(int(n * (1.0 - LATE_FRACTION)), n - 1)
            seed_late_median = float(np.median(s.update_norms[cut:]))
            lines.append(
                f"    seed={s.seed} episodes={n} per_seed_raw_late_median="
                f"{_format_norm(seed_late_median)} path={s.path}"
            )
        lines.append(
            f"  smoothed-median late-tail median: {_format_norm(rd.late_median)}"
        )
        lines.append("")

    # Cross-regime ratios anchored on CIFAR.
    cifar = by_key.get("cifar_clean_aees_lr")
    sst2 = by_key.get("sst2_aees_dual_wl")
    agn = by_key.get("agnews_sym20_aees_dual_wl")
    lines.append("Cross-regime ratios (anchored on CIFAR clean / AEES-LR):")
    if cifar is not None and sst2 is not None:
        lines.append(
            f"  CIFAR / SST-2  = {_ratio_str(cifar.late_median, sst2.late_median)}  (~"
            f"{np.log10(cifar.late_median / sst2.late_median):.2f} dex)"
        )
    if cifar is not None and agn is not None:
        lines.append(
            f"  CIFAR / AGNews = {_ratio_str(cifar.late_median, agn.late_median)}  (~"
            f"{np.log10(cifar.late_median / agn.late_median):.2f} dex)"
        )
    lines.append("")

    # Hypothesis A1 callout. The cross-regime ratio is informative but cannot
    # by itself establish the sigma-to-signal ratio: a non-trivial slice of it
    # is just the eta ratio between regimes, and AdamW's per-coordinate
    # rescaling means this is the post-rescaling update norm rather than the
    # raw gradient norm. So we phrase it as "consistent with" not "confirms".
    callout = "Hypothesis A1 callout: insufficient data."
    if cifar is not None and sst2 is not None and agn is not None:
        r_sst = cifar.late_median / sst2.late_median
        r_agn = cifar.late_median / agn.late_median
        worst = min(r_sst, r_agn)
        if worst >= 10.0:
            callout = (
                f"Hypothesis A1: CONSISTENT (not confirmed). "
                f"CIFAR update-norm is {r_sst:.1f}x SST-2 and {r_agn:.1f}x AG News at "
                f"late training (>= 1 dex on both axes), directionally matching A1. "
                "Caveat: ~20x of this gap is just the eta ratio between regimes, and "
                "AdamW's per-coordinate rescaling means the post-rescaling update norm "
                "does not directly establish the sigma-to-signal ratio. Direct "
                "confirmation requires raw gradient norms, which are not in the "
                "archived runs."
            )
        else:
            callout = (
                f"Hypothesis A1: WEAKENED. CIFAR/SST-2 ratio = {r_sst:.1f}x, "
                f"CIFAR/AGNews ratio = {r_agn:.1f}x -- both within ~1 dex; the discussion "
                "paragraph in axis_dependence.tex should be reworded. Note: direct "
                "confirmation still requires raw gradient norms, which are not in the "
                "archived runs."
            )
    lines.append(callout)
    lines.append("")

    lines.append("outputs:")
    lines.append(f"  - {pdf_path}")
    lines.append(f"  - {png_path}")
    return lines


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=pathlib.Path,
        required=True,
        help="Repository root containing results/cifar_clean, results/sst2, results/noisy_agnews.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="Directory in which to write the update_norm_trajectory outputs.",
    )
    args = parser.parse_args(argv)

    try:
        regime_data: list[RegimeData] = []
        missing_all: list[str] = []
        for regime in REGIMES:
            series, missing = _load_seed_series(args.runs_root, regime)
            if missing:
                missing_all.extend(missing)
                continue
            # All 5 files must succeed for the regime; otherwise it's missing.
            if len(series) != len(regime.rel_paths):
                missing_all.append(
                    f"{regime.key}: only {len(series)}/{len(regime.rel_paths)} seeds loaded"
                )
                continue
            # All seeds in a regime must share the same episode length for the
            # mean overlay to be meaningful.
            lens = {len(s.update_norms) for s in series}
            if len(lens) != 1:
                missing_all.append(
                    f"{regime.key}: inconsistent episode counts across seeds: {sorted(lens)}"
                )
                continue
            regime_data.append(_aggregate_regime(regime, series))

        if missing_all:
            reason = (
                "Required AEES run files for the update-norm trajectory figure were "
                "missing or unreadable:\n\n"
                + "\n".join(f"- {m}" for m in missing_all)
                + f"\n\nrunsRoot: {args.runs_root.resolve()}\n"
            )
            write_missing(args.out_dir, NAME, reason)
            return 1

        fig = _build_figure(regime_data)
    except Exception as exc:
        reason = (
            "Failed to build update_norm_trajectory figure.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, NAME, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, NAME)
    summary_lines = _build_summary(
        args.runs_root, regime_data, pdf_path, png_path)
    summary_path = write_summary(args.out_dir, NAME, summary_lines)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
