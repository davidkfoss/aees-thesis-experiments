"""Selected-arm trajectory over training (stacked-area, Task 6).

Two settings produced by one CLI invocation each:
    --setting cifar_sym40    → LR-multiplier axis on noisy CIFAR-100 40% symmetric
                                (AdamW, AEES-LR flagship adamw_aees_ep200_lr05102).
    --setting agnews_noisy   → σ and LR axes on noisy AG News 20% symmetric
                                (AEES-Dual + warmup-linear flagship). Two-panel
                                figure: both axes shown because the σ-only
                                presentation hid that the LR axis is also
                                near-uniform.

For each setting and active axis, every episode is binned by the candidate-arm
value the controller picked at that episode. We stack the cross-seed
proportions to visualise whether the controller settled on one arm.

A centered rolling-mean (window ≈ 5% of episode count, odd) is applied for
display only — the summary file reports raw proportions. The smoothing
preserves the columns-sum-to-1 property by averaging full episode slices.

CLI:
    python -m scripts.plots_claude.controllers.plot_arm_selection \\
        --runs-root results/cifar_noisy --out-dir results/plots/controllers \\
        --setting cifar_sym40

    python -m scripts.plots_claude.controllers.plot_arm_selection \\
        --runs-root results/noisy_agnews --out-dir results/plots/controllers \\
        --setting agnews_noisy

Outputs (on success):
    arm_selection_<setting>.pdf
    arm_selection_<setting>.png
    arm_selection_<setting>.summary.txt

Outputs (on failure):
    arm_selection_<setting>.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from typing import Any

import numpy as np

from scripts.plots_claude._common import (
    RunInfo,
    identify,
    walk_runs,
)
from scripts.plots_claude._style import (
    arm_palette,
    save_figure,
    write_missing,
    write_summary,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Per-setting / per-axis configuration. Each setting holds a list of axes; a
# multi-axis setting renders one stacked-area panel per axis (side-by-side,
# shared y).
# ---------------------------------------------------------------------------


AXIS_SPECS: dict[str, dict[str, Any]] = {
    "lr": {
        "base_key": "aees_lr",
        "selected_values_key": "selected_lr_values",
        "controller_logs_key": "lr_controller_logs",
        "axis_title": "LR multiplier",
        "arm_label_fmt": lambda v: f"m={float(v):.1f}",
    },
    "noise": {
        "base_key": "aees_noise",
        "selected_values_key": "selected_noise_values",
        "controller_logs_key": "noise_controller_logs",
        "axis_title": "Gradient noise σ",
        "arm_label_fmt": lambda v: ("σ=0" if float(v) == 0.0 else f"σ={v:g}"),
    },
}


SETTING_SPEC: dict[str, dict[str, Any]] = {
    "cifar_sym40": {
        "name": "arm_selection_cifar_sym40",
        "axes": ["lr"],
    },
    "cifar_sym40_cosine": {
        # Same layout as cifar_sym40 but for the Cosine + AEES flagship.
        # Used to show that the controller's preferred arm pattern is
        # consistent with or modulated by the global decay schedule.
        "name": "arm_selection_cifar_sym40_cosine",
        "axes": ["lr"],
    },
    "cifar_sym40_paired": {
        # Side-by-side comparison: AEES-LR alone vs Cosine + AEES on the
        # 40% symmetric label noise setting. Renders the arm-selection
        # trajectories for both runs in a single two-panel figure, sharing
        # the y-axis so the AEES-alone vs Cosine+AEES contrast (preference
        # cements vs preference inverts) reads in a single glance.
        "name": "arm_selection_cifar_sym40_paired",
        "axes": ["lr"],  # both panels show the LR axis
        "component_settings": ["cifar_sym40", "cifar_sym40_cosine"],
        "panel_titles": ["AEES (no scheduler)", "Cosine + AEES"],
    },
    "agnews_noisy": {
        # Two panels: σ axis on the left, LR axis on the right. Showing both
        # is critical for the AG News chapter — the σ-only view hid that the
        # LR axis is also near-uniform on this task.
        "name": "arm_selection_agnews_noisy",
        "axes": ["noise", "lr"],
    },
}


# ---------------------------------------------------------------------------
# Run-filtering predicates (per setting).
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


def _predicate_cifar_sym40_cosine(info: RunInfo, _record: dict[str, Any]) -> bool:
    if info.task != "cifar100":
        return False
    if info.noise_setting != "sym40":
        return False
    if info.optimizer != "AdamW":
        return False
    if info.path is None:
        return False
    if info.path.name != "adamw_cosine_aees_ep200_lr05102":
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
    # Need both axes active to plot both panels.
    if len(lr_cands) <= 1 or len(noise_cands) <= 1:
        return False
    return True


PREDICATES = {
    "cifar_sym40": _predicate_cifar_sym40,
    "cifar_sym40_cosine": _predicate_cifar_sym40_cosine,
    "agnews_noisy": _predicate_agnews_noisy,
}


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------


def _load_runs(
    runs_root: pathlib.Path, setting: str
) -> list[tuple[RunInfo, dict[str, Any]]]:
    """Walk runs_root and filter to the flagship file(s) for the chosen setting."""
    pred = PREDICATES[setting]
    out: list[tuple[RunInfo, dict[str, Any]]] = []
    for path, record in walk_runs(runs_root):
        if record.get("controller_logs") is None:
            continue
        info = identify(record, path)
        if not pred(info, record):
            continue
        out.append((info, record))
    return out


# ---------------------------------------------------------------------------
# Arm-fraction matrix.
# ---------------------------------------------------------------------------


def _arm_fractions(
    items: list[tuple[RunInfo, dict[str, Any]]],
    selected_key: str,
    arm_values: list[float],
) -> tuple[np.ndarray, int, list[int]]:
    """Stack-friendly fraction matrix.

    Returns:
        fractions: ndarray of shape [n_arms, n_episodes], each column sums to 1.
        n_episodes: episode count after truncating to the shortest seed.
        seeds: sorted seed list.
    """
    items_sorted = sorted(
        items, key=lambda iv: (iv[0].seed if iv[0].seed is not None else -1)
    )
    seeds = [info.seed for info, _ in items_sorted]
    selected_lists = [r["episode_logs"][selected_key] for _, r in items_sorted]
    lengths = [len(s) for s in selected_lists]
    n_episodes = min(lengths)
    if n_episodes <= 0:
        raise ValueError(
            f"degenerate episode count after truncation: {n_episodes}")

    n_seeds = len(items_sorted)
    n_arms = len(arm_values)

    selected = np.array(
        [s[:n_episodes] for s in selected_lists], dtype=float
    )  # [n_seeds, n_episodes]

    arms_arr = np.asarray(arm_values, dtype=float)
    fractions = np.zeros((n_arms, n_episodes), dtype=float)
    for a_idx, a_val in enumerate(arms_arr):
        match = np.isclose(selected, a_val, atol=1e-9)
        fractions[a_idx, :] = match.sum(axis=0) / n_seeds

    col_sums = fractions.sum(axis=0)
    if not np.allclose(col_sums, 1.0, atol=1e-9):
        bad = np.where(~np.isclose(col_sums, 1.0, atol=1e-9))[0]
        raise ValueError(
            f"selected values include arm not in arm_values; "
            f"{len(bad)} episodes have column-sum != 1, "
            f"first bad column index = {int(bad[0])} sum = {col_sums[bad[0]]:.6f}"
        )

    return fractions, n_episodes, seeds


# ---------------------------------------------------------------------------
# Smoothing. Ported from scripts/plots_gpt/controllers/plot_arm_selection.py
# (window = ~5% of episode count, odd). Visual-only — summary reports raw
# proportions.
# ---------------------------------------------------------------------------


def _display_window(n_episodes: int) -> int:
    """Window size (odd, ≥5) for centered rolling-mean smoothing."""
    window = max(5, int(round(n_episodes * 0.05)))
    return window if window % 2 == 1 else window + 1


def _per_seed_smoothed_fractions(
    items: list[tuple[RunInfo, dict[str, Any]]],
    selected_key: str,
    arm_values: list[float],
    n_episodes: int,
    window: int,
) -> np.ndarray:
    """[n_seeds, n_arms, n_episodes] of smoothed per-seed selection fractions.

    For each seed, each arm-row is a 0/1 indicator vector over episodes (1 if
    that seed picked that arm in that episode), centered-rolling-meaned with
    the same window used for the stacked-area display. The resulting per-seed
    fractions allow drawing cross-seed mean ± SD bands per arm, which is the
    correct view for the "did the controller learn a preference?" question:
    the SD band shows how consistent the preference is across seeds.
    """
    items_sorted = sorted(
        items, key=lambda iv: (iv[0].seed if iv[0].seed is not None else -1)
    )
    n_seeds = len(items_sorted)
    n_arms = len(arm_values)
    arms_arr = np.asarray(arm_values, dtype=float)

    indicators = np.zeros((n_seeds, n_arms, n_episodes), dtype=float)
    for s_idx, (_, record) in enumerate(items_sorted):
        s_vals = np.asarray(
            record["episode_logs"][selected_key][:n_episodes], dtype=float
        )
        for a_idx, a_val in enumerate(arms_arr):
            indicators[s_idx, a_idx, :] = np.isclose(
                s_vals, a_val, atol=1e-9
            ).astype(float)

    if window <= 1:
        return indicators

    half = window // 2
    smoothed = np.empty_like(indicators)
    for i in range(n_episodes):
        lo = max(0, i - half)
        hi = min(n_episodes, i + half + 1)
        smoothed[:, :, i] = indicators[:, :, lo:hi].mean(axis=2)
    return smoothed


def _centered_rolling_mean(fractions: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean along axis=1 (time). Renormalises columns to 1.

    Renormalisation is a no-op when the input columns already sum to 1 and
    every output bin uses the same time window, but we keep the explicit
    divide so the function tolerates ragged windows at the boundaries without
    drifting away from 1.
    """
    if window <= 1:
        return fractions
    n_arms, n_eps = fractions.shape
    out = np.empty_like(fractions, dtype=float)
    half = window // 2
    for i in range(n_eps):
        lo = max(0, i - half)
        hi = min(n_eps, i + half + 1)
        out[:, i] = fractions[:, lo:hi].mean(axis=1)
    col_sums = out.sum(axis=0, keepdims=True)
    return np.divide(
        out, col_sums, out=np.zeros_like(out), where=col_sums > 0
    )


# ---------------------------------------------------------------------------
# Per-seed disagreement diagnostic (raw, unsmoothed).
# ---------------------------------------------------------------------------


def _seed_disagreement(
    items: list[tuple[RunInfo, dict[str, Any]]],
    selected_key: str,
    arm_values: list[float],
    n_episodes: int,
) -> list[tuple[int, dict[float, float], float]]:
    items_sorted = sorted(
        items, key=lambda iv: (iv[0].seed if iv[0].seed is not None else -1)
    )
    selected_lists = [
        np.asarray(r["episode_logs"][selected_key][:n_episodes], dtype=float)
        for _, r in items_sorted
    ]
    arms_arr = np.asarray(arm_values, dtype=float)
    per_seed: list[dict[float, float]] = []
    for s_vals in selected_lists:
        seed_fracs = {}
        for a_val in arms_arr:
            seed_fracs[float(a_val)] = float(
                np.mean(np.isclose(s_vals, a_val, atol=1e-9))
            )
        per_seed.append(seed_fracs)
    group = {
        float(a): float(np.mean([s[float(a)] for s in per_seed])) for a in arms_arr
    }
    out: list[tuple[int, dict[float, float], float]] = []
    for (info, _), seed_fracs in zip(items_sorted, per_seed):
        l1 = sum(abs(seed_fracs[k] - group[k]) for k in group)
        out.append((info.seed, seed_fracs, l1))
    return out


# ---------------------------------------------------------------------------
# Plot construction.
# ---------------------------------------------------------------------------


def _make_multi_panel(n_panels: int):
    """N-panel figure with shared y."""
    if n_panels == 1:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        return fig, [ax]
    width = 3.5 * n_panels + 0.5  # ~3.5in per panel
    fig, axes = plt.subplots(
        1, n_panels, sharey=True, figsize=(width, 4.0)
    )
    return fig, list(axes)


def _draw_per_arm_lines(
    ax,
    per_seed_smoothed: np.ndarray,
    arm_values: list[float],
    axis_spec: dict[str, Any],
    representative_record: dict[str, Any],
) -> None:
    """Per-arm cross-seed mean trajectory with ±1 SD band; uniform reference.

    Reads more directly than the stacked-area display when the controller
    *does* learn a preference: one line lifts above the 1/K uniform line
    while the others drop below, and the SD band shows whether seeds agree.
    Used for the cifar_sym40 setting; the agnews_noisy setting keeps the
    stacked area because the story there is that nothing concentrates.
    """
    n_seeds, n_arms, n_episodes = per_seed_smoothed.shape
    mean = per_seed_smoothed.mean(axis=0)
    std = per_seed_smoothed.std(axis=0, ddof=0)
    # Override the standard arm_palette here: this figure is the only place
    # in the thesis that puts three LR-arm trajectories on a single set of
    # axes (the other arm-selection figure uses stacked areas), and the
    # light-to-dark single-hue palette becomes ambiguous when the three lines
    # are stacked at similar y. Use hue-separated colors instead so each arm
    # is unambiguously identifiable in print and at a glance. Order:
    # m=0.5 (preferred) → orange, m=1.0 (neutral) → blue, m=2.0 (suppressed)
    # → red. Falls back to arm_palette for n_arms ≠ 3 or non-aees_lr.
    if axis_spec["base_key"] == "aees_lr" and n_arms == 3:
        colors = ["#ff7f0e", "#1f77b4", "#d62728"]
    else:
        colors = arm_palette(axis_spec["base_key"], n_arms)

    # x-axis: convert episode index to epoch using the run's total_epochs,
    # so the figure shares its time unit with every other CIFAR plot.
    total_epochs = representative_record.get("total_epochs")
    if total_epochs and n_episodes >= 2:
        eps_per_epoch = n_episodes / float(total_epochs)
        x = np.arange(1, n_episodes + 1) / eps_per_epoch
        x_label = "Epoch"
        x_lim = (0.0, float(total_epochs))
    else:
        x = np.arange(1, n_episodes + 1, dtype=float)
        x_label = "Episode"
        x_lim = (1.0, float(n_episodes))

    for a_idx, (arm_val, color) in enumerate(zip(arm_values, colors)):
        lo = np.clip(mean[a_idx] - std[a_idx], 0.0, 1.0)
        hi = np.clip(mean[a_idx] + std[a_idx], 0.0, 1.0)
        ax.fill_between(x, lo, hi, color=color,
                        alpha=0.18, linewidth=0, zorder=1)
        ax.plot(
            x, mean[a_idx],
            color=color, linewidth=1.8, zorder=3,
            label=axis_spec["arm_label_fmt"](arm_val),
        )

    # Uniform-random reference line. If the controller learns anything, lines
    # diverge from this line; if it does not, lines hug it.

    ax.set_xlabel(x_label)
    ax.set_ylabel("Per-episode selection fraction")
    ax.set_xlim(*x_lim)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(axis_spec["axis_title"], fontsize=10, pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def _draw_panel(
    ax,
    fractions_display: np.ndarray,
    n_episodes: int,
    arm_values: list[float],
    axis_spec: dict[str, Any],
    representative_record: dict[str, Any],
    is_first_panel: bool,
):
    x = np.arange(1, n_episodes + 1)
    colors = arm_palette(axis_spec["base_key"], len(arm_values))
    labels = [axis_spec["arm_label_fmt"](v) for v in arm_values]

    ax.stackplot(x, *fractions_display, colors=colors, labels=labels)
    ax.set_xlabel("Episode")
    if is_first_panel:
        ax.set_ylabel("Seed fraction")
    ax.set_xlim(1, n_episodes)
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0, y=0)
    ax.set_title(axis_spec["axis_title"], fontsize=10, pad=12)

    ax.legend(
        loc="lower center",
        ncol=len(arm_values),
        bbox_to_anchor=(0.5, -0.30),
        frameon=False,
    )

    total_epochs = representative_record["total_epochs"]
    if total_epochs >= 1 and n_episodes >= 2:
        eps_per_epoch = n_episodes / total_epochs

        def ep_to_epoch(episode_idx):
            return episode_idx / eps_per_epoch

        def epoch_to_ep(epoch):
            return epoch * eps_per_epoch

        if total_epochs <= 10:
            epoch_ticks = list(range(1, total_epochs + 1))
        else:
            step = max(10, int(round(total_epochs / 5 / 10)) * 10)
            epoch_ticks = list(range(step, total_epochs + 1, step))

        secax = ax.secondary_xaxis("top", functions=(ep_to_epoch, epoch_to_ep))
        secax.set_xticks(epoch_ticks)
        secax.set_xlim(ep_to_epoch(0), ep_to_epoch(n_episodes))
        secax.set_xlabel("Epoch")
        secax.grid(False)


# ---------------------------------------------------------------------------
# Summary text and entry point.
# ---------------------------------------------------------------------------


def _settled_callout(
    fractions: np.ndarray, arm_values: list[float], axis: str
) -> str:
    """One-line objective callout for the summary (raw proportions)."""
    n_arms, n_eps = fractions.shape
    overall = fractions.mean(axis=1)
    dominant_idx = int(np.argmax(overall))
    dominant_frac = float(overall[dominant_idx])
    dominant_val = arm_values[dominant_idx]
    settled = dominant_frac >= 0.70

    extra = ""
    q = n_eps // 4
    if q > 0:
        first_q = fractions[:, :q].mean(axis=1)
        last_q = fractions[:, -q:].mean(axis=1)
        first_high = float(first_q[-1])
        last_high = float(last_q[-1])
        first_low = float(first_q[0])
        last_low = float(last_q[0])
        delta_high = last_high - first_high
        delta_low = last_low - first_low
        if axis == "noise":
            if delta_high <= -0.15 and delta_low >= 0.15:
                extra = (
                    f" σ pattern: EARLY-HIGH / LATE-LOW "
                    f"(high-σ share {first_high:.2f} → {last_high:.2f}, "
                    f"low-σ share {first_low:.2f} → {last_low:.2f})."
                )
            elif delta_high >= 0.15 and delta_low <= -0.15:
                extra = (
                    f" σ pattern: EARLY-LOW / LATE-HIGH "
                    f"(high-σ share {first_high:.2f} → {last_high:.2f}, "
                    f"low-σ share {first_low:.2f} → {last_low:.2f})."
                )
            else:
                extra = (
                    f" σ pattern: no clear early/late skew "
                    f"(high-σ share {first_high:.2f} → {last_high:.2f}, "
                    f"low-σ share {first_low:.2f} → {last_low:.2f})."
                )
        elif axis == "lr":
            extra = (
                f" LR drift early→late: m={arm_values[-1]:g} share "
                f"{first_high:.2f}→{last_high:.2f}, m={arm_values[0]:g} share "
                f"{first_low:.2f}→{last_low:.2f}."
            )

    if settled:
        return (
            f"settled? YES — dominant arm {dominant_val:g} averages "
            f"{dominant_frac:.2f} across all episodes.{extra}"
        )
    return (
        f"settled? NO — top arm {dominant_val:g} only averages "
        f"{dominant_frac:.2f}; bands stay mixed.{extra}"
    )


def _quarter_means(
    fractions: np.ndarray, arm_values: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean fraction per arm: (overall, first-25%, middle-50%, last-25%)."""
    n_eps = fractions.shape[1]
    q = max(n_eps // 4, 1)
    return (
        fractions.mean(axis=1),
        fractions[:, :q].mean(axis=1),
        fractions[:, q: 3 * q].mean(axis=1),
        fractions[:, -q:].mean(axis=1),
    )


def _run_paired(args, spec: dict[str, Any]) -> int:
    """Render a two-panel arm-selection figure comparing two component settings.

    Each panel shows per-arm trajectories (mean ± SD across seeds) for one
    component setting. Both panels share the y-axis so the cross-context
    contrast (preference cements vs preference inverts) reads directly off
    the page.
    """
    name = spec["name"]
    component_keys: list[str] = spec["component_settings"]
    panel_titles: list[str] = spec["panel_titles"]
    axis_key: str = spec["axes"][0]
    axis_spec = AXIS_SPECS[axis_key]

    try:
        # Load runs and compute per-axis data for each component.
        per_component: list[dict[str, Any]] = []
        for comp_key in component_keys:
            items = _load_runs(args.runs_root, comp_key)
            if not items:
                write_missing(
                    args.out_dir, name,
                    f"No flagship runs found for component '{comp_key}' "
                    f"under {args.runs_root.resolve()}.\n",
                )
                return 1
            items_sorted = sorted(
                items, key=lambda iv: (
                    iv[0].seed if iv[0].seed is not None else -1
                )
            )
            ctrl_key = axis_spec["controller_logs_key"]
            arm_value_sets = {
                tuple(r["controller_logs"][ctrl_key]["arm_values"])
                for _, r in items
            }
            if len(arm_value_sets) != 1:
                raise ValueError(
                    f"inconsistent {ctrl_key}.arm_values across seeds for "
                    f"component '{comp_key}': {sorted(arm_value_sets)}"
                )
            (arm_values_tuple,) = arm_value_sets
            arm_values = sorted(arm_values_tuple)

            fractions, n_episodes, seeds = _arm_fractions(
                items, axis_spec["selected_values_key"], arm_values
            )
            window = _display_window(n_episodes)
            per_seed = _per_seed_smoothed_fractions(
                items,
                axis_spec["selected_values_key"],
                arm_values,
                n_episodes,
                window,
            )
            per_component.append({
                "comp_key": comp_key,
                "items": items,
                "items_sorted": items_sorted,
                "arm_values": arm_values,
                "fractions": fractions,
                "per_seed": per_seed,
                "n_episodes": n_episodes,
                "seeds": seeds,
                "window": window,
                "representative_record": items_sorted[0][1],
            })

        # Two-panel figure, shared y so the cross-context contrast reads
        # directly. ~3.5in per panel matches _make_multi_panel for n=2.
        fig, panels = plt.subplots(
            1, 2, sharey=True, figsize=(7.5, 4.0)
        )
        for ax, comp, title in zip(panels, per_component, panel_titles):
            _draw_per_arm_lines(
                ax,
                comp["per_seed"],
                comp["arm_values"],
                axis_spec,
                comp["representative_record"],
            )
            # Override the per-axis title set by _draw_per_arm_lines so the
            # panel headers identify the scheduler context, not just the axis.
            ax.set_title(title, fontsize=10, pad=8)

        # Drop redundant y-label from the right panel (sharey already
        # suppresses tick labels but the ylabel call inside _draw_per_arm_lines
        # sets it on both axes).
        panels[1].set_ylabel("")

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

    repo_root = pathlib.Path(__file__).resolve().parents[3]

    def _rel(p: pathlib.Path) -> str:
        try:
            return str(p.resolve().relative_to(repo_root))
        except ValueError:
            return str(p.resolve())

    summary_lines: list[str] = []
    summary_lines.append(f"runs-root: {_rel(args.runs_root)}")
    summary_lines.append(f"setting: {spec['name']}")
    summary_lines.append(f"component_settings: {component_keys}")
    summary_lines.append("")
    for comp, title in zip(per_component, panel_titles):
        summary_lines.append(
            f"=== Panel: {title} (component={comp['comp_key']}) ===")
        for info, _ in comp["items_sorted"]:
            summary_lines.append(f"  - {_rel(info.path)} (seed={info.seed})")
        overall, first_q, mid, last_q = _quarter_means(
            comp["fractions"], comp["arm_values"]
        )
        summary_lines.append("")
        summary_lines.append("Mean fraction per arm (raw, unsmoothed):")
        for a_idx, a_val in enumerate(comp["arm_values"]):
            summary_lines.append(
                f"  arm {a_val:g}: overall={overall[a_idx]:.3f}  "
                f"first25%={first_q[a_idx]:.3f}  middle50%={mid[a_idx]:.3f}  "
                f"last25%={last_q[a_idx]:.3f}  "
                f"drift(last-first)={last_q[a_idx] - first_q[a_idx]:+.3f}"
            )
        summary_lines.append("")

    summary_lines.append("outputs:")
    summary_lines.append(f"  - {pdf_path}")
    summary_lines.append(f"  - {png_path}")
    write_summary(args.out_dir, name, summary_lines)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=pathlib.Path, required=True,
        help="Root directory containing the flagship runs.",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, required=True,
        help="Directory to write the arm_selection_<setting> outputs.",
    )
    parser.add_argument(
        "--setting", choices=sorted(SETTING_SPEC.keys()), required=True,
        help="Which setting to plot.",
    )
    args = parser.parse_args(argv)

    spec = SETTING_SPEC[args.setting]
    name = spec["name"]
    axis_keys: list[str] = spec["axes"]

    # Paired-figure path: load each component setting independently and
    # render their per-arm-lines into a shared two-panel figure. Bypasses
    # the per-axis loop used by the other settings.
    if "component_settings" in spec:
        return _run_paired(args, spec)

    try:
        items = _load_runs(args.runs_root, args.setting)
        if not items:
            write_missing(
                args.out_dir, name,
                f"No flagship runs found for setting '{args.setting}' under "
                f"{args.runs_root.resolve()}.\n",
            )
            return 1

        items_sorted = sorted(
            items, key=lambda iv: (
                iv[0].seed if iv[0].seed is not None else -1)
        )

        # Per-axis: arm values, raw fractions, smoothed display fractions.
        per_axis: dict[str, dict[str, Any]] = {}
        for axis_key in axis_keys:
            axis_spec = AXIS_SPECS[axis_key]
            ctrl_key = axis_spec["controller_logs_key"]
            arm_value_sets = {
                tuple(r["controller_logs"][ctrl_key]["arm_values"])
                for _, r in items
            }
            if len(arm_value_sets) != 1:
                raise ValueError(
                    f"inconsistent {ctrl_key}.arm_values across seeds: "
                    f"{sorted(arm_value_sets)}"
                )
            (arm_values_tuple,) = arm_value_sets
            arm_values = sorted(arm_values_tuple)

            fractions, n_episodes, seeds = _arm_fractions(
                items, axis_spec["selected_values_key"], arm_values
            )
            window = _display_window(n_episodes)
            display_fractions = _centered_rolling_mean(fractions, window)

            per_axis[axis_key] = {
                "axis_spec": axis_spec,
                "arm_values": arm_values,
                "fractions": fractions,           # raw, for summary
                "display": display_fractions,     # smoothed, for plot
                "n_episodes": n_episodes,
                "seeds": seeds,
                "window": window,
            }

        representative_info, representative_record = items_sorted[0]
        fig, panels = _make_multi_panel(len(axis_keys))

        # Both cifar_sym40 settings (AEES alone and Cosine + AEES) use per-arm
        # trajectories with ±SD bands — the right view when the controller
        # *does* learn a preference. agnews_noisy keeps the stacked-area
        # display because its story is that the bands stay roughly equal
        # across all arms.
        use_lines = args.setting in ("cifar_sym40", "cifar_sym40_cosine")

        for ax, axis_key in zip(panels, axis_keys):
            entry = per_axis[axis_key]
            if use_lines:
                per_seed = _per_seed_smoothed_fractions(
                    items,
                    entry["axis_spec"]["selected_values_key"],
                    entry["arm_values"],
                    entry["n_episodes"],
                    entry["window"],
                )
                _draw_per_arm_lines(
                    ax,
                    per_seed,
                    entry["arm_values"],
                    entry["axis_spec"],
                    representative_record,
                )
            else:
                _draw_panel(
                    ax,
                    entry["display"],
                    entry["n_episodes"],
                    entry["arm_values"],
                    entry["axis_spec"],
                    representative_record,
                    is_first_panel=(axis_key == axis_keys[0]),
                )

        if use_lines:
            fig.tight_layout()
        else:
            fig.subplots_adjust(bottom=0.24, top=0.85, wspace=0.10)
    except Exception as exc:
        reason = (
            f"Failed to build {name}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, name, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, name)

    repo_root = pathlib.Path(__file__).resolve().parents[3]

    def _rel(p: pathlib.Path) -> str:
        try:
            return str(p.resolve().relative_to(repo_root))
        except ValueError:
            return str(p.resolve())

    summary_lines: list[str] = []
    summary_lines.append(f"runs-root: {_rel(args.runs_root)}")
    summary_lines.append(f"setting: {args.setting}")
    summary_lines.append(f"axes plotted: {axis_keys}")
    summary_lines.append("")
    summary_lines.append("Files loaded:")
    for info, _ in items_sorted:
        summary_lines.append(f"  - {_rel(info.path)} (seed={info.seed})")
    summary_lines.append("")

    for axis_key in axis_keys:
        entry = per_axis[axis_key]
        axis_spec = entry["axis_spec"]
        arm_values = entry["arm_values"]
        fractions = entry["fractions"]

        summary_lines.append(
            f"=== Axis: {axis_key} ({axis_spec['axis_title']}) ===")
        summary_lines.append(f"n_seeds: {len(entry['seeds'])}")
        summary_lines.append(
            f"n_episodes_after_truncation: {entry['n_episodes']}")
        summary_lines.append(f"arm_values: {arm_values}")
        summary_lines.append(
            f"smoothing_window (display only): {entry['window']} "
            f"(centered rolling mean, ≈5% of episodes)"
        )

        overall, first_q, mid, last_q = _quarter_means(fractions, arm_values)
        summary_lines.append("")
        summary_lines.append("Mean fraction per arm (raw, unsmoothed):")
        for a_idx, a_val in enumerate(arm_values):
            summary_lines.append(
                f"  arm {a_val:g}: overall={overall[a_idx]:.3f}  "
                f"first25%={first_q[a_idx]:.3f}  middle50%={mid[a_idx]:.3f}  "
                f"last25%={last_q[a_idx]:.3f}  "
                f"drift(last-first)={last_q[a_idx] - first_q[a_idx]:+.3f}"
            )

        disagreement = _seed_disagreement(
            items, axis_spec["selected_values_key"], arm_values, entry["n_episodes"]
        )
        summary_lines.append("")
        summary_lines.append(
            "Per-seed mean-fraction (and L1 distance from group mean):"
        )
        for seed, fracs_dict, l1 in disagreement:
            frac_str = " ".join(f"{k:g}:{v:.2f}" for k,
                                v in fracs_dict.items())
            summary_lines.append(f"  seed {seed}: {frac_str}  L1={l1:.3f}")

        callout = _settled_callout(fractions, arm_values, axis_key)
        summary_lines.append("")
        summary_lines.append(f"callout: {callout}")
        summary_lines.append("")

    summary_lines.append("outputs:")
    summary_lines.append(f"  - {_rel(pdf_path)}")
    summary_lines.append(f"  - {_rel(png_path)}")

    summary_path = write_summary(args.out_dir, name, summary_lines)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
