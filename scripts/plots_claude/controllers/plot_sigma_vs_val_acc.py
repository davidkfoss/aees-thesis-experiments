"""Task 8 — Selected sigma vs. epoch validation accuracy (noisy AG News).

For every AEES-Noise and AEES-Dual run on noisy AG News (sym20) with
len(noise_candidates) > 1, pair up the per-epoch mean of the controller's
selected sigma values with the validation accuracy at the end of that epoch,
then render a 2-D hexbin density. Falls back to a strip plot when the hexbin
ends up too sparse to be interpretable.

Outputs (on success):
    results/plots/controllers/sigma_vs_val_acc_agnews_noisy.pdf
    results/plots/controllers/sigma_vs_val_acc_agnews_noisy.png
    results/plots/controllers/sigma_vs_val_acc_agnews_noisy.summary.txt

On failure:
    results/plots/controllers/sigma_vs_val_acc_agnews_noisy.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

import numpy as np

from scripts.plots_claude._common import (
    walk_runs,
    identify,
    load_set,
    RunInfo,
    episode_to_epoch,
    epoch_step_range,
)
from scripts.plots_claude._style import (
    PALETTE,
    LABEL,
    save_figure,
    write_summary,
    write_missing,
    mean_std_band,
    mark_peak_final,
    short_trajectory_kwargs,
    arm_palette,
    make_figure,
)


NAME = "sigma_vs_val_acc_agnews_noisy"


def _predicate(info: RunInfo, record: dict) -> bool:
    if info.task != "agnews":
        return False
    if info.noise_setting != "sym20":
        return False
    if not info.is_aees:
        return False
    noise_cands = record.get("config", {}).get("noise_candidates") or []
    if len(noise_cands) <= 1:
        return False
    return True


def _collect_pairs(
    runs: list[tuple[RunInfo, dict]],
) -> tuple[list[float], list[float], list[dict]]:
    """Return (x_sigmas, y_accs, per_run_info_dicts).

    Each run contributes total_epochs (seed, epoch) samples; epochs with no
    episodes are skipped.
    """
    x: list[float] = []
    y: list[float] = []
    per_run: list[dict] = []
    for info, r in runs:
        ep_logs = r.get("episode_logs", {})
        end_steps = ep_logs.get("episode_end_steps") or []
        sel_noise = ep_logs.get("selected_noise_values") or []
        val_accs = r.get("val_accuracies") or []
        total_epochs = r.get("total_epochs")
        if total_epochs is None or not end_steps or not sel_noise or not val_accs:
            raise RuntimeError(
                f"Run {info.path} missing required fields "
                f"(episode_end_steps/selected_noise_values/val_accuracies/total_epochs)."
            )
        if len(end_steps) != len(sel_noise):
            raise RuntimeError(
                f"Run {info.path}: episode_end_steps/selected_noise_values length mismatch."
            )
        per_epoch_means: list[tuple[int, float, float]] = []
        for k in range(int(total_epochs)):
            lo, hi = epoch_step_range(r, k)
            vals = [
                float(sel_noise[i])
                for i, es in enumerate(end_steps)
                if lo <= int(es) < hi
            ]
            if not vals:
                continue
            mean_sigma = float(np.mean(vals))
            if k >= len(val_accs):
                continue
            acc = float(val_accs[k]) * 100.0
            x.append(mean_sigma)
            y.append(acc)
            per_epoch_means.append((k, mean_sigma, acc))
        per_run.append(
            {
                "path": info.path,
                "seed": info.seed,
                "variant_key": info.variant_key,
                "n_epochs_kept": len(per_epoch_means),
            }
        )
    return x, y, per_run


def _per_candidate_aggregate(
    runs: list[tuple[RunInfo, dict]],
    candidates: list[float],
) -> dict[float, tuple[float, int]]:
    """Mean val-acc-at-epoch-end across epochs where each candidate was the
    *modal* (most-selected) sigma in that epoch.

    Returns {candidate: (mean_val_acc_pct, n_epochs)}.
    """
    buckets: dict[float, list[float]] = {c: [] for c in candidates}
    for info, r in runs:
        ep_logs = r.get("episode_logs", {})
        end_steps = ep_logs.get("episode_end_steps") or []
        sel_noise = ep_logs.get("selected_noise_values") or []
        val_accs = r.get("val_accuracies") or []
        total_epochs = int(r.get("total_epochs") or 0)
        for k in range(total_epochs):
            lo, hi = epoch_step_range(r, k)
            vals = [
                float(sel_noise[i])
                for i, es in enumerate(end_steps)
                if lo <= int(es) < hi
            ]
            if not vals or k >= len(val_accs):
                continue
            # Modal candidate (snap each selected value to nearest candidate).
            counts: dict[float, int] = defaultdict(int)
            for v in vals:
                nearest = min(candidates, key=lambda c: abs(c - v))
                counts[nearest] += 1
            modal = max(counts.items(), key=lambda kv: kv[1])[0]
            buckets[modal].append(float(val_accs[k]) * 100.0)
    out: dict[float, tuple[float, int]] = {}
    for c in candidates:
        if buckets[c]:
            out[c] = (float(np.mean(buckets[c])), len(buckets[c]))
        else:
            out[c] = (float("nan"), 0)
    return out


def _classify_relationship(
    candidates: list[float],
    per_cand: dict[float, tuple[float, int]],
) -> str:
    """Return a one-line callout describing the shape (monotone/single-peak/flat)."""
    pairs = [
        (c, per_cand[c][0]) for c in sorted(candidates) if per_cand[c][1] > 0
    ]
    if len(pairs) < 2:
        return "Too few populated sigma buckets to classify shape."
    means = [m for _, m in pairs]
    spread = max(means) - min(means)
    if spread < 0.3:
        return f"Relationship is approximately FLAT (spread = {spread:.2f} pp across candidates)."
    diffs = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    all_up = all(d > 0 for d in diffs)
    all_down = all(d < 0 for d in diffs)
    if all_up:
        return f"Relationship is MONOTONE INCREASING (val-acc rises with sigma; spread = {spread:.2f} pp)."
    if all_down:
        return f"Relationship is MONOTONE DECREASING (val-acc falls as sigma rises; spread = {spread:.2f} pp)."
    # Single peak: exactly one direction change, and the extremum is interior.
    argmax = int(np.argmax(means))
    argmin = int(np.argmin(means))
    if 0 < argmax < len(means) - 1:
        return (
            f"Relationship is SINGLE-PEAK at sigma={pairs[argmax][0]:.3g} "
            f"(peak mean = {pairs[argmax][1]:.2f}%; spread = {spread:.2f} pp)."
        )
    if 0 < argmin < len(means) - 1:
        return (
            f"Relationship is SINGLE-TROUGH at sigma={pairs[argmin][0]:.3g} "
            f"(trough mean = {pairs[argmin][1]:.2f}%; spread = {spread:.2f} pp)."
        )
    return f"Relationship is NON-MONOTONIC but not cleanly single-peak (spread = {spread:.2f} pp)."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        required=True,
        help="Directory containing the noisy AG News result JSONs (e.g. results/noisy_agnews).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where the PDF/PNG/summary (or MISSING.md) get written.",
    )
    args = parser.parse_args(argv)

    runs_root = pathlib.Path(args.runs_root).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()

    if not runs_root.exists():
        write_missing(out_dir, NAME, f"Runs root does not exist: {runs_root}")
        return 1

    grouped = load_set(runs_root, predicate=_predicate)

    # Flatten across variant_keys — every passing variant contributes to the
    # same density.
    all_runs: list[tuple[RunInfo, dict]] = []
    for v, runs in grouped.items():
        all_runs.extend(runs)

    if not all_runs:
        write_missing(
            out_dir,
            NAME,
            "No AEES-Noise or AEES-Dual runs found on noisy AG News (sym20) "
            f"with len(noise_candidates)>1 under {runs_root}.",
        )
        return 1

    # Sanity: every run should share the same candidate set (the noisy AG News
    # bundle uses [0.0, 0.005, 0.01]). If not, take the union.
    cand_sets: list[tuple[float, ...]] = []
    for _info, r in all_runs:
        nc = tuple(sorted(r.get("config", {}).get("noise_candidates") or []))
        cand_sets.append(nc)
    candidates_union = sorted({c for s in cand_sets for c in s})

    if len({s for s in cand_sets}) != 1:
        # Not fatal — but worth flagging. We continue with the union.
        pass

    # ---- Collect (sigma, val-acc) pairs --------------------------------
    try:
        xs, ys, per_run = _collect_pairs(all_runs)
    except RuntimeError as e:
        write_missing(out_dir, NAME, str(e))
        return 1

    if not xs:
        write_missing(
            out_dir,
            NAME,
            "Filter matched runs but produced zero (seed, epoch) pairs "
            "(every epoch had no episodes — unexpected at AG News scale).",
        )
        return 1

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)

    # ---- Per-candidate aggregate (always computed for the summary) -----
    per_cand = _per_candidate_aggregate(all_runs, candidates_union)
    shape_callout = _classify_relationship(candidates_union, per_cand)

    # ---- Variant / seed inventory --------------------------------------
    variant_to_seeds: dict[str, list[int]] = defaultdict(list)
    for info, _r in all_runs:
        variant_to_seeds[info.variant_key].append(
            info.seed if info.seed is not None else -1
        )
    for v in variant_to_seeds:
        variant_to_seeds[v] = sorted(set(variant_to_seeds[v]))

    # ---- Figure: hexbin first, strip-plot fallback ---------------------
    fig, axes = make_figure(n_panels=1)
    ax = axes[0]

    # If sigma has effectively no spread, the hexbin will collapse to a line —
    # go straight to the strip plot.
    sigma_spread = float(x_arr.max() - x_arr.min())
    use_strip = sigma_spread <= 1e-9

    gridsize = 20
    non_empty_bins = 0
    if not use_strip:
        hb = ax.hexbin(
            x_arr,
            y_arr,
            gridsize=gridsize,
            cmap="viridis",
            mincnt=1,
        )
        counts = hb.get_array()
        non_empty_bins = int(np.sum(counts > 0))
        if non_empty_bins < 8:
            # Reset axis and fall back.
            ax.clear()
            use_strip = True

    fallback_note = ""
    if not use_strip:
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("(seed, epoch) pairs")

        # ---- Overlay: per-sigma-bin mean val-acc -----------------------
        sigma_min = float(x_arr.min())
        sigma_max = float(x_arr.max())
        edges = np.linspace(sigma_min, sigma_max, 7)
        bin_centers: list[float] = []
        bin_means: list[float] = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if i == len(edges) - 2:
                mask = (x_arr >= lo) & (x_arr <= hi)
            else:
                mask = (x_arr >= lo) & (x_arr < hi)
            if not np.any(mask):
                continue
            bin_centers.append(0.5 * (lo + hi))
            bin_means.append(float(y_arr[mask].mean()))
        if bin_centers:
            ax.plot(
                bin_centers,
                bin_means,
                color="black",
                linewidth=1.4,
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor="black",
                label="Per-bin mean val-acc",
            )
            ax.legend(loc="lower left")

        ax.set_xlabel("Mean σ in epoch")
        ax.set_ylabel("Validation accuracy (%)")
    else:
        # ---- Strip plot fallback ---------------------------------------
        # Snap each sample's mean-sigma to its nearest candidate (across runs
        # the controller selects among len(candidates) values per episode, so
        # epoch-averaged sigmas may be between candidates — snap for the strip).
        cands = candidates_union
        rng = np.random.default_rng(0)
        for i, c in enumerate(cands):
            # Find samples whose nearest candidate is c.
            nearest_idx = np.array(
                [int(np.argmin(np.abs(np.asarray(cands) - v))) for v in x_arr]
            )
            mask = nearest_idx == i
            if not np.any(mask):
                continue
            n = int(mask.sum())
            jitter = rng.uniform(-0.18, 0.18, size=n)
            ax.scatter(
                np.full(n, i) + jitter,
                y_arr[mask],
                s=18,
                alpha=0.65,
                color=PALETTE["aees_noise"],
                edgecolor="none",
            )
            ax.scatter(
                [i],
                [float(y_arr[mask].mean())],
                s=70,
                marker="_",
                color="black",
                linewidths=2.0,
                zorder=5,
            )
        ax.set_xticks(range(len(cands)))
        ax.set_xticklabels([f"{c:g}" for c in cands])
        ax.set_xlabel("σ candidate (nearest)")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_xlim(-0.6, len(cands) - 0.4)
        fallback_note = (
            f"rendered as strip plot due to sparse hexbin "
            f"(non_empty_bins={non_empty_bins}, sigma_spread={sigma_spread:.4g})"
        )

    fig.tight_layout()

    pdf, png = save_figure(fig, out_dir, NAME)

    # ---- Summary --------------------------------------------------------
    lines: list[str] = []
    lines.append(f"Resolved --runs-root: {runs_root}")
    lines.append(f"Resolved --out-dir:   {out_dir}")
    lines.append("")
    lines.append("Files included by variant:")
    for v in sorted(variant_to_seeds):
        seeds = variant_to_seeds[v]
        lines.append(
            f"  [{v}] (display='{LABEL.get(v, v)}') n_runs={len(grouped.get(v, []))} "
            f"seeds={seeds}"
        )
    lines.append("")
    lines.append(f"Total (seed, epoch) pairs: {len(xs)}")
    lines.append(f"sigma candidates (union across runs): {candidates_union}")
    if len({s for s in cand_sets}) != 1:
        lines.append(
            f"  WARNING: noise_candidates differ across runs; saw {sorted(set(cand_sets))}"
        )
    lines.append(f"sigma min/max observed (epoch-mean): "
                 f"{x_arr.min():.4g} / {x_arr.max():.4g}")
    lines.append(
        f"val-acc min/max observed (%): {y_arr.min():.2f} / {y_arr.max():.2f}")
    lines.append("")
    if not use_strip:
        lines.append(
            f"Hexbin grid: gridsize={gridsize} (≈{gridsize*gridsize} bins), "
            f"non-empty bins = {non_empty_bins}"
        )
    else:
        lines.append(f"Hexbin fallback: {fallback_note}")
    lines.append("")
    lines.append(
        "Per-σ-candidate aggregate val-acc (mean across epochs whose modal selected σ was that candidate):")
    for c in candidates_union:
        mean_pct, n_ep = per_cand[c]
        if n_ep == 0:
            lines.append(
                f"  σ={c:g}: n_epochs=0  (no epochs preferred this candidate)")
        else:
            lines.append(
                f"  σ={c:g}: mean val-acc = {mean_pct:.2f}%  (n_epochs={n_ep})")
    lines.append("")
    lines.append(shape_callout)

    # Headline contradiction check: does σ=0 (no noise) win?
    if 0.0 in candidates_union and per_cand[0.0][1] > 0:
        means_by_cand = {c: per_cand[c][0]
                         for c in candidates_union if per_cand[c][1] > 0}
        best_cand = max(means_by_cand, key=means_by_cand.get)
        if abs(best_cand - 0.0) < 1e-12:
            lines.append(
                "FLAG: σ=0 has the highest mean val-acc among candidates — "
                "this contradicts the 'noise helps' headline."
            )
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"  {pdf}")
    lines.append(f"  {png}")
    write_summary(out_dir, NAME, lines)

    return 0


if __name__ == "__main__":
    sys.exit(main())
