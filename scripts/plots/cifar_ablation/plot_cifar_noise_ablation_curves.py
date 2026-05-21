"""Noisy CIFAR-100 small-noise-injection ablation: validation trajectories.

Compares two AEES controller configurations on noisy CIFAR-100 (symmetric 20%
label noise), matched on episode length (200), context mode (trend), and the
100-epoch budget, over 5 seeds:

  - No injection (``lronly``):        noise_candidates = [0.0]            (LR-only)
  - Small-noise injection (``smallnoise``): noise_candidates = [0.0, 0.0025, 0.005]

The figure plots validation accuracy over the 100 epochs as a cross-seed mean
with a +/- 1 sample-SD band per condition. It shows that allowing the
controller to inject small gradient noise yields a lower validation-accuracy
trajectory (lower peak and lower final) than the no-injection controller.

Caveat: both conditions use the LR candidate grid {0.5, 1.0, 2.0} except the
no-injection seed 0, which used {0.5, 1.0, 1.5}. The gradient-noise injection
axis is the systematic difference between the two conditions.

This module also hosts the shared data loader used by the companion barplot
(``plot_cifar_noise_ablation_barplot.py``).

Typical reproduction command:
    uv run python -m scripts.plots.cifar_ablation.plot_cifar_noise_ablation_curves \
        --runs-root archived_results/cifar_noise_ablation \
        --out-dir reproduced_artifacts/figures/cifar_ablation

Outputs on success:
    <out-dir>/cifar_noise_ablation_curves.pdf
    <out-dir>/cifar_noise_ablation_curves.png
    <out-dir>/cifar_noise_ablation_curves.summary.txt

On failure:
    <out-dir>/cifar_noise_ablation_curves.MISSING.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np

from scripts.plots._style import (
    PALETTE,
    make_figure,
    mark_peak_final,
    mean_std_band,
    save_figure,
    write_missing,
    write_summary,
)


NAME = "cifar_noise_ablation_curves"

# Two conditions, in display order. Color keys reuse the shared palette:
# aees_lr for the LR-only (no-injection) controller, aees_noise for injection.
TAGS: list[str] = ["lronly", "smallnoise"]
LABEL: dict[str, str] = {
    "lronly": "No gradient noise",
    "smallnoise": "Gradient-noise injection",
}
COLOR_KEY: dict[str, str] = {"lronly": "aees_lr", "smallnoise": "aees_noise"}

_SEED_RE = re.compile(r"_seed(\d+)$")


@dataclass
class ConditionData:
    tag: str
    label: str
    color_key: str
    seeds: list[int]
    best: np.ndarray   # [n_seeds], percent
    final: np.ndarray  # [n_seeds], percent
    traj: np.ndarray   # [n_seeds, n_epochs], percent
    lr_candidates: list        # modal (most common) grid across seeds
    noise_candidates: list     # modal grid across seeds
    lr_grids: list             # all distinct LR grids seen (usually one)
    noise_grids: list          # all distinct noise grids seen


def _grid_stats(grids) -> tuple[list, list]:
    """Given an iterable of per-seed candidate grids, return (modal, distinct).

    ``modal`` is the most common grid (the representative shown in summaries);
    ``distinct`` is every unique grid seen, sorted, so a per-seed oddball is
    surfaced rather than silently dropped.
    """
    counter = Counter(tuple(g) for g in grids if g is not None)
    if not counter:
        return [], []
    modal = list(counter.most_common(1)[0][0])
    distinct = [list(g) for g in sorted(counter)]
    return modal, distinct


def load_conditions(
    runs_root: pathlib.Path,
) -> tuple[dict[str, ConditionData], list[str]]:
    """Load both conditions from ``<runs-root>/seed_*/noisy_coarse_<tag>_ep200_trend_seed*.json``.

    Returns (conditions_by_tag, missing_messages). A non-empty missing list
    signals failure.
    """
    runs_root = pathlib.Path(runs_root)
    conditions: dict[str, ConditionData] = {}
    missing: list[str] = []

    for tag in TAGS:
        pattern = f"seed_*/noisy_coarse_{tag}_ep200_trend_seed*.json"
        files = sorted(runs_root.glob(pattern))
        if not files:
            missing.append(
                f"{tag}: no files match {pattern} under {runs_root}")
            continue

        per_seed: dict[int, dict] = {}
        for f in files:
            try:
                record = json.loads(f.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                missing.append(f"{tag}: unreadable ({exc!r}) at {f}")
                continue
            m = _SEED_RE.search(f.stem)
            seed = int(m.group(1)) if m else record.get("seed")
            best = record.get("best_val_accuracy")
            final = record.get("final_val_accuracy")
            vals = record.get("val_accuracies")
            if best is None or final is None or not vals:
                missing.append(
                    f"{tag} seed{seed}: missing best/final_val_accuracy or "
                    f"val_accuracies in {f}"
                )
                continue
            cfg = record.get("config", {})
            per_seed[int(seed)] = {
                "best": float(best) * 100.0,
                "final": float(final) * 100.0,
                "vals": [float(v) * 100.0 for v in vals],
                "lr": cfg.get("lr_candidates"),
                "noise": cfg.get("noise_candidates"),
            }

        if not per_seed:
            missing.append(f"{tag}: no usable runs loaded under {runs_root}")
            continue

        seeds = sorted(per_seed)
        # Truncate trajectories to the shortest seed so they stack cleanly.
        min_len = min(len(per_seed[s]["vals"]) for s in seeds)
        traj = np.array([per_seed[s]["vals"][:min_len]
                        for s in seeds], dtype=float)
        lr_modal, lr_grids = _grid_stats(per_seed[s]["lr"] for s in seeds)
        noise_modal, noise_grids = _grid_stats(
            per_seed[s]["noise"] for s in seeds)
        conditions[tag] = ConditionData(
            tag=tag,
            label=LABEL[tag],
            color_key=COLOR_KEY[tag],
            seeds=seeds,
            best=np.array([per_seed[s]["best"] for s in seeds], dtype=float),
            final=np.array([per_seed[s]["final"] for s in seeds], dtype=float),
            traj=traj,
            lr_candidates=lr_modal,
            noise_candidates=noise_modal,
            lr_grids=lr_grids,
            noise_grids=noise_grids,
        )

    return conditions, missing


def _std(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Sample SD (ddof=1); zero where there is a single observation."""
    n = arr.shape[axis]
    if n < 2:
        return np.zeros(arr.shape[1] if arr.ndim > 1 else (), dtype=float)
    return np.std(arr, axis=axis, ddof=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default="archived_results/cifar_noise_ablation",
        help="Directory containing the cifar_noise_ablation seed_*/ result JSONs.",
    )
    parser.add_argument(
        "--out-dir",
        default="reproduced_artifacts/figures/cifar_ablation",
        help="Directory where the PDF/PNG/summary (or MISSING.md) get written.",
    )
    args = parser.parse_args(argv)

    runs_root = pathlib.Path(args.runs_root).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()

    if not runs_root.exists():
        write_missing(out_dir, NAME, f"Runs root does not exist: {runs_root}")
        return 1

    conditions, missing = load_conditions(runs_root)
    if missing or any(t not in conditions for t in TAGS):
        write_missing(
            out_dir, NAME,
            "Could not load both conditions:\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + f"\n\nResolved runs root: {runs_root}",
        )
        return 1

    # ---- Figure --------------------------------------------------------
    fig, axes = make_figure(n_panels=1)
    ax = axes[0]

    for tag in TAGS:
        c = conditions[tag]
        color = PALETTE[c.color_key]
        n_epochs = c.traj.shape[1]
        x = np.arange(1, n_epochs + 1)
        mean = c.traj.mean(axis=0)
        std = _std(c.traj, axis=0)
        mean_std_band(ax, x, mean, std, color=color,
                      label=f"{c.label} (n={len(c.seeds)})")
        peak_i = int(np.argmax(mean))
        mark_peak_final(ax, (x[peak_i], mean[peak_i]),
                        (x[-1], mean[-1]), color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xlim(1, max(c.traj.shape[1] for c in conditions.values()))
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dir, NAME)

    # ---- Summary -------------------------------------------------------
    lines: list[str] = []
    lines.append(f"Resolved --runs-root: {runs_root}")
    lines.append(
        "Noisy CIFAR-100 (symmetric 20%), ep200, trend context, 100 epochs.")
    lines.append(
        "Markers: hollow circle = mean-curve peak, filled square = final epoch.")
    lines.append("")
    for tag in TAGS:
        c = conditions[tag]
        mean = c.traj.mean(axis=0)
        peak_i = int(np.argmax(mean))
        lines.append(f"[{c.label}] ({tag}) n={len(c.seeds)} seeds={c.seeds}")
        lines.append(
            f"  lr_candidates(modal)={c.lr_candidates}  noise_candidates(modal)={c.noise_candidates}")
        if len(c.lr_grids) > 1:
            lines.append(f"  NOTE: LR grid varies across seeds: {c.lr_grids}")
        lines.append(
            f"  peak(best_val) mean={c.best.mean():.2f}  final mean={c.final.mean():.2f}  "
            f"mean-curve peak at epoch {peak_i + 1} ({mean[peak_i]:.2f}%)"
        )
        lines.append("")
    lines.append(
        "Caveat: both conditions use the LR grid {0.5,1.0,2.0} except no-injection "
        "seed 0 ({0.5,1.0,1.5}); the gradient-noise injection axis is the systematic difference."
    )
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"  {pdf}")
    lines.append(f"  {png}")
    write_summary(out_dir, NAME, lines)

    return 0


if __name__ == "__main__":
    sys.exit(main())
