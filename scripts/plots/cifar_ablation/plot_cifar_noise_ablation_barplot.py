"""Noisy CIFAR-100 small-noise-injection ablation: peak vs. final bar chart.

Companion to ``plot_cifar_noise_ablation_curves.py``. Two conditions on noisy
CIFAR-100 (symmetric 20% label noise), matched on episode length (200), context
mode (trend), and the 100-epoch budget, over 5 seeds:

  - No injection (``lronly``):              noise_candidates = [0.0]
  - Small-noise injection (``smallnoise``): noise_candidates = [0.0, 0.0025, 0.005]

Each condition shows two bars: solid for peak (best) validation accuracy and
hatched ("////") for final-epoch validation accuracy. Error bars are +/- 1
sample SD across the seeds. The figure shows that injecting small gradient
noise lowers both peak and final validation accuracy.

Caveat: both conditions use the LR candidate grid {0.5, 1.0, 2.0} except the
no-injection seed 0, which used {0.5, 1.0, 1.5}. The gradient-noise injection
axis is the systematic difference between the two conditions.

Typical reproduction command:
    uv run python -m scripts.plots.cifar_ablation.plot_cifar_noise_ablation_barplot \
        --runs-root archived_results/cifar_noise_ablation \
        --out-dir reproduced_artifacts/figures/cifar_ablation

Outputs on success:
    <out-dir>/cifar_noise_ablation_barplot.pdf
    <out-dir>/cifar_noise_ablation_barplot.png
    <out-dir>/cifar_noise_ablation_barplot.summary.txt

On failure:
    <out-dir>/cifar_noise_ablation_barplot.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from matplotlib.patches import Patch

from scripts.plots._style import (
    PALETTE,
    make_figure,
    save_figure,
    write_missing,
    write_summary,
)
from scripts.plots.cifar_ablation.plot_cifar_noise_ablation_curves import (
    TAGS,
    load_conditions,
)


NAME = "cifar_noise_ablation_barplot"


def _std(arr: np.ndarray) -> float:
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


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

    x = np.arange(len(TAGS))
    bar_w = 0.38

    peak_means = np.array([conditions[t].best.mean() for t in TAGS])
    peak_stds = np.array([_std(conditions[t].best) for t in TAGS])
    final_means = np.array([conditions[t].final.mean() for t in TAGS])
    final_stds = np.array([_std(conditions[t].final) for t in TAGS])
    colors = [PALETTE[conditions[t].color_key] for t in TAGS]

    ax.bar(
        x - bar_w / 2, peak_means, bar_w, yerr=peak_stds,
        color=colors, edgecolor="black", linewidth=0.6,
        capsize=3, ecolor="black", error_kw={"linewidth": 0.8},
    )
    ax.bar(
        x + bar_w / 2, final_means, bar_w, yerr=final_stds,
        color=colors, edgecolor="black", linewidth=0.6, hatch="////",
        capsize=3, ecolor="black", error_kw={"linewidth": 0.8},
    )

    for xi, mean, std in zip(x - bar_w / 2, peak_means, peak_stds):
        ax.text(xi, mean + std + 0.2, f"{mean:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, mean, std in zip(x + bar_w / 2, final_means, final_stds):
        ax.text(xi, mean + std + 0.2, f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([conditions[t].label for t in TAGS])
    ax.set_ylabel("Validation accuracy (%)")
    all_vals = np.concatenate([peak_means, final_means])
    all_err = np.concatenate([peak_stds, final_stds])
    ax.set_ylim(max(0.0, float((all_vals - all_err).min()) - 4.0),
                float((all_vals + all_err).max()) + 4.0)
    ax.grid(axis="x", visible=False)

    legend_handles = [
        Patch(facecolor="#888888", edgecolor="black", linewidth=0.6, label="Peak"),
        Patch(facecolor="#888888", edgecolor="black", linewidth=0.6,
              hatch="////", label="Final"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dir, NAME)

    # ---- Summary -------------------------------------------------------
    lines: list[str] = []
    lines.append(f"Resolved --runs-root: {runs_root}")
    lines.append("Noisy CIFAR-100 (symmetric 20%), ep200, trend context, 100 epochs.")
    lines.append("Bars: solid = peak (best val), hatched = final; error bars = +/- 1 sample SD.")
    lines.append("")
    for t in TAGS:
        c = conditions[t]
        gap = c.best.mean() - c.final.mean()
        lines.append(f"[{c.label}] ({t}) n={len(c.seeds)} seeds={c.seeds}")
        lines.append(f"  noise_candidates(modal)={c.noise_candidates}  lr_candidates(modal)={c.lr_candidates}")
        if len(c.lr_grids) > 1:
            lines.append(f"  NOTE: LR grid varies across seeds: {c.lr_grids}")
        lines.append(f"  peak  mean={c.best.mean():.2f}%  std={_std(c.best):.2f}%")
        lines.append(f"  final mean={c.final.mean():.2f}%  std={_std(c.final):.2f}%")
        lines.append(f"  peak-to-final gap = {gap:.2f} pp")
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
