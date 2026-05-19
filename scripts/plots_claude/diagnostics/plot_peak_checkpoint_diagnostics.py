"""Peak-checkpoint memorization diagnostics on noisy \\cifar{} (§5.3.4).

Visualises Table~5.6 of the thesis as a scatter of corrupted-subset accuracies
at the peak validation checkpoint. The two axes are the two corrupted-subset
diagnostics:

    x-axis: Corr. vs noisy (accuracy of the model against the corrupted
            training label). Lower is better — high values indicate the
            model has memorized the corrupted targets.
    y-axis: Corr. vs clean (accuracy against the original clean label).
            Higher is better — high values indicate the model preserves the
            underlying clean-label structure on examples that were corrupted.

A method that finds a useful checkpoint *before* memorizing the corrupted
labels sits in the upper-left of each panel (low Corr. vs noisy, high
Corr. vs clean). A method whose peak coincides with full memorization sits
in the lower-right (Corr. vs noisy near 100%, Corr. vs clean near 0%).

Two panels:
    Asym. 20% — structured corruption; Cosine reaches its peak validation
                checkpoint late, after the model has fully fit the
                corrupted labels. AEES variants reach peaks early and
                preserve substantial clean-label accuracy.
    Sym. 40%  — random corruption; the model cannot fit the corrupted
                labels through any compact mapping, so Corr. vs noisy
                stays low for every method. The Corr. vs clean column is
                the more informative axis here.

CLI:
    python -m scripts.plots_claude.diagnostics.plot_peak_checkpoint_diagnostics \\
        --out-dir results/plots_claude/diagnostics
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.plots_claude._style import (
    save_figure,
    write_missing,
    write_summary,
)


NAME = "peak_checkpoint_diagnostics"


# Slot colors match the drop_summary figure for cross-figure consistency.
SLOT_COLORS = {
    "flat":         "#666666",
    "cosine":       "#1f77b4",
    "aees":         "#d62728",
    "cosine_aees":  "#ff7f0e",
}
SLOT_LABELS = {
    "flat":         "Flat",
    "cosine":       "Cosine",
    "aees":         "AEES",
    "cosine_aees":  "Cosine + AEES",
}
# Marker per optimizer.
OPT_MARKERS = {"AdamW": "o", "SGD+M": "s"}


@dataclass(frozen=True)
class Point:
    setting: str
    optimizer: str
    method: str   # one of SLOT_COLORS keys
    corr_noisy_mean: float
    corr_noisy_std: float
    corr_clean_mean: float
    corr_clean_std: float


# Data sourced verbatim from Table 5.6 of the thesis. We plot AdamW only:
# the SGD+M Cosine + AEES point on asym20 has a bimodal seed distribution
# (some seeds peak early, some at the end) and a 36.1 pp std on Corr. noisy,
# which produces error bars that span most of the panel and obscure the
# clean Cosine-vs-AEES contrast. The SGD+M numbers are still reported in
# Table 5.6 for reference.
POINTS: tuple[Point, ...] = (
    # Asym. 20% — AdamW.
    Point("asym20", "AdamW", "flat",         37.0, 5.8, 45.8, 3.6),
    Point("asym20", "AdamW", "aees",         29.0, 8.6, 53.8, 5.1),
    Point("asym20", "AdamW", "cosine",      100.0, 0.0,  0.0, 0.0),
    Point("asym20", "AdamW", "cosine_aees",  24.6, 7.7, 57.3, 4.5),
    # Sym. 40% — AdamW.
    Point("sym40",  "AdamW", "flat",          3.6, 1.7, 48.4, 1.3),
    Point("sym40",  "AdamW", "aees",          3.1, 1.1, 52.8, 0.9),
    Point("sym40",  "AdamW", "cosine",        1.9, 0.2, 53.0, 0.5),
    Point("sym40",  "AdamW", "cosine_aees",   1.9, 0.4, 53.5, 0.7),
)


def _draw_panel(ax, points: list[Point], setting: str):
    """One scatter panel. Color = method; marker = optimizer."""
    for p in points:
        color = SLOT_COLORS[p.method]
        marker = OPT_MARKERS[p.optimizer]
        ax.errorbar(
            p.corr_noisy_mean, p.corr_clean_mean,
            xerr=p.corr_noisy_std, yerr=p.corr_clean_std,
            fmt=marker, color=color, markeredgecolor="black",
            markeredgewidth=0.6, markersize=10,
            ecolor=color, elinewidth=1.0, capsize=3,
            alpha=0.9, zorder=3,
        )

    # Shade the "good" corner (low Corr.noisy, high Corr.clean) and the
    # "bad" corner (high Corr.noisy, low Corr.clean) for visual orientation.
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    # We'll set these explicitly per panel after drawing.

    ax.set_xlabel(r"Corr. vs noisy (%) $\downarrow$")
    if setting == "asym20":
        ax.set_ylabel(r"Corr. vs clean (%) $\uparrow$")
        ax.set_title("Asym. 20%")
    else:
        ax.set_title("Sym. 40%")


def _build_figure():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    asym20_pts = [p for p in POINTS if p.setting == "asym20"]
    sym40_pts = [p for p in POINTS if p.setting == "sym40"]

    # Asym 20%: widely spread (Cosine at 100,0 vs AEES at ~25,55).
    # Full 0–100 range on both axes is needed to show the contrast.
    axes[0].set_xlim(-5, 110)
    axes[0].set_ylim(-5, 70)
    _draw_panel(axes[0], asym20_pts, "asym20")

    # Sym 40%: tight cluster (Corr.noisy ~2–7%, Corr.clean ~47–53%).
    # Zoom in so the differences are visible.
    axes[1].set_xlim(0, 14)
    axes[1].set_ylim(40, 60)
    _draw_panel(axes[1], sym40_pts, "sym40")

    # Direction-of-preference annotations on the asym20 panel only — the
    # sym40 panel is too zoomed-in for them to fit cleanly.
    ax0 = axes[0]
    ax0.annotate(
        "better",
        xy=(0, 65), xytext=(0, 65),
        ha="left", va="top", fontsize=9, fontstyle="italic",
        color="#226622",
    )
    ax0.annotate(
        "memorized",
        xy=(100, 0), xytext=(100, 5),
        ha="right", va="bottom", fontsize=9, fontstyle="italic",
        color="#882222",
    )

    # Method-only legend (AdamW everywhere; no need for an optimizer key).
    method_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=SLOT_COLORS[key],
            markeredgecolor="black", markeredgewidth=0.5,
            markersize=10, label=SLOT_LABELS[key],
        )
        for key in ("flat", "cosine", "aees", "cosine_aees")
    ]
    fig.legend(
        handles=method_handles,
        loc="lower center", ncol=len(method_handles),
        bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=pathlib.Path, required=True,
        help="Output directory for the peak_checkpoint_diagnostics outputs.",
    )
    args = parser.parse_args(argv)

    try:
        fig = _build_figure()
    except Exception as exc:
        reason = (
            f"Failed to build {NAME}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, NAME, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, NAME)

    summary_lines: list[str] = [
        "Peak-checkpoint memorization diagnostics (CIFAR-100 noisy labels)",
        "Source: Table 5.6 of the thesis (corrupted-subset accuracies at the",
        "        peak-validation checkpoint, mean ± SD over 5 seeds).",
        "",
    ]
    for setting in ("asym20", "sym40"):
        summary_lines.append(f"Setting: {setting}")
        for opt in ("AdamW", "SGD+M"):
            summary_lines.append(f"  Optimizer: {opt}")
            for p in POINTS:
                if p.setting != setting or p.optimizer != opt:
                    continue
                summary_lines.append(
                    f"    {SLOT_LABELS[p.method]:18s} "
                    f"Corr. noisy = {p.corr_noisy_mean:5.1f} ± {p.corr_noisy_std:4.1f}  "
                    f"Corr. clean = {p.corr_clean_mean:5.1f} ± {p.corr_clean_std:4.1f}"
                )
        summary_lines.append("")
    summary_lines.append("outputs:")
    summary_lines.append(f"  - {pdf_path}")
    summary_lines.append(f"  - {png_path}")
    write_summary(args.out_dir, NAME, summary_lines)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
