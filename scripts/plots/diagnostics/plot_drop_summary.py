"""Cross-regime peak-to-final drop ($\Delta_{drop}$) summary.

Single-figure summary of the $\Delta_{drop} = \\mathrm{Acc}_\\mathrm{best} -
\\mathrm{Acc}_\\mathrm{final}$ metric across all tested regimes. This is the
metric §6.1 (Peak vs Final) argues for: a configuration that improves peak
without preserving final has high $\Delta_{drop}$; one that improves both
has low $\Delta_{drop}$.

Two panels, separate y-axis scales:

    CIFAR-100 panel  — AdamW only, settings: clean, asym20, sym20, sym40.
                       Wide $\Delta_{drop}$ range (0–16 pp) requires its own
                       scale; bars are dominated by AEES-alone and Flat under
                       sym40.

    NLP panel        — AdamW, settings: AG News clean, AG News noisy (sym20),
                       SST-2 clean. Narrow $\Delta_{drop}$ range (0–3 pp) so a
                       shared scale would render the differences invisible.

Per panel, methods are grouped at each x position: Flat, the standard
scheduler for that domain (Cosine for CIFAR, Linear for NLP), AEES alone,
and the combination (Cosine+AEES / Linear+AEES). All values are mean ± 1 SD
across 5 seeds and are sourced from the canonical results tables that the
chapter-5 figures and chapter-6 discussion rely on.

CLI:
    python -m scripts.plots.diagnostics.plot_drop_summary \\
        --out-dir results/plots/diagnostics
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from dataclasses import dataclass

import numpy as np
from matplotlib.lines import Line2D

from scripts.plots._style import (
    PALETTE,
    save_figure,
    write_missing,
    write_summary,
)
import matplotlib.pyplot as plt


NAME = "drop_summary"


# ---------------------------------------------------------------------------
# Canonical regime / method data, sourced from the chapter-5 results tables.
# Each entry is (mean_drop_pp, std_drop_pp) across 5 seeds. Methods that do
# not exist for a given regime (e.g., Cosine on noisy AG News) are omitted.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodBar:
    key: str            # internal key for color lookup
    label: str          # what appears in the legend
    mean: float
    std: float


@dataclass(frozen=True)
class RegimeCol:
    label: str          # short label drawn on the x-axis
    methods: tuple[MethodBar, ...]


# CIFAR-100 — AdamW only across noise settings. The "standard scheduler"
# for CIFAR is Cosine in the thesis, so that's the second bar in each group.
CIFAR_REGIMES: tuple[RegimeCol, ...] = (
    RegimeCol(
        label="Clean",
        methods=(
            MethodBar("flat", "Flat", 0.7, 0.4),
            MethodBar("cosine", "Cosine", 0.2, 0.1),
            MethodBar("aees_lr", "AEES", 3.1, 3.6),
            MethodBar("cosine_aees", "Cosine + AEES", 0.1, 0.1),
        ),
    ),
    RegimeCol(
        label="Asym. 20%",
        methods=(
            MethodBar("flat", "Flat", 2.2, 1.0),
            MethodBar("cosine", "Cosine", 0.2, 0.1),
            MethodBar("aees_lr", "AEES", 3.7, 0.4),
            MethodBar("cosine_aees", "Cosine + AEES", 2.3, 0.7),
        ),
    ),
    RegimeCol(
        label="Sym. 20%",
        methods=(
            MethodBar("flat", "Flat", 4.5, 0.6),
            MethodBar("cosine", "Cosine", 0.6, 0.5),
            MethodBar("aees_lr", "AEES", 6.6, 0.7),
            MethodBar("cosine_aees", "Cosine + AEES", 3.6, 0.8),
        ),
    ),
    RegimeCol(
        label="Sym. 40%",
        methods=(
            MethodBar("flat", "Flat", 13.6, 0.6),
            MethodBar("cosine", "Cosine", 9.5, 1.1),
            MethodBar("aees_lr", "AEES", 15.1, 1.7),
            MethodBar("cosine_aees", "Cosine + AEES", 12.2, 1.1),
        ),
    ),
)

# NLP — AdamW only. The "standard scheduler" is Linear (warmup-linear);
# AEES-Dual is the canonical AEES variant for NLP.
NLP_REGIMES: tuple[RegimeCol, ...] = (
    RegimeCol(
        label="AG News (clean)",
        methods=(
            MethodBar("flat", "Flat", 0.46, 0.31),
            MethodBar("warmup_linear", "Linear", 0.09, 0.13),
            MethodBar("aees_dual", "AEES", 0.28, 0.27),
            MethodBar("warmup_linear_aees", "Linear + AEES", 0.04, 0.08),
        ),
    ),
    RegimeCol(
        label="AG News (sym20)",
        methods=(
            MethodBar("flat", "Flat", 2.59, 0.69),
            MethodBar("warmup_linear", "Linear", 2.40, 0.28),
            MethodBar("aees_dual", "AEES", 0.47, 0.56),
            MethodBar("warmup_linear_aees", "Linear + AEES", 0.05, 0.07),
        ),
    ),
    RegimeCol(
        label="SST-2",
        methods=(
            MethodBar("flat", "Flat", 1.56, 0.80),
            MethodBar("warmup_linear", "Linear", 0.64, 0.40),
            MethodBar("aees_dual", "AEES", 0.78, 0.61),
            MethodBar("warmup_linear_aees", "Linear + AEES", 0.37, 0.38),
        ),
    ),
)


# Map method keys to the canonical legend buckets used across both panels.
# CIFAR uses cosine / cosine_aees; NLP uses warmup_linear / warmup_linear_aees;
# but the *interpretation* of those slots is the same in each domain:
# "standard scheduler" and "standard + AEES". The legend uses the abstract
# slot label rather than the per-domain scheduler name.
LEGEND_BUCKETS: tuple[tuple[str, str, str], ...] = (
    # (key for color, key for cifar, key for nlp) ... we'll resolve a single
    # legend entry per slot rather than per regime-specific name.
    ("flat", "flat", "flat"),
    ("standard", "cosine", "warmup_linear"),
    ("aees", "aees_lr", "aees_dual"),
    ("standard_aees", "cosine_aees", "warmup_linear_aees"),
)

LEGEND_LABELS = {
    "flat": "Flat",
    "standard": "Standard scheduler",
    "aees": "AEES (alone)",
    "standard_aees": "Standard scheduler + AEES",
}

# Match the per-panel scheduler color to whichever scheme is in play. CIFAR
# uses Cosine -> blue; NLP uses Linear -> teal (per the thesis palette). We
# pick *one* color per slot for the legend; the actual bar uses the
# regime-specific PALETTE entry to stay consistent with other plots.
LEGEND_COLORS = {
    "flat": PALETTE["flat"],
    "standard": PALETTE["cosine"],
    "aees": PALETTE["aees_lr"],
    "standard_aees": PALETTE["cosine_aees"],
}


def _draw_panel(ax, regimes: tuple[RegimeCol, ...], title: str):
    """Draw one panel: grouped bars for several regimes, methods within each."""
    n_regimes = len(regimes)
    n_methods = max(len(r.methods) for r in regimes)
    bar_w = 0.18

    x = np.arange(n_regimes, dtype=float)

    # Use slot-keyed colors so the bar color matches the abstract legend
    # regardless of which concrete method (cosine vs warmup_linear,
    # aees_lr vs aees_dual) fills the slot. Order: flat / standard /
    # aees / standard+aees — same as LEGEND_BUCKETS.
    slot_colors = [LEGEND_COLORS[slot_key]
                   for slot_key, _, _ in LEGEND_BUCKETS]

    for slot in range(n_methods):
        means = []
        stds = []
        for r in regimes:
            if slot < len(r.methods):
                m = r.methods[slot]
                means.append(m.mean)
                stds.append(m.std)
            else:
                means.append(np.nan)
                stds.append(0.0)
        offset = (slot - (n_methods - 1) / 2.0) * bar_w
        ax.bar(
            x + offset, means, bar_w,
            yerr=stds, color=slot_colors[slot],
            edgecolor="black", linewidth=0.5,
            capsize=2, ecolor="black",
            error_kw={"linewidth": 0.7},
        )
        # Annotate each bar with its mean to one decimal place. Smaller font
        # because there are many bars.
        for xi, mean, std in zip(x + offset, means, stds):
            if np.isnan(mean):
                continue
            ax.text(
                xi, mean + std + 0.05 * max(1.0, max(means)),
                f"{mean:.1f}",
                ha="center", va="bottom", fontsize=6.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([r.label for r in regimes])
    ax.set_xlabel("Regime")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def _build_figure():
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.0, 0.85]}
    )

    _draw_panel(axes[0], CIFAR_REGIMES, "CIFAR-100 (AdamW)")
    _draw_panel(axes[1], NLP_REGIMES, "NLP fine-tuning (AdamW)")

    # Per-panel y-limits chosen from data with headroom for top-of-bar labels.
    axes[0].set_ylim(0, 18.5)
    axes[1].set_ylim(0, 3.5)

    axes[0].set_ylabel(r"$\Delta_{\mathrm{drop}} = "
                       r"\mathrm{Acc}_{\mathrm{best}} - "
                       r"\mathrm{Acc}_{\mathrm{final}}$ (pp)")

    # Shared abstract-slot legend at the bottom: Flat, Standard scheduler,
    # AEES, Standard + AEES. The colors are pulled from the canonical slot
    # color map rather than from any specific scheduler choice, so the
    # legend communicates the *role* of each bar rather than its specific
    # configuration. The text in the caption notes that "standard scheduler"
    # = Cosine on CIFAR and Linear on NLP.
    legend_handles = [
        Line2D(
            [0], [0], marker="s", linestyle="none",
            markerfacecolor=LEGEND_COLORS[slot],
            markeredgecolor="black", markeredgewidth=0.5,
            markersize=10,
            label=LEGEND_LABELS[slot],
        )
        for slot, _, _ in LEGEND_BUCKETS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.02), frameon=False,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=pathlib.Path, required=True,
        help="Output directory for the drop_summary outputs.",
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
        f"Drop summary across regimes (mean ± 1 SD over 5 seeds, AdamW)",
        "",
        "CIFAR-100 (AdamW):",
    ]
    for r in CIFAR_REGIMES:
        summary_lines.append(f"  Regime: {r.label}")
        for m in r.methods:
            summary_lines.append(
                f"    {m.label:25s} drop = {m.mean:5.2f} ± {m.std:.2f} pp"
            )
    summary_lines.append("")
    summary_lines.append("NLP fine-tuning (AdamW):")
    for r in NLP_REGIMES:
        summary_lines.append(f"  Regime: {r.label}")
        for m in r.methods:
            summary_lines.append(
                f"    {m.label:25s} drop = {m.mean:5.2f} ± {m.std:.2f} pp"
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
