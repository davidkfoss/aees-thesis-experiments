"""Peak-checkpoint memorization diagnostics on noisy CIFAR-100.

Scatters the corrupted-subset accuracies at the peak-validation checkpoint,
computed directly from the rerun JSONs under ``archived_results/checkpointing``.
The rerun files stop at the peak-validation checkpoint, so each file's final
diagnostics are the peak-checkpoint diagnostics. Loading and aggregation are
delegated to ``scripts.tables.make_checkpointing_tables``, the source of truth
for the corresponding checkpoint-diagnostics table, so this figure and that
table use the same aggregated values.

The two axes are the corrupted-subset diagnostics:

    x-axis: Corr. vs noisy, the accuracy on corrupted training examples against
            their corrupted labels. Lower values indicate less prediction of
            the imposed noisy targets.
    y-axis: Corr. vs clean, the accuracy on the same corrupted examples against
            their original clean labels. Higher values indicate better
            preservation of the underlying clean-label structure.

A method that finds a useful checkpoint before memorizing the corrupted labels
sits toward the upper-left of each panel: low Corr. vs noisy and high Corr. vs
clean. A method whose peak coincides with full corrupted-label fitting sits
toward the lower-right: Corr. vs noisy near 100% and Corr. vs clean near 0%.

Two panels:
    Asym. 20% — structured corruption; Cosine reaches its peak-validation
                checkpoint late, after the model has fully fit the corrupted
                labels. AEES variants reach peaks earlier and preserve
                substantial clean-label accuracy.
    Sym. 40%  — random corruption; corrupted-label accuracy remains low for
                every method, so Corr. vs clean is the more informative axis.

Typical reproduction command:
    uv run python -m scripts.plots.diagnostics.plot_peak_checkpoint_diagnostics \\
        --runs-root archived_results/checkpointing \\
        --out-dir reproduced_artifacts/figures/diagnostics

Outputs on success:
    <out-dir>/peak_checkpoint_diagnostics.pdf
    <out-dir>/peak_checkpoint_diagnostics.png
    <out-dir>/peak_checkpoint_diagnostics.summary.txt

On failure:
    <out-dir>/peak_checkpoint_diagnostics.MISSING.md
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

from scripts.plots._style import (
    save_figure,
    write_missing,
    write_summary,
)
from scripts.tables.make_checkpointing_tables import aggregate, parse_result


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

# Translate the table module's aggregation keys onto this figure's vocabulary.
# Optimizer keys -> the labels OPT_MARKERS expects; variant keys -> the SLOT_*
# method keys (note the table's "baseline" is this figure's "flat").
OPT_KEY_TO_LABEL = {"adamw": "AdamW", "sgd": "SGD+M"}
VARIANT_KEY_TO_SLOT = {
    "baseline":    "flat",
    "aees":        "aees",
    "cosine":      "cosine",
    "cosine_aees": "cosine_aees",
}

# Every cell the figure relies on: both optimizers (the .summary.txt reports
# both) for both settings and all four methods. Missing/short cells fail loudly.
EXPECTED_SETTINGS = ("asym20", "sym40")
EXPECTED_OPTIMIZERS = ("adamw", "sgd")
EXPECTED_VARIANTS = ("baseline", "aees", "cosine", "cosine_aees")

# Order in which methods are listed in the summary (table's variant order).
SUMMARY_METHOD_ORDER = ("flat", "aees", "cosine", "cosine_aees")


@dataclass(frozen=True)
class Point:
    setting: str
    optimizer: str
    method: str   # one of SLOT_COLORS keys
    corr_noisy_mean: float
    corr_noisy_std: float
    corr_clean_mean: float
    corr_clean_std: float


def _load_points(runs_root: pathlib.Path) -> list[Point]:
    """Read and aggregate the checkpointing reruns into per-cell Points.

    Globs ``<runs_root>/cifar100_asym20/*.json`` and ``.../cifar100_sym40``,
    reuses the table module's ``parse_result``/``aggregate`` so the numbers are
    identical to Table 5.6, then maps each aggregated cell onto a Point. The
    aggregated means/SDs are fractions in [0, 1]; we scale to percent here.

    Fails loudly: missing input directories, parse errors, and any
    (setting, optimizer, method) cell that is absent or has <2 seeds all raise.
    """
    json_paths = sorted(
        list((runs_root / "cifar100_asym20").glob("*.json"))
        + list((runs_root / "cifar100_sym40").glob("*.json"))
    )
    if not json_paths:
        raise FileNotFoundError(
            f"No JSON files found under {runs_root / 'cifar100_asym20'} or "
            f"{runs_root / 'cifar100_sym40'}"
        )

    rows = [parse_result(path) for path in json_paths]
    aggregated = aggregate(rows)

    # Index by the table's keys so completeness can be checked exactly.
    cells = {
        (c["setting_key"], c["optimizer_key"], c["variant_key"]): c
        for c in aggregated
    }

    problems: list[str] = []
    for setting in EXPECTED_SETTINGS:
        for optimizer in EXPECTED_OPTIMIZERS:
            for variant in EXPECTED_VARIANTS:
                cell = cells.get((setting, optimizer, variant))
                if cell is None:
                    problems.append(
                        f"absent cell: {setting}/{optimizer}/{variant}")
                elif cell["n"] < 2:
                    problems.append(
                        f"cell {setting}/{optimizer}/{variant} has only "
                        f"{cell['n']} seed(s); need >= 2 for a sample SD"
                    )
    if problems:
        raise ValueError(
            "Incomplete checkpointing data; cannot build "
            f"{NAME}:\n  - " + "\n  - ".join(problems)
        )

    points: list[Point] = []
    for cell in aggregated:
        points.append(Point(
            setting=cell["setting_key"],
            optimizer=OPT_KEY_TO_LABEL[cell["optimizer_key"]],
            method=VARIANT_KEY_TO_SLOT[cell["variant_key"]],
            corr_noisy_mean=cell["corr_noisy_mean"] * 100.0,
            corr_noisy_std=cell["corr_noisy_sd"] * 100.0,
            corr_clean_mean=cell["corr_clean_mean"] * 100.0,
            corr_clean_std=cell["corr_clean_sd"] * 100.0,
        ))
    return points


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


def _build_figure(points: list[Point]):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # We plot AdamW only: the SGD+M Cosine + AEES point on asym20 has a bimodal
    # seed distribution (some seeds peak early, some at the end) and a ~36 pp std
    # on Corr. noisy, which produces error bars that span most of the panel and
    # obscure the clean Cosine-vs-AEES contrast. The SGD+M numbers are still
    # reported in the .summary.txt for reference.
    asym20_pts = [p for p in points if p.setting ==
                  "asym20" and p.optimizer == "AdamW"]
    sym40_pts = [p for p in points if p.setting ==
                 "sym40" and p.optimizer == "AdamW"]

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
        "--runs-root", type=pathlib.Path, default=pathlib.Path("results/checkpointing"),
        help="Directory holding cifar100_asym20/ and cifar100_sym40/ rerun JSONs.",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, required=True,
        help="Output directory for the peak_checkpoint_diagnostics outputs.",
    )
    args = parser.parse_args(argv)

    try:
        points = _load_points(args.runs_root)
        fig = _build_figure(points)
    except Exception as exc:
        reason = (
            f"Failed to build {NAME}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, NAME, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, NAME)

    by_cell = {(p.setting, p.optimizer, p.method): p for p in points}

    summary_lines: list[str] = [
        "Peak-checkpoint memorization diagnostics (CIFAR-100 noisy labels)",
        f"Source: {args.runs_root}/ reruns (final diagnostics == peak-checkpoint",
        "        diagnostics), aggregated via scripts.tables.make_checkpointing_tables.",
        "        Values are mean ± sample SD (ddof=1) over seeds.",
        "",
    ]
    for setting in ("asym20", "sym40"):
        summary_lines.append(f"Setting: {setting}")
        for opt in ("AdamW", "SGD+M"):
            summary_lines.append(f"  Optimizer: {opt}")
            for method in SUMMARY_METHOD_ORDER:
                p = by_cell[(setting, opt, method)]
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
