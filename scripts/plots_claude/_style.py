"""Shared matplotlib styling for thesis figure scripts.

Every script under scripts/plots/<category>/ should `from scripts.plots_claude._style
import ...` rather than re-deriving fonts, palette, or save behavior. Keeps
the 10 figures consumed by the AEES thesis visually consistent.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pathlib
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless; never try to open a GUI


# ---------------------------------------------------------------------------
# Global rcParams. Applied once at import time.
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        # Sizing
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Fonts
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        # Spines and grid
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        # Lines and markers
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "patch.linewidth": 0.8,
        # Legend
        "legend.frameon": False,
        # PDF/PS: embed TrueType so the thesis-side latexmk doesn't substitute fonts.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# ---------------------------------------------------------------------------
# Color palette. Internal variant keys -> hex.
# Keep the mapping flat and stable; every figure uses the same color for the
# same variant. `warmup_linear` shares cosine's blue and `cosine_aees` shares
# `warmup_linear_aees`'s orange — the two never appear in the same panel.
# ---------------------------------------------------------------------------

PALETTE: dict[str, str] = {
    "flat":               "#666666",  # gray
    "cosine":             "#1f77b4",  # blue
    "linear":             "#17becf",  # teal
    "warmup_linear":      "#1f77b4",  # blue
    "aees":               "#d62728",  # red
    "aees_lr":            "#d62728",  # red
    "aees_noise":         "#9467bd",  # purple
    "aees_dual":          "#8c564b",  # brown
    "cosine_aees":        "#ff7f0e",  # orange
    "linear_aees":        "#ff7f0e",  # orange
    "warmup_linear_aees": "#ff7f0e",  # orange
}

# ---------------------------------------------------------------------------
# Display-name map: internal key -> exactly what appears in legends/ticks.
# These strings must match the prose in the thesis verbatim.
# ---------------------------------------------------------------------------

LABEL: dict[str, str] = {
    "flat":               "Flat",
    "cosine":             "Cosine",
    "linear":             "Linear",
    "warmup_linear":      "Warmup-linear",
    "aees":               "AEES",
    "aees_lr":            "AEES-LR",
    "aees_noise":         "AEES-Noise",
    "aees_dual":          "AEES-Dual",
    "cosine_aees":        "Cosine + AEES",
    "linear_aees":        "Linear + AEES",
    "warmup_linear_aees": "Warmup-linear + AEES",
}

# ---------------------------------------------------------------------------
# Arm-color helpers (Task 6: stacked-area arm-selection plot).
# Generates n light->dark shades from the base variant's color.
# ---------------------------------------------------------------------------


def arm_palette(base_key: str, n: int) -> list[str]:
    """n shades of `PALETTE[base_key]` for stacking arms low -> high.

    Lighter at the bottom, darker at the top. Always returns hex strings.
    """
    base = mcolors.to_rgb(PALETTE[base_key])
    shades: list[str] = []
    for i in range(n):
        # t in [0.25, 0.85] — avoids near-white and near-pure-base extremes.
        t = 0.25 + 0.6 * (i / max(n - 1, 1))
        rgb = tuple(1.0 - (1.0 - b) * t for b in base)
        shades.append(mcolors.to_hex(rgb))
    return shades


# ---------------------------------------------------------------------------
# Figure factories.
# ---------------------------------------------------------------------------


def make_figure(n_panels: int = 1, panel_height: float = 3.5):
    """Single- or multi-panel figure with the standard 6.5in width.

    Returns (fig, list[ax]) — even for n_panels=1, the second element is a list
    so callers don't branch on type.
    """
    if n_panels == 1:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        return fig, [ax]
    fig, axes = plt.subplots(
        1, n_panels, sharey=True, figsize=(6.5, panel_height)
    )
    return fig, list(axes)


# ---------------------------------------------------------------------------
# Saving helpers. Every figure task must emit {pdf, png, summary.txt}, or
# {MISSING.md} on failure. These are the canonical writers.
# ---------------------------------------------------------------------------


def save_figure(
    fig, out_dir: str | pathlib.Path, name: str
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write <out_dir>/<name>.pdf and <out_dir>/<name>.png. Closes the figure."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf, png


def write_summary(
    out_dir: str | pathlib.Path, name: str, lines: Iterable[str]
) -> pathlib.Path:
    """Write <out_dir>/<name>.summary.txt with one entry per line."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.summary.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_missing(
    out_dir: str | pathlib.Path, name: str, reason: str
) -> pathlib.Path:
    """Write <out_dir>/<name>.MISSING.md explaining what was missing.

    Use this in place of a partial PDF/PNG when required input data is absent.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.MISSING.md"
    path.write_text(f"# {name} — missing\n\n{reason}\n")
    return path


# ---------------------------------------------------------------------------
# Small per-figure conveniences used by multiple tasks.
# ---------------------------------------------------------------------------


def mean_std_band(ax, x, mean, std, color: str, label: str | None = None):
    """Mean line + 1-σ shaded band, with consistent alpha and width."""
    line, = ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - std, mean + std,
                    color=color, alpha=0.18, linewidth=0)
    return line


def mark_peak_final(ax, peak_xy: tuple[float, float], final_xy: tuple[float, float], color: str):
    """Hollow circle at the peak, filled square at the final epoch.

    Drawn at zorder=5 with a slightly heavier hollow ring so it remains
    visible above the size-4 filled data dots used by
    ``short_trajectory_kwargs`` on the 5-epoch AG News / SST-2 panels.
    """
    ax.plot(*peak_xy, marker="o", markerfacecolor="none",
            markeredgecolor=color, markersize=9, markeredgewidth=1.6,
            linestyle="none", zorder=5)
    ax.plot(*final_xy, marker="s", color=color, markeredgecolor="black",
            markeredgewidth=0.6, markersize=6, linestyle="none", zorder=5)


def short_trajectory_kwargs() -> dict:
    """Use these for AG News (5 epochs) and SST-2: marker-anchored line, no smoothing."""
    return {"marker": "o", "markersize": 4, "linewidth": 1.4}
