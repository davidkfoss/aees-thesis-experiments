"""Noisy AG News sigma-axis falsification: peak vs. final grouped bar chart.

Companion to plot_agnews_noise_ablation_curves.py. Each condition shows two
bars: solid for peak validation accuracy and hatched ("////") for final
validation accuracy. Error bars show +/- 1 SD across five seeds. The sigma=0
baselines are shown on the left, and the nonzero-noise conditions are grouped
on the right.

Typical reproduction command:
    uv run python -m scripts.plots.nlp_ablation.plot_agnews_noise_ablation_barplot \\
        --runs-root archived_results \\
        --out-dir reproduced_artifacts/figures/nlp_ablation

Outputs on success:
    <out-dir>/agnews_noise_ablation_barplot.pdf
    <out-dir>/agnews_noise_ablation_barplot.png
    <out-dir>/agnews_noise_ablation_barplot.summary.txt

On failure:
    <out-dir>/agnews_noise_ablation_barplot.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback

import numpy as np
from matplotlib.patches import Patch

from scripts.plots._style import (
    make_figure,
    save_figure,
    write_missing,
    write_summary,
)
from scripts.plots.nlp_ablation.plot_agnews_noise_ablation_curves import (
    PLOT_LINES,
    N_EXPECTED_SEEDS,
    LineSpec,
    _aggregate,
    _collect,
)


NAME = "agnews_noise_ablation_barplot"


def _build_figure(stats):
    fig, axes = make_figure(n_panels=1)
    ax = axes[0]

    n = len(stats)
    x = np.arange(n)
    bar_w = 0.38

    peak_means = np.array([st.mean_peak_acc * 100.0 for st in stats])
    peak_stds = np.array([st.std_peak_acc * 100.0 for st in stats])
    final_means = np.array([st.mean_final_acc * 100.0 for st in stats])
    final_stds = np.array([st.std[-1] * 100.0 for st in stats])
    colors = [st.spec.color for st in stats]

    ax.bar(
        x - bar_w / 2,
        peak_means,
        bar_w,
        yerr=peak_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        ecolor="black",
        error_kw={"linewidth": 0.8},
    )
    ax.bar(
        x + bar_w / 2,
        final_means,
        bar_w,
        yerr=final_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        hatch="////",
        capsize=3,
        ecolor="black",
        error_kw={"linewidth": 0.8},
    )

    # Annotate each bar's mean above the cap to 1 decimal place.
    for xi, mean, std in zip(x - bar_w / 2, peak_means, peak_stds):
        ax.text(xi, mean + std + 0.15, f"{mean:.1f}",
                ha="center", va="bottom", fontsize=7.5)
    for xi, mean, std in zip(x + bar_w / 2, final_means, final_stds):
        ax.text(xi, mean + std + 0.15, f"{mean:.1f}",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [st.spec.label for st in stats],
        rotation=20,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
    )
    ax.set_ylabel("Validation accuracy (%)")
    # Without Flat the lowest bar+error_bar is Baseline final (~90.2);
    # 90.0 fits with a small margin while preserving resolution in the
    # σ>0 cluster around 93.5.
    ax.set_ylim(90.0, 94.5)
    ax.grid(axis="x", visible=False)

    legend_handles = [
        Patch(facecolor="#888888", edgecolor="black",
              linewidth=0.6, label="Peak"),
        Patch(facecolor="#888888", edgecolor="black",
              linewidth=0.6, hatch="////", label="Final"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")
    fig.tight_layout()
    return fig


def _summary_lines(
    runs_root: pathlib.Path,
    stats,
    pdf_path: pathlib.Path,
    png_path: pathlib.Path,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"runs-root: {runs_root}")
    lines.append(
        "Conditions (left to right): "
        + ", ".join(st.spec.label for st in stats)
    )
    lines.append("")
    for st in stats:
        gap = (st.mean_peak_acc - st.mean_final_acc) * 100.0
        lines.append(
            f"[{st.spec.key}] (display='{st.spec.label}') "
            f"n={len(st.seeds)} seeds={st.seeds}"
        )
        lines.append(
            f"  peak  mean={st.mean_peak_acc * 100.0:.2f}%  "
            f"std={st.std_peak_acc * 100.0:.2f}%"
        )
        lines.append(
            f"  final mean={st.mean_final_acc * 100.0:.2f}%  "
            f"std={st.std[-1] * 100.0:.2f}%"
        )
        lines.append(f"  peak-to-final gap = {gap:.2f} pp")
        lines.append("")

    finals_pos = [
        st.mean_final_acc * 100.0
        for st in stats
        if st.spec.key != "adamw_wl"
    ]
    spread = max(finals_pos) - min(finals_pos)
    lines.append(f"σ>0 cluster final-epoch spread (max-min): {spread:.3f} pp")
    lines.append("")
    lines.append("outputs:")
    lines.append(f"  {pdf_path}")
    lines.append(f"  {png_path}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=pathlib.Path, required=True,
        help="Parent directory; walked recursively for noisy AG News runs.",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, required=True,
        help=f"Output directory for {NAME}.{{pdf,png,summary.txt}}",
    )
    args = parser.parse_args(argv)

    try:
        if not args.runs_root.exists():
            write_missing(
                args.out_dir, NAME,
                f"--runs-root does not exist: {args.runs_root}",
            )
            return 1

        per_key, errors = _collect(args.runs_root)
        if errors:
            reason_lines = [
                "Required runs are missing or malformed; refusing to emit a partial PDF.",
                "",
                f"--runs-root: {args.runs_root}",
                "",
                "Issues:",
                *[f"- {e}" for e in errors],
            ]
            write_missing(args.out_dir, NAME, "\n".join(reason_lines))
            return 1

        stats = [_aggregate(spec, per_key[spec.key]) for spec in PLOT_LINES]
        fig = _build_figure(stats)
    except Exception as exc:
        reason = (
            f"Failed to build {NAME}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, NAME, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, NAME)
    summary_path = write_summary(
        args.out_dir, NAME, _summary_lines(
            args.runs_root, stats, pdf_path, png_path)
    )
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
