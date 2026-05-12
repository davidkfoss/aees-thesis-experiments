"""AEES method overview block diagram.

Pure schematic — no input data. Renders a single-panel block diagram with
three labeled regions:

  1. Training loop (left): mini-batch -> forward/backward -> optimizer ->
     parameter update.
  2. Episode controller (right): two stacked Episodic Bandit Controllers,
     one per active axis (LR-multiplier, Gradient noise).
  3. Bridge (center): selected arm values flow controller -> training; reward
     computed from the EMA-loss trace flows training -> controllers after
     each episode of E steps.

CLI:
    python -m scripts.plots_claude.method.plot_aees_overview --out-dir results/plots/method/

Outputs (on success):
    aees_overview.pdf
    aees_overview.png
    aees_overview.summary.txt

Outputs (on failure):
    aees_overview.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.plots_claude._style import make_figure, save_figure, write_missing, write_summary


# ---------------------------------------------------------------------------
# Required exact labels (eight). The acceptance criteria checks these strings
# verbatim, so we keep them as constants and reuse for both rendering and the
# summary file.
# ---------------------------------------------------------------------------

LABEL_MINIBATCH = "Mini-batch"
LABEL_FORWARD = "Forward / backward"
LABEL_OPTIMIZER = "Base optimizer"
LABEL_UPDATE = "Parameter update"
LABEL_BANDIT_LR = "Episodic Bandit Controller (LR-multiplier)"
LABEL_BANDIT_NOISE = "Episodic Bandit Controller (Gradient noise)"
LABEL_ARM_VALUES = "Selected arm values"
LABEL_REWARD = "Reward $r_e$ from EMA loss"

REQUIRED_LABELS = [
    LABEL_MINIBATCH,
    LABEL_FORWARD,
    LABEL_OPTIMIZER,
    LABEL_UPDATE,
    LABEL_BANDIT_LR,
    LABEL_BANDIT_NOISE,
    LABEL_ARM_VALUES,
    LABEL_REWARD,
]

# Two-tone palette: cool slate for the training loop, warm sand for the
# bandit controllers. Matches the thesis "schematic" aesthetic.
COLOR_TRAIN_FILL = "#dde6ef"
COLOR_TRAIN_EDGE = "#3b5266"
COLOR_BANDIT_FILL = "#f3e2c4"
COLOR_BANDIT_EDGE = "#8a6a2b"
COLOR_TEXT = "#1a1a1a"
COLOR_ARROW_FWD = "#3b5266"  # arm-values direction (controller -> train)
COLOR_ARROW_REV = "#8a6a2b"  # reward direction (train -> controller)


def _add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 9.0,
    fontweight: str = "normal",
):
    """Draw a rounded rectangle centered at xy with the given label."""
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=COLOR_TEXT,
        wrap=True,
    )


def _add_arrow(
    ax,
    p_from: tuple[float, float],
    p_to: tuple[float, float],
    *,
    color: str,
    style: str = "-|>",
    linewidth: float = 1.2,
    linestyle: str = "-",
    mutation_scale: float = 12.0,
    connectionstyle: str = "arc3,rad=0.0",
):
    arrow = FancyArrowPatch(
        p_from,
        p_to,
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        mutation_scale=mutation_scale,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def _draw_region_outline(ax, xy_lo, xy_hi, *, label, color):
    """Light dashed bracket for a region with a small italic label."""
    x0, y0 = xy_lo
    x1, y1 = xy_hi
    rect = FancyBboxPatch(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        linewidth=0.8,
        linestyle=(0, (4, 3)),
        facecolor="none",
        edgecolor=color,
    )
    ax.add_patch(rect)
    ax.text(
        (x0 + x1) / 2,
        y1 + 0.10,
        label,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontstyle="italic",
        color=color,
    )


def _build_figure():
    fig, axes = make_figure(n_panels=1)
    ax = axes[0]
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 6.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    # ------------------------------------------------------------------ training loop (left column)
    train_x = 1.9
    box_w = 2.6
    box_h = 0.7

    # Stacked top-to-bottom.
    train_ys = [5.4, 4.2, 3.0, 1.8]
    train_labels = [
        LABEL_MINIBATCH,
        LABEL_FORWARD,
        LABEL_OPTIMIZER + "\n(AdamW or SGD+M)",
        LABEL_UPDATE,
    ]
    for y, label in zip(train_ys, train_labels):
        _add_box(
            ax,
            (train_x, y),
            box_w,
            box_h if "\n" not in label else 0.85,
            label,
            facecolor=COLOR_TRAIN_FILL,
            edgecolor=COLOR_TRAIN_EDGE,
        )

    # Arrows linking the training-loop boxes top -> bottom.
    arrow_pairs = [
        (5.4, 4.2),
        (4.2, 3.0),
        (3.0, 1.8),
    ]
    for y_top, y_bot in arrow_pairs:
        # Account for the slightly taller optimizer box (h=0.85) when laying
        # out arrow endpoints so we don't poke into the box.
        h_top = 0.85 if "Base optimizer" in train_labels[train_ys.index(
            y_top)] else box_h
        h_bot = 0.85 if "Base optimizer" in train_labels[train_ys.index(
            y_bot)] else box_h
        _add_arrow(
            ax,
            (train_x, y_top - h_top / 2 - 0.02),
            (train_x, y_bot + h_bot / 2 + 0.02),
            color=COLOR_TRAIN_EDGE,
            mutation_scale=10.0,
        )

    # Region outline + label for training loop.
    _draw_region_outline(
        ax,
        (train_x - box_w / 2 - 0.25, 1.30),
        (train_x + box_w / 2 + 0.25, 5.95),
        label="Training loop (per step)",
        color=COLOR_TRAIN_EDGE,
    )

    # ------------------------------------------------------------------ bandit controllers (right column)
    # Boxes widened ~20% so the controller labels fit on TWO lines inside the
    # box without overflowing the left edge (Round 3 fix). Vertical spacing
    # between the LR and Noise controller boxes increased ~30% to give the
    # trunk, the curved arrow into the optimizer, and the bridge annotations
    # room to breathe.
    bandit_x = 8.1
    bandit_w = 3.7
    bandit_h = 0.95
    bandit_ys = [4.68, 2.62]
    # Force the controller name onto two lines so the wider box accommodates
    # it without crowding the box edges.
    bandit_labels = [
        "Episodic Bandit\nController (LR-multiplier)",
        "Episodic Bandit\nController (Gradient noise)",
    ]
    for y, label in zip(bandit_ys, bandit_labels):
        _add_box(
            ax,
            (bandit_x, y),
            bandit_w,
            bandit_h,
            label,
            facecolor=COLOR_BANDIT_FILL,
            edgecolor=COLOR_BANDIT_EDGE,
            fontsize=8.6,
        )

    bandit_region_x0 = bandit_x - bandit_w / 2 - 0.25
    bandit_region_y0 = 1.97
    bandit_region_y1 = 5.33
    _draw_region_outline(
        ax,
        (bandit_region_x0, bandit_region_y0),
        (bandit_x + bandit_w / 2 + 0.25, bandit_region_y1),
        label="Episode controller (per episode of $E$ steps)",
        color=COLOR_BANDIT_EDGE,
    )

    # ------------------------------------------------------------------ bridge: arm values (controllers -> training)
    # Both bandits feed into a small junction in the middle column, then a
    # single arrow into the optimizer box. We use a single labeled "rail" to
    # keep the diagram readable at half-textwidth.
    rail_x = 5.2
    rail_top_y = bandit_ys[0]
    rail_bot_y = bandit_ys[1]

    # Outgoing from each controller's left edge to the rail.
    _add_arrow(
        ax,
        (bandit_x - bandit_w / 2 - 0.02, rail_top_y),
        (rail_x, rail_top_y),
        color=COLOR_ARROW_FWD,
        mutation_scale=10.0,
    )
    _add_arrow(
        ax,
        (bandit_x - bandit_w / 2 - 0.02, rail_bot_y),
        (rail_x, rail_bot_y),
        color=COLOR_ARROW_FWD,
        mutation_scale=10.0,
    )
    # Vertical rail joining the two contributions.
    ax.plot(
        [rail_x, rail_x],
        [rail_bot_y, rail_top_y],
        color=COLOR_ARROW_FWD,
        linewidth=1.2,
    )
    # From the rail (mid-height) into the optimizer box on the left.
    rail_mid_y = (rail_top_y + rail_bot_y) / 2  # ≈ 3.65
    _add_arrow(
        ax,
        (rail_x, rail_mid_y),
        (train_x + box_w / 2 + 0.02, 3.0),
        color=COLOR_ARROW_FWD,
        mutation_scale=12.0,
        connectionstyle="arc3,rad=-0.15",
    )

    # Both bridge labels sit in the vertical gap between the two controller
    # boxes, centered horizontally between the trunk (rail_x) and the LR
    # controller's left edge so they do not touch either box edge nor cross
    # the curved arrow into the optimizer. Reading order top-to-bottom:
    # "Selected arm values" (what the controllers output) over the italic
    # "Apply ..." pair (what the training loop does with them). Both anchored
    # with va="bottom" so the text grows upward from each y-line.
    label_x = rail_x + (bandit_x - bandit_w / 2 - rail_x) / 2
    # "Apply ..." annotation: ABOVE the arrow midpoint (which sits at about
    # y=3.3 along the curved arrow from rail_mid_y=3.65 to optimizer y=3.0).
    # Two lines, italic, va="bottom".
    ax.text(
        label_x,
        3.50,
        "Apply learning-rate multiplier\nApply gradient noise",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=COLOR_ARROW_FWD,
        fontstyle="italic",
    )
    # "Selected arm values" — placed in the same gap, ABOVE the "Apply ..."
    # annotation. The wider top-of-gap headroom (with the ~30% gap increase)
    # keeps the label well clear of the top controller box's bottom edge.
    ax.text(
        label_x,
        3.97,
        LABEL_ARM_VALUES,
        ha="center",
        va="bottom",
        fontsize=8.6,
        color=COLOR_ARROW_FWD,
    )

    # ------------------------------------------------------------------ bridge: reward (training -> controllers)
    # Reward arrow leaves the parameter-update box, runs along the bottom of
    # the figure, then terminates on the bottom edge of the dashed bandit
    # region rectangle — making it visually clear that the SAME reward
    # $r_e$ from the EMA loss is consumed by BOTH controllers inside that
    # region.
    reward_y = 0.95
    _add_arrow(
        ax,
        (train_x + box_w / 2 + 0.02, 1.8),
        (train_x + box_w / 2 + 0.6, reward_y),
        color=COLOR_ARROW_REV,
        linestyle="--",
        mutation_scale=10.0,
        connectionstyle="arc3,rad=-0.2",
    )
    ax.plot(
        [train_x + box_w / 2 + 0.6, bandit_x],
        [reward_y, reward_y],
        color=COLOR_ARROW_REV,
        linewidth=1.2,
        linestyle="--",
    )
    # Terminate on the dashed bandit-region rectangle's bottom edge (centered
    # on the column), so the arrowhead clearly enters the enclosing region
    # rather than picking a single controller.
    _add_arrow(
        ax,
        (bandit_x, reward_y),
        (bandit_x, bandit_region_y0 - 0.02),
        color=COLOR_ARROW_REV,
        linestyle="--",
        mutation_scale=12.0,
    )

    # Reward-rail label. Identical phrasing applies to both controllers.
    ax.text(
        (train_x + bandit_x) / 2,
        reward_y - 0.30,
        LABEL_REWARD,
        ha="center",
        va="center",
        fontsize=8.6,
        color=COLOR_ARROW_REV,
    )
    ax.text(
        (train_x + bandit_x) / 2,
        reward_y - 0.58,
        "(applied to both controllers; from EMA-loss trace over $E$-step episode)",
        ha="center",
        va="center",
        fontsize=7.4,
        fontstyle="italic",
        color=COLOR_ARROW_REV,
    )

    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="Directory in which to write aees_overview.{pdf,png,summary.txt}.",
    )
    args = parser.parse_args(argv)
    name = "aees_overview"

    try:
        fig = _build_figure()
    except Exception as exc:  # pragma: no cover - defensive
        reason = (
            "Failed to build the AEES overview schematic.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, name, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, name)

    summary_lines = [
        "schematic - no input data",
        "labels rendered:",
        *[f"  - {label}" for label in REQUIRED_LABELS],
        "outputs:",
        f"  - {pdf_path}",
        f"  - {png_path}",
    ]
    summary_path = write_summary(args.out_dir, name, summary_lines)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
