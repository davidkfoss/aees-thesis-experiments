"""Render the AEES method overview schematic.

This is a pure diagram: it accepts ``--runs-root`` for CLI consistency with the
data-driven plot scripts, but does not read run data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.plots_gpt._common import (
    epoch_step_range,
    episode_to_epoch,
    identify,
    load_set,
    walk_runs,
)
from scripts.plots_gpt._style import (
    LABEL,
    PALETTE,
    arm_palette,
    make_figure,
    mark_peak_final,
    mean_std_band,
    save_figure,
    short_trajectory_kwargs,
    write_missing,
    write_summary,
)


NAME = "aees_overview"

LABEL_MINIBATCH = "Mini-batch"
LABEL_FORWARD = "Forward / backward"
LABEL_OPTIMIZER = "Base optimizer"
LABEL_UPDATE = "Parameter update"
LABEL_BANDIT_LR = "Episodic Bandit Controller (LR-multiplier)"
LABEL_BANDIT_NOISE = "Episodic Bandit Controller (Gradient noise)"
LABEL_ARM_VALUES = "Selected arm values"
LABEL_REWARD = "Reward $r_e$ from EMA loss"

REQUIRED_LABELS = (
    LABEL_MINIBATCH,
    LABEL_FORWARD,
    LABEL_OPTIMIZER,
    LABEL_UPDATE,
    LABEL_BANDIT_LR,
    LABEL_BANDIT_NOISE,
    LABEL_ARM_VALUES,
    LABEL_REWARD,
)

TRAIN_FILL = "#dfeaf2"
TRAIN_EDGE = "#355169"
CONTROL_FILL = "#f3e4c8"
CONTROL_EDGE = "#80601e"
BRIDGE_FILL = "#f7f7f4"
ARM_COLOR = PALETTE["aees_lr"]
REWARD_COLOR = PALETTE["cosine_aees"]
TEXT_COLOR = "#171717"


def _box(
    ax,
    *,
    center: tuple[float, float],
    size: tuple[float, float],
    label: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 9.0,
    weight: str = "normal",
    mutation: float = 0.09,
) -> None:
    """Add a rounded labeled box centered at ``center``."""

    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.03,rounding_size={mutation}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.05,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=TEXT_COLOR,
        wrap=True,
    )


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.25,
    linestyle: str = "-",
    rad: float = 0.0,
    scale: float = 12.0,
) -> None:
    """Add a consistent thesis-style arrow."""

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            mutation_scale=scale,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def _region(
    ax,
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    color: str,
) -> None:
    """Draw a subtle dashed region boundary and title."""

    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="none",
            edgecolor=color,
            linewidth=0.8,
            linestyle=(0, (4, 3)),
            alpha=0.85,
        )
    )
    ax.text(
        x + width / 2,
        y + height + 0.14,
        label,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontstyle="italic",
        color=color,
    )


def build_figure():
    """Build the AEES overview diagram."""

    fig, axes = make_figure(n_panels=1)
    ax = axes[0]
    ax.set_xlim(0, 10.9)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.grid(False)

    # Left region: step-level training loop.
    train_x = 1.75
    train_w = 2.35
    train_h = 0.64
    train_y = [5.15, 4.05, 2.95, 1.85]
    for y, label in zip(
        train_y,
        (LABEL_MINIBATCH, LABEL_FORWARD, LABEL_OPTIMIZER, LABEL_UPDATE),
    ):
        _box(
            ax,
            center=(train_x, y),
            size=(train_w, train_h),
            label=label,
            facecolor=TRAIN_FILL,
            edgecolor=TRAIN_EDGE,
        )

    for upper, lower in zip(train_y, train_y[1:]):
        _arrow(
            ax,
            (train_x, upper - train_h / 2 - 0.03),
            (train_x, lower + train_h / 2 + 0.03),
            color=TRAIN_EDGE,
            scale=10.5,
        )

    _region(
        ax,
        xy=(0.28, 1.28),
        width=2.94,
        height=4.30,
        label="Training loop",
        color=TRAIN_EDGE,
    )

    # Right region: episode controllers. Long acceptance labels are rendered
    # verbatim and wrapped by Matplotlib inside wide boxes.
    control_x = 8.35
    control_w = 4.05
    control_h = 0.92
    lr_y = 4.55
    noise_y = 2.65
    _box(
        ax,
        center=(control_x, lr_y),
        size=(control_w, control_h),
        label=LABEL_BANDIT_LR,
        facecolor=CONTROL_FILL,
        edgecolor=CONTROL_EDGE,
        fontsize=6.9,
    )
    _box(
        ax,
        center=(control_x, noise_y),
        size=(control_w, control_h),
        label=LABEL_BANDIT_NOISE,
        facecolor=CONTROL_FILL,
        edgecolor=CONTROL_EDGE,
        fontsize=6.9,
    )
    _region(
        ax,
        xy=(6.08, 1.92),
        width=4.55,
        height=3.12,
        label="Episode controller",
        color=CONTROL_EDGE,
    )

    # Center bridge: controller choices feed into the optimizer/noise injection.
    bridge_x = 4.80
    bridge_w = 2.10
    _box(
        ax,
        center=(bridge_x, 3.56),
        size=(bridge_w, 0.62),
        label=LABEL_ARM_VALUES,
        facecolor=BRIDGE_FILL,
        edgecolor=ARM_COLOR,
        fontsize=8.0,
        weight="bold",
        mutation=0.06,
    )
    ax.text(
        bridge_x,
        3.02,
        "Apply learning-rate multiplier\nApply gradient noise",
        ha="center",
        va="center",
        fontsize=7.6,
        fontstyle="italic",
        color=ARM_COLOR,
    )

    controller_left = control_x - control_w / 2
    bridge_right = bridge_x + bridge_w / 2
    _arrow(ax, (controller_left - 0.04, lr_y),
           (bridge_right + 0.04, 3.77), color=ARM_COLOR)
    _arrow(
        ax,
        (controller_left - 0.04, noise_y),
        (bridge_right + 0.04, 3.35),
        color=ARM_COLOR,
        rad=-0.10,
    )
    _arrow(
        ax,
        (bridge_x - bridge_w / 2 - 0.04, 3.56),
        (train_x + train_w / 2 + 0.07, 2.95),
        color=ARM_COLOR,
        linewidth=1.45,
        rad=0.10,
        scale=13,
    )

    # Dashed reward path: after an episode, EMA-loss improvement returns to
    # both controllers via the enclosing controller region rather than one arm.
    reward_y = 0.90
    reward_start_x = train_x + train_w / 2 + 0.08
    controller_mid_x = control_x
    _arrow(
        ax,
        (reward_start_x, 1.85),
        (reward_start_x + 0.55, reward_y),
        color=REWARD_COLOR,
        linestyle="--",
        rad=-0.18,
        scale=10.5,
    )
    ax.plot(
        [reward_start_x + 0.55, controller_mid_x],
        [reward_y, reward_y],
        color=REWARD_COLOR,
        linewidth=1.25,
        linestyle="--",
    )
    _arrow(
        ax,
        (controller_mid_x, reward_y),
        (controller_mid_x, 1.88),
        color=REWARD_COLOR,
        linestyle="--",
        scale=12,
    )
    ax.text(
        5.08,
        0.49,
        LABEL_REWARD,
        ha="center",
        va="center",
        fontsize=8.5,
        color=REWARD_COLOR,
        fontweight="bold",
    )
    ax.text(
        5.08,
        0.22,
        "computed after each episode of $E$ training steps",
        ha="center",
        va="center",
        fontsize=7.4,
        color=REWARD_COLOR,
        fontstyle="italic",
    )

    fig.tight_layout(pad=0.2)
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results"),
        help="Accepted for consistency with data-driven scripts; unused.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plots/method"),
        help="Output directory for aees_overview.{pdf,png,summary.txt}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        fig = build_figure()
        pdf_path, png_path = save_figure(fig, args.out_dir, NAME)
        write_summary(
            args.out_dir,
            NAME,
            [
                "status: rendered",
                "input data: none (pure schematic)",
                f"runs_root accepted but unused: {args.runs_root}",
                "labels:",
                *[f"- {label}" for label in REQUIRED_LABELS],
                f"pdf: {pdf_path}",
                f"png: {png_path}",
            ],
        )
    except Exception as exc:  # pragma: no cover - defensive CLI behavior
        write_missing(
            args.out_dir,
            NAME,
            "Failed to render AEES method overview.\n\n"
            f"Error: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```",
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
