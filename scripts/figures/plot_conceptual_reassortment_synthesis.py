#!/usr/bin/env python3
"""Render a conceptual synthesis figure for the reassortment manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "notebooks" / "figures" / "conceptual_reassortment_synthesis.png"
DEFAULT_OUT_SVG = REPO_ROOT / "notebooks" / "figures" / "conceptual_reassortment_synthesis.svg"


BG = "#f8f5ef"
INK = "#2f2d2a"
MUTED = "#7a766f"
GREEN = "#dcefd8"
BLUE = "#dcecf6"
PEACH = "#f4dfcb"
PINK = "#ead8ea"
SAND = "#efe6d5"
SOFT_RED = "#e6a19c"
SOFT_TEAL = "#96c8bf"
SOFT_GOLD = "#e8cb8d"
SOFT_LAV = "#b8abd6"
SOFT_GRAY = "#d7d0c6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-svg", type=Path, default=DEFAULT_OUT_SVG)
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Helvetica Neue",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": BG,
            "text.color": INK,
        }
    )


def rounded_panel(ax: plt.Axes, x: float, y: float, w: float, h: float, color: str, alpha: float = 1.0) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor=color,
            edgecolor="none",
            linewidth=0,
            alpha=alpha,
            zorder=0,
        )
    )


def draw_stage_label(ax: plt.Axes, x: float, y: float, text: str, color: str = INK, size: float = 17, weight: str = "bold") -> None:
    ax.text(x, y, text, ha="left", va="top", fontsize=size, fontweight=weight, color=color)


def draw_body_text(ax: plt.Axes, x: float, y: float, text: str, size: float = 10.5, color: str = MUTED, ha: str = "left") -> None:
    ax.text(x, y, text, ha=ha, va="top", fontsize=size, color=color, linespacing=1.25)


def draw_dot_cloud(
    ax: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    n: int,
    colors: list[str],
    alpha: float = 0.9,
    seed: int = 0,
    scale: tuple[float, float] = (0.010, 0.018),
    z: int = 3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = []
    cx, cy = center
    for i in range(n):
        while True:
            px = rng.uniform(cx - width / 2, cx + width / 2)
            py = rng.uniform(cy - height / 2, cy + height / 2)
            dx = (px - cx) / (width / 2)
            dy = (py - cy) / (height / 2)
            if dx * dx + dy * dy <= 1:
                break
        r = rng.uniform(scale[0], scale[1])
        color = colors[i % len(colors)]
        ax.add_patch(Circle((px, py), r, facecolor=color, edgecolor="white", linewidth=0.9, alpha=alpha, zorder=z))
        pts.append((px, py, r))
    return np.asarray(pts)


def draw_funnel(ax: plt.Axes) -> None:
    verts = [
        (0.49, 0.72),
        (0.59, 0.72),
        (0.66, 0.64),
        (0.66, 0.57),
        (0.59, 0.48),
        (0.49, 0.48),
        (0.55, 0.57),
        (0.49, 0.72),
    ]
    patch = Polygon(verts, closed=True, facecolor=PEACH, edgecolor="white", linewidth=2.2, alpha=0.98, zorder=1)
    ax.add_patch(patch)
    ax.add_patch(
        FancyBboxPatch(
            (0.47, 0.44),
            0.20,
            0.05,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=SAND,
            edgecolor="none",
            alpha=0.75,
            zorder=0,
        )
    )


def draw_flow_arrows(ax: plt.Axes) -> None:
    arrows = [
        ((0.28, 0.69), (0.40, 0.63), GREEN),
        ((0.28, 0.54), (0.40, 0.58), BLUE),
        ((0.68, 0.60), (0.83, 0.60), PINK),
    ]
    for start, end, color in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2.2,
                color=color,
                connectionstyle="arc3,rad=0.0",
                zorder=2,
            )
        )


def draw_candidate_space(ax: plt.Axes) -> None:
    rounded_panel(ax, 0.33, 0.48, 0.14, 0.26, "#f7f2e8", alpha=0.92)
    draw_dot_cloud(
        ax,
        center=(0.40, 0.60),
        width=0.11,
        height=0.19,
        n=30,
        colors=[SOFT_RED, SOFT_TEAL, SOFT_GOLD, SOFT_LAV],
        seed=13,
        scale=(0.008, 0.014),
        alpha=0.95,
        z=4,
    )
    draw_stage_label(ax, 0.33, 0.77, "Candidate\nreassortment space", size=14)


def draw_filter_losses(ax: plt.Axes) -> None:
    draw_dot_cloud(
        ax,
        center=(0.59, 0.77),
        width=0.10,
        height=0.07,
        n=8,
        colors=[SOFT_GRAY],
        seed=4,
        scale=(0.006, 0.010),
        alpha=0.9,
        z=4,
    )
    draw_dot_cloud(
        ax,
        center=(0.60, 0.43),
        width=0.11,
        height=0.07,
        n=9,
        colors=[SOFT_GRAY],
        seed=7,
        scale=(0.006, 0.010),
        alpha=0.9,
        z=4,
    )
    for start, end in [((0.57, 0.61), (0.60, 0.75)), ((0.57, 0.56), (0.60, 0.44))]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-",
                mutation_scale=12,
                linewidth=1.4,
                linestyle=(0, (3, 3)),
                color=MUTED,
                alpha=0.85,
                zorder=3,
            )
        )


def draw_realized_space(ax: plt.Axes) -> None:
    rounded_panel(ax, 0.76, 0.47, 0.18, 0.26, "#f6f0f6", alpha=0.98)
    draw_dot_cloud(
        ax,
        center=(0.84, 0.60),
        width=0.12,
        height=0.16,
        n=11,
        colors=[SOFT_RED, SOFT_LAV, SOFT_TEAL],
        seed=19,
        scale=(0.009, 0.015),
        alpha=0.98,
        z=4,
    )
    draw_stage_label(ax, 0.77, 0.77, "Observed\nreassortants", size=14)
    ax.text(0.845, 0.44, "effective reassortment space", ha="center", va="top", fontsize=10.5, color=MUTED)


def draw_support_badges(ax: plt.Axes) -> None:
    badges = [
        (0.16, 0.33, 0.13, 0.052, GREEN, "strongest support"),
        (0.32, 0.33, 0.12, 0.052, BLUE, "contextual"),
        (0.54, 0.33, 0.16, 0.052, PEACH, "species-level filter"),
    ]
    for x, y, w, h, color, label in badges:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=2,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.2, zorder=3)


def draw_filter_annotations(ax: plt.Axes) -> None:
    draw_stage_label(ax, 0.475, 0.81, "Molecular\ncompatibility filter", size=15)
    draw_body_text(
        ax,
        0.475,
        0.69,
        "Compatibility is summarized by\nUTR structure, LD burden, and\ntopological discordance proxies.",
        size=10.0,
    )
    label_specs = [
        (0.485, 0.505, "UTR\nstructure", SOFT_RED),
        (0.555, 0.505, "LD\nburden", SOFT_GOLD),
        (0.625, 0.505, "tree\ndiscordance", SOFT_TEAL),
    ]
    for x, y, text, color in label_specs:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.055,
                0.065,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=4,
                alpha=0.92,
            )
        )
        ax.text(x + 0.0275, y + 0.032, text, ha="center", va="center", fontsize=8.7, zorder=5)


def draw_opportunity_side(ax: plt.Axes) -> None:
    rounded_panel(ax, 0.06, 0.60, 0.20, 0.18, GREEN, alpha=0.98)
    rounded_panel(ax, 0.06, 0.42, 0.20, 0.15, BLUE, alpha=0.98)

    draw_stage_label(ax, 0.08, 0.76, "Ecological opportunity", size=15)
    draw_body_text(
        ax,
        0.08,
        0.67,
        "Host overlap, contact structure,\nand shared opportunity create the\nspace in which reassortment can occur.",
        size=10.0,
    )

    draw_stage_label(ax, 0.08, 0.56, "Local prevalence", size=15)
    draw_body_text(
        ax,
        0.08,
        0.49,
        "Co-circulation changes how much of\nthat opportunity is actually occupied,\nbut with weaker directionality.",
        size=10.0,
    )


def draw_baseline_track(ax: plt.Axes) -> None:
    verts = [
        (0.08, 0.25),
        (0.24, 0.25),
        (0.34, 0.25),
        (0.49, 0.25),
        (0.58, 0.25),
        (0.76, 0.25),
        (0.90, 0.25),
    ]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    path = MplPath(verts, codes)
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor="#d8d0c3", linewidth=6, capstyle="round", zorder=1, alpha=0.8))
    ax.text(0.08, 0.22, "more possible combinations", ha="left", va="top", fontsize=10.4, color=MUTED)
    ax.text(0.90, 0.22, "fewer realized combinations", ha="right", va="top", fontsize=10.4, color=MUTED)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out_svg:
        args.out_svg.parent.mkdir(parents=True, exist_ok=True)

    set_style()

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_panel(ax, 0.04, 0.08, 0.92, 0.84, "#fffdfa", alpha=0.94)

    draw_opportunity_side(ax)
    draw_candidate_space(ax)
    draw_funnel(ax)
    draw_filter_annotations(ax)
    draw_filter_losses(ax)
    draw_realized_space(ax)
    draw_flow_arrows(ax)
    draw_support_badges(ax)
    draw_baseline_track(ax)

    ax.text(
        0.51,
        0.15,
        "Ecology generates candidate reassortment space; molecular compatibility narrows that space into the reassortants that are actually realized and sampled.",
        ha="center",
        va="center",
        fontsize=11.2,
        color=INK,
    )

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    if args.out_svg:
        fig.savefig(args.out_svg, bbox_inches="tight")
    plt.close(fig)
    print(args.out.resolve())
    if args.out_svg:
        print(args.out_svg.resolve())


if __name__ == "__main__":
    main()
