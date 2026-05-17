#!/usr/bin/env python3
"""Render a manuscript-style figure for the piecewise path sensitivity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERM_TABLE = REPO_ROOT / "results" / "modeling" / "piecewise_path_sensitivity" / "path_term_summary.tsv"
DEFAULT_EVENT_TABLE = REPO_ROOT / "results" / "modeling" / "piecewise_path_sensitivity" / "event_analysis_dataset.tsv"
DEFAULT_MOLECULAR_TABLE = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "final_hierarchical_block_comparison"
    / "fixed_effect_comparison.tsv"
)
DEFAULT_OUT = REPO_ROOT / "results" / "figures" / "piecewise_path_sensitivity.png"
DEFAULT_OUT_SVG = REPO_ROOT / "results" / "figures" / "piecewise_path_sensitivity.svg"


BG = "#fbf8f4"
PANEL = "#fffdf9"
TEXT = "#2c2c2c"
MUTED = "#7d7a74"
POS = "#e7a39f"
NEG = "#9bc7be"
UNCERTAIN = "#c7c2ba"
NODE_ECO = "#dcefd8"
NODE_PREV = "#d8ebf1"
NODE_DISC = "#f6e3d1"
NODE_OUTCOME = "#f3dce7"
ACCENT = "#f3cc89"
ACCENT2 = "#ddd7ee"

MOLECULAR_LABELS = {
    "species_feat_mean_cramers_v_overall_z": "LD burden",
    "species_feat_prop_significant_overall_z": "Significant LD fraction",
    "species_feat_utr5_S_structdist_mean_z": "UTR5 S",
    "species_feat_utr5_M_structdist_mean_z": "UTR5 M",
    "species_feat_utr5_L_structdist_mean_z": "UTR5 L",
    "species_feat_utr3_S_structdist_mean_z": "UTR3 S",
    "species_feat_utr3_M_structdist_mean_z": "UTR3 M",
    "species_feat_utr3_L_structdist_mean_z": "UTR3 L",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--term-table", type=Path, default=DEFAULT_TERM_TABLE)
    parser.add_argument("--event-table", type=Path, default=DEFAULT_EVENT_TABLE)
    parser.add_argument("--molecular-table", type=Path, default=DEFAULT_MOLECULAR_TABLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-svg", type=Path, default=DEFAULT_OUT_SVG)
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "Helvetica Neue",
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": TEXT,
            "axes.edgecolor": BG,
        }
    )


def fmt_p(value: float) -> str:
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def fmt_effect(row: pd.Series) -> str:
    if pd.notna(row.get("odds_ratio", np.nan)):
        return f"OR {row['odds_ratio']:.2f}"
    return f"b {row['estimate']:.2f}"


def significance_style(row: pd.Series) -> tuple[str, str, float, str]:
    estimate = float(row["estimate"])
    p_value = float(row["p_value"])
    color = POS if estimate > 0 else NEG
    if p_value >= 0.05:
        color = UNCERTAIN
    width = 1.6 + min(abs(estimate) * 2.2, 3.8)
    linestyle = "-" if p_value < 0.05 else (0, (4, 3))
    alpha = 0.95 if p_value < 0.05 else 0.8
    return color, linestyle, width * alpha, fmt_effect(row)


def add_panel(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.04",
            facecolor=PANEL,
            edgecolor="none",
            zorder=0,
        )
    )


def add_node(ax: plt.Axes, xy: tuple[float, float], label: str, facecolor: str, width: float = 0.17, height: float = 0.10) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor=facecolor,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=11, zorder=4)


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    row: pd.Series,
    rad: float,
    label_xy: tuple[float, float],
) -> None:
    color, linestyle, width, effect_text = significance_style(row)
    patch = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=width,
        linestyle=linestyle,
        color=color,
        alpha=0.92,
        zorder=2,
        shrinkA=18,
        shrinkB=18,
    )
    ax.add_patch(patch)
    lx, ly = label_xy
    ax.text(
        lx,
        ly,
        f"{effect_text}\n{fmt_p(float(row['p_value']))}",
        ha="center",
        va="center",
        fontsize=8.8,
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.3,rounding_size=0.12",
            "facecolor": BG,
            "edgecolor": "none",
            "alpha": 0.94,
        },
        zorder=5,
    )


def term(df: pd.DataFrame, model_name: str, term_name: str) -> pd.Series:
    out = df[(df["model_name"] == model_name) & (df["term"] == term_name)]
    if out.empty:
        raise KeyError(f"Missing term {term_name} for model {model_name}")
    return out.iloc[0]


def add_legend(ax: plt.Axes) -> None:
    x0, y0 = 0.06, 0.05
    ax.text(x0, y0 + 0.08, "Path readout", fontsize=9.5, color=MUTED, ha="left")
    ax.plot([x0, x0 + 0.07], [y0 + 0.045, y0 + 0.045], color=POS, lw=3.4, solid_capstyle="round")
    ax.text(x0 + 0.09, y0 + 0.045, "positive", va="center", fontsize=8.6)
    ax.plot([x0, x0 + 0.07], [y0 + 0.015, y0 + 0.015], color=NEG, lw=3.4, solid_capstyle="round")
    ax.text(x0 + 0.09, y0 + 0.015, "negative", va="center", fontsize=8.6)
    ax.plot([x0 + 0.28, x0 + 0.35], [y0 + 0.045, y0 + 0.045], color=TEXT, lw=2.3, solid_capstyle="round")
    ax.text(x0 + 0.37, y0 + 0.045, "solid = 95% CI excludes null", va="center", fontsize=8.6)
    ax.plot([x0 + 0.28, x0 + 0.35], [y0 + 0.015, y0 + 0.015], color=TEXT, lw=2.3, dashes=(4, 3), solid_capstyle="round")
    ax.text(x0 + 0.37, y0 + 0.015, "dashed = uncertain", va="center", fontsize=8.6)


def load_molecular_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df[df["model"] == "molecular_plus_utr3_block"].copy()
    df = df[df["term"].isin(MOLECULAR_LABELS)].copy()
    df["label"] = df["term"].map(MOLECULAR_LABELS)
    df["direction"] = np.where(df["mean"] >= 0, "positive", "negative")
    df["certain"] = (df["hdi_3%"] > 0) | (df["hdi_97%"] < 0)
    order = ["LD burden", "Significant LD fraction", "UTR5 L", "UTR5 M", "UTR5 S", "UTR3 L", "UTR3 M", "UTR3 S"]
    df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
    return df.sort_values("label").reset_index(drop=True)


def add_molecular_panel(fig: plt.Figure, molecular_df: pd.DataFrame) -> None:
    axm = fig.add_axes([0.71, 0.17, 0.22, 0.66], facecolor=PANEL)
    axm.axvline(0, color="#b7b0a7", lw=1.1, zorder=0)

    y = np.arange(len(molecular_df))[::-1]
    for yi, (_, row) in zip(y, molecular_df.iterrows()):
        color = POS if row["mean"] > 0 else NEG
        if not row["certain"]:
            color = UNCERTAIN
        axm.plot([row["hdi_3%"], row["hdi_97%"]], [yi, yi], color=color, lw=3.0, solid_capstyle="round", zorder=2)
        axm.scatter(
            [row["mean"]],
            [yi],
            s=70 + 70 * abs(row["posterior_prob_positive"] - 0.5),
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        prob_text = f"P(+){row['posterior_prob_positive']:.2f}"
        axm.text(
            row["hdi_97%"] + 0.08,
            yi,
            prob_text,
            va="center",
            ha="left",
            fontsize=8.1,
            color=MUTED,
        )

    axm.set_yticks(y)
    axm.set_yticklabels(list(molecular_df["label"]), fontsize=9.2)
    axm.set_xlim(-2.6, 2.6)
    axm.set_ylim(-0.7, len(molecular_df) - 0.3)
    axm.set_xlabel("Posterior effect on log-odds", fontsize=9.2)
    axm.tick_params(axis="x", labelsize=8.4)
    axm.tick_params(axis="y", length=0)
    for spine in axm.spines.values():
        spine.set_visible(False)
    axm.grid(axis="x", color="#ece7e0", lw=0.8)

    axm.text(0.00, 1.06, "Species-level molecular background", transform=axm.transAxes, fontsize=12, ha="left")
    axm.text(
        0.00,
        1.00,
        "Repeated summaries from the molecular+UTR3 Bayesian block.\nShown here even when directionality remains uncertain.",
        transform=axm.transAxes,
        fontsize=8.5,
        ha="left",
        va="top",
        color=MUTED,
    )


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out_svg:
        args.out_svg.parent.mkdir(parents=True, exist_ok=True)

    set_style()

    term_table = pd.read_csv(args.term_table, sep="\t")
    event_df = pd.read_csv(args.event_table, sep="\t")
    molecular_df = load_molecular_table(args.molecular_table)
    mixed_events = int((event_df["target_nunique"] > 1).sum())
    n_events = int(len(event_df))

    fig = plt.figure(figsize=(14.0, 8.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_panel(ax, 0.04, 0.54, 0.63, 0.36)
    add_panel(ax, 0.04, 0.13, 0.63, 0.33)
    add_panel(ax, 0.69, 0.13, 0.27, 0.75)

    ax.text(0.07, 0.86, "Event-aware weighted positive fraction", fontsize=12.5, ha="left")
    ax.text(0.07, 0.44, "Sequence-level sensitivity clustered by event", fontsize=12.5, ha="left")
    ax.text(0.07, 0.81, f"{mixed_events}/{n_events} events mix positive and negative labels", fontsize=9.2, color=MUTED)
    ax.text(0.07, 0.39, "Same raw predictors, but no hard event collapse for the outcome", fontsize=9.2, color=MUTED)

    top_nodes = {
        "overlap": (0.15, 0.69),
        "prev": (0.15, 0.58),
        "disc": (0.37, 0.74),
        "out": (0.58, 0.64),
    }
    bottom_nodes = {
        "overlap": (0.15, 0.29),
        "prev": (0.15, 0.19),
        "disc": (0.37, 0.34),
        "out": (0.58, 0.24),
    }

    for nodes in (top_nodes, bottom_nodes):
        add_node(ax, nodes["overlap"], "Overlap\nopportunity", NODE_ECO)
        add_node(ax, nodes["prev"], "Local\nprevalence", NODE_PREV)
        add_node(ax, nodes["disc"], "Tree\ndiscordance", NODE_DISC)
        add_node(ax, nodes["out"], "Reassortment", NODE_OUTCOME)

    add_arrow(
        ax,
        top_nodes["overlap"],
        top_nodes["disc"],
        term(term_table, "event_discordance", "overlap_mean_z"),
        0.08,
        (0.25, 0.76),
    )
    add_arrow(
        ax,
        top_nodes["prev"],
        top_nodes["disc"],
        term(term_table, "event_discordance", "prev_mean_z"),
        -0.08,
        (0.25, 0.57),
    )
    add_arrow(
        ax,
        top_nodes["disc"],
        top_nodes["out"],
        term(term_table, "event_fraction_weighted", "discord_mean_z"),
        -0.03,
        (0.47, 0.75),
    )
    add_arrow(
        ax,
        top_nodes["overlap"],
        top_nodes["out"],
        term(term_table, "event_fraction_weighted", "overlap_mean_z"),
        0.12,
        (0.35, 0.61),
    )
    add_arrow(
        ax,
        top_nodes["prev"],
        top_nodes["out"],
        term(term_table, "event_fraction_weighted", "prev_mean_z"),
        -0.16,
        (0.36, 0.47),
    )

    add_arrow(
        ax,
        bottom_nodes["overlap"],
        bottom_nodes["disc"],
        term(term_table, "sequence_discordance_clustered", "overlap_opportunity_local_z"),
        0.08,
        (0.25, 0.35),
    )
    add_arrow(
        ax,
        bottom_nodes["prev"],
        bottom_nodes["disc"],
        term(term_table, "sequence_discordance_clustered", "arha_prev_local_z"),
        -0.08,
        (0.25, 0.17),
    )
    add_arrow(
        ax,
        bottom_nodes["disc"],
        bottom_nodes["out"],
        term(term_table, "sequence_target_clustered", "tree_discordance_score_z"),
        -0.03,
        (0.47, 0.35),
    )
    add_arrow(
        ax,
        bottom_nodes["overlap"],
        bottom_nodes["out"],
        term(term_table, "sequence_target_clustered", "overlap_opportunity_local_z"),
        0.12,
        (0.35, 0.23),
    )
    add_arrow(
        ax,
        bottom_nodes["prev"],
        bottom_nodes["out"],
        term(term_table, "sequence_target_clustered", "arha_prev_local_z"),
        -0.16,
        (0.36, 0.16),
    )

    ax.text(
        0.61,
        0.86,
        "Event view: overlap is the only\nstable direct positive path.",
        ha="right",
        va="top",
        fontsize=10.5,
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.2",
            "facecolor": ACCENT,
            "edgecolor": "none",
            "alpha": 0.55,
        },
    )
    ax.text(
        0.61,
        0.45,
        "Discordance does not hold up\nas a strong mediator.",
        ha="right",
        va="top",
        fontsize=10.5,
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.2",
            "facecolor": "#e5e0f2",
            "edgecolor": "none",
            "alpha": 0.55,
        },
    )

    add_legend(ax)
    add_molecular_panel(fig, molecular_df)

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    if args.out_svg:
        fig.savefig(args.out_svg, bbox_inches="tight")
    plt.close(fig)
    print(args.out.resolve())
    if args.out_svg:
        print(args.out_svg.resolve())


if __name__ == "__main__":
    main()
