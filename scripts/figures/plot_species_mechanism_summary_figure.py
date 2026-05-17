#!/usr/bin/env python3
"""Render a species-by-mechanism summary figure for the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODULE_TABLE = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "final_hierarchical_block_comparison"
    / "figures"
    / "manuscript_utr3_integrated"
    / "integrated_module_contributions.tsv"
)
DEFAULT_RESPONSE_METRICS = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "species_varying_slope_block_model"
    / "species_response_metrics.tsv"
)
DEFAULT_DATASET = (
    REPO_ROOT
    / "data"
    / "processed"
    / "final_sequence_dataset.tsv"
)
DEFAULT_OUTPATH = REPO_ROOT / "results" / "figures" / "figure5_species_mechanism_summary.png"


STARFISH = ["#24492e", "#015b58", "#2c6184", "#59629b", "#89689d", "#ba7999", "#e69b99", "#d9b24c"]
PAL = [STARFISH[i - 1] for i in [7, 4, 5, 3, 1, 6, 2]]
SPECIES_LABELS = {
    "orthohantavirus_andesense": "ANDV",
    "orthohantavirus_dobravaense": "DOBV",
    "orthohantavirus_hantanense": "HTNV",
    "orthohantavirus_puumalaense": "PUUV",
    "orthohantavirus_seoulense": "SEOV",
    "orthohantavirus_sinnombreense": "SNV",
    "orthohantavirus_tulaense": "TULV",
}
SPECIES_COLORS = {
    "orthohantavirus_andesense": PAL[5],
    "orthohantavirus_dobravaense": PAL[0],
    "orthohantavirus_hantanense": PAL[1],
    "orthohantavirus_puumalaense": PAL[3],
    "orthohantavirus_seoulense": PAL[4],
    "orthohantavirus_sinnombreense": PAL[6],
    "orthohantavirus_tulaense": PAL[2],
}
MODULE_ORDER = [
    "Overlap opportunity",
    "Host dispersal",
    "Host body mass",
    "Host litter size",
    "Host gestation length",
    "LD burden",
    "Tree discordance",
]
MODULE_SHORT = {
    "Overlap opportunity": "Overlap",
    "Host dispersal": "Dispersal",
    "Host body mass": "Body mass",
    "Host litter size": "Litter size",
    "Host gestation length": "Gestation",
    "LD burden": "LD burden",
    "Tree discordance": "Discordance",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-table", type=Path, default=DEFAULT_MODULE_TABLE)
    parser.add_argument("--response-metrics", type=Path, default=DEFAULT_RESPONSE_METRICS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--outpath", type=Path, default=DEFAULT_OUTPATH)
    return parser.parse_args(argv)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    sns.set_style("white")


def panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def load_observed_rates(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, sep="\t", low_memory=False)
    df = df[df["is_reassortant_final"].notna()].copy()
    df = df[df["target_source"].isin(["sequence_true_label", "isolate_true_label"])].copy()
    df["target"] = pd.to_numeric(df["is_reassortant_final"], errors="coerce")
    rates = (
        df.groupby("species", dropna=False)["target"]
        .agg(n="size", positives="sum", observed_rate="mean")
        .reset_index()
    )
    rates["species_label"] = rates["species"].map(SPECIES_LABELS)
    return rates.sort_values("observed_rate").reset_index(drop=True)


def build_heatmap(module_table_path: Path, species_order: list[str]) -> pd.DataFrame:
    mods = pd.read_csv(module_table_path, sep="\t")
    keep = ["species", *MODULE_ORDER]
    mods = mods[keep].copy()
    heat = mods.groupby("species", dropna=False)[MODULE_ORDER].mean()
    heat = heat.reindex(species_order)
    # Standardize within module so the heatmap shows relative species differences.
    heat = heat.apply(
        lambda col: (col - col.mean()) / (col.std(ddof=0) if col.std(ddof=0) else 1.0),
        axis=0,
    )
    heat.columns = [MODULE_SHORT[c] for c in heat.columns]
    heat.index = [SPECIES_LABELS[s] for s in heat.index]
    return heat


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_style()

    rates = load_observed_rates(args.dataset)
    species_order = rates["species"].tolist()

    heat = build_heatmap(args.module_table, species_order)
    resp = pd.read_csv(args.response_metrics, sep="\t").set_index("species").loc[species_order].reset_index()
    resp["species_label"] = resp["species"].map(SPECIES_LABELS)
    resp["color"] = resp["species"].map(SPECIES_COLORS)

    fig = plt.figure(figsize=(10.8, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.9, 1.25], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[1, 0])

    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    panel_label(ax_c, "C")

    # Panel A: observed reassortment rate with sample sizes.
    y = np.arange(len(rates))
    ax_a.barh(
        y,
        rates["observed_rate"],
        color=[SPECIES_COLORS[s] for s in rates["species"]],
        edgecolor="none",
        alpha=0.92,
        height=0.72,
    )
    ax_a.set_yticks(y, rates["species_label"])
    ax_a.set_xlim(0, 0.75)
    ax_a.set_xlabel("Observed reassortment rate")
    for i, row in rates.iterrows():
        ax_a.text(
            min(row["observed_rate"] + 0.02, 0.73),
            i,
            f"{int(row['positives'])}/{int(row['n'])}",
            va="center",
            ha="left",
            fontsize=8.1,
            color="#2b2b2b",
        )

    # Panel B: species x module heatmap.
    sns.heatmap(
        heat,
        ax=ax_b,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Species-standardized module contribution"},
        square=False,
    )
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.tick_params(axis="x", rotation=35)
    ax_b.tick_params(axis="y", rotation=0)

    # Panel C: ecology vs molecular slope magnitude and context dependence.
    x1 = resp["mean_abs_ecology_gradient"].to_numpy()
    x2 = resp["mean_abs_molecular_gradient"].to_numpy()
    yy = np.arange(len(resp))
    for i, row in resp.iterrows():
        ax_c.plot(
            [row["mean_abs_ecology_gradient"], row["mean_abs_molecular_gradient"]],
            [i, i],
            color="#bdbdbd",
            lw=2.0,
            zorder=1,
        )
    ax_c.scatter(x1, yy, s=48, color="#4c9a92", label="Ecology gradient", zorder=3)
    ax_c.scatter(x2, yy, s=48, color="#b85c5c", label="Molecular gradient", zorder=3)
    ax_c.set_yticks(yy, resp["species_label"])
    ax_c.set_xlabel("Mean absolute response gradient")
    ax_c.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax_c.invert_yaxis()

    # Add species labels repelled near the dominant point to make the panel self-contained.
    texts = []
    for i, row in resp.iterrows():
        x_anchor = max(row["mean_abs_ecology_gradient"], row["mean_abs_molecular_gradient"])
        texts.append(
            ax_c.text(
                x_anchor + 0.01,
                i,
                row["species_label"],
                fontsize=7.6,
                color=row["color"],
                va="center",
            )
        )
    adjust_text(
        texts,
        ax=ax_c,
        only_move={"points": "x", "text": "x"},
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.45),
    )

    fig.savefig(args.outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(args.outpath.resolve())


if __name__ == "__main__":
    main()
