#!/usr/bin/env python3
"""Render baltic tree-length diagnostics for species-by-segment trees."""

from __future__ import annotations

import argparse
from pathlib import Path

import baltic as bt
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TREE_ROOT = (
    REPO_ROOT
    / "model_test"
    / "runs"
    / "direct516_plus_pathoplexus_andes_egt"
    / "tree_roots_common61"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "specific_evolutionary_analyses"
    / "outputs_direct516_pathoplexus_andes"
    / "tree_length_diagnostics"
)

SPECIES = [
    ("orthohantavirus_andesense", "Andes"),
    ("orthohantavirus_dobravaense", "Dobrava"),
    ("orthohantavirus_hantanense", "Hantaan"),
    ("orthohantavirus_puumalaense", "Puumala"),
    ("orthohantavirus_seoulense", "Seoul"),
    ("orthohantavirus_sinnombreense", "Sin Nombre"),
    ("orthohantavirus_tulaense", "Tula"),
]
SEGMENTS = ["S", "M", "L"]
SEGMENT_COLORS = {"S": "#8DD3C7", "M": "#80B1D3", "L": "#FCCDE5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-root", type=Path, default=DEFAULT_TREE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_tree(path: Path):
    tree = bt.loadNewick(str(path), absoluteTime=False)
    tree.sortBranches()
    tree.drawTree()
    return tree


def tree_metrics(tree) -> dict[str, float]:
    lengths = [max(float(getattr(obj, "length", 0.0) or 0.0), 0.0) for obj in tree.Objects]
    leaves = [obj for obj in tree.Objects if getattr(obj, "branchType", "") == "leaf"]
    internal = [obj for obj in tree.Objects if getattr(obj, "branchType", "") == "node"]
    positive_lengths = [x for x in lengths if x > 0]
    return {
        "n_tips": len(leaves),
        "n_internal": len(internal),
        "tree_height": float(getattr(tree, "treeHeight", max((obj.height for obj in leaves), default=0.0))),
        "total_branch_length": float(sum(lengths)),
        "median_positive_branch_length": float(pd.Series(positive_lengths).median()) if positive_lengths else 0.0,
        "max_branch_length": float(max(lengths) if lengths else 0.0),
        "zero_branch_fraction": float(sum(x == 0 for x in lengths) / len(lengths)) if lengths else 0.0,
    }


def clean_axis(ax, show_xlabel: bool) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=7, width=0.6, length=2.5, color="#808080")
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.spines["bottom"].set_linewidth(0.6)
    if show_xlabel:
        ax.set_xlabel("substitutions/site", fontsize=8, labelpad=2)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
    mpl.rcParams["axes.titlesize"] = 0
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["svg.fonttype"] = "none"

    loaded = {}
    rows = []
    for species, label in SPECIES:
        for segment in SEGMENTS:
            path = args.tree_root / species / "augur" / segment / f"{segment}_tree_refined.newick"
            if not path.exists():
                rows.append(
                    {
                        "species": species,
                        "species_label": label,
                        "segment": segment,
                        "tree_path": str(path),
                        "status": "missing",
                    }
                )
                continue
            tree = load_tree(path)
            loaded[(species, segment)] = tree
            rows.append(
                {
                    "species": species,
                    "species_label": label,
                    "segment": segment,
                    "tree_path": str(path),
                    "status": "ok",
                    **tree_metrics(tree),
                }
            )

    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "tree_length_diagnostics_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    x_max = {}
    for segment in SEGMENTS:
        heights = [
            summary.loc[
                (summary["segment"] == segment) & (summary["status"] == "ok"), "tree_height"
            ].max()
        ]
        x_max[segment] = max([x for x in heights if pd.notna(x)] + [1e-6]) * 1.08

    fig, axes = plt.subplots(
        len(SPECIES),
        len(SEGMENTS),
        figsize=(10.2, 12.0),
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
        constrained_layout=False,
    )

    for row_idx, (species, label) in enumerate(SPECIES):
        for col_idx, segment in enumerate(SEGMENTS):
            ax = axes[row_idx, col_idx]
            tree = loaded.get((species, segment))
            if tree is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, color="#808080")
                ax.set_xlim(0, x_max[segment])
                ax.set_ylim(0, 1)
            else:
                color = SEGMENT_COLORS[segment]
                tree.plotTree(
                    ax,
                    x_attr=lambda k: k.height,
                    width=lambda k: 0.42,
                    colour=lambda k: color,
                    connection_type="baltic",
                )
                tree.plotPoints(
                    ax,
                    x_attr=lambda k: k.height,
                    target=lambda k: getattr(k, "branchType", "") == "leaf",
                    size=lambda k: 2.6,
                    colour=lambda k: color,
                    outline=False,
                )
                metric = summary[
                    (summary["species"] == species)
                    & (summary["segment"] == segment)
                    & (summary["status"] == "ok")
                ].iloc[0]
                ax.text(
                    0.985,
                    0.08,
                    f"n={int(metric.n_tips)}  h={metric.tree_height:.3f}  sum={metric.total_branch_length:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=6.5,
                    color="#555555",
                )
                ax.set_xlim(0, x_max[segment])
                ax.set_ylim(0, max(1.0, tree.ySpan + 1.0))
            clean_axis(ax, show_xlabel=row_idx == len(SPECIES) - 1)
            if row_idx == 0:
                ax.text(
                    0.02,
                    0.96,
                    segment,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="#333333",
                    fontweight="bold",
                )
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                    color="#333333",
                )

    fig.subplots_adjust(left=0.12, right=0.985, top=0.985, bottom=0.055)
    png_path = args.output_dir / "all_species_segment_trees_shared_x.png"
    svg_path = args.output_dir / "all_species_segment_trees_shared_x.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)

    print(summary_path)
    print(png_path)
    print(svg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
