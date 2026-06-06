#!/usr/bin/env python3
"""Plot corrected common-tip FEL results across S/M/L segments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIR = (
    REPO_ROOT
    / "specific_evolutionary_analyses"
    / "outputs_direct516_pathoplexus_andes"
    / "hyphy_fel_selection_scan"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "specific_evolutionary_analyses"
    / "outputs_direct516_pathoplexus_andes"
    / "fel_selection_overview"
)

SEGMENTS = [
    ("S", "S / NP", "#8DD3C7"),
    ("M", "M / GPC", "#CDB4DB"),
    ("L", "L / RdRp", "#FCCDE5"),
]

SPECIES_LABELS = {
    "orthohantavirus_andesense": "andes",
    "orthohantavirus_dobravaense": "dobrava",
    "orthohantavirus_hantanense": "hantaan",
    "orthohantavirus_puumalaense": "puumala",
    "orthohantavirus_seoulense": "seoul",
    "orthohantavirus_sinnombreense": "sin_nombre",
    "orthohantavirus_tulaense": "tula",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        nargs="+",
        default=["orthohantavirus_andesense"],
        help="Species IDs matching FEL JSON prefixes.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def slug_for_species(species: str) -> str:
    return SPECIES_LABELS.get(species, species.replace("orthohantavirus_", "").replace("ense", ""))


def load_fel(species: str, segment: str) -> pd.DataFrame:
    path = SCAN_DIR / "reports" / f"{species}_{segment}_FEL.json"
    data = json.loads(path.read_text())
    headers = [h[0] for h in data["MLE"]["headers"]]
    df = pd.DataFrame(data["MLE"]["content"]["0"], columns=headers)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["segment"] = segment
    df["species"] = species
    df["site"] = range(1, len(df) + 1)
    df["effect"] = df["beta"] - df["alpha"]
    df["minus_log10_p"] = -df["p-value"].clip(lower=1e-300).map(math.log10)
    df["q_value"] = multipletests(df["p-value"].fillna(1.0), method="fdr_bh")[1]
    df["negative_p"] = (df["beta"] < df["alpha"]) & (df["p-value"] <= 0.1)
    df["positive_p"] = (df["beta"] > df["alpha"]) & (df["p-value"] <= 0.1)
    df["negative_q"] = (df["beta"] < df["alpha"]) & (df["q_value"] <= 0.1)
    df["positive_q"] = (df["beta"] > df["alpha"]) & (df["q_value"] <= 0.1)
    df["n_sequences"] = data["input"]["number of sequences"]
    return df


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C1D1")
    ax.spines["bottom"].set_color("#B8C1D1")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", labelsize=8, colors="#6E7890", width=0.6, length=3)
    ax.grid(True, axis="both", color="#D8E0EF", linewidth=0.55, alpha=0.9)


def add_segment_backgrounds(ax, spans) -> None:
    for _, _, start, end, color in spans:
        ax.axvspan(start, end, color=color, alpha=0.18, lw=0, zorder=0)
    for _, _, start, _, _ in spans[1:]:
        ax.axvline(start, color="#D7D2A8", lw=1.0, alpha=0.75, zorder=1)


def render_species(species: str, output_dir: Path) -> tuple[Path, Path]:
    frames = []
    spans = []
    offset = 0
    for segment, label, color in SEGMENTS:
        df = load_fel(species, segment)
        df["x"] = df["site"] + offset
        frames.append(df)
        spans.append((segment, label, offset + 1, offset + len(df), color))
        offset += len(df)
    data = pd.concat(frames, ignore_index=True)
    slug = slug_for_species(species)
    data.to_csv(output_dir / f"{slug}_common_tip_fel_site_table.tsv", sep="\t", index=False)

    summary = (
        data.groupby("segment")
        .agg(
            n_sequences=("n_sequences", "first"),
            sites=("site", "size"),
            negative_p=("negative_p", "sum"),
            negative_q=("negative_q", "sum"),
            positive_p=("positive_p", "sum"),
            positive_q=("positive_q", "sum"),
            median_alpha=("alpha", "median"),
            median_beta=("beta", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / f"{slug}_common_tip_fel_overview_summary.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[0.65, 2.5, 1.75],
        hspace=0.28,
        left=0.075,
        right=0.985,
        top=0.965,
        bottom=0.085,
    )
    ax_bar = fig.add_subplot(gs[0])
    ax_effect = fig.add_subplot(gs[1], sharex=ax_bar)
    ax_evidence = fig.add_subplot(gs[2], sharex=ax_bar)

    for ax in [ax_effect, ax_evidence]:
        add_segment_backgrounds(ax, spans)
        style_axis(ax)

    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlim(1, offset)
    ax_bar.axis("off")
    for segment, label, start, end, color in spans:
        ax_bar.add_patch(
            plt.Rectangle((start, 0.32), end - start + 1, 0.36, color=color, ec="none", alpha=0.95)
        )
        ax_bar.text((start + end) / 2, 0.72, label, ha="center", va="bottom", fontsize=9, color="#303640")
        ax_bar.text(
            (start + end) / 2,
            0.28,
            f"{end - start + 1:,} codons",
            ha="center",
            va="top",
            fontsize=8,
            color="#6E7890",
        )
    for _, _, start, _, _ in spans[1:]:
        ax_bar.axvline(start, color="#D7D2A8", lw=1.1, alpha=0.85)

    clipped_low = -25.0
    plot_y = data["effect"].clip(lower=clipped_low)
    neutral = ~(data["negative_p"] | data["positive_p"])
    ax_effect.scatter(
        data.loc[neutral, "x"],
        plot_y.loc[neutral],
        s=20,
        color="#BFC4C9",
        alpha=0.65,
        edgecolor="none",
        zorder=2,
    )
    neg = data["negative_p"]
    ax_effect.scatter(
        data.loc[neg, "x"],
        plot_y.loc[neg],
        s=50,
        facecolor="#17BEEA",
        alpha=0.85,
        edgecolor="#111111",
        linewidth=0.9,
        zorder=3,
    )
    q = data["negative_q"] | data["positive_q"]
    ax_effect.scatter(
        data.loc[q, "x"],
        plot_y.loc[q],
        s=92,
        facecolor="none",
        edgecolor="#111111",
        linewidth=1.5,
        zorder=4,
    )
    clipped = data["effect"] < clipped_low
    ax_effect.scatter(
        data.loc[clipped, "x"],
        [clipped_low] * int(clipped.sum()),
        marker="v",
        s=58,
        facecolor="#17BEEA",
        edgecolor="#111111",
        linewidth=0.9,
        zorder=5,
    )
    ax_effect.axhline(0, color="#7D8798", lw=0.9, alpha=0.85)
    ax_effect.set_ylim(clipped_low - 1.5, 4.2)
    ax_effect.set_ylabel("FEL beta-alpha", fontsize=9, color="#566178")
    ax_effect.tick_params(axis="x", labelbottom=False)

    neutral_evidence = ~data["negative_p"]
    ax_evidence.scatter(
        data.loc[neutral_evidence, "x"],
        data.loc[neutral_evidence, "minus_log10_p"],
        s=19,
        color="#D0D3D6",
        alpha=0.62,
        edgecolor="none",
        zorder=2,
    )
    ax_evidence.scatter(
        data.loc[neg, "x"],
        data.loc[neg, "minus_log10_p"],
        s=42,
        facecolor="#17BEEA",
        alpha=0.82,
        edgecolor="#111111",
        linewidth=0.75,
        zorder=3,
    )
    ax_evidence.scatter(
        data.loc[q, "x"],
        data.loc[q, "minus_log10_p"],
        s=82,
        facecolor="none",
        edgecolor="#111111",
        linewidth=1.4,
        zorder=4,
    )
    ax_evidence.axhline(-math.log10(0.1), color="#9CA6B8", lw=0.8, ls="--", alpha=0.8)
    ax_evidence.set_ylabel("FEL -log10(p-value)", fontsize=9, color="#566178")
    ax_evidence.set_xlabel("Common-tip codon position, concatenated by segment", fontsize=9, color="#6E7890")

    for ax in [ax_effect, ax_evidence]:
        for segment, label, start, end, _ in spans:
            seg_summary = summary[summary["segment"] == segment].iloc[0]
            ax.text(
                end - 8,
                ax.get_ylim()[1] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"{segment}: p-neg {int(seg_summary.negative_p)}, q-neg {int(seg_summary.negative_q)}",
                ha="right",
                va="top",
                fontsize=7.5,
                color="#4C566A",
            )

    png = output_dir / f"{slug}_common_tip_fel_selection_overview.png"
    svg = output_dir / f"{slug}_common_tip_fel_selection_overview.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
    mpl.rcParams["axes.titlesize"] = 0
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["svg.fonttype"] = "none"

    for species in args.species:
        png, svg = render_species(species, args.output_dir)
        print(png)
        print(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
