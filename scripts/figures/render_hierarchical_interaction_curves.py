#!/usr/bin/env python3
"""Render direct posterior interaction curves for a hierarchical variant."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANT_DIR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "hierarchical_block_ablation"
    / "combo_interactions_biology"
)
DEFAULT_OUTPATH = (
    REPO_ROOT / "results" / "figures" / "figure6_combo_overlap_utr5L_curves.png"
)

SPECIES_ORDER = [
    "orthohantavirus_andesense",
    "orthohantavirus_dobravaense",
    "orthohantavirus_hantanense",
    "orthohantavirus_puumalaense",
    "orthohantavirus_seoulense",
    "orthohantavirus_sinnombreense",
    "orthohantavirus_tulaense",
]

SPECIES_LABELS = {
    "orthohantavirus_andesense": "Andes",
    "orthohantavirus_dobravaense": "Dobrava",
    "orthohantavirus_hantanense": "Hantaan",
    "orthohantavirus_puumalaense": "Puumala",
    "orthohantavirus_seoulense": "Seoul",
    "orthohantavirus_sinnombreense": "Sin Nombre",
    "orthohantavirus_tulaense": "Tula",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--outpath", type=Path, default=DEFAULT_OUTPATH)
    parser.add_argument("--n-grid", type=int, default=120)
    return parser.parse_args(argv)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_style()

    analysis = pd.read_csv(args.variant_dir / "analysis_dataset.tsv", sep="\t")
    idata = az.from_netcdf(args.variant_dir / "posterior.nc")
    post = idata.posterior

    overlap_grid = np.linspace(
        float(analysis["overlap_opportunity_local_z"].quantile(0.05)),
        float(analysis["overlap_opportunity_local_z"].quantile(0.95)),
        args.n_grid,
    )
    observed_utr = (
        analysis[["species", "species_feat_utr5_L_structdist_mean_z"]]
        .drop_duplicates()
        ["species_feat_utr5_L_structdist_mean_z"]
    )
    low_utr = float(observed_utr.quantile(0.25))
    high_utr = float(observed_utr.quantile(0.75))

    intercept = np.asarray(post["Intercept"]).reshape(-1)
    beta_overlap = np.asarray(post["overlap_opportunity_local_z"]).reshape(-1)
    beta_utr = np.asarray(post["species_feat_utr5_L_structdist_mean_z"]).reshape(-1)
    beta_inter = np.asarray(
        post["overlap_opportunity_local_z:species_feat_utr5_L_structdist_mean_z"]
    ).reshape(-1)
    species_effects = np.asarray(post["1|species"]).reshape(-1, len(post.coords["species__factor_dim"]))
    species_idx = {str(s): i for i, s in enumerate(np.asarray(post.coords["species__factor_dim"]))}

    curve_rows = []
    for species in SPECIES_ORDER:
        if species not in species_idx:
            continue
        sp = species_effects[:, species_idx[species]]
        for level_name, utr_value, color in [
            ("Low incompatibility", low_utr, "#9ecae1"),
            ("High incompatibility", high_utr, "#f4b6c2"),
        ]:
            logits = (
                intercept[:, None]
                + sp[:, None]
                + beta_overlap[:, None] * overlap_grid[None, :]
                + beta_utr[:, None] * utr_value
                + beta_inter[:, None] * (overlap_grid[None, :] * utr_value)
            )
            probs = sigmoid(logits)
            mean = probs.mean(axis=0)
            lo = np.quantile(probs, 0.03, axis=0)
            hi = np.quantile(probs, 0.97, axis=0)
            for x, m, l, h in zip(overlap_grid, mean, lo, hi):
                curve_rows.append(
                    {
                        "species": species,
                        "species_label": SPECIES_LABELS.get(species, species),
                        "curve": level_name,
                        "utr5_l_structdist_value": utr_value,
                        "overlap_opportunity_local_z": x,
                        "probability_mean": m,
                        "probability_hdi_03": l,
                        "probability_hdi_97": h,
                        "color": color,
                    }
                )

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(args.variant_dir / "figures" / "combo_overlap_utr5L_curves.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(8.2, 7.2), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.ravel()
    for ax in axes[len(SPECIES_ORDER):]:
        ax.axis("off")

    for ax, species in zip(axes, SPECIES_ORDER):
        sub = curve_df[curve_df["species"] == species].copy()
        if sub.empty:
            ax.axis("off")
            continue
        for level_name, color in [("Low incompatibility", "#9ecae1"), ("High incompatibility", "#f4b6c2")]:
            ss = sub[sub["curve"] == level_name].sort_values("overlap_opportunity_local_z")
            ax.fill_between(
                ss["overlap_opportunity_local_z"],
                ss["probability_hdi_03"],
                ss["probability_hdi_97"],
                color=color,
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(
                ss["overlap_opportunity_local_z"],
                ss["probability_mean"],
                color=color,
                linewidth=2.0,
                label=level_name,
            )
        ax.text(
            0.03,
            0.97,
            SPECIES_LABELS.get(species, species),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="#3a3a3a",
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.axhline(0.5, color="#d9d9d9", linewidth=0.6, zorder=0)

    for idx in [0, 3, 6]:
        axes[idx].set_ylabel("Posterior probability")
    for idx in [6, 7, 8]:
        if idx < len(axes):
            axes[idx].set_xlabel("Overlap opportunity (z)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))

    args.outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(args.outpath)


if __name__ == "__main__":
    main()
