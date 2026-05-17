#!/usr/bin/env python3
"""Render species-overlay interaction curves for hierarchical ablation variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANT_DIR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "hierarchical_block_ablation"
    / "combo_interactions_biology"
)
DEFAULT_OUTPATH = REPO_ROOT / "results" / "figures" / "figure6_combo_interaction_species_overlays.png"

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

SPECIES_COLORS = {
    "orthohantavirus_andesense": "#a7d8c7",
    "orthohantavirus_dobravaense": "#f3c48d",
    "orthohantavirus_hantanense": "#c8b7e6",
    "orthohantavirus_puumalaense": "#cbe7a1",
    "orthohantavirus_seoulense": "#f1b7cf",
    "orthohantavirus_sinnombreense": "#e6c6a5",
    "orthohantavirus_tulaense": "#a7c8ea",
}

INTERACTIONS = [
    {
        "base_term": "species_feat_utr5_L_structdist_mean_z",
        "interaction_term": "overlap_opportunity_local_z:species_feat_utr5_L_structdist_mean_z",
        "label": "UTR5 L incompatibility",
        "level_name_low": "Low UTR5 L",
        "level_name_high": "High UTR5 L",
    },
    {
        "base_term": "species_feat_utr3_M_structdist_mean_z",
        "interaction_term": "overlap_opportunity_local_z:species_feat_utr3_M_structdist_mean_z",
        "label": "UTR3 M incompatibility",
        "level_name_low": "Low UTR3 M",
        "level_name_high": "High UTR3 M",
    },
    {
        "base_term": "species_feat_mean_cramers_v_ML_z",
        "interaction_term": "overlap_opportunity_local_z:species_feat_mean_cramers_v_ML_z",
        "label": "ML LD burden",
        "level_name_low": "Low ML LD",
        "level_name_high": "High ML LD",
    },
]


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
            "legend.fontsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gray_tint(color: str, gray: str = "#9aa1a8", mix: float = 0.58) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(color))
    gray_rgb = np.asarray(to_rgb(gray))
    return tuple((1.0 - mix) * base + mix * gray_rgb)


def compute_curve_table(variant_dir: Path, n_grid: int) -> pd.DataFrame:
    analysis = pd.read_csv(variant_dir / "analysis_dataset.tsv", sep="\t")
    idata = az.from_netcdf(variant_dir / "posterior.nc")
    post = idata.posterior

    overlap_grid = np.linspace(
        float(analysis["overlap_opportunity_local_z"].quantile(0.05)),
        float(analysis["overlap_opportunity_local_z"].quantile(0.95)),
        n_grid,
    )

    intercept = np.asarray(post["Intercept"]).reshape(-1)
    beta_overlap = np.asarray(post["overlap_opportunity_local_z"]).reshape(-1)
    species_effects = np.asarray(post["1|species"]).reshape(-1, len(post.coords["species__factor_dim"]))
    species_idx = {str(s): i for i, s in enumerate(np.asarray(post.coords["species__factor_dim"]))}

    rows = []
    for spec in INTERACTIONS:
        base_term = spec["base_term"]
        interaction_term = spec["interaction_term"]
        observed = analysis[["species", base_term]].drop_duplicates()[base_term]
        low_value = float(observed.quantile(0.25))
        high_value = float(observed.quantile(0.75))
        beta_base = np.asarray(post[base_term]).reshape(-1)
        beta_inter = np.asarray(post[interaction_term]).reshape(-1)

        for species in SPECIES_ORDER:
            if species not in species_idx:
                continue
            sp = species_effects[:, species_idx[species]]
            for level_name, level_value in [
                (spec["level_name_low"], low_value),
                (spec["level_name_high"], high_value),
            ]:
                logits = (
                    intercept[:, None]
                    + sp[:, None]
                    + beta_overlap[:, None] * overlap_grid[None, :]
                    + beta_base[:, None] * level_value
                    + beta_inter[:, None] * (overlap_grid[None, :] * level_value)
                )
                probs = sigmoid(logits)
                mean = probs.mean(axis=0)
                for x, m in zip(overlap_grid, mean):
                    rows.append(
                        {
                            "interaction_label": spec["label"],
                            "species": species,
                            "species_label": SPECIES_LABELS.get(species, species),
                            "curve": level_name,
                            "level_value": level_value,
                            "overlap_opportunity_local_z": x,
                            "probability_mean": m,
                        }
                    )
    return pd.DataFrame(rows)


def render(curves: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, interaction in zip(axes, [spec["label"] for spec in INTERACTIONS]):
        sub = curves[curves["interaction_label"] == interaction]
        spec = next(s for s in INTERACTIONS if s["label"] == interaction)
        for species in SPECIES_ORDER:
            ss = sub[sub["species"] == species]
            if ss.empty:
                continue
            color = SPECIES_COLORS[species]
            muted = gray_tint(color)
            low = ss[ss["curve"] == spec["level_name_low"]].sort_values("overlap_opportunity_local_z")
            high = ss[ss["curve"] == spec["level_name_high"]].sort_values("overlap_opportunity_local_z")
            ax.plot(
                low["overlap_opportunity_local_z"],
                low["probability_mean"],
                color=color,
                linewidth=1.5,
                alpha=0.95,
            )
            ax.plot(
                high["overlap_opportunity_local_z"],
                high["probability_mean"],
                color=muted,
                linewidth=1.5,
                linestyle="--",
                alpha=0.95,
            )
        ax.text(0.02, 0.97, interaction, transform=ax.transAxes, ha="left", va="top", color="#3a3a3a")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.axhline(0.5, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_xlabel("Overlap opportunity (z)")

    axes[0].set_ylabel("Posterior probability")

    from matplotlib.lines import Line2D

    species_handles = [
        Line2D([0], [0], color=SPECIES_COLORS[s], linewidth=2, label=SPECIES_LABELS[s]) for s in SPECIES_ORDER
    ]
    style_handles = [
        Line2D([0], [0], color="#666666", linewidth=1.8, linestyle="-", label="Low"),
        Line2D([0], [0], color="#9aa1a8", linewidth=1.8, linestyle="--", label="High"),
    ]
    fig.legend(
        handles=species_handles + style_handles,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        columnspacing=0.9,
        handlelength=1.6,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_style()
    curves = compute_curve_table(args.variant_dir, args.n_grid)
    curves.to_csv(args.variant_dir / "figures" / "combo_interaction_species_overlays.tsv", sep="\t", index=False)
    render(curves, args.outpath)
    print(args.outpath)


if __name__ == "__main__":
    main()
