#!/usr/bin/env python3
"""Create manuscript-style figures from a hierarchical block-comparison run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "final_hierarchical_block_comparison"
)

MODEL_LABELS = {
    "species_only": "Species only",
    "ecology_block": "Ecology",
    "ecology_interaction_block": "Ecology + interaction",
    "molecular_block": "Molecular",
    "integrated_block": "Integrated",
    "integrated_tree_sensitivity": "Integrated + tree",
}

TERM_LABELS = {
    "hosttrait_combine_gestation_length_d_z": "Host gestation length",
    "hosttrait_combine_litter_size_n_z": "Host litter size",
    "hosttrait_combine_dispersal_km_z": "Host dispersal",
    "species_feat_hostdiv_faith_pd_z": "Host Faith's PD",
    "overlap_opportunity_local_z": "Overlap opportunity",
    "arha_prev_local_z": "Local host prevalence",
    "overlap_opportunity_local_z:species_feat_hostdiv_faith_pd_z": "Overlap x host PD",
    "species_feat_mean_cramers_v_overall_z": "LD burden (Cramer's V)",
    "species_feat_prop_significant_overall_z": "LD significant-pair fraction",
    "species_feat_utr5_M_structdist_mean_z": "UTR5 M structure distance",
    "species_feat_utr5_L_structdist_mean_z": "UTR5 L structure distance",
    "tree_discordance_score_z": "Tree discordance score",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _model_order(summary_df: pd.DataFrame) -> list[str]:
    primary = summary_df.sort_values(["is_primary_model", "loo_elpd"], ascending=[False, False])
    return primary["model"].tolist()


def plot_model_comparison(summary_df: pd.DataFrame, outpath: Path) -> None:
    order = _model_order(summary_df)
    plot_df = summary_df.set_index("model").loc[order].reset_index()
    y = np.arange(len(plot_df))
    colors = np.where(plot_df["is_primary_model"].astype(int) == 1, "#1f4e79", "#9b2226")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    axes[0].barh(y, plot_df["loo_elpd"], color=colors)
    axes[0].set_yticks(y, [MODEL_LABELS.get(m, m) for m in plot_df["model"]])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("LOO expected log predictive density")

    axes[1].barh(y, plot_df["sigma_reduction_fraction_vs_species_only"], color=colors)
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Species random-effect reduction vs species-only")

    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_effect_forest(effect_df: pd.DataFrame, outpath: Path) -> None:
    selected_models = ["ecology_block", "ecology_interaction_block", "molecular_block", "integrated_block", "integrated_tree_sensitivity"]
    selected_terms = list(TERM_LABELS.keys())
    plot_df = effect_df[
        effect_df["model"].isin(selected_models)
        & effect_df["term"].isin(selected_terms)
    ].copy()
    plot_df = plot_df.sort_values(["model", "term"]).reset_index(drop=True)

    if plot_df.empty:
        return

    plot_df["label"] = plot_df["term"].map(TERM_LABELS)
    plot_df["model_label"] = plot_df["model"].map(MODEL_LABELS).fillna(plot_df["model"])
    plot_df["y_label"] = plot_df["label"] + " | " + plot_df["model_label"]

    y = np.arange(len(plot_df))
    fig_height = max(7, len(plot_df) * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height), constrained_layout=True)
    ax.hlines(y, plot_df["hdi_3%"], plot_df["hdi_97%"], color="#577590", linewidth=2)
    ax.scatter(plot_df["mean"], y, color="#bc4749", s=28, zorder=3)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, plot_df["y_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Posterior coefficient")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_species_ppc(ppc_df: pd.DataFrame, outpath: Path) -> None:
    plot_df = ppc_df.sort_values("observed_rate").reset_index(drop=True).copy()
    x = plot_df["observed_rate"].to_numpy()
    y = plot_df["predicted_rate_mean"].to_numpy()
    err_low = y - plot_df["predicted_rate_hdi_3"].to_numpy()
    err_high = plot_df["predicted_rate_hdi_97"].to_numpy() - y

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.errorbar(x, y, yerr=[err_low, err_high], fmt="o", color="#2a6f97", ecolor="#89c2d9", capsize=3)
    lim_min = min(x.min(), y.min(), plot_df["predicted_rate_hdi_3"].min()) - 0.03
    lim_max = max(x.max(), y.max(), plot_df["predicted_rate_hdi_97"].max()) + 0.03
    lim_min = max(0.0, lim_min)
    lim_max = min(1.0, lim_max)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="black", linewidth=0.9)
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["species"].replace("orthohantavirus_", ""),
            (row["observed_rate"], row["predicted_rate_mean"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Observed reassortment rate")
    ax.set_ylabel("Posterior predictive mean")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    figures_dir = args.figures_dir or (args.run_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.run_dir / "block_comparison_summary.tsv", sep="\t")
    effect_df = pd.read_csv(args.run_dir / "fixed_effect_comparison.tsv", sep="\t")
    ppc_df = pd.read_csv(args.run_dir / "integrated_block" / "ppc_by_species.tsv", sep="\t")

    plot_model_comparison(summary_df, figures_dir / "hierarchical_model_comparison.png")
    plot_effect_forest(effect_df, figures_dir / "hierarchical_effect_forest.png")
    plot_species_ppc(ppc_df, figures_dir / "integrated_species_ppc.png")


if __name__ == "__main__":
    main()
