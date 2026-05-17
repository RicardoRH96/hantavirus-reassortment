#!/usr/bin/env python3
"""Render alternate hierarchy figures for an ablation variant.

This keeps the promoted hierarchy outputs untouched while generating
Figure 2- and Figure 6-style artifacts for an exploratory variant.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABLATION_ROOT = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "hierarchical_block_ablation"
)
DEFAULT_VARIANT = "combo_interactions_biology"
DEFAULT_DATASET = (
    REPO_ROOT
    / "data"
    / "processed"
    / "final_sequence_dataset.tsv"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_ABLATION_ROOT / DEFAULT_VARIANT / "figures" / "manuscript_combo_interactions_biology_alt"
)
DEFAULT_FIG2 = REPO_ROOT / "results" / "figures" / "figure2_mechanism_inference_combo_interactions_biology.png"
DEFAULT_FIG6 = REPO_ROOT / "results" / "figures" / "figure6_species_block_surface_facets_combo_interactions_biology.png"


MODEL_LABELS = {
    "baseline_integrated_plus_utr3": "Baseline + UTR3",
    "enh02_interactions_primary": "Interactions primary",
    "combo_interactions_biology": "Best biology combo",
    "all_enhancements": "All enhancements",
}

MODEL_COLORS = {
    "Baseline + UTR3": "#9bb7d4",
    "Interactions primary": "#a8d8b9",
    "Best biology combo": "#f2a6a6",
    "All enhancements": "#c9b6e4",
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure2-path", type=Path, default=DEFAULT_FIG2)
    parser.add_argument("--figure6-path", type=Path, default=DEFAULT_FIG6)
    parser.add_argument("--varying-slope-draws", type=int, default=1500)
    parser.add_argument("--varying-slope-tune", type=int, default=1500)
    parser.add_argument("--varying-slope-chains", type=int, default=4)
    parser.add_argument("--varying-slope-cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.03, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def species_short(name: str) -> str:
    return name.replace("orthohantavirus_", "").replace("ense", ".")


def pretty_term(term: str) -> str:
    mapping = {
        "hosttrait_combine_adult_mass_g_z": "Host body mass",
        "hosttrait_combine_litter_size_n_z": "Host litter size",
        "hosttrait_combine_gestation_length_d_z": "Host gestation length",
        "hosttrait_combine_dispersal_km_z": "Host dispersal",
        "overlap_opportunity_local_z": "Overlap opportunity",
        "arha_prev_local_z": "Local host prevalence",
        "species_feat_hostdiv_faith_pd_z": "Host phylogenetic diversity",
        "species_feat_mean_cramers_v_ML_z": "ML LD burden",
        "species_feat_prop_significant_ML_z": "ML significant LD fraction",
        "species_feat_cramers_v_variance_z": "LD burden variance",
        "species_feat_utr5_L_length_mean_z": "UTR5 L length",
        "species_feat_utr5_L_seqdist_mean_z": "UTR5 L sequence distance",
        "species_feat_utr5_L_structdist_mean_z": "UTR5 L structural distance",
        "species_feat_utr3_M_length_mean_z": "UTR3 M length",
        "species_feat_utr3_M_seqdist_mean_z": "UTR3 M sequence distance",
        "species_feat_utr3_M_structdist_mean_z": "UTR3 M structural distance",
        "overlap_opportunity_local_z:species_feat_mean_cramers_v_ML_z": "Overlap × ML LD burden",
        "overlap_opportunity_local_z:species_feat_utr5_L_structdist_mean_z": "Overlap × UTR5 L structure",
        "overlap_opportunity_local_z:species_feat_utr3_M_structdist_mean_z": "Overlap × UTR3 M structure",
    }
    return mapping.get(term, term.replace("_z", "").replace("_", " "))


def term_module(term: str) -> Optional[str]:
    if term.startswith("hosttrait_combine_adult_mass_g"):
        return "Host body mass"
    if term.startswith("hosttrait_combine_litter_size_n"):
        return "Host litter size"
    if term.startswith("hosttrait_combine_gestation_length_d"):
        return "Host gestation length"
    if term.startswith("hosttrait_combine_dispersal_km"):
        return "Host dispersal"
    if term.startswith("overlap_opportunity_local"):
        return "Overlap opportunity"
    if term.startswith("arha_prev_local"):
        return "Local host prevalence"
    if term.startswith("species_feat_hostdiv_faith_pd"):
        return "Host phylogenetic diversity"
    if term.startswith("species_feat_mean_cramers_v_") or term.startswith("species_feat_cramers_v_"):
        return "LD burden"
    if term.startswith("species_feat_prop_significant_"):
        return "Significant LD fraction"
    if term.startswith("species_feat_utr5_S_"):
        return "UTR5 S structure"
    if term.startswith("species_feat_utr5_M_"):
        return "UTR5 M structure"
    if term.startswith("species_feat_utr5_L_"):
        return "UTR5 L structure"
    if term.startswith("species_feat_utr3_S_"):
        return "UTR3 S structure"
    if term.startswith("species_feat_utr3_M_"):
        return "UTR3 M structure"
    if term.startswith("species_feat_utr3_L_"):
        return "UTR3 L structure"
    return None


def load_fixed_effects(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "variant_fixed_effects.tsv", sep="\t")


def load_posterior_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "posterior_summary.tsv", sep="\t")


def load_analysis_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "analysis_dataset.tsv", sep="\t")


def build_module_table(variant_dir: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_analysis_dataset(variant_dir).copy()
    idata = az.from_netcdf(variant_dir / "posterior.nc")
    post = idata.posterior

    def flat_draws(name: str) -> np.ndarray:
        return np.asarray(post[name]).reshape(-1)

    module_names = [
        "Host gestation length",
        "Host litter size",
        "Host dispersal",
        "Host body mass",
        "Host phylogenetic diversity",
        "Overlap opportunity",
        "Local host prevalence",
        "LD burden",
        "Significant LD fraction",
        "UTR5 S structure",
        "UTR5 M structure",
        "UTR5 L structure",
        "UTR3 S structure",
        "UTR3 M structure",
        "UTR3 L structure",
    ]
    modules = pd.DataFrame(0.0, index=df.index, columns=module_names)

    species_labels = [str(x) for x in np.asarray(post.coords["species__factor_dim"])]
    species_to_idx = {species: idx for idx, species in enumerate(species_labels)}
    species_idx = df["species"].map(species_to_idx).to_numpy(dtype=int)

    intercept_draws = flat_draws("Intercept")
    species_draws = np.asarray(post["1|species"]).reshape(-1, len(species_labels))
    linear_draws = intercept_draws[:, None] + species_draws[:, species_idx]

    skip_terms = {"Intercept", "1|species_sigma", "1|species"}
    term_names = [name for name in post.data_vars if name not in skip_terms]
    for term in term_names:
        if term in skip_terms:
            continue
        coef_draws = flat_draws(term)
        if ":" in term:
            lhs, rhs = term.split(":", 1)
            if lhs not in df.columns or rhs not in df.columns:
                continue
            value = df[lhs].to_numpy(dtype=float) * df[rhs].to_numpy(dtype=float)
            contribution_draws = coef_draws[:, None] * value[None, :]
            linear_draws += contribution_draws
            lhs_mod = term_module(lhs)
            rhs_mod = term_module(rhs)
            mean_contrib = contribution_draws.mean(axis=0)
            if lhs_mod:
                modules[lhs_mod] += 0.5 * mean_contrib
            if rhs_mod:
                modules[rhs_mod] += 0.5 * mean_contrib
            continue
        if term not in df.columns:
            continue
        value = df[term].to_numpy(dtype=float)
        contribution_draws = coef_draws[:, None] * value[None, :]
        linear_draws += contribution_draws
        module = term_module(term)
        if module:
            modules[module] += contribution_draws.mean(axis=0)

    out = modules.copy()
    prob_draws = sigmoid(linear_draws)
    out.insert(0, "probability_hdi_97", np.quantile(prob_draws, 0.97, axis=0))
    out.insert(0, "probability_hdi_03", np.quantile(prob_draws, 0.03, axis=0))
    out.insert(0, "probability", prob_draws.mean(axis=0))
    out.insert(0, "target", df["target"].to_numpy(dtype=int))
    out.insert(0, "species", df["species"].to_numpy())
    out.insert(0, "tip_name", [f"{variant_dir.name}|{i:04d}" for i in range(len(df))])
    out.to_csv(out_dir / "integrated_module_contributions.tsv", sep="\t", index=False)
    return out


def build_species_random_effects(ablation_root: Path, variants: list[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    for variant in variants:
        post = load_posterior_summary(ablation_root / variant)
        for _, r in post.iterrows():
            term = r["term"]
            if not term.startswith("1|species["):
                continue
            rows.append(
                {
                    "model": variant,
                    "species": term.split("[", 1)[1].rstrip("]"),
                    "mean": r["mean"],
                    "hdi_3%": r["hdi_3%"],
                    "hdi_97%": r["hdi_97%"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "species_random_effects.tsv", sep="\t", index=False)
    return out


def render_figure2(ablation_root: Path, variant: str, dataset_path: Path, out_dir: Path, outpath: Path) -> None:
    fx = load_fixed_effects(ablation_root)
    species_re = build_species_random_effects(
        ablation_root,
        ["baseline_integrated_plus_utr3", "enh02_interactions_primary", variant, "all_enhancements"],
        out_dir,
    )
    raw = pd.read_csv(dataset_path, sep="\t", low_memory=False)
    raw = raw[raw["is_reassortant_final"].notna()].copy()
    raw = raw[raw["target_source"].isin(["sequence_true_label", "isolate_true_label"])].copy()
    raw["target"] = pd.to_numeric(raw["is_reassortant_final"], errors="coerce")

    fig = plt.figure(figsize=(10.6, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.28, 1.0])
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])
    panel_label(axA, "A")
    panel_label(axB, "B")
    panel_label(axC, "C")

    forest_model_order = ["baseline_integrated_plus_utr3", "enh02_interactions_primary", variant, "all_enhancements"]
    preferred_terms = [
        "hosttrait_combine_adult_mass_g_z",
        "hosttrait_combine_litter_size_n_z",
        "hosttrait_combine_gestation_length_d_z",
        "hosttrait_combine_dispersal_km_z",
        "species_feat_hostdiv_faith_pd_z",
        "overlap_opportunity_local_z",
        "arha_prev_local_z",
        "species_feat_mean_cramers_v_ML_z",
        "species_feat_prop_significant_ML_z",
        "species_feat_cramers_v_variance_z",
        "species_feat_utr5_L_length_mean_z",
        "species_feat_utr5_L_seqdist_mean_z",
        "species_feat_utr5_L_structdist_mean_z",
        "species_feat_utr3_M_length_mean_z",
        "species_feat_utr3_M_seqdist_mean_z",
        "species_feat_utr3_M_structdist_mean_z",
        "overlap_opportunity_local_z:species_feat_mean_cramers_v_ML_z",
        "overlap_opportunity_local_z:species_feat_utr5_L_structdist_mean_z",
        "overlap_opportunity_local_z:species_feat_utr3_M_structdist_mean_z",
    ]
    forest = fx[fx["variant"].isin(forest_model_order) & fx["term"].isin(preferred_terms)].copy()
    forest["term_label"] = forest["term"].map(pretty_term)
    forest["model_label"] = forest["variant"].map(MODEL_LABELS)
    forest["term_label"] = pd.Categorical(
        forest["term_label"],
        categories=[pretty_term(t) for t in preferred_terms if t in forest["term"].unique()],
        ordered=True,
    )
    forest["model_label"] = pd.Categorical(
        forest["model_label"],
        categories=[MODEL_LABELS[m] for m in forest_model_order if m in forest["variant"].unique()],
        ordered=True,
    )
    forest = forest.sort_values(["model_label", "term_label"]).reset_index(drop=True)
    y = np.arange(len(forest))
    for block_idx, (_, block) in enumerate(forest.groupby("model_label", sort=False)):
        if block_idx % 2 == 0:
            axA.axhspan(block.index.min() - 0.5, block.index.max() + 0.5, color="#faf7fb", zorder=0)
    axA.hlines(y, forest["hdi_3%"], forest["hdi_97%"], color=[MODEL_COLORS[m] for m in forest["model_label"]], linewidth=2)
    axA.scatter(forest["mean"], y, color=[MODEL_COLORS[m] for m in forest["model_label"]], s=24, zorder=3)
    axA.axvline(0.0, linestyle="--", color="#666666", linewidth=0.9)
    axA.set_yticks(y, forest["term_label"])
    axA.invert_yaxis()
    axA.set_xlabel("Posterior coefficient")
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5.3, markerfacecolor=MODEL_COLORS[label], markeredgecolor=MODEL_COLORS[label], label=label)
        for label in forest["model_label"].cat.categories
    ]
    axA.legend(handles=legend_handles, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.08), mode="expand", borderaxespad=0.0)

    shrink_models = ["baseline_integrated_plus_utr3", variant]
    shrink = species_re[species_re["model"].isin(shrink_models)].copy()
    species_order = shrink[shrink["model"] == shrink_models[0]].sort_values("mean")["species"].tolist()
    ypos = np.arange(len(species_order))
    offsets = {shrink_models[0]: -0.12, shrink_models[-1]: 0.12}
    point_colors = {
        shrink_models[0]: MODEL_COLORS[MODEL_LABELS[shrink_models[0]]],
        shrink_models[-1]: MODEL_COLORS[MODEL_LABELS[shrink_models[-1]]],
    }
    legend_labels = {m: MODEL_LABELS[m] for m in shrink_models}
    for model in shrink_models:
        sub = shrink[shrink["model"] == model].set_index("species").loc[species_order].reset_index()
        yy = ypos + offsets[model]
        axB.hlines(yy, sub["hdi_3%"], sub["hdi_97%"], color=point_colors[model], linewidth=2, alpha=0.88)
        axB.scatter(sub["mean"], yy, color=point_colors[model], s=28, label=legend_labels[model])
    axB.axvline(0.0, linestyle="--", color="#666666", linewidth=0.9)
    axB.set_yticks(ypos, [species_short(s) for s in species_order])
    axB.set_xlabel("Species random effect")
    axB.legend(frameon=False, loc="lower right")

    combo_df = load_analysis_dataset(ablation_root / variant)
    heat_features = [
        "observed_rate",
        "overlap_opportunity_local_z",
        "arha_prev_local_z",
        "species_feat_hostdiv_faith_pd_z",
        "species_feat_mean_cramers_v_ML_z",
        "species_feat_prop_significant_ML_z",
        "species_feat_utr5_L_length_mean_z",
        "species_feat_utr5_L_seqdist_mean_z",
        "species_feat_utr5_L_structdist_mean_z",
        "species_feat_utr3_M_length_mean_z",
        "species_feat_utr3_M_seqdist_mean_z",
        "species_feat_utr3_M_structdist_mean_z",
    ]
    species_feature = combo_df.groupby("species").agg(observed_rate=("target", "mean")).reset_index()
    feature_cols = [c for c in heat_features if c != "observed_rate" and c in combo_df.columns]
    feature_table = combo_df.groupby("species")[feature_cols].first().reset_index()
    heat = species_feature.merge(feature_table, on="species", how="left").sort_values("observed_rate").reset_index(drop=True)
    heat_mat = heat[["observed_rate", *feature_cols]].copy()
    for col in heat_mat.columns:
        if col == "observed_rate":
            continue
        series = pd.to_numeric(heat_mat[col], errors="coerce")
        std = series.std(ddof=0)
        heat_mat[col] = 0.0 if pd.isna(std) or std == 0 else (series - series.mean()) / std
    data = heat_mat.to_numpy(dtype=float)
    cmap = mpl.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("#e5e5e5")
    im = axC.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, vmin=-2.2, vmax=2.2)
    axC.set_yticks(np.arange(len(heat)), [species_short(s) for s in heat["species"]])
    xlabels = ["Observed rate"] + [pretty_term(c) for c in feature_cols]
    axC.set_xticks(np.arange(len(xlabels)), xlabels, rotation=45, ha="right")
    cbar = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.03)
    cbar.set_label("Species-standardized value")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_figure6(module_table_path: Path, out_dir: Path, outpath: Path, args: argparse.Namespace) -> None:
    response_dir = out_dir / "species_varying_slope_combo_interactions_biology_alt"
    response_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "modeling" / "fit_species_varying_slope_block_model.py"),
        "--module-table",
        str(module_table_path),
        "--output-dir",
        str(response_dir),
        "--figure-path",
        str(outpath),
        "--draws",
        str(args.varying_slope_draws),
        "--tune",
        str(args.varying_slope_tune),
        "--chains",
        str(args.varying_slope_chains),
        "--cores",
        str(args.varying_slope_cores),
        "--target-accept",
        str(args.target_accept),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant_dir = args.ablation_root / args.variant
    module_df = build_module_table(variant_dir, args.output_dir)
    render_figure2(args.ablation_root, args.variant, args.dataset, args.output_dir, args.figure2_path)
    render_figure6(args.output_dir / "integrated_module_contributions.tsv", args.output_dir, args.figure6_path, args)

    manifest = {
        "ablation_root": str(args.ablation_root),
        "variant": args.variant,
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "figure2_path": str(args.figure2_path),
        "figure6_path": str(args.figure6_path),
        "module_rows": int(len(module_df)),
    }
    (args.output_dir / "render_manifest.tsv").write_text(
        "key\tvalue\n" + "\n".join(f"{k}\t{v}" for k, v in manifest.items())
    )
    print(args.figure2_path)
    print(args.figure6_path)


if __name__ == "__main__":
    main()
