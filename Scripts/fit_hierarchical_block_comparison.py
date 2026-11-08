#!/usr/bin/env python3
"""Hierarchical block-comparison for reassortment mechanism inference."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    os.environ.get(
        "HANTA_EVO_DATASET",
        REPO_ROOT
        / "model_test"
        / "runs"
        / "model_site"
        / "datasets"
        / "sequence_level_final_dataset_combined_relaxed_andesense.tsv",
    )
).expanduser()
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "model_test"
    / "runs"
    / "model_site"
    / "hierarchical_block_comparison_combined_relaxed_andesense"
)

ECOLOGY_FEATURES = [
    "hosttrait_combine_adult_mass_g",
    "hosttrait_combine_litter_size_n",
    "hosttrait_combine_gestation_length_d",
    "hosttrait_combine_dispersal_km",
    "species_feat_hostdiv_faith_pd",
    "overlap_opportunity_local",
    "arha_prev_local",
]

MOLECULAR_FEATURES = [
    "species_feat_mean_cramers_v_overall",
    "species_feat_prop_significant_overall",
    "species_feat_utr5_S_structdist_mean",
    "species_feat_utr5_M_structdist_mean",
    "species_feat_utr5_L_structdist_mean",
]

UTR3_FEATURES = [
    "species_feat_utr3_S_structdist_mean",
    "species_feat_utr3_M_structdist_mean",
    "species_feat_utr3_L_structdist_mean",
]

MOLECULAR_FEATURES_PLUS_UTR3 = MOLECULAR_FEATURES + UTR3_FEATURES

TREE_SENSITIVITY_FEATURES = [
    "tree_discordance_score",
]


@dataclass(frozen=True)
class BlockModelSpec:
    name: str
    predictors: List[str]
    description: str
    interaction_pairs: List[tuple[str, str]] = field(default_factory=list)
    is_primary: bool = True


MODEL_SPECS = [
    BlockModelSpec("species_only", [], "Species random intercept only."),
    BlockModelSpec("ecology_block", ECOLOGY_FEATURES, "Ecology and host-trait mechanisms with species random intercept."),
    BlockModelSpec(
        "ecology_interaction_block",
        ECOLOGY_FEATURES,
        "Ecology block with predeclared overlap x host-PD interaction.",
        interaction_pairs=[("overlap_opportunity_local", "species_feat_hostdiv_faith_pd")],
    ),
    BlockModelSpec("molecular_block", MOLECULAR_FEATURES, "Molecular compatibility proxies with species random intercept."),
    BlockModelSpec(
        "molecular_plus_utr3_block",
        MOLECULAR_FEATURES_PLUS_UTR3,
        "Molecular compatibility proxies plus 3' UTR summaries with species random intercept.",
    ),
    BlockModelSpec(
        "integrated_block",
        ECOLOGY_FEATURES + MOLECULAR_FEATURES,
        "Integrated ecology and molecular mechanism model with species random intercept.",
    ),
    BlockModelSpec(
        "integrated_plus_utr3_block",
        ECOLOGY_FEATURES + MOLECULAR_FEATURES_PLUS_UTR3,
        "Integrated ecology, molecular, and 3' UTR mechanism model with species random intercept.",
    ),
    BlockModelSpec(
        "integrated_overlap_ld_interaction",
        ECOLOGY_FEATURES + MOLECULAR_FEATURES,
        "Integrated model with overlap opportunity x LD burden interaction.",
        interaction_pairs=[("overlap_opportunity_local", "species_feat_mean_cramers_v_overall")],
        is_primary=False,
    ),
    BlockModelSpec(
        "integrated_overlap_utrL_interaction",
        ECOLOGY_FEATURES + MOLECULAR_FEATURES,
        "Integrated model with overlap opportunity x UTR5 L structural distance interaction.",
        interaction_pairs=[("overlap_opportunity_local", "species_feat_utr5_L_structdist_mean")],
        is_primary=False,
    ),
    BlockModelSpec(
        "integrated_hostpd_ld_interaction",
        ECOLOGY_FEATURES + MOLECULAR_FEATURES,
        "Integrated model with host phylogenetic diversity x LD burden interaction.",
        interaction_pairs=[("species_feat_hostdiv_faith_pd", "species_feat_mean_cramers_v_overall")],
        is_primary=False,
    ),
    BlockModelSpec(
        "integrated_tree_sensitivity",
        ECOLOGY_FEATURES + MOLECULAR_FEATURES + TREE_SENSITIVITY_FEATURES,
        "Integrated model plus tree-discordance sensitivity term.",
        is_primary=False,
    ),
]


COMMON_PRIORS = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1.5),
    "common": bmb.Prior("Normal", mu=0, sigma=1.0),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-source",
        choices=["all", "true_label", "sequence_true_label", "isolate_true_label"],
        default="true_label",
    )
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--tune", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.97)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args(argv)


def load_dataset(path: Path, target_source: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["is_reassortant_final"].notna()].copy()
    if target_source != "all":
        if target_source == "true_label":
            df = df[df["target_source"].isin(["sequence_true_label", "isolate_true_label"])].copy()
        else:
            df = df[df["target_source"] == target_source].copy()
    df["target"] = pd.to_numeric(df["is_reassortant_final"], errors="coerce")
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)
    df["species"] = df["species"].astype(str)
    return df


def add_scaled_feature(frame: pd.DataFrame, raw: pd.Series, name: str) -> List[str]:
    created: List[str] = []
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().sum() == 0:
        return created
    missing = numeric.isna().astype(int)
    if missing.sum() > 0 and missing.sum() < len(missing):
        miss_name = f"{name}_missing"
        frame[miss_name] = missing
        created.append(miss_name)
    filled = numeric.fillna(numeric.median())
    std = float(filled.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return created
    z_name = f"{name}_z"
    frame[z_name] = (filled - filled.mean()) / std
    created.insert(0, z_name)
    return created


def prepare_analysis_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    frame = df[["species", "target"]].copy()
    frame["has_host_traits"] = pd.to_numeric(df.get("hosttrait_has_any_traits"), errors="coerce").fillna(0).astype(int)

    feature_map: Dict[str, List[str]] = {}
    all_features = sorted(
        {
            feature
            for spec in MODEL_SPECS
            for feature in spec.predictors
        }
    )
    for feature in all_features:
        if feature not in df.columns:
            continue
        created = add_scaled_feature(frame, df[feature], feature)
        if created:
            feature_map[feature] = created
    return frame, feature_map


def build_formula(spec: BlockModelSpec, feature_map: Dict[str, List[str]]) -> str:
    predictors = ["has_host_traits"]
    for feature in spec.predictors:
        predictors.extend(feature_map.get(feature, []))
    for left, right in spec.interaction_pairs:
        left_z = f"{left}_z"
        right_z = f"{right}_z"
        if left_z in predictors and right_z in predictors:
            predictors.append(f"{left_z}:{right_z}")
    predictors = list(dict.fromkeys(predictors))
    rhs = " + ".join(predictors + ["(1|species)"]) if predictors else "(1|species)"
    return f"target ~ {rhs}"


def extract_sign_probability(idata: az.InferenceData, term: str) -> float:
    values = np.asarray(idata.posterior[term]).reshape(-1)
    return float((values > 0).mean())


def extract_sigma_mean(idata: az.InferenceData) -> float:
    return float(np.asarray(idata.posterior["1|species_sigma"]).reshape(-1).mean())


def build_model(spec: BlockModelSpec, analysis_df: pd.DataFrame, feature_map: Dict[str, List[str]]) -> tuple[bmb.Model, str]:
    formula = build_formula(spec, feature_map)
    model = bmb.Model(formula, analysis_df, family="bernoulli", priors=COMMON_PRIORS)
    return model, formula


def compute_diagnostics(
    idata: az.InferenceData,
    analysis_df: pd.DataFrame,
) -> tuple[Dict[str, float], pd.DataFrame]:
    diagnostics: Dict[str, float] = {}
    pareto_df = pd.DataFrame(columns=["row_index", "species", "target", "pareto_k", "log_likelihood_mean"])
    try:
        loo = az.loo(idata, pointwise=True)
        diagnostics["loo_elpd"] = float(loo.elpd_loo)
        diagnostics["loo_p"] = float(loo.p_loo)
        diagnostics["loo_se"] = float(loo.se)
        pareto = np.asarray(loo.pareto_k)
        log_likelihood = np.asarray(idata.log_likelihood["target"]).mean(axis=(0, 1))
        pareto_df = pd.DataFrame(
            {
                "row_index": np.arange(len(analysis_df)),
                "species": analysis_df["species"].to_numpy(),
                "target": analysis_df["target"].to_numpy(),
                "pareto_k": pareto,
                "log_likelihood_mean": log_likelihood,
            }
        )
        diagnostics["pareto_k_max"] = float(np.nanmax(pareto))
        diagnostics["pareto_k_n_gt_0_7"] = int(np.sum(pareto > 0.7))
        diagnostics["pareto_k_n_gt_1_0"] = int(np.sum(pareto > 1.0))
    except Exception:
        diagnostics["loo_elpd"] = np.nan
        diagnostics["loo_p"] = np.nan
        diagnostics["loo_se"] = np.nan
        diagnostics["pareto_k_max"] = np.nan
        diagnostics["pareto_k_n_gt_0_7"] = np.nan
        diagnostics["pareto_k_n_gt_1_0"] = np.nan

    try:
        waic = az.waic(idata)
        diagnostics["waic_elpd"] = float(waic.elpd_waic)
        diagnostics["waic_p"] = float(waic.p_waic)
        diagnostics["waic_se"] = float(waic.se)
    except Exception:
        diagnostics["waic_elpd"] = np.nan
        diagnostics["waic_p"] = np.nan
        diagnostics["waic_se"] = np.nan

    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None and "diverging" in sample_stats:
        divergences = np.asarray(sample_stats["diverging"]).sum()
        diagnostics["n_divergences"] = int(divergences)
    else:
        diagnostics["n_divergences"] = np.nan
    return diagnostics, pareto_df


def summarize_ppc(
    model: bmb.Model,
    idata: az.InferenceData,
    analysis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_df = analysis_df.reset_index(drop=True).copy()
    pred_idata = model.predict(idata, kind="response_params", inplace=False)
    p_draws = np.asarray(pred_idata.posterior["p"])
    draw_axis = p_draws.reshape(-1, p_draws.shape[-1])

    overall_rate = draw_axis.mean(axis=1)
    overall_df = pd.DataFrame(
        [
            {
                "scope": "overall",
                "observed_rate": float(analysis_df["target"].mean()),
                "predicted_rate_mean": float(overall_rate.mean()),
                "predicted_rate_hdi_3": float(np.quantile(overall_rate, 0.03)),
                "predicted_rate_hdi_97": float(np.quantile(overall_rate, 0.97)),
            }
        ]
    )

    species_rows: List[Dict[str, object]] = []
    for species, group in analysis_df.groupby("species", dropna=False):
        idx = group.index.to_numpy(dtype=int)
        species_draws = draw_axis[:, idx].mean(axis=1)
        species_rows.append(
            {
                "species": species,
                "n_rows": int(len(group)),
                "observed_rate": float(group["target"].mean()),
                "predicted_rate_mean": float(species_draws.mean()),
                "predicted_rate_hdi_3": float(np.quantile(species_draws, 0.03)),
                "predicted_rate_hdi_97": float(np.quantile(species_draws, 0.97)),
            }
        )
    return overall_df, pd.DataFrame(species_rows).sort_values("species").reset_index(drop=True)


def build_report_markdown(
    comparison_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
) -> str:
    primary = comparison_df[comparison_df["is_primary_model"] == 1].copy()
    best = primary.sort_values("loo_elpd", ascending=False).iloc[0]
    lines = [
        "# Hierarchical Block Comparison",
        "",
        "## Model Ranking",
        f"- Best primary-model expected out-of-sample fit by LOO: `{best['model']}`",
        f"- Species sigma reduction vs species-only: `{best['sigma_reduction_fraction_vs_species_only']:.3f}`",
        "",
        "## Block Comparison",
    ]
    for _, row in comparison_df.sort_values("loo_elpd", ascending=False).iterrows():
        tag = " [sensitivity]" if int(row["is_primary_model"]) == 0 else ""
        lines.append(
            f"- `{row['model']}`{tag}: loo_elpd={row['loo_elpd']:.3f}, species_sigma_mean={row['species_sigma_mean']:.3f}, sigma_reduction={row['sigma_reduction_fraction_vs_species_only']:.3f}"
        )
    lines.extend(["", "## Diagnostics"])
    for _, row in diagnostics_df.sort_values("model").iterrows():
        max_pareto = "NA" if pd.isna(row["pareto_k_max"]) else f"{float(row['pareto_k_max']):.3f}"
        n_gt_07 = "NA" if pd.isna(row["pareto_k_n_gt_0_7"]) else str(int(row["pareto_k_n_gt_0_7"]))
        divergences = "NA" if pd.isna(row["n_divergences"]) else str(int(row["n_divergences"]))
        lines.append(
            f"- `{row['model']}`: divergences={divergences}, max Pareto-k={max_pareto}, n(Pareto-k>0.7)={n_gt_07}"
        )
    lines.extend(["", "## Main Fixed Effects"])
    filtered = effect_df[
        effect_df["term"].isin(
            [
                "hosttrait_combine_gestation_length_d_z",
                "hosttrait_combine_litter_size_n_z",
                "hosttrait_combine_dispersal_km_z",
                "overlap_opportunity_local_z",
                "arha_prev_local_z",
                "species_feat_hostdiv_faith_pd_z",
                "overlap_opportunity_local_z:species_feat_hostdiv_faith_pd_z",
                "species_feat_mean_cramers_v_overall_z",
                "species_feat_prop_significant_overall_z",
                "species_feat_utr5_S_structdist_mean_z",
                "species_feat_utr5_M_structdist_mean_z",
                "species_feat_utr5_L_structdist_mean_z",
                "species_feat_utr3_S_structdist_mean_z",
                "species_feat_utr3_M_structdist_mean_z",
                "species_feat_utr3_L_structdist_mean_z",
                "tree_discordance_score_z",
            ]
        )
    ]
    for _, row in filtered.sort_values(["model", "term"]).iterrows():
        lines.append(
            f"- `{row['model']}` / `{row['term']}`: mean={row['mean']:.3f}, 94% HDI=({row['hdi_3%']:.3f}, {row['hdi_97%']:.3f}), P(>0)={row['posterior_prob_positive']:.3f}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(args.dataset, args.target_source)
    analysis_df, feature_map = prepare_analysis_frame(raw)
    analysis_df.to_csv(args.output_dir / "analysis_dataset.tsv", sep="\t", index=False)

    comparison_rows: List[Dict[str, object]] = []
    effect_rows: List[Dict[str, object]] = []
    species_rows: List[Dict[str, object]] = []
    diagnostics_rows: List[Dict[str, object]] = []
    formulas: Dict[str, str] = {}
    sigma_means: Dict[str, float] = {}
    observation_manifest = (
        raw[["tip_name", "accession", "canonical_isolate", "species", "host", "country", "target_source"]]
        .reset_index(drop=True)
        .copy()
    )
    observation_manifest["row_index"] = np.arange(len(observation_manifest))
    observation_manifest = observation_manifest[
        ["row_index", "species", "tip_name", "accession", "canonical_isolate", "host", "country", "target_source"]
    ]
    observation_manifest.to_csv(args.output_dir / "observation_manifest.tsv", sep="\t", index=False)

    for spec in MODEL_SPECS:
        model, formula = build_model(spec, analysis_df, feature_map)
        formulas[spec.name] = formula
        idata = model.fit(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            random_seed=args.random_seed,
            target_accept=args.target_accept,
            idata_kwargs={"log_likelihood": True},
        )

        model_dir = args.output_dir / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "formula.txt").write_text(formula + "\n")
        idata.to_netcdf(model_dir / "posterior.nc")

        diagnostics, pareto_df = compute_diagnostics(idata, analysis_df)
        if not pareto_df.empty:
            pareto_df.to_csv(model_dir / "pareto_k_by_observation.tsv", sep="\t", index=False)
            enriched_pareto = pareto_df.merge(observation_manifest, on=["row_index", "species"], how="left")
            enriched_pareto.sort_values("pareto_k", ascending=False).head(25).to_csv(
                model_dir / "influential_observations_top25.tsv",
                sep="\t",
                index=False,
            )
        summary = az.summary(idata, var_names=list(idata.posterior.data_vars), round_to=4).reset_index().rename(
            columns={"index": "term"}
        )
        summary.to_csv(model_dir / "posterior_summary.tsv", sep="\t", index=False)
        overall_ppc_df, species_ppc_df = summarize_ppc(model, idata, analysis_df)
        overall_ppc_df.to_csv(model_dir / "ppc_overall.tsv", sep="\t", index=False)
        species_ppc_df.to_csv(model_dir / "ppc_by_species.tsv", sep="\t", index=False)

        sigma_mean = extract_sigma_mean(idata)
        sigma_means[spec.name] = sigma_mean
        comparison_rows.append(
            {
                "model": spec.name,
                "description": spec.description,
                "formula": formula,
                "rows": int(len(analysis_df)),
                "positives": int(analysis_df["target"].sum()),
                "species_levels": int(analysis_df["species"].nunique()),
                "species_sigma_mean": sigma_mean,
                "loo_elpd": diagnostics["loo_elpd"],
                "loo_p": diagnostics["loo_p"],
                "loo_se": diagnostics["loo_se"],
                "waic_elpd": diagnostics["waic_elpd"],
                "waic_p": diagnostics["waic_p"],
                "waic_se": diagnostics["waic_se"],
                "is_primary_model": int(spec.is_primary),
            }
        )
        diagnostics_row = {"model": spec.name, **diagnostics}
        diagnostics_rows.append(diagnostics_row)

        fixed_terms = ["Intercept", "has_host_traits"]
        for feature in spec.predictors:
            fixed_terms.extend(feature_map.get(feature, []))
        for left, right in spec.interaction_pairs:
            fixed_terms.append(f"{left}_z:{right}_z")
        fixed_terms = list(dict.fromkeys(fixed_terms))
        for term in fixed_terms:
            if term not in summary["term"].values:
                continue
            row = summary.loc[summary["term"] == term].iloc[0].to_dict()
            row["model"] = spec.name
            row["posterior_prob_positive"] = extract_sign_probability(idata, term)
            effect_rows.append(row)

        species_terms = [term for term in summary["term"] if term.startswith("1|species[")]
        for term in species_terms:
            row = summary.loc[summary["term"] == term].iloc[0].to_dict()
            row["model"] = spec.name
            row["species"] = term.removeprefix("1|species[").removesuffix("]")
            species_rows.append(row)

    baseline_sigma = sigma_means["species_only"]
    variance_df = pd.DataFrame(
        [
            {
                "model": name,
                "species_sigma_mean": sigma,
                "sigma_reduction_fraction_vs_species_only": np.nan if baseline_sigma == 0 else 1.0 - (sigma / baseline_sigma),
            }
            for name, sigma in sigma_means.items()
        ]
    )

    comparison_df = pd.DataFrame(comparison_rows).merge(variance_df, on=["model", "species_sigma_mean"], how="left")
    effect_df = pd.DataFrame(effect_rows)
    species_df = pd.DataFrame(species_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    comparison_df.to_csv(args.output_dir / "block_comparison_summary.tsv", sep="\t", index=False)
    effect_df.to_csv(args.output_dir / "fixed_effect_comparison.tsv", sep="\t", index=False)
    species_df.to_csv(args.output_dir / "species_random_effects.tsv", sep="\t", index=False)
    variance_df.to_csv(args.output_dir / "species_variance_reduction.tsv", sep="\t", index=False)
    diagnostics_df.to_csv(args.output_dir / "model_diagnostics.tsv", sep="\t", index=False)
    (args.output_dir / "model_formulas.json").write_text(json.dumps(formulas, indent=2, sort_keys=True))
    (args.output_dir / "report.md").write_text(build_report_markdown(comparison_df, effect_df, diagnostics_df))

    run_info = {
        "dataset": str(args.dataset),
        "target_source_filter": args.target_source,
        "rows": int(len(analysis_df)),
        "positives": int(analysis_df["target"].sum()),
        "positive_rate": float(analysis_df["target"].mean()),
        "species": analysis_df["species"].value_counts().to_dict(),
        "ecology_features_raw": ECOLOGY_FEATURES,
        "molecular_features_raw": MOLECULAR_FEATURES,
        "molecular_features_plus_utr3_raw": MOLECULAR_FEATURES_PLUS_UTR3,
        "tree_sensitivity_features_raw": TREE_SENSITIVITY_FEATURES,
        "model_specs": [
            {
                "name": spec.name,
                "predictors": spec.predictors,
                "interaction_pairs": spec.interaction_pairs,
                "is_primary": spec.is_primary,
                "description": spec.description,
            }
            for spec in MODEL_SPECS
        ],
        "feature_map": feature_map,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "random_seed": args.random_seed,
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True))

    print(comparison_df.to_string(index=False))
    print(effect_df.to_string(index=False))


if __name__ == "__main__":
    main()
