#!/usr/bin/env python3
"""Bayesian LD-vs-epistasis block comparison.

This is a focused companion to `fit_hierarchical_block_comparison.py`. It uses
the curated final epistasis dataset and compares equivalent molecular blocks:
species-level LD summaries, species-level epistasis summaries, and numeric
selected-pair epistasis summaries suitable for a hierarchical Bernoulli model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT
    / "model_test"
    / "runs"
    / "intersegment_epistasis_final_direct553"
    / "datasets"
    / "sequence_level_final_dataset_direct553_epistasis_final.tsv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "model_test"
    / "runs"
    / "intersegment_epistasis_final_direct553"
    / "bayesian_ld_epistasis_comparison"
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
LD_SUMMARY_FEATURES = [
    "species_feat_mean_cramers_v_overall",
    "species_feat_prop_significant_overall",
]
EPISTASIS_SUMMARY_FEATURES = [
    "species_feat_epi_time_mean_score",
    "species_feat_epi_time_max_score",
    "species_feat_epi_div_mean_score",
    "species_feat_epi_div_max_score",
    "species_feat_epi_final_time_selected_pair_per_100k_site_pairs",
    "species_feat_epi_final_time_min_fdr_neglog10",
    "species_feat_epi_final_time_best_robust_positive_fraction_neglog10p",
    "species_feat_epi_final_div_min_fdr_neglog10",
]
EPISTASIS_PAIR_NUMERIC_FEATURES = [
    "epi_selected_time_applicable_pair_n",
    "epi_selected_time_aa_major_state_fraction",
    "epi_selected_time_aa_rare_state_fraction",
    "epi_selected_time_aa_problem_state_fraction",
    "epi_selected_time_nt_major_state_fraction",
    "epi_selected_time_nt_rare_state_fraction",
]
TREE_FEATURES = ["tree_discordance_score"]


@dataclass(frozen=True)
class BayesianSpec:
    name: str
    predictors: Tuple[str, ...]
    description: str
    interaction_pairs: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    is_primary: bool = True


MODEL_SPECS: Tuple[BayesianSpec, ...] = (
    BayesianSpec("species_only", tuple(), "Species random intercept only."),
    BayesianSpec("ecology_block", tuple(ECOLOGY_FEATURES), "Ecology and host-trait mechanisms."),
    BayesianSpec("ld_summary_block", tuple(LD_SUMMARY_FEATURES), "Species-level LD summary block."),
    BayesianSpec("epistasis_summary_block", tuple(EPISTASIS_SUMMARY_FEATURES), "Species-level time/div epistasis summary block."),
    BayesianSpec(
        "epistasis_pair_numeric_block",
        tuple(EPISTASIS_PAIR_NUMERIC_FEATURES),
        "Sequence-level numeric summaries over selected epistatic pair states.",
    ),
    BayesianSpec(
        "integrated_ld_summary",
        tuple(ECOLOGY_FEATURES + LD_SUMMARY_FEATURES),
        "Integrated ecology plus species-level LD summaries.",
    ),
    BayesianSpec(
        "integrated_epistasis_summary",
        tuple(ECOLOGY_FEATURES + EPISTASIS_SUMMARY_FEATURES),
        "Integrated ecology plus species-level epistasis summaries.",
    ),
    BayesianSpec(
        "integrated_epistasis_summary_pairnumeric",
        tuple(ECOLOGY_FEATURES + EPISTASIS_SUMMARY_FEATURES + EPISTASIS_PAIR_NUMERIC_FEATURES),
        "Integrated ecology plus species-level epistasis and selected-pair numeric summaries.",
    ),
    BayesianSpec(
        "integrated_overlap_ld_interaction",
        tuple(ECOLOGY_FEATURES + LD_SUMMARY_FEATURES),
        "Integrated LD model with overlap opportunity x LD burden interaction.",
        interaction_pairs=(("overlap_opportunity_local", "species_feat_mean_cramers_v_overall"),),
        is_primary=False,
    ),
    BayesianSpec(
        "integrated_overlap_epistasis_interaction",
        tuple(ECOLOGY_FEATURES + EPISTASIS_SUMMARY_FEATURES + EPISTASIS_PAIR_NUMERIC_FEATURES),
        "Integrated epistasis model with overlap opportunity x selected-pair epistasis burden interaction.",
        interaction_pairs=(("overlap_opportunity_local", "epi_selected_time_aa_rare_state_fraction"),),
        is_primary=False,
    ),
    BayesianSpec(
        "integrated_ld_tree_sensitivity",
        tuple(ECOLOGY_FEATURES + LD_SUMMARY_FEATURES + TREE_FEATURES),
        "Integrated LD model plus tree-discordance sensitivity.",
        is_primary=False,
    ),
    BayesianSpec(
        "integrated_epistasis_tree_sensitivity",
        tuple(ECOLOGY_FEATURES + EPISTASIS_SUMMARY_FEATURES + EPISTASIS_PAIR_NUMERIC_FEATURES + TREE_FEATURES),
        "Integrated epistasis model plus tree-discordance sensitivity.",
        is_primary=False,
    ),
)

COMMON_PRIORS = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1.5),
    "common": bmb.Prior("Normal", mu=0, sigma=1.0),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-source", choices=["all", "true_label", "sequence_true_label", "isolate_true_label"], default="true_label")
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--tune", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.97)
    parser.add_argument("--random-seed", type=int, default=20260605)
    parser.add_argument("--model", action="append", dest="models", default=[])
    parser.add_argument("--force", action="store_true", help="Refit models even when posterior.nc already exists.")
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
    return df.reset_index(drop=True)


def add_scaled_feature(frame: pd.DataFrame, raw: pd.Series, name: str) -> List[str]:
    created: List[str] = []
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().sum() == 0:
        return created
    missing = numeric.isna().astype(int)
    if 0 < missing.sum() < len(missing):
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


def prepare_analysis_frame(df: pd.DataFrame, specs: Sequence[BayesianSpec]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    frame = df[["species", "target"]].copy()
    frame["has_host_traits"] = pd.to_numeric(df.get("hosttrait_has_any_traits"), errors="coerce").fillna(0).astype(int)
    all_features = sorted({feature for spec in specs for feature in spec.predictors})
    feature_map: Dict[str, List[str]] = {}
    for feature in all_features:
        if feature not in df.columns:
            continue
        created = add_scaled_feature(frame, df[feature], feature)
        if created:
            feature_map[feature] = created
    return frame, feature_map


def build_formula(spec: BayesianSpec, feature_map: Mapping[str, List[str]]) -> str:
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


def sign_probability(idata: az.InferenceData, term: str) -> float:
    values = np.asarray(idata.posterior[term]).reshape(-1)
    return float((values > 0).mean())


def sigma_mean(idata: az.InferenceData) -> float:
    return float(np.asarray(idata.posterior["1|species_sigma"]).reshape(-1).mean())


def diagnostics(idata: az.InferenceData) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        loo = az.loo(idata, pointwise=False)
        out["loo_elpd"] = float(loo.elpd_loo)
        out["loo_p"] = float(loo.p_loo)
        out["loo_se"] = float(loo.se)
    except Exception:
        out["loo_elpd"] = np.nan
        out["loo_p"] = np.nan
        out["loo_se"] = np.nan
    try:
        waic = az.waic(idata)
        out["waic_elpd"] = float(waic.elpd_waic)
        out["waic_p"] = float(waic.p_waic)
        out["waic_se"] = float(waic.se)
    except Exception:
        out["waic_elpd"] = np.nan
        out["waic_p"] = np.nan
        out["waic_se"] = np.nan
    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None and "diverging" in sample_stats:
        out["n_divergences"] = int(np.asarray(sample_stats["diverging"]).sum())
    else:
        out["n_divergences"] = np.nan
    return out


def model_terms(spec: BayesianSpec, formula: str) -> List[str]:
    rhs = formula.split("~", 1)[1]
    terms = []
    for item in rhs.split("+"):
        item = item.strip()
        if not item or item.startswith("("):
            continue
        terms.append(item)
    return list(dict.fromkeys(["Intercept"] + terms))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    selected_specs = [spec for spec in MODEL_SPECS if not args.models or spec.name in set(args.models)]
    missing_specs = sorted(set(args.models) - {spec.name for spec in MODEL_SPECS})
    if missing_specs:
        raise ValueError(f"Unknown model spec(s): {missing_specs}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(args.dataset, args.target_source)
    analysis_df, feature_map = prepare_analysis_frame(raw, selected_specs)
    analysis_df.to_csv(args.output_dir / "analysis_dataset.tsv", sep="\t", index=False)
    raw[["species", "tip_name", "accession", "canonical_isolate", "target_source"]].to_csv(
        args.output_dir / "observation_manifest.tsv",
        sep="\t",
        index=True,
        index_label="row_index",
    )

    comparison_rows: List[Dict[str, object]] = []
    effect_rows: List[Dict[str, object]] = []
    formulas: Dict[str, str] = {}
    sigma_by_model: Dict[str, float] = {}
    for spec in selected_specs:
        formula = build_formula(spec, feature_map)
        formulas[spec.name] = formula
        model_dir = args.output_dir / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "formula.txt").write_text(formula + "\n")
        posterior_path = model_dir / "posterior.nc"
        if posterior_path.exists() and not args.force:
            idata = az.from_netcdf(posterior_path)
        else:
            model = bmb.Model(formula, analysis_df, family="bernoulli", priors=COMMON_PRIORS)
            idata = model.fit(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                random_seed=args.random_seed,
                target_accept=args.target_accept,
                idata_kwargs={"log_likelihood": True},
            )
            idata.to_netcdf(posterior_path)
        summary = az.summary(idata, var_names=list(idata.posterior.data_vars), round_to=4).reset_index().rename(
            columns={"index": "term"}
        )
        summary.to_csv(model_dir / "posterior_summary.tsv", sep="\t", index=False)

        diag = diagnostics(idata)
        sig = sigma_mean(idata)
        sigma_by_model[spec.name] = sig
        comparison_rows.append(
            {
                "model": spec.name,
                "description": spec.description,
                "formula": formula,
                "rows": int(len(analysis_df)),
                "positives": int(analysis_df["target"].sum()),
                "species_levels": int(analysis_df["species"].nunique()),
                "species_sigma_mean": sig,
                "is_primary_model": int(spec.is_primary),
                **diag,
            }
        )

        for term in model_terms(spec, formula):
            if term not in summary["term"].values:
                continue
            row = summary.loc[summary["term"] == term].iloc[0].to_dict()
            row["model"] = spec.name
            row["posterior_prob_positive"] = sign_probability(idata, term)
            effect_rows.append(row)

    baseline_sigma = sigma_by_model.get("species_only", np.nan)
    comparison = pd.DataFrame(comparison_rows)
    comparison["sigma_reduction_fraction_vs_species_only"] = (
        np.nan
        if not np.isfinite(baseline_sigma) or baseline_sigma == 0
        else 1.0 - comparison["species_sigma_mean"] / baseline_sigma
    )
    effects = pd.DataFrame(effect_rows)

    comparison.to_csv(args.output_dir / "block_comparison_summary.tsv", sep="\t", index=False)
    effects.to_csv(args.output_dir / "fixed_effect_comparison.tsv", sep="\t", index=False)
    (args.output_dir / "model_formulas.json").write_text(json.dumps(formulas, indent=2, sort_keys=True) + "\n")
    run_info = {
        "dataset": str(args.dataset),
        "target_source_filter": args.target_source,
        "rows": int(len(analysis_df)),
        "positives": int(analysis_df["target"].sum()),
        "positive_rate": float(analysis_df["target"].mean()),
        "draws": int(args.draws),
        "tune": int(args.tune),
        "chains": int(args.chains),
        "cores": int(args.cores),
        "target_accept": float(args.target_accept),
        "feature_map": feature_map,
        "model_specs": [
            {
                "name": spec.name,
                "predictors": list(spec.predictors),
                "interaction_pairs": [list(pair) for pair in spec.interaction_pairs],
                "description": spec.description,
                "is_primary": spec.is_primary,
            }
            for spec in selected_specs
        ],
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n")
    print(comparison.sort_values("loo_elpd", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
