#!/usr/bin/env python3
"""Ablation runner for hierarchical mechanistic models.

Explores targeted enhancements to the current integrated+UTR hierarchy
without overwriting the canonical manuscript run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT
    / "data"
    / "processed"
    / "final_sequence_dataset.tsv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "hierarchical_block_ablation"
)

ROW_VARYING_ECOLOGY = [
    "hosttrait_combine_adult_mass_g",
    "hosttrait_combine_litter_size_n",
    "hosttrait_combine_gestation_length_d",
    "hosttrait_combine_dispersal_km",
    "overlap_opportunity_local",
    "arha_prev_local",
]
SPECIES_LEVEL_ECOLOGY = [
    "species_feat_hostdiv_faith_pd",
]
BASE_LD_COARSE = [
    "species_feat_mean_cramers_v_overall",
    "species_feat_prop_significant_overall",
]
LD_REFINED = [
    "species_feat_mean_cramers_v_SM",
    "species_feat_mean_cramers_v_SL",
    "species_feat_mean_cramers_v_ML",
    "species_feat_prop_significant_SM",
    "species_feat_prop_significant_SL",
    "species_feat_prop_significant_ML",
    "species_feat_cramers_v_variance",
    "species_feat_cramers_v_skewness",
]
UTR_BASE = [
    "species_feat_utr5_S_structdist_mean",
    "species_feat_utr5_M_structdist_mean",
    "species_feat_utr5_L_structdist_mean",
    "species_feat_utr3_S_structdist_mean",
    "species_feat_utr3_M_structdist_mean",
    "species_feat_utr3_L_structdist_mean",
]
UTR_EXPANDED = [
    "species_feat_utr5_S_length_mean",
    "species_feat_utr5_S_seqdist_mean",
    "species_feat_utr5_S_structdist_mean",
    "species_feat_utr5_M_length_mean",
    "species_feat_utr5_M_seqdist_mean",
    "species_feat_utr5_M_structdist_mean",
    "species_feat_utr5_L_length_mean",
    "species_feat_utr5_L_seqdist_mean",
    "species_feat_utr5_L_structdist_mean",
    "species_feat_utr3_S_length_mean",
    "species_feat_utr3_S_seqdist_mean",
    "species_feat_utr3_S_structdist_mean",
    "species_feat_utr3_M_length_mean",
    "species_feat_utr3_M_seqdist_mean",
    "species_feat_utr3_M_structdist_mean",
    "species_feat_utr3_L_length_mean",
    "species_feat_utr3_L_seqdist_mean",
    "species_feat_utr3_L_structdist_mean",
]
TREE_FEATURE = "tree_discordance_score"
CONTROL_FLAG = "has_host_traits"
MISSING_FLAG_SUFFIX = "_missing"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    enhancements: Tuple[str, ...]


VARIANTS: List[VariantSpec] = [
    VariantSpec("baseline_integrated_plus_utr3", "Current additive integrated+UTR baseline.", tuple()),
    VariantSpec("enh01_within_between", "Within/between decomposition for row-varying ecology.", ("within_between",)),
    VariantSpec("enh02_interactions_primary", "Promote overlap x compatibility interactions to the primary model.", ("interactions_primary",)),
    VariantSpec("enh03_drop_controls", "Remove control and missingness terms from the primary hierarchy.", ("drop_controls",)),
    VariantSpec("enh04_expand_utr", "Expand UTR biology beyond structural means.", ("expand_utr",)),
    VariantSpec("enh05_refine_ld", "Replace coarse overall LD with segment-pair LD summaries.", ("refine_ld",)),
    VariantSpec(
        "combo_wb_dropcontrols",
        "Within/between ecology plus control stripping.",
        ("within_between", "drop_controls"),
    ),
    VariantSpec(
        "combo_wb_expandutr",
        "Within/between ecology plus expanded UTR module.",
        ("within_between", "expand_utr"),
    ),
    VariantSpec(
        "combo_wb_refineld",
        "Within/between ecology plus refined LD module.",
        ("within_between", "refine_ld"),
    ),
    VariantSpec(
        "combo_interactions_biology",
        "Interactions as primary plus biology-only cleanup and richer molecular summaries.",
        ("interactions_primary", "drop_controls", "expand_utr", "refine_ld"),
    ),
    VariantSpec(
        "all_enhancements",
        "All mechanistic enhancements combined.",
        ("within_between", "interactions_primary", "drop_controls", "expand_utr", "refine_ld"),
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
    parser.add_argument("--target-source", choices=["true_label"], default="true_label")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--variant", action="append", dest="variants", default=[])
    return parser.parse_args(argv)


def load_dataset(path: Path, target_source: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["is_reassortant_final"].notna()].copy()
    if target_source == "true_label":
        df = df[df["target_source"].isin(["sequence_true_label", "isolate_true_label"])].copy()
    df["target"] = pd.to_numeric(df["is_reassortant_final"], errors="coerce")
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)
    df["species"] = df["species"].astype(str)
    return df.reset_index(drop=True)


def zscore(series: pd.Series) -> pd.Series:
    filled = pd.to_numeric(series, errors="coerce")
    std = float(filled.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (filled - filled.mean()) / std


def add_standard_feature(frame: pd.DataFrame, raw: pd.Series, name: str, include_missing: bool) -> List[str]:
    created: List[str] = []
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().sum() == 0:
        return created
    missing = numeric.isna().astype(int)
    if include_missing and 0 < int(missing.sum()) < len(missing):
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


def add_within_between_feature(frame: pd.DataFrame, df: pd.DataFrame, name: str, include_missing: bool) -> List[str]:
    created: List[str] = []
    numeric = pd.to_numeric(df[name], errors="coerce")
    if numeric.notna().sum() == 0:
        return created
    missing = numeric.isna().astype(int)
    if include_missing and 0 < int(missing.sum()) < len(missing):
        miss_name = f"{name}_missing"
        frame[miss_name] = missing
        created.append(miss_name)
    filled = numeric.fillna(numeric.median())
    species_mean = filled.groupby(df["species"]).transform("mean")
    within = filled - species_mean
    between = species_mean
    within_std = float(within.std(ddof=0))
    between_std = float(between.std(ddof=0))
    if np.isfinite(within_std) and within_std > 0:
        w_name = f"{name}_within_z"
        frame[w_name] = (within - within.mean()) / within_std
        created.insert(0, w_name)
    if np.isfinite(between_std) and between_std > 0:
        b_name = f"{name}_between_z"
        frame[b_name] = (between - between.mean()) / between_std
        insert_at = 1 if created and created[0].endswith("_within_z") else 0
        created.insert(insert_at, b_name)
    return created


def select_utr_features(expanded: bool) -> List[str]:
    return UTR_EXPANDED if expanded else UTR_BASE


def select_ld_features(refined: bool) -> List[str]:
    return LD_REFINED if refined else BASE_LD_COARSE


def build_variant_frame(df: pd.DataFrame, variant: VariantSpec) -> tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    enhancements = set(variant.enhancements)
    include_missing = "drop_controls" not in enhancements
    frame = df[["species", "target"]].copy()
    if "drop_controls" not in enhancements:
        frame[CONTROL_FLAG] = pd.to_numeric(df.get("hosttrait_has_any_traits"), errors="coerce").fillna(0).astype(int)

    feature_map: Dict[str, List[str]] = {}
    row_ecology_terms: List[str] = []
    for feature in ROW_VARYING_ECOLOGY:
        if feature not in df.columns:
            continue
        if "within_between" in enhancements:
            created = add_within_between_feature(frame, df, feature, include_missing=include_missing)
        else:
            created = add_standard_feature(frame, df[feature], feature, include_missing=include_missing)
        if created:
            feature_map[feature] = created
            row_ecology_terms.extend([c for c in created if c.endswith("_z") and not c.endswith("_missing")])

    for feature in SPECIES_LEVEL_ECOLOGY + select_ld_features("refine_ld" in enhancements) + select_utr_features("expand_utr" in enhancements):
        if feature not in df.columns:
            continue
        created = add_standard_feature(frame, df[feature], feature, include_missing=False)
        if created:
            feature_map[feature] = created

    return frame, feature_map, row_ecology_terms


def build_formula(variant: VariantSpec, feature_map: Dict[str, List[str]]) -> str:
    enhancements = set(variant.enhancements)
    predictors: List[str] = []
    if "drop_controls" not in enhancements:
        predictors.append(CONTROL_FLAG)

    for feature in ROW_VARYING_ECOLOGY:
        predictors.extend(feature_map.get(feature, []))
    for feature in SPECIES_LEVEL_ECOLOGY:
        predictors.extend(feature_map.get(feature, []))
    for feature in select_ld_features("refine_ld" in enhancements):
        predictors.extend(feature_map.get(feature, []))
    for feature in select_utr_features("expand_utr" in enhancements):
        predictors.extend(feature_map.get(feature, []))

    if "interactions_primary" in enhancements:
        overlap_term = "overlap_opportunity_local_within_z" if "within_between" in enhancements else "overlap_opportunity_local_z"
        for partner in [
            "species_feat_mean_cramers_v_ML_z" if "refine_ld" in enhancements else "species_feat_mean_cramers_v_overall_z",
            "species_feat_utr5_L_structdist_mean_z",
            "species_feat_utr3_M_structdist_mean_z",
        ]:
            if overlap_term in predictors and partner in predictors:
                predictors.append(f"{overlap_term}:{partner}")

    predictors = list(dict.fromkeys(predictors))
    rhs = " + ".join(predictors + ["(1|species)"])
    return f"target ~ {rhs}"


def compute_diagnostics(idata: az.InferenceData, analysis_df: pd.DataFrame) -> tuple[Dict[str, float], pd.DataFrame]:
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
    except Exception:
        diagnostics["loo_elpd"] = np.nan
        diagnostics["loo_p"] = np.nan
        diagnostics["loo_se"] = np.nan
        diagnostics["pareto_k_max"] = np.nan
        diagnostics["pareto_k_n_gt_0_7"] = np.nan
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
    diagnostics["n_divergences"] = (
        int(np.asarray(sample_stats["diverging"]).sum())
        if sample_stats is not None and "diverging" in sample_stats
        else np.nan
    )
    return diagnostics, pareto_df


def summarize_ppc(model: bmb.Model, idata: az.InferenceData, analysis_df: pd.DataFrame) -> pd.DataFrame:
    pred_idata = model.predict(idata, kind="response_params", inplace=False)
    p_draws = np.asarray(pred_idata.posterior["p"])
    draw_axis = p_draws.reshape(-1, p_draws.shape[-1])
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
    return pd.DataFrame(species_rows).sort_values("species").reset_index(drop=True)


def extract_sign_probability(idata: az.InferenceData, term: str) -> float:
    values = np.asarray(idata.posterior[term]).reshape(-1)
    return float((values > 0).mean())


def fit_species_aggregate_model(
    variant_dir: Path,
    raw_df: pd.DataFrame,
    variant: VariantSpec,
    feature_map: Dict[str, List[str]],
    analysis_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, float]:
    # Species-level robustness companion for between-species biology only.
    aggregate_rows = []
    for species, group in raw_df.groupby("species", dropna=False):
        row = {"species": species, "successes": int(group["target"].sum()), "trials": int(len(group))}
        for col in analysis_df.columns:
            if col in {"species", "target"}:
                continue
            if col.endswith("_within_z"):
                continue
            row[col] = analysis_df.loc[group.index[0], col]
        aggregate_rows.append(row)
    species_df = pd.DataFrame(aggregate_rows)
    candidate_terms = [c for c in species_df.columns if c not in {"species", "successes", "trials"}]
    # Keep only non-constant species-level predictors.
    terms = [c for c in candidate_terms if species_df[c].nunique(dropna=False) > 1]
    if not terms:
        return {"species_agg_loo_elpd": np.nan, "species_agg_pareto_k_max": np.nan}
    formula = "p(successes, trials) ~ " + " + ".join(terms)
    model = bmb.Model(formula, species_df, family="binomial", priors=COMMON_PRIORS)
    idata = model.fit(
        draws=max(500, args.draws // 2),
        tune=max(500, args.tune // 2),
        chains=args.chains,
        cores=args.cores,
        random_seed=args.random_seed,
        target_accept=args.target_accept,
        idata_kwargs={"log_likelihood": True},
    )
    species_dir = variant_dir / "species_aggregate_check"
    species_dir.mkdir(parents=True, exist_ok=True)
    az.summary(idata, var_names=list(idata.posterior.data_vars), round_to=4).reset_index().rename(
        columns={"index": "term"}
    ).to_csv(species_dir / "posterior_summary.tsv", sep="\t", index=False)
    try:
        loo = az.loo(idata)
        out = {"species_agg_loo_elpd": float(loo.elpd_loo), "species_agg_pareto_k_max": float(np.nanmax(np.asarray(loo.pareto_k)))}
    except Exception:
        out = {"species_agg_loo_elpd": np.nan, "species_agg_pareto_k_max": np.nan}
    (species_dir / "formula.txt").write_text(formula + "\n")
    species_df.to_csv(species_dir / "analysis_dataset.tsv", sep="\t", index=False)
    return out


def selected_variants(names: Iterable[str]) -> List[VariantSpec]:
    if not names:
        return VARIANTS
    lookup = {variant.name: variant for variant in VARIANTS}
    return [lookup[name] for name in names if name in lookup]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(args.dataset, args.target_source)

    summary_rows: List[Dict[str, object]] = []
    coefficient_rows: List[Dict[str, object]] = []
    feature_map_dump: Dict[str, Dict[str, List[str]]] = {}

    for variant in selected_variants(args.variants):
        analysis_df, feature_map, _row_terms = build_variant_frame(raw, variant)
        formula = build_formula(variant, feature_map)
        variant_dir = args.output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        analysis_df.to_csv(variant_dir / "analysis_dataset.tsv", sep="\t", index=False)
        (variant_dir / "formula.txt").write_text(formula + "\n")

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
        idata.to_netcdf(variant_dir / "posterior.nc")

        diagnostics, pareto_df = compute_diagnostics(idata, analysis_df)
        if not pareto_df.empty:
            pareto_df.to_csv(variant_dir / "pareto_k_by_observation.tsv", sep="\t", index=False)
        ppc_df = summarize_ppc(model, idata, analysis_df)
        ppc_df.to_csv(variant_dir / "ppc_by_species.tsv", sep="\t", index=False)
        summary = az.summary(idata, var_names=list(idata.posterior.data_vars), round_to=4).reset_index().rename(columns={"index": "term"})
        summary.to_csv(variant_dir / "posterior_summary.tsv", sep="\t", index=False)

        sigma_mean = float(np.asarray(idata.posterior["1|species_sigma"]).reshape(-1).mean())
        species_check = fit_species_aggregate_model(variant_dir, raw, variant, feature_map, analysis_df, args)
        summary_rows.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "enhancements": "|".join(variant.enhancements),
                "formula": formula,
                "rows": int(len(analysis_df)),
                "species_levels": int(analysis_df["species"].nunique()),
                "species_sigma_mean": sigma_mean,
                **diagnostics,
                **species_check,
            }
        )

        fixed_terms = [term for term in summary["term"] if not term.startswith("1|species[")]
        for term in fixed_terms:
            row = summary.loc[summary["term"] == term].iloc[0].to_dict()
            row["variant"] = variant.name
            row["posterior_prob_positive"] = extract_sign_probability(idata, term)
            coefficient_rows.append(row)
        feature_map_dump[variant.name] = feature_map

    summary_df = pd.DataFrame(summary_rows).sort_values("loo_elpd", ascending=False).reset_index(drop=True)
    coef_df = pd.DataFrame(coefficient_rows)
    summary_df.to_csv(args.output_dir / "variant_summary.tsv", sep="\t", index=False)
    coef_df.to_csv(args.output_dir / "variant_fixed_effects.tsv", sep="\t", index=False)
    (args.output_dir / "feature_map_by_variant.json").write_text(json.dumps(feature_map_dump, indent=2, sort_keys=True))
    (args.output_dir / "run_info.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "draws": args.draws,
                "tune": args.tune,
                "chains": args.chains,
                "cores": args.cores,
                "target_accept": args.target_accept,
                "variants": [variant.name for variant in selected_variants(args.variants)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
