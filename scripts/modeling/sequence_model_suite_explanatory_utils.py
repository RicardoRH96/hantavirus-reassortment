#!/usr/bin/env python3
"""Helpers for the mechanism-focused explanatory sequence model suite."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
import re
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ECOLOGY_SEED_COLUMNS = [
    "has_arha_host_match",
    "overlap_opportunity_local",
    "hosttrait_has_any_traits",
    "local_latitude",
    "local_longitude",
    "arha_prev_local",
]

ECOLOGY_PREFIX = "hosttrait_"
ECOLOGY_SPECIES_PREFIXES = (
    "species_feat_hostdiv_",
    "species_feat_overlapopp_",
)

LINEAGE_ECOLOGY_CORE_SEED_COLUMNS = [
    "overlap_opportunity_local",
    "arha_prev_local",
]

LINEAGE_ECOLOGY_CONTEXT_COLUMNS = [
    "local_latitude",
    "local_longitude",
    "year",
    "has_arha_host_match",
    "arha_sequence_match_count",
    "hosttrait_has_any_traits",
    "hosttrait_has_direct_any_traits",
    "hosttrait_fill_donor_count",
]

LINEAGE_ECOLOGY_CONTEXT_PREFIXES = (
    "species_feat_eco_",
)

LINEAGE_MOLECULAR_DIRECT_PREFIXES = (
    "discord_",
    "utrterm_",
)

ECOLOGY_EXCLUDE = {
    "hosttrait_fill_anchor_name",
    "hosttrait_fill_anchor_ott_id",
    "hosttrait_resolved_host_species",
    "hosttrait_combine_order",
    "hosttrait_combine_family",
    "hosttrait_combine_genus",
    "hosttrait_combine_species",
    "hosttrait_combine_iucn2020_binomial",
    "hosttrait_combine_phylacine_binomial",
    "hosttrait_combine_match_source",
    "hosttrait_combine_match_key",
    "hosttrait_combine_norm_species",
    "hosttrait_combine_norm_phylacine_binomial",
    "hosttrait_elton_MSW3_ID",
    "hosttrait_elton_Scientific",
    "hosttrait_elton_MSWFamilyLatin",
    "hosttrait_elton_Diet-Source",
    "hosttrait_elton_Diet-Certainty",
    "hosttrait_elton_ForStrat-Certainty",
    "hosttrait_elton_ForStrat-Comment",
    "hosttrait_elton_Activity-Source",
    "hosttrait_elton_Activity-Certainty",
    "hosttrait_elton_BodyMass-Source",
    "hosttrait_elton_match_source",
    "hosttrait_elton_match_key",
    "hosttrait_source",
    "hosttrait_fill_method",
    "hosttrait_resolution_stage",
    "hosttrait_combine_teat_number_n",
    "species_feat_hostdiv_n_labels_missing",
}

MOLECULAR_SEED_COLUMNS = [
    "segment_completeness",
    "tree_discordance_score",
    "has_S",
    "has_M",
    "has_L",
]

MOLECULAR_PREFIX = "ld_site_"
MOLECULAR_SPECIES_PREFIXES = (
    "species_feat_spr_",
    "species_feat_mean_spr_",
    "species_feat_max_spr_",
    "species_feat_min_spr_",
    "species_feat_std_spr_",
    "species_feat_utr5_",
    "species_feat_utr3_",
    "species_feat_mean_cramers_v_",
    "species_feat_median_cramers_v_",
    "species_feat_mean_nmi_",
    "species_feat_prop_significant_",
    "species_feat_n_site_pairs_",
    "species_feat_n_significant_",
    "species_feat_cramers_v_",
)
MOLECULAR_SPECIES_COLUMNS = [
    "species_feat_n_isolates",
    "species_feat_n_spr_pairs",
]
MOLECULAR_EXCLUDE = {
    "species_feat_reassortment_proxy_mean_spr",
    "species_feat_reassortment_proxy_mean_spr_per_isolate",
    "species_feat_target_proxy_available",
}

LD_MAIN_SITE_PATTERN = re.compile(r"^ld_site_[SML]_\d+$")

CONTROL_SEED_COLUMNS = [
    "year",
    "segment_completeness",
    "has_S",
    "has_M",
    "has_L",
    "arha_sequence_match_count",
    "has_arha_sequence_match",
    "has_arha_pathogen_match",
    "has_arha_host_match",
    "has_arha_host_overlap",
    "has_host_traits",
    "hosttrait_has_any_traits",
    "hosttrait_has_direct_any_traits",
    "hosttrait_fill_donor_count",
    "hosttrait_resolution_score",
]
CONTROL_SPECIES_COLUMNS = [
    "species_feat_matched_isolates",
    "species_feat_eco_n_records",
    "species_feat_eco_n_unique_accessions",
    "species_feat_eco_country_richness",
    "species_feat_eco_geo_richness",
    "species_feat_eco_year_span",
    "species_feat_isolate_matrix_exists",
    "species_feat_gate_pass",
    "species_feat_all_refined_trees",
    "species_feat_artifact_pass",
    "species_feat_overall_pass",
    "species_feat_arg_status_present",
    "species_feat_run_arg_status_present",
]

CONTROL_EXCLUDE = {
    # Constant or all-missing in the final defended dataset.
    "segment_completeness",
    "has_S",
    "has_M",
    "has_L",
    "has_arha_host_overlap",
    "hosttrait_resolution_score",
    "species_feat_isolate_matrix_exists",
    "species_feat_gate_pass",
    "species_feat_all_refined_trees",
    "species_feat_artifact_pass",
    "species_feat_overall_pass",
    # Redundant bookkeeping/provenance flags.
    "has_host_traits",
    "has_arha_sequence_match",
    "has_arha_pathogen_match",
    "has_arha_host_match",
    "hosttrait_fill_donor_count",
    "hosttrait_has_direct_any_traits",
    "species_feat_arg_status_present",
    "species_feat_eco_n_unique_accessions",
    "species_feat_run_arg_status_present",
}


@dataclass(frozen=True)
class FeatureSet:
    name: str
    description: str
    block_names: Tuple[str, ...]


FEATURE_SETS: Tuple[FeatureSet, ...] = (
    FeatureSet(
        name="species_only",
        description="Taxon-only baseline using the held-out species label.",
        block_names=("species",),
    ),
    FeatureSet(
        name="ecology_only",
        description="Host ecology and host-overlap mechanisms plus control covariates, without species identity.",
        block_names=("ecology", "control"),
    ),
    FeatureSet(
        name="molecular_only",
        description="Sequence-derived molecular signals plus control covariates, without ecology or species identity.",
        block_names=("molecular", "control"),
    ),
    FeatureSet(
        name="ecology_molecular",
        description="Combined ecology, molecular, and control signals without species identity.",
        block_names=("ecology", "molecular", "control"),
    ),
    FeatureSet(
        name="integrated_taxon_mechanism",
        description="Species identity plus ecology, molecular, and control mechanism signals.",
        block_names=("species", "ecology", "molecular", "control"),
    ),
    FeatureSet(
        name="ecology_molecular_lineage_core",
        description="Lineage-focused local ecology plus direct per-isolate molecular signals, excluding geography/provenance and species-level summaries.",
        block_names=("ecology_core", "molecular_core"),
    ),
    FeatureSet(
        name="ecology_molecular_lineage_plus_context",
        description="Lineage-focused local ecology and direct molecular signals plus geography, time, provenance, and sampling context.",
        block_names=("ecology_core", "molecular_core", "context"),
    ),
    FeatureSet(
        name="ecology_molecular_lineage_plus_species_summaries",
        description="Lineage-focused core with context plus species-level ecology and molecular summary blocks for ablation.",
        block_names=("ecology_core", "molecular_core", "context", "ecology_species", "molecular_species"),
    ),
)


def unique_in_order(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def infer_column_type(series: pd.Series) -> str | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    numeric_candidate = pd.to_numeric(series, errors="coerce")
    numeric_fraction = float(numeric_candidate.notna().sum()) / float(non_null.shape[0])
    if pd.api.types.is_numeric_dtype(series) or numeric_fraction >= 0.95:
        return "numeric"
    return "categorical"


def classify_columns(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    numeric: List[str] = []
    categorical: List[str] = []
    for col in unique_in_order(columns):
        if col not in df.columns:
            continue
        column_type = infer_column_type(df[col])
        if column_type is None:
            continue
        if df[col].dropna().nunique() <= 1:
            continue
        if column_type == "numeric":
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def detect_feature_blocks(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    ecology_candidates = ECOLOGY_SEED_COLUMNS + [
        col for col in df.columns if col.startswith(ECOLOGY_PREFIX) and col not in ECOLOGY_EXCLUDE
    ] + [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in ECOLOGY_SPECIES_PREFIXES)
        and col not in ECOLOGY_EXCLUDE
    ]
    molecular_candidates = MOLECULAR_SEED_COLUMNS + [
        col for col in df.columns if col.startswith(MOLECULAR_PREFIX)
    ] + [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in MOLECULAR_SPECIES_PREFIXES) and col not in MOLECULAR_EXCLUDE
    ] + [
        col for col in MOLECULAR_SPECIES_COLUMNS if col in df.columns
    ]
    control_candidates = [
        col for col in CONTROL_SEED_COLUMNS if col not in CONTROL_EXCLUDE
    ] + [
        col for col in CONTROL_SPECIES_COLUMNS
        if col in df.columns and col not in CONTROL_EXCLUDE
    ]
    ecology_core_candidates = LINEAGE_ECOLOGY_CORE_SEED_COLUMNS + [
        col
        for col in df.columns
        if (
            (
                col.startswith("hosttrait_combine_")
                or col.startswith("hosttrait_elton_")
            )
            and col not in ECOLOGY_EXCLUDE
            and not col.startswith("hosttrait_fill_")
            and not col.startswith("hosttrait_has_")
            and "match_source" not in col
            and "match_key" not in col
            and "resolved_host_species" not in col
            and "resolution_" not in col
            and not col.endswith("_source")
        )
    ]
    context_candidates = LINEAGE_ECOLOGY_CONTEXT_COLUMNS + [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in LINEAGE_ECOLOGY_CONTEXT_PREFIXES)
    ] + [
        col
        for col in CONTROL_SEED_COLUMNS + CONTROL_SPECIES_COLUMNS
        if col in df.columns and col not in {"hosttrait_has_any_traits", "hosttrait_has_direct_any_traits", "hosttrait_fill_donor_count", "year"}
    ]
    ecology_species_candidates = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in ECOLOGY_SPECIES_PREFIXES)
        and col not in ECOLOGY_EXCLUDE
    ]
    molecular_core_candidates = [
        "tree_discordance_score",
    ] + [
        col for col in df.columns if col.startswith(MOLECULAR_PREFIX)
    ] + [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in LINEAGE_MOLECULAR_DIRECT_PREFIXES)
    ]
    molecular_species_candidates = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in MOLECULAR_SPECIES_PREFIXES)
        and col not in MOLECULAR_EXCLUDE
    ] + [
        col for col in MOLECULAR_SPECIES_COLUMNS if col in df.columns
    ]
    species_candidates = ["species"]

    ecology_numeric, ecology_categorical = classify_columns(df, ecology_candidates)
    molecular_numeric, molecular_categorical = classify_columns(df, molecular_candidates)
    control_numeric, control_categorical = classify_columns(df, control_candidates)
    ecology_core_numeric, ecology_core_categorical = classify_columns(df, ecology_core_candidates)
    context_numeric, context_categorical = classify_columns(df, context_candidates)
    ecology_species_numeric, ecology_species_categorical = classify_columns(df, ecology_species_candidates)
    molecular_core_numeric, molecular_core_categorical = classify_columns(df, molecular_core_candidates)
    molecular_species_numeric, molecular_species_categorical = classify_columns(df, molecular_species_candidates)
    species_numeric, species_categorical = classify_columns(df, species_candidates)

    return {
        "species": {
            "numeric": species_numeric,
            "categorical": species_categorical,
        },
        "ecology": {
            "numeric": ecology_numeric,
            "categorical": ecology_categorical,
        },
        "molecular": {
            "numeric": molecular_numeric,
            "categorical": molecular_categorical,
        },
        "control": {
            "numeric": control_numeric,
            "categorical": control_categorical,
        },
        "ecology_core": {
            "numeric": ecology_core_numeric,
            "categorical": ecology_core_categorical,
        },
        "context": {
            "numeric": context_numeric,
            "categorical": context_categorical,
        },
        "ecology_species": {
            "numeric": ecology_species_numeric,
            "categorical": ecology_species_categorical,
        },
        "molecular_core": {
            "numeric": molecular_core_numeric,
            "categorical": molecular_core_categorical,
        },
        "molecular_species": {
            "numeric": molecular_species_numeric,
            "categorical": molecular_species_categorical,
        },
    }


def build_feature_frame(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    blocks: Mapping[str, Mapping[str, Sequence[str]]],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    numeric: List[str] = []
    categorical: List[str] = []
    for block_name in feature_set.block_names:
        block = blocks[block_name]
        numeric.extend(block["numeric"])
        categorical.extend(block["categorical"])
    numeric = unique_in_order(numeric)
    categorical = [col for col in unique_in_order(categorical) if col not in numeric]
    frame = df[numeric + categorical].copy()
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.loc[~np.isfinite(frame[col]), col] = np.nan
    for col in categorical:
        frame[col] = frame[col].astype(object)
        frame.loc[frame[col].isna(), col] = np.nan
        frame[col] = frame[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return frame, numeric, categorical


def apply_ld_encoding(
    blocks: Mapping[str, Mapping[str, Sequence[str]]],
    encoding: str = "full",
) -> Dict[str, Dict[str, List[str]]]:
    """Return a copy of feature blocks with the requested LD encoding.

    `full` keeps single-site LD genotype terms and LD pair/interactions.
    `interaction_only` removes the single-site LD genotype terms while
    retaining the pair/interactions and all species-level LD summaries.
    """
    if encoding == "full":
        return {
            name: {
                "numeric": list(block["numeric"]),
                "categorical": list(block["categorical"]),
            }
            for name, block in blocks.items()
        }
    if encoding != "interaction_only":
        raise ValueError(f"Unsupported LD encoding: {encoding}")

    filtered: Dict[str, Dict[str, List[str]]] = {
        name: {
            "numeric": list(block["numeric"]),
            "categorical": list(block["categorical"]),
        }
        for name, block in blocks.items()
    }
    for block_name in ("molecular", "molecular_core"):
        molecular = filtered.get(block_name)
        if molecular is None:
            continue
        drop = {
            col
            for col in molecular["numeric"] + molecular["categorical"]
            if LD_MAIN_SITE_PATTERN.match(col)
        }
        molecular["numeric"] = [col for col in molecular["numeric"] if col not in drop]
        molecular["categorical"] = [col for col in molecular["categorical"] if col not in drop]
    return filtered


def build_feature_catalog(
    feature_sets: Sequence[FeatureSet],
    blocks: Mapping[str, Mapping[str, Sequence[str]]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feature_set in feature_sets:
        numeric: List[str] = []
        categorical: List[str] = []
        for block_name in feature_set.block_names:
            block = blocks[block_name]
            numeric.extend(block["numeric"])
            categorical.extend(block["categorical"])
        numeric = unique_in_order(numeric)
        categorical = [col for col in unique_in_order(categorical) if col not in numeric]
        rows.append(
            {
                "feature_set": feature_set.name,
                "description": feature_set.description,
                "blocks": "|".join(feature_set.block_names),
                "numeric_features": "|".join(numeric),
                "categorical_features": "|".join(categorical),
                "n_numeric": len(numeric),
                "n_categorical": len(categorical),
            }
        )
    return pd.DataFrame(rows)


def try_import_bayesian_tooling() -> Dict[str, str]:
    status: MutableMapping[str, str] = {}
    for mod in ["pymc", "bambi", "arviz", "statsmodels"]:
        try:
            __import__(mod)
        except Exception as exc:
            status[mod] = f"missing: {type(exc).__name__}: {exc}"
        else:
            status[mod] = "available"
    return dict(status)


def build_fixed_effect_preprocessor(numeric: Sequence[str], categorical: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric),
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=1,
                                drop="first",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                list(categorical),
            ),
        ],
    )


def fit_fixed_effect_logistic(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric: Sequence[str],
    categorical: Sequence[str],
    kind: str,
    seed: int,
) -> Pipeline:
    preprocessor = build_fixed_effect_preprocessor(numeric, categorical)
    if kind == "logistic_l2":
        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            penalty="l2",
            solver="liblinear",
            random_state=seed,
            max_iter=4000,
        )
    elif kind == "logistic_elasticnet":
        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            random_state=seed,
            max_iter=6000,
        )
    else:
        raise ValueError(f"Unsupported fixed-effect kind: {kind}")
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipeline.fit(X, y)
    return pipeline


def _species_effect_frame(pipeline: Pipeline) -> pd.DataFrame:
    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    coef = pipeline.named_steps["model"].coef_.reshape(-1)
    out = pd.DataFrame({"feature": feature_names, "coefficient": coef})
    mask = out["feature"].str.startswith("cat__species_")
    species = out.loc[mask].copy()
    if species.empty:
        return pd.DataFrame(columns=["species", "coefficient", "abs_coefficient"])
    species["species"] = species["feature"].str.removeprefix("cat__species_")
    species["abs_coefficient"] = species["coefficient"].abs()
    return species[["species", "coefficient", "abs_coefficient"]].sort_values("species").reset_index(drop=True)


def summarize_species_effects(
    df: pd.DataFrame,
    y: np.ndarray,
    blocks: Mapping[str, Mapping[str, Sequence[str]]],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summaries: List[Dict[str, object]] = []
    coefficients: List[pd.DataFrame] = []
    feature_set_lookup = {feature_set.name: feature_set for feature_set in FEATURE_SETS}
    target_sets = {
        "species_only": feature_set_lookup["species_only"],
        "integrated_taxon_mechanism": feature_set_lookup["integrated_taxon_mechanism"],
    }
    for kind in ["logistic_l2", "logistic_elasticnet"]:
        fit_results: Dict[str, pd.DataFrame] = {}
        for feature_set_name, feature_set in target_sets.items():
            X, numeric, categorical = build_feature_frame(df, feature_set, blocks)
            pipeline = fit_fixed_effect_logistic(X, y, numeric, categorical, kind=kind, seed=seed)
            species = _species_effect_frame(pipeline)
            species.insert(0, "model", kind)
            species.insert(0, "feature_set", feature_set_name)
            coefficients.append(species)
            fit_results[feature_set_name] = species

        base = fit_results["species_only"].set_index("species")
        integrated = fit_results["integrated_taxon_mechanism"].set_index("species")
        joined = base.join(
            integrated,
            how="outer",
            lsuffix="_species_only",
            rsuffix="_integrated",
        ).fillna(0.0)
        base_norm = float(np.linalg.norm(joined["coefficient_species_only"].to_numpy(), ord=2))
        integrated_norm = float(np.linalg.norm(joined["coefficient_integrated"].to_numpy(), ord=2))
        base_mean_abs = float(joined["abs_coefficient_species_only"].mean())
        integrated_mean_abs = float(joined["abs_coefficient_integrated"].mean())
        base_max_abs = float(joined["abs_coefficient_species_only"].max())
        integrated_max_abs = float(joined["abs_coefficient_integrated"].max())
        summaries.append(
            {
                "model": kind,
                "fallback_method": "regularized_fixed_effect_logistic",
                "regularization_strength_c": 0.1,
                "comparison": "species_only_vs_integrated_taxon_mechanism",
                "species_levels_compared": int(joined.shape[0]),
                "species_only_l2_norm": base_norm,
                "integrated_l2_norm": integrated_norm,
                "l2_reduction_fraction": np.nan if base_norm == 0.0 else 1.0 - (integrated_norm / base_norm),
                "species_only_mean_abs": base_mean_abs,
                "integrated_mean_abs": integrated_mean_abs,
                "mean_abs_reduction_fraction": np.nan
                if base_mean_abs == 0.0
                else 1.0 - (integrated_mean_abs / base_mean_abs),
                "species_only_max_abs": base_max_abs,
                "integrated_max_abs": integrated_max_abs,
                "max_abs_reduction_fraction": np.nan
                if base_max_abs == 0.0
                else 1.0 - (integrated_max_abs / base_max_abs),
            }
        )

    coefficient_df = pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    return summary_df, coefficient_df
