#!/usr/bin/env python3
"""Build the final epistasis modeling dataset.

"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISTASIS_ROOT = REPO_ROOT / "model_test" / "runs" / "intersegment_epistasis_fullscan_direct553"
DEFAULT_INPUT_DATASET = (
    DEFAULT_EPISTASIS_ROOT
    / "datasets"
    / "sequence_level_final_dataset_direct553_plus_selected_epistasis_interactions.tsv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "model_test" / "runs" / "intersegment_epistasis_final_direct553"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate final epistasis modeling dataset and feature manifests.")
    parser.add_argument("--epistasis-root", type=Path, default=DEFAULT_EPISTASIS_ROOT)
    parser.add_argument("--input-dataset", type=Path, default=DEFAULT_INPUT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fdr-alpha", type=float, default=0.01)
    parser.add_argument("--analysis-fdr-threshold", type=float, default=0.20)
    parser.add_argument("--robust-p-threshold", type=float, default=0.10)
    parser.add_argument("--rare-state-frequency-threshold", type=float, default=0.10)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


SELECTED_PAIR_SUMMARY_SUFFIXES = [
    "applicable_pair_n",
    "aa_major_state_n",
    "aa_rare_state_n",
    "aa_problem_state_n",
    "aa_major_state_fraction",
    "aa_rare_state_fraction",
    "aa_problem_state_fraction",
    "nt_major_state_n",
    "nt_rare_state_n",
    "nt_problem_state_n",
    "nt_major_state_fraction",
    "nt_rare_state_fraction",
    "nt_problem_state_fraction",
]


def read_table(path: Path, allow_empty: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path, sep="\t", low_memory=False)
    except pd.errors.EmptyDataError:
        if allow_empty:
            return pd.DataFrame()
        raise


def parse_state_counts(value: object) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if pd.isna(value):
        return counts
    for item in str(value).split(";"):
        if ":" not in item:
            continue
        state, raw_count = item.rsplit(":", 1)
        state = state.strip()
        try:
            counts[state] = int(float(raw_count))
        except ValueError:
            continue
    return counts


def major_and_rare_states(counts: Mapping[str, int], rare_threshold: float) -> Tuple[str, set[str]]:
    if not counts:
        return "missing", set()
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = sum(counts.values())
    rare = {state for state, count in counts.items() if total > 0 and (count / total) <= rare_threshold}
    return ordered[0][0], rare


def is_problem_state(value: object) -> bool:
    text = str(value)
    parts = text.split("|")
    return any(part in {"missing", "ambiguous", "gap", "N", "not_coding"} for part in parts)


def add_selected_pair_numeric_summaries(
    df: pd.DataFrame,
    selected: pd.DataFrame,
    rare_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    summary_rows: List[Dict[str, object]] = []
    if selected.empty:
        for mode in ["time", "div"]:
            prefix = f"epi_selected_{mode}"
            for suffix in SELECTED_PAIR_SUMMARY_SUFFIXES:
                out[f"{prefix}_{suffix}"] = 0.0
        return out, pd.DataFrame(summary_rows)

    for mode, mode_pairs in selected.groupby("mode", sort=True):
        mode = str(mode)
        aa_applicable = np.zeros(len(out), dtype=float)
        aa_major = np.zeros(len(out), dtype=float)
        aa_rare = np.zeros(len(out), dtype=float)
        aa_problem = np.zeros(len(out), dtype=float)
        nt_applicable = np.zeros(len(out), dtype=float)
        nt_major = np.zeros(len(out), dtype=float)
        nt_rare = np.zeros(len(out), dtype=float)
        nt_problem = np.zeros(len(out), dtype=float)

        for pair in mode_pairs.itertuples(index=False):
            aa_feature = str(pair.aa_feature)
            nt_feature = str(pair.nt_feature)
            if aa_feature not in out.columns or nt_feature not in out.columns:
                continue
            aa_counts = parse_state_counts(pair.top_aa_states)
            nt_counts = parse_state_counts(pair.top_nt_states)
            aa_major_state, aa_rare_states = major_and_rare_states(aa_counts, rare_threshold)
            nt_major_state, nt_rare_states = major_and_rare_states(nt_counts, rare_threshold)

            aa_values = out[aa_feature].astype(str)
            nt_values = out[nt_feature].astype(str)
            aa_mask = ~aa_values.eq("not_applicable")
            nt_mask = ~nt_values.eq("not_applicable")

            aa_applicable += aa_mask.to_numpy(dtype=float)
            aa_major += (aa_mask & aa_values.eq(aa_major_state)).to_numpy(dtype=float)
            aa_rare += (aa_mask & aa_values.isin(aa_rare_states)).to_numpy(dtype=float)
            aa_problem += (aa_mask & aa_values.map(is_problem_state)).to_numpy(dtype=float)
            nt_applicable += nt_mask.to_numpy(dtype=float)
            nt_major += (nt_mask & nt_values.eq(nt_major_state)).to_numpy(dtype=float)
            nt_rare += (nt_mask & nt_values.isin(nt_rare_states)).to_numpy(dtype=float)
            nt_problem += (nt_mask & nt_values.map(is_problem_state)).to_numpy(dtype=float)

            summary_rows.append(
                {
                    "interaction_id": pair.interaction_id,
                    "mode": mode,
                    "aa_feature": aa_feature,
                    "nt_feature": nt_feature,
                    "major_aa_state": aa_major_state,
                    "rare_aa_states": "|".join(sorted(aa_rare_states)),
                    "major_nt_state": nt_major_state,
                    "rare_nt_states": "|".join(sorted(nt_rare_states)),
                    "rare_state_frequency_threshold": float(rare_threshold),
                }
            )

        prefix = f"epi_selected_{mode}"
        out[f"{prefix}_applicable_pair_n"] = aa_applicable
        out[f"{prefix}_aa_major_state_n"] = aa_major
        out[f"{prefix}_aa_rare_state_n"] = aa_rare
        out[f"{prefix}_aa_problem_state_n"] = aa_problem
        out[f"{prefix}_aa_major_state_fraction"] = np.divide(
            aa_major,
            aa_applicable,
            out=np.zeros_like(aa_major),
            where=aa_applicable > 0,
        )
        out[f"{prefix}_aa_rare_state_fraction"] = np.divide(
            aa_rare,
            aa_applicable,
            out=np.zeros_like(aa_rare),
            where=aa_applicable > 0,
        )
        out[f"{prefix}_aa_problem_state_fraction"] = np.divide(
            aa_problem,
            aa_applicable,
            out=np.zeros_like(aa_problem),
            where=aa_applicable > 0,
        )
        out[f"{prefix}_nt_major_state_n"] = nt_major
        out[f"{prefix}_nt_rare_state_n"] = nt_rare
        out[f"{prefix}_nt_problem_state_n"] = nt_problem
        out[f"{prefix}_nt_major_state_fraction"] = np.divide(
            nt_major,
            nt_applicable,
            out=np.zeros_like(nt_major),
            where=nt_applicable > 0,
        )
        out[f"{prefix}_nt_rare_state_fraction"] = np.divide(
            nt_rare,
            nt_applicable,
            out=np.zeros_like(nt_rare),
            where=nt_applicable > 0,
        )
        out[f"{prefix}_nt_problem_state_fraction"] = np.divide(
            nt_problem,
            nt_applicable,
            out=np.zeros_like(nt_problem),
            where=nt_applicable > 0,
        )

    return out, pd.DataFrame(summary_rows)


def add_species_fdr_features(
    df: pd.DataFrame,
    fdr_summary: pd.DataFrame,
    selected: pd.DataFrame,
    alpha: float,
    fdr_threshold: float,
    robust_p_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    work = fdr_summary.copy()
    work = work[np.isclose(pd.to_numeric(work["alpha"], errors="coerce"), alpha)].copy()
    numeric_cols = [
        "fdr",
        "observed_positive_count_p",
        "robust_positive_fraction_p",
        "observed_positives",
        "expected_false_positives",
        "n_site_pairs",
        "robust_observed_positives",
        "robust_total_pairs",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    selected_counts = (
        selected.groupby(["species", "mode"], dropna=False)
        .size()
        .rename("selected_pair_n")
        .reset_index()
        if not selected.empty
        else pd.DataFrame(columns=["species", "mode", "selected_pair_n"])
    )

    feature_rows: List[Dict[str, object]] = []
    species_feature_map: Dict[str, Dict[str, object]] = {}
    for (species, mode), group in work.groupby(["species", "mode"], sort=True):
        species = str(species)
        total_site_pairs = float(group["n_site_pairs"].sum())
        observed = float(group["observed_positives"].sum())
        efp = float(group["expected_false_positives"].sum())
        selected_pair_n = int(
            selected_counts.loc[
                (selected_counts["species"].astype(str) == str(species))
                & (selected_counts["mode"].astype(str) == str(mode)),
                "selected_pair_n",
            ].sum()
        )
        min_fdr = float(group["fdr"].min()) if group["fdr"].notna().any() else np.nan
        best_count_p = (
            float(group["observed_positive_count_p"].min())
            if group["observed_positive_count_p"].notna().any()
            else np.nan
        )
        best_robust_p = (
            float(group["robust_positive_fraction_p"].min())
            if group["robust_positive_fraction_p"].notna().any()
            else np.nan
        )
        prefix = f"species_feat_epi_final_{mode}"
        row = {
            "species": species,
            f"{prefix}_alpha": float(alpha),
            f"{prefix}_analysis_n": int(group.shape[0]),
            f"{prefix}_analysis_fdr_le_{str(fdr_threshold).replace('.', '_')}_n": int(group["fdr"].le(fdr_threshold).sum()),
            f"{prefix}_analysis_robust_p_le_{str(robust_p_threshold).replace('.', '_')}_n": int(
                group["robust_positive_fraction_p"].le(robust_p_threshold).sum()
            ),
            f"{prefix}_observed_positive_n": observed,
            f"{prefix}_expected_false_positive_n": efp,
            f"{prefix}_selected_pair_n": selected_pair_n,
            f"{prefix}_site_pair_n": total_site_pairs,
            f"{prefix}_selected_pair_per_100k_site_pairs": 0.0 if total_site_pairs <= 0 else selected_pair_n / total_site_pairs * 100000.0,
            f"{prefix}_min_fdr": min_fdr,
            f"{prefix}_min_fdr_neglog10": -math.log10(max(min_fdr, 1e-12)) if np.isfinite(min_fdr) else np.nan,
            f"{prefix}_best_observed_positive_count_p": best_count_p,
            f"{prefix}_best_observed_positive_count_neglog10p": -math.log10(max(best_count_p, 1e-12)) if np.isfinite(best_count_p) else np.nan,
            f"{prefix}_best_robust_positive_fraction_p": best_robust_p,
            f"{prefix}_best_robust_positive_fraction_neglog10p": -math.log10(max(best_robust_p, 1e-12)) if np.isfinite(best_robust_p) else np.nan,
        }
        species_feature_map.setdefault(species, {"species": species}).update(row)
        for feature, value in row.items():
            if feature == "species":
                continue
            feature_rows.append(
                {
                    "feature": feature,
                    "species": species,
                    "mode": mode,
                    "value": value,
                    "source": "simultaneous_fdr_summary",
                }
            )
    species_df = pd.DataFrame(species_feature_map.values())
    if not species_df.empty:
        out = out.merge(species_df, on="species", how="left")
    final_cols = [col for col in out.columns if col.startswith("species_feat_epi_final_")]
    for col in final_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out, pd.DataFrame(feature_rows)


def drop_empty_epistasis_placeholders(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    drop_cols = [
        col
        for col in df.columns
        if col.startswith("species_feat_epi_")
        and not col.startswith("species_feat_epi_final_")
        and df[col].isna().all()
    ]
    return df.drop(columns=drop_cols), drop_cols


def feature_scope(feature: str) -> str:
    if feature.startswith("species_feat_"):
        return "species"
    if feature.startswith("epi_pair_state_") or feature.startswith("epi_selected_"):
        return "sequence"
    if feature.startswith("ld_site_"):
        return "sequence"
    return "other"


def feature_kind(df: pd.DataFrame, feature: str) -> str:
    if feature not in df.columns:
        return "missing"
    if pd.api.types.is_numeric_dtype(df[feature]):
        return "numeric"
    numeric = pd.to_numeric(df[feature], errors="coerce")
    non_null = df[feature].notna().sum()
    if non_null and numeric.notna().sum() / non_null >= 0.95:
        return "numeric"
    return "categorical"


def build_feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feature in df.columns:
        source = None
        description = None
        if feature.startswith("species_feat_epi_final_"):
            source = "epistasis_fdr_species_summary"
            description = "Species-level simultaneous-null FDR epistasis summary."
        elif feature.startswith("species_feat_epi_"):
            source = "epistasis_observed_species_summary"
            description = "Species-level observed full-scan epistasis score summary."
        elif feature.startswith("epi_pair_state_"):
            source = "selected_epistatic_pair_state"
            description = "Per-sequence categorical state at a selected FDR-gated intersegment epistatic pair."
        elif feature.startswith("epi_selected_"):
            source = "selected_epistatic_pair_numeric_summary"
            description = "Per-sequence numeric summary over selected FDR-gated epistatic pair states."
        elif feature.startswith("ld_site_"):
            source = "ld_pair_or_site_state"
            description = "Per-sequence LD-selected site or site-pair state."
        elif (
            feature.startswith("species_feat_mean_cramers_v_")
            or feature.startswith("species_feat_prop_significant_")
            or feature.startswith("species_feat_n_site_pairs_")
            or feature.startswith("species_feat_n_significant_")
            or feature.startswith("species_feat_mean_nmi_")
            or feature.startswith("species_feat_median_cramers_v_")
            or feature.startswith("species_feat_cramers_v_")
        ):
            source = "ld_species_summary"
            description = "Species-level LD summary."
        if source:
            rows.append(
                {
                    "feature": feature,
                    "scope": feature_scope(feature),
                    "kind": feature_kind(df, feature),
                    "source": source,
                    "n_nonmissing": int(df[feature].notna().sum()),
                    "n_unique": int(df[feature].nunique(dropna=True)),
                    "description": description,
                }
            )
    return pd.DataFrame(rows)


def build_equivalence_map(df: pd.DataFrame) -> pd.DataFrame:
    ld_pair_cols = [col for col in df.columns if col.startswith("ld_site_")]
    epi_pair_cols = [col for col in df.columns if col.startswith("epi_pair_state_")]
    epi_selected_cols = [col for col in df.columns if col.startswith("epi_selected_")]
    ld_summary_coarse = [
        col
        for col in ["species_feat_mean_cramers_v_overall", "species_feat_prop_significant_overall"]
        if col in df.columns
    ]
    ld_summary_segment = [
        col
        for col in df.columns
        if re.match(r"species_feat_(mean_cramers_v|prop_significant)_(SM|SL|ML)$", col)
    ]
    epi_summary_observed = [
        col
        for col in df.columns
        if col.startswith("species_feat_epi_")
        and not col.startswith("species_feat_epi_final_")
        and not col.startswith("species_feat_epi_mutcount_audit_")
    ]
    epi_summary_fdr = [col for col in df.columns if col.startswith("species_feat_epi_final_")]
    rows = [
        {
            "comparison_block": "species_summary_coarse",
            "ld_features": "|".join(ld_summary_coarse),
            "epistasis_features": "|".join(epi_summary_fdr + [c for c in epi_summary_observed if c.endswith(("_mean_score", "_max_score"))]),
            "recommended_model_use": "Bayesian block comparison, EGT molecular-permissiveness comparison, and tree-model summary block.",
        },
        {
            "comparison_block": "species_summary_segment_pair",
            "ld_features": "|".join(ld_summary_segment),
            "epistasis_features": "|".join([c for c in epi_summary_observed if c.endswith(("_SM", "_SL", "_ML"))]),
            "recommended_model_use": "Segment-pair sensitivity or tree-model feature block.",
        },
        {
            "comparison_block": "sequence_pair_state_categorical",
            "ld_features": "|".join(ld_pair_cols),
            "epistasis_features": "|".join(epi_pair_cols),
            "recommended_model_use": "LightGBM/XGBoost/CatBoost categorical comparison.",
        },
        {
            "comparison_block": "sequence_pair_state_numeric_summary",
            "ld_features": "",
            "epistasis_features": "|".join(epi_selected_cols),
            "recommended_model_use": "Bayesian/EGT-compatible numeric summary of selected epistatic pair states.",
        },
    ]
    return pd.DataFrame(rows)


def validate_dataset(original: pd.DataFrame, final: pd.DataFrame, dropped: Sequence[str], selected_n: int) -> pd.DataFrame:
    checks: List[Dict[str, object]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": int(bool(passed)), "detail": detail})

    add("row_count_preserved", len(original) == len(final), f"{len(original)}->{len(final)}")
    if "tip_name" in original.columns and "tip_name" in final.columns:
        add("tip_order_preserved", original["tip_name"].astype(str).equals(final["tip_name"].astype(str)), "tip_name")
    for label in ["is_reassortant_final", "is_reassortant", "target_source"]:
        if label in original.columns and label in final.columns:
            add(
                f"{label}_preserved",
                original[label].astype(str).equals(final[label].astype(str)),
                label,
            )
    epi_cols = [
        col
        for col in final.columns
        if col.startswith("species_feat_epi_") or col.startswith("epi_pair_state_") or col.startswith("epi_selected_")
    ]
    all_missing = [col for col in epi_cols if final[col].isna().all()]
    add("no_all_missing_epistasis_columns", not all_missing, "|".join(all_missing[:20]))
    add("dropped_empty_epistasis_placeholders", len(dropped) > 0, len(dropped))
    has_pair_states = any(col.startswith("epi_pair_state_") for col in final.columns)
    add(
        "selected_pair_state_columns_consistent",
        has_pair_states if selected_n > 0 else not has_pair_states,
        f"selected_interactions={selected_n}",
    )
    add("has_selected_pair_numeric_summaries", any(col.startswith("epi_selected_") for col in final.columns), "")
    add("has_fdr_species_summaries", any(col.startswith("species_feat_epi_final_") for col in final.columns), "")
    return pd.DataFrame(checks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    reports_dir = args.output_root / "reports"
    datasets_dir = args.output_root / "datasets"
    reports_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    original = read_table(args.input_dataset)
    selected = read_table(args.epistasis_root / "reports" / "intersegment_epistasis_selected_interactions.tsv", allow_empty=True)
    fdr_summary = read_table(args.epistasis_root / "reports" / "intersegment_epistasis_fdr_summary.tsv")

    final, selected_state_manifest = add_selected_pair_numeric_summaries(
        original,
        selected,
        rare_threshold=args.rare_state_frequency_threshold,
    )
    final, fdr_feature_values = add_species_fdr_features(
        final,
        fdr_summary,
        selected,
        alpha=args.fdr_alpha,
        fdr_threshold=args.analysis_fdr_threshold,
        robust_p_threshold=args.robust_p_threshold,
    )
    final, dropped = drop_empty_epistasis_placeholders(final)

    dataset_path = datasets_dir / "sequence_level_final_dataset_direct553_epistasis_final.tsv"
    final.to_csv(dataset_path, sep="\t", index=False)

    feature_dictionary = build_feature_dictionary(final)
    feature_dictionary.to_csv(reports_dir / "epistasis_feature_dictionary.tsv", sep="\t", index=False)
    selected_state_manifest.to_csv(reports_dir / "selected_pair_state_manifest.tsv", sep="\t", index=False)
    fdr_feature_values.to_csv(reports_dir / "species_epistasis_fdr_feature_values.tsv", sep="\t", index=False)
    equivalence = build_equivalence_map(final)
    equivalence.to_csv(reports_dir / "ld_epistasis_equivalence_map.tsv", sep="\t", index=False)
    validation = validate_dataset(original, final, dropped, selected_n=int(selected.shape[0]))
    validation.to_csv(reports_dir / "final_epistasis_dataset_validation.tsv", sep="\t", index=False)

    summary = {
        "input_dataset": str(args.input_dataset),
        "output_dataset": str(dataset_path),
        "rows": int(final.shape[0]),
        "columns": int(final.shape[1]),
        "dropped_empty_epistasis_placeholder_columns": len(dropped),
        "epistasis_feature_dictionary_rows": int(feature_dictionary.shape[0]),
        "selected_interactions": int(selected.shape[0]),
        "selected_pair_state_columns": int(sum(col.startswith("epi_pair_state_") for col in final.columns)),
        "selected_pair_numeric_summary_columns": int(sum(col.startswith("epi_selected_") for col in final.columns)),
        "species_fdr_summary_columns": int(sum(col.startswith("species_feat_epi_final_") for col in final.columns)),
        "validation_passed": bool(validation["passed"].eq(1).all()),
    }
    (reports_dir / "final_epistasis_dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    pd.DataFrame({"dropped_column": list(dropped)}).to_csv(
        reports_dir / "dropped_empty_epistasis_placeholder_columns.tsv",
        sep="\t",
        index=False,
    )

    if args.strict and not summary["validation_passed"]:
        raise RuntimeError(f"Final epistasis dataset validation failed: {validation.to_dict(orient='records')}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
