#!/usr/bin/env python3
"""Build a sequence-level final modeling dataset with ArHa host/pathogen joins."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from build_case_study_site_level_dataset import (
    annotate_target_availability,
    build_isolate_table,
    load_species_feature_table,
    merge_species_features,
    read_case_species,
    slugify_species,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENRICHED_HOST = REPO_ROOT / "model_test" / "runs" / "model_site" / "datasets" / "arha_host_with_overlap.csv"
DEFAULT_RAW_HOST = REPO_ROOT / "Data" / "ArHa_data" / "host.csv"

SPECIES_TRAIT_ALIASES = {
    "peromyscus sonoriensis": "peromyscus maniculatus",
}


def default_host_path() -> Path:
    return DEFAULT_ENRICHED_HOST if DEFAULT_ENRICHED_HOST.exists() else DEFAULT_RAW_HOST


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final sequence-level dataset by joining isolate outputs to ArHa sequence/pathogen/host tables."
    )
    parser.add_argument(
        "--isolate-dataset",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "model_site" / "datasets" / "isolate_level_with_proxy_full.tsv",
        help="Existing isolate-level dataset. If absent, the script rebuilds it from the Espalier inputs.",
    )
    parser.add_argument(
        "--case-species",
        type=Path,
        default=REPO_ROOT / "model_test" / "config" / "case_species.txt",
    )
    parser.add_argument(
        "--espalier-root",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "espalier_pipeline",
    )
    parser.add_argument(
        "--species-input-root",
        type=Path,
        default=REPO_ROOT / "Data" / "TreeGrafting" / "espalier_pipeline" / "inputs" / "species",
    )
    parser.add_argument(
        "--proxy-quantile",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--proxy-min-discordance",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--isolate-labels",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "model_site" / "datasets" / "isolate_true_labels.tsv",
        help="Optional true isolate labels. If missing, proxy labels remain available.",
    )
    parser.add_argument(
        "--sequence-true-labels",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "model_sequence" / "datasets" / "sequence_true_labels.tsv",
        help="Optional true sequence labels keyed by tip_name / sequence_id.",
    )
    parser.add_argument(
        "--sequence-ld-features",
        type=Path,
        default=REPO_ROOT / "Data" / "ML" / "ld_features" / "sequence_ld_features.csv",
    )
    parser.add_argument(
        "--species-feature-table",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "model" / "datasets" / "case_species_model_dataset_proxy_combined_relaxed_andesense_all_backfills.tsv",
        help="Optional species-level explanatory feature table to merge by species.",
    )
    parser.add_argument(
        "--arha-sequence",
        type=Path,
        default=REPO_ROOT / "Data" / "ArHa_data" / "sequence.csv",
    )
    parser.add_argument(
        "--arha-pathogen",
        type=Path,
        default=REPO_ROOT / "Data" / "ArHa_data" / "pathogen_taxonomy_pcr.csv",
    )
    parser.add_argument(
        "--arha-host",
        type=Path,
        default=default_host_path(),
    )
    parser.add_argument(
        "--sequence-host-traits",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "traits" / "sequence_host_with_traits.tsv",
    )
    parser.add_argument(
        "--host-species-traits",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "traits" / "arha_host_species_traits.tsv",
    )
    parser.add_argument(
        "--host-overlap-column",
        type=str,
        default="sampling_coordinate_overlap",
    )
    parser.add_argument(
        "--local-prevalence-exact-radius-km",
        type=float,
        default=25.0,
        help="Radius for exact local ArHa prevalence pooling.",
    )
    parser.add_argument(
        "--local-prevalence-max-radius-km",
        type=float,
        default=1000.0,
        help="Maximum donor radius for nearest/IDW local ArHa prevalence fallback.",
    )
    parser.add_argument(
        "--local-prevalence-idw-min-neighbors",
        type=int,
        default=2,
        help="Minimum donor count inside the fallback radius required to use IDW instead of nearest-neighbor prevalence.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "model_site" / "datasets" / "sequence_level_final_dataset.tsv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=REPO_ROOT / "model_test" / "reports" / "sequence_level_final_dataset_summary.tsv",
    )
    return parser.parse_args(argv)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_accession(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "null", "<na>"}:
        return pd.NA
    return re.sub(r"\.\d+$", "", text.upper())


def normalize_isolate_token(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if not text or text in {"nan", "na", "none", "null", "<na>"}:
        return pd.NA
    text = text.replace("'", "").replace(".", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or pd.NA


def normalize_species_name(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip().lower().replace("_", " ")
    if not text or text in {"nan", "na", "none", "null", "<na>", "unknown"}:
        return pd.NA
    text = re.sub(r"\s+", " ", text)
    text = SPECIES_TRAIT_ALIASES.get(text, text)
    return text or pd.NA


def first_valid(series: pd.Series) -> object:
    for value in series:
        if pd.notna(value) and str(value).strip() != "":
            return value
    return pd.NA


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def series_or_na(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(pd.NA, index=df.index, dtype="object")


def load_or_build_isolates(args: argparse.Namespace) -> pd.DataFrame:
    if args.isolate_dataset.exists():
        return pd.read_csv(args.isolate_dataset, sep="\t", low_memory=False)

    case_species = read_case_species(args.case_species)
    labels_path = args.isolate_labels if args.isolate_labels.exists() else None
    return build_isolate_table(
        case_species=case_species,
        espalier_root=args.espalier_root,
        species_input_root=args.species_input_root,
        proxy_quantile=args.proxy_quantile,
        proxy_min_discordance=args.proxy_min_discordance,
        isolate_labels_path=labels_path,
        sequence_ld_path=args.sequence_ld_features,
    )


def reset_existing_true_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any pre-existing true-label columns before merging current sources.

    Some persisted isolate-level datasets already carry historical true-label
    fields. If those survive into a rebuild, later left-merges can leave stale
    values in place and make the final dataset look synchronized when it is not.
    """
    drop_cols = [
        col
        for col in [
            "is_reassortant",
            "is_reassortant_final",
            "target_source",
            "sequence_true_is_reassortant",
            "sequence_true_n_segments_labeled",
            "sequence_true_n_segments_positive",
            "sequence_true_min_distance_to_reassort_node",
            "sequence_true_has_reassortment_distance",
            "sequence_true_distance_status",
            "sequence_true_descriptive_distance_to_reassort_node",
            "sequence_true_sources",
            "isolate_true_is_reassortant",
            "isolate_true_n_segments_labeled",
            "isolate_true_n_sequences",
            "isolate_true_sources",
        ]
        if col in df.columns
    ]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def merge_isolate_true_labels(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return df
    labels = pd.read_csv(path, sep="\t", low_memory=False).copy()
    if "species" in labels.columns:
        labels["species"] = labels["species"].map(slugify_species)
    labels["canonical_isolate_norm"] = labels["canonical_isolate"].map(normalize_isolate_token)
    dedupe_keys = ["canonical_isolate_norm"]
    if "species" in labels.columns and "species" in df.columns:
        dedupe_keys = ["species", "canonical_isolate_norm"]
    labels = labels.drop_duplicates(subset=dedupe_keys)
    out = df.copy()
    if "species" in out.columns:
        out["species"] = out["species"].map(slugify_species)
    out["canonical_isolate_norm"] = out["canonical_isolate"].map(normalize_isolate_token)
    merge_keys = ["canonical_isolate_norm"]
    label_cols = [
        "canonical_isolate_norm",
        "is_reassortant",
        "n_segments_labeled",
        "n_sequences",
        "sources",
    ]
    if "species" in labels.columns and "species" in out.columns:
        merge_keys = ["species", "canonical_isolate_norm"]
        label_cols = ["species"] + label_cols
    out = out.merge(
        labels[label_cols].rename(
            columns={
                "is_reassortant": "isolate_true_is_reassortant",
                "n_segments_labeled": "isolate_true_n_segments_labeled",
                "n_sequences": "isolate_true_n_sequences",
                "sources": "isolate_true_sources",
            }
        ),
        on=merge_keys,
        how="left",
    )
    if "is_reassortant" not in out.columns:
        out["is_reassortant"] = out["isolate_true_is_reassortant"]
    else:
        out["is_reassortant"] = out["is_reassortant"].where(
            out["is_reassortant"].notna(),
            out["isolate_true_is_reassortant"],
        )
    return out


def choose_reassortment_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sequence_true_vals = pd.to_numeric(series_or_na(df, "sequence_true_is_reassortant"), errors="coerce")
    isolate_true_vals = pd.to_numeric(series_or_na(df, "is_reassortant"), errors="coerce")
    proxy_vals = pd.to_numeric(series_or_na(df, "is_reassortant_proxy"), errors="coerce")
    df["is_reassortant_final"] = np.where(
        sequence_true_vals.notna(),
        sequence_true_vals,
        np.where(isolate_true_vals.notna(), isolate_true_vals, proxy_vals),
    )
    df["target_source"] = np.where(
        sequence_true_vals.notna(),
        "sequence_true_label",
        np.where(
            isolate_true_vals.notna(),
            "isolate_true_label",
            np.where(proxy_vals.notna(), "proxy_label", "missing"),
        ),
    )
    return df


def annotate_sequence_target_availability(df: pd.DataFrame) -> pd.DataFrame:
    out = annotate_target_availability(
        df,
        true_label_col="is_reassortant",
        proxy_label_col="is_reassortant_proxy",
    )
    sequence_true_vals = pd.to_numeric(series_or_na(out, "sequence_true_is_reassortant"), errors="coerce")
    isolate_true_vals = pd.to_numeric(series_or_na(out, "is_reassortant"), errors="coerce")
    final_vals = pd.to_numeric(series_or_na(out, "is_reassortant_final"), errors="coerce")
    out["target_available_sequence_true"] = sequence_true_vals.notna().astype(int)
    out["target_available_isolate_true"] = isolate_true_vals.notna().astype(int)
    out["target_available_true"] = (sequence_true_vals.notna() | isolate_true_vals.notna()).astype(int)
    out["target_available_final"] = final_vals.notna().astype(int)
    out["target_available_main_explanatory"] = out["target_available_true"].astype(int)
    out["proxy_only_label"] = (
        final_vals.notna()
        & (out["target_available_true"] == 0)
        & (pd.to_numeric(series_or_na(out, "is_reassortant_proxy"), errors="coerce").notna())
    ).astype(int)
    return out


def load_sequence_true_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["tip_name"])
    df = pd.read_csv(path, sep="\t", low_memory=False).copy()
    if "species" in df.columns:
        df["species"] = df["species"].map(slugify_species)
    df = df.rename(
        columns={
            "sequence_id": "tip_name",
            "is_reassortant": "sequence_true_is_reassortant",
            "n_segments_labeled": "sequence_true_n_segments_labeled",
            "n_segments_positive": "sequence_true_n_segments_positive",
            "min_distance_to_reassort_node": "sequence_true_min_distance_to_reassort_node",
            "has_reassortment_distance": "sequence_true_has_reassortment_distance",
            "distance_status": "sequence_true_distance_status",
            "descriptive_distance_to_reassort_node": "sequence_true_descriptive_distance_to_reassort_node",
            "sources": "sequence_true_sources",
        }
    )
    keep = [
        "species",
        "tip_name",
        "sequence_true_is_reassortant",
        "sequence_true_n_segments_labeled",
        "sequence_true_n_segments_positive",
        "sequence_true_min_distance_to_reassort_node",
        "sequence_true_has_reassortment_distance",
        "sequence_true_distance_status",
        "sequence_true_descriptive_distance_to_reassort_node",
        "sequence_true_sources",
    ]
    keep = [c for c in keep if c in df.columns]
    dedupe_keys = ["tip_name"]
    if "species" in keep:
        dedupe_keys = ["species", "tip_name"]
    return df[keep].drop_duplicates(subset=dedupe_keys)


def load_sequence_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ["accession_primary", "query_accession", "isolate", "pathogen_species_clean"]:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.copy()
    df["accession_primary_key"] = df["accession_primary"].map(normalize_accession)
    df["query_accession_key"] = df["query_accession"].map(normalize_accession)
    df["sequence_isolate_norm"] = df["isolate"].map(normalize_isolate_token)
    df["sequence_species_slug"] = df["pathogen_species_clean"].map(slugify_species)
    return df


def choose_best_sequence_matches(isolate_df: pd.DataFrame, sequence_df: pd.DataFrame) -> pd.DataFrame:
    isolates = isolate_df.reset_index(drop=True).copy()
    isolates["_isolate_row_id"] = np.arange(len(isolates))
    isolates["accession_key"] = isolates.get("accession", pd.Series(index=isolates.index)).map(normalize_accession)
    isolates["canonical_isolate_norm"] = isolates["canonical_isolate"].map(normalize_isolate_token)
    isolates["isolate_raw_norm"] = isolates.get("isolate_raw", pd.Series(index=isolates.index)).map(normalize_isolate_token)

    accession_primary = isolates[isolates["accession_key"].notna()].merge(
        sequence_df,
        left_on="accession_key",
        right_on="accession_primary_key",
        how="left",
    )
    accession_primary["match_source"] = "accession_primary"

    accession_query = isolates[isolates["accession_key"].notna()].merge(
        sequence_df,
        left_on="accession_key",
        right_on="query_accession_key",
        how="left",
    )
    accession_query["match_source"] = "query_accession"

    raw_isolate = isolates[isolates["isolate_raw_norm"].notna()].merge(
        sequence_df,
        left_on="isolate_raw_norm",
        right_on="sequence_isolate_norm",
        how="left",
    )
    raw_isolate["match_source"] = "isolate_raw"

    canonical_isolate = isolates[isolates["canonical_isolate_norm"].notna()].merge(
        sequence_df,
        left_on="canonical_isolate_norm",
        right_on="sequence_isolate_norm",
        how="left",
    )
    canonical_isolate["match_source"] = "canonical_isolate"

    candidates = pd.concat(
        [accession_primary, accession_query, raw_isolate, canonical_isolate],
        ignore_index=True,
        sort=False,
    )
    if candidates.empty:
        out = isolates.copy()
        out["arha_sequence_match_source"] = pd.NA
        out["arha_sequence_match_count"] = 0
        return out

    candidates = candidates[candidates["sequence_record_id"].notna()].copy()
    if candidates.empty:
        out = isolates.copy()
        out["arha_sequence_match_source"] = pd.NA
        out["arha_sequence_match_count"] = 0
        return out

    candidates["species_match"] = (
        candidates["species"].map(slugify_species) == candidates["sequence_species_slug"]
    ).astype(int)
    candidates["host_record_present"] = candidates["host_record_id"].notna().astype(int)
    match_rank = {"accession_primary": 0, "query_accession": 1, "isolate_raw": 2, "canonical_isolate": 3}
    candidates["match_source_rank"] = candidates["match_source"].map(match_rank).fillna(99).astype(int)
    candidates["sequence_record_id_num"] = pd.to_numeric(candidates["sequence_record_id"], errors="coerce")
    candidates = candidates[
        (candidates["match_source_rank"] <= 1) | (candidates["species_match"] == 1)
    ].copy()

    if candidates.empty:
        out = isolates.copy()
        out["arha_sequence_match_source"] = pd.NA
        out["arha_sequence_match_count"] = 0
        return out

    match_summary = (
        candidates.groupby("_isolate_row_id", dropna=False)
        .agg(
            arha_sequence_match_count=("sequence_record_id", "nunique"),
            arha_sequence_match_sources=("match_source", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
        )
        .reset_index()
    )

    candidates = candidates.sort_values(
        by=[
            "_isolate_row_id",
            "species_match",
            "host_record_present",
            "match_source_rank",
            "sequence_record_id_num",
        ],
        ascending=[True, False, False, True, True],
    )
    best = candidates.groupby("_isolate_row_id", dropna=False).head(1).copy()
    best = best.merge(match_summary, on="_isolate_row_id", how="left")

    keep_cols = [
        "_isolate_row_id",
        "sequence_record_id",
        "host_record_id",
        "pathogen_record_id",
        "match_source",
        "arha_sequence_match_count",
        "arha_sequence_match_sources",
        "accession_primary",
        "query_accession",
        "sequence_type",
        "resolution_status",
        "host",
        "pathogen",
        "isolate",
        "country",
        "iso3c",
        "locality",
        "collection_date",
        "host_species_clean",
        "pathogen_species_clean",
        "species_match",
    ]
    keep_cols = [c for c in keep_cols if c in best.columns]
    best = best[keep_cols].copy()
    best = best.rename(
        columns={
            "match_source": "arha_sequence_match_source",
            "host": "arha_sequence_host_raw",
            "pathogen": "arha_sequence_pathogen_raw",
            "isolate": "arha_sequence_isolate_raw",
            "country": "arha_sequence_country",
            "iso3c": "arha_sequence_iso3c",
            "locality": "arha_sequence_locality",
            "collection_date": "arha_sequence_collection_date",
            "host_species_clean": "arha_sequence_host_species_clean",
            "pathogen_species_clean": "arha_sequence_pathogen_species_clean",
        }
    )

    out = isolates.merge(best, on="_isolate_row_id", how="left")
    out["arha_sequence_match_count"] = pd.to_numeric(
        out.get("arha_sequence_match_count"),
        errors="coerce",
    ).fillna(0).astype(int)
    return out


def load_pathogen_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ["assigned_name", "pathogen_species_cleaned", "pathogen_species_ncbi"]:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.copy()
    df["arha_pathogen_harmonized_name"] = (
        df["assigned_name"]
        .fillna(df["pathogen_species_cleaned"])
        .fillna(df["pathogen_species_ncbi"])
    )
    df["arha_pathogen_harmonized_slug"] = df["arha_pathogen_harmonized_name"].map(slugify_species)
    return df


def load_host_table(path: Path, overlap_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if overlap_col not in df.columns:
        df[overlap_col] = pd.NA
    keep = [
        "host_record_id",
        "host_species",
        "country",
        "iso3c",
        "latitude",
        "longitude",
        overlap_col,
        "start_date",
        "end_date",
        "number_of_hosts",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    out = df[keep].copy()
    return out.rename(
        columns={
            "host_species": "arha_host_species",
            "country": "arha_host_country",
            "iso3c": "arha_host_iso3c",
            "latitude": "arha_host_latitude",
            "longitude": "arha_host_longitude",
            overlap_col: "arha_host_sampling_overlap",
            "start_date": "arha_host_start_date",
            "end_date": "arha_host_end_date",
            "number_of_hosts": "arha_host_number_of_hosts",
        }
    )


def load_prevalence_donors(pathogen_path: Path, host_path: Path) -> pd.DataFrame:
    pathogen = pd.read_csv(pathogen_path, low_memory=False).copy()
    host = pd.read_csv(host_path, low_memory=False).copy()

    species_raw = (
        pathogen.get("assigned_name", pd.Series(index=pathogen.index, dtype=object))
        .fillna(pathogen.get("pathogen_species_cleaned", pd.Series(index=pathogen.index, dtype=object)))
        .fillna(pathogen.get("pathogen_species_ncbi", pd.Series(index=pathogen.index, dtype=object)))
    )
    pathogen["species"] = species_raw.map(slugify_species)
    pathogen["number_tested_num"] = pd.to_numeric(pathogen.get("number_tested"), errors="coerce")
    pathogen["number_positive_num"] = pd.to_numeric(pathogen.get("number_positive"), errors="coerce")

    for col in ["host_record_id", "host_species", "latitude", "longitude"]:
        if col not in host.columns:
            host[col] = pd.NA
    host["host_norm"] = host["host_species"].map(normalize_species_name)
    host["host_latitude"] = pd.to_numeric(host["latitude"], errors="coerce")
    host["host_longitude"] = pd.to_numeric(host["longitude"], errors="coerce")

    merged = pathogen.merge(
        host[["host_record_id", "host_species", "host_norm", "host_latitude", "host_longitude"]],
        on="host_record_id",
        how="left",
    )
    merged = merged[
        merged["species"].notna()
        & merged["host_latitude"].notna()
        & merged["host_longitude"].notna()
        & (merged["number_tested_num"].fillna(0) > 0)
    ].copy()
    if merged.empty:
        return merged

    grouped = (
        merged.groupby(
            ["species", "host_record_id", "host_species", "host_norm", "host_latitude", "host_longitude"],
            dropna=False,
            as_index=False,
        )
        .agg(
            number_tested_num=("number_tested_num", "sum"),
            number_positive_num=("number_positive_num", "sum"),
        )
    )
    grouped = grouped[grouped["number_tested_num"] > 0].copy()
    grouped["prevalence_point"] = grouped["number_positive_num"] / grouped["number_tested_num"]
    return grouped.reset_index(drop=True)


def assign_local_ecology_features(
    df: pd.DataFrame,
    donors: pd.DataFrame,
    *,
    exact_radius_km: float,
    max_radius_km: float,
    idw_min_neighbors: int,
) -> pd.DataFrame:
    out = df.copy()

    geo_overlap = pd.to_numeric(series_or_na(out, "geoquery_overlap"), errors="coerce")
    arha_overlap = pd.to_numeric(series_or_na(out, "arha_host_sampling_overlap"), errors="coerce")
    out["overlap_opportunity_local"] = geo_overlap.fillna(arha_overlap)
    out["overlap_opportunity_provenance"] = np.where(
        geo_overlap.notna(),
        "sequence_georef",
        np.where(arha_overlap.notna(), "arha_host_match", "missing"),
    )

    seq_lat = pd.to_numeric(series_or_na(out, "latitude"), errors="coerce")
    seq_lon = pd.to_numeric(series_or_na(out, "longitude"), errors="coerce")
    arha_lat = pd.to_numeric(series_or_na(out, "arha_host_latitude"), errors="coerce")
    arha_lon = pd.to_numeric(series_or_na(out, "arha_host_longitude"), errors="coerce")
    out["local_latitude"] = arha_lat.fillna(seq_lat)
    out["local_longitude"] = arha_lon.fillna(seq_lon)
    out["local_coordinate_provenance"] = np.where(
        arha_lat.notna() & arha_lon.notna() & pd.to_numeric(series_or_na(out, "geoquery_used_for_main_overlap"), errors="coerce").fillna(0).eq(0),
        "arha_host_match",
        np.where(seq_lat.notna() & seq_lon.notna(), "sequence_georef", "missing"),
    )

    out["query_host_norm"] = series_or_na(out, "hosttrait_resolved_host_species").map(normalize_species_name)

    out["arha_prev_local"] = np.nan
    out["arha_prev_distance_km"] = np.nan
    out["arha_prev_neighbor_count"] = np.nan
    out["arha_prev_provenance"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["arha_prev_host_match_quality"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["arha_prev_tested_total"] = np.nan
    out["arha_prev_positive_total"] = np.nan

    donors_by_species: Dict[str, pd.DataFrame] = {
        str(species): sub.reset_index(drop=True)
        for species, sub in donors.groupby("species", dropna=False)
    }

    def evaluate_subset(sub: pd.DataFrame, dists: np.ndarray) -> Tuple[float, float, int, str, float, float]:
        within_exact = dists <= float(exact_radius_km)
        if within_exact.any():
            exact = sub.loc[within_exact].copy()
            tested = float(exact["number_tested_num"].sum())
            positive = float(exact["number_positive_num"].sum())
            prevalence = positive / tested if tested > 0 else np.nan
            return prevalence, float(np.nanmin(dists[within_exact])), int(within_exact.sum()), "prevalence_exact", tested, positive

        within_cap = dists <= float(max_radius_km)
        if within_cap.sum() >= int(idw_min_neighbors):
            local = sub.loc[within_cap].copy()
            local_dists = dists[within_cap]
            tested_weights = local["number_tested_num"].to_numpy(dtype=float)
            tested_weights = np.where(np.isfinite(tested_weights) & (tested_weights > 0), tested_weights, 1.0)
            idw_weights = tested_weights / np.maximum(local_dists, 1.0) ** 2
            prevalence = float(np.average(local["prevalence_point"].to_numpy(dtype=float), weights=idw_weights))
            tested = float(local["number_tested_num"].sum())
            positive = float(local["number_positive_num"].sum())
            return prevalence, float(np.nanmin(local_dists)), int(within_cap.sum()), "prevalence_idw", tested, positive

        if within_cap.any():
            nearest_idx = int(np.nanargmin(np.where(within_cap, dists, np.inf)))
            nearest = sub.iloc[nearest_idx]
            tested = float(nearest["number_tested_num"])
            positive = float(nearest["number_positive_num"])
            prevalence = float(nearest["prevalence_point"])
            return prevalence, float(dists[nearest_idx]), 1, "prevalence_nearest", tested, positive

        return np.nan, np.nan, 0, "prevalence_missing", np.nan, np.nan

    for idx, row in out.iterrows():
        species = row.get("species")
        lat = pd.to_numeric(pd.Series([row.get("local_latitude")]), errors="coerce").iloc[0]
        lon = pd.to_numeric(pd.Series([row.get("local_longitude")]), errors="coerce").iloc[0]
        if pd.isna(species) or not np.isfinite(lat) or not np.isfinite(lon):
            out.at[idx, "arha_prev_provenance"] = "prevalence_missing"
            continue

        species_donors = donors_by_species.get(str(species))
        if species_donors is None or species_donors.empty:
            out.at[idx, "arha_prev_provenance"] = "prevalence_missing"
            continue

        query_host = row.get("query_host_norm")
        use_same_host = pd.notna(query_host) and str(query_host) != "homo sapiens"
        candidate = species_donors
        match_quality = "pathogen_only"
        if use_same_host:
            host_specific = species_donors[species_donors["host_norm"] == query_host].copy()
            if not host_specific.empty:
                candidate = host_specific
                match_quality = "same_host"

        dists = haversine_km(
            float(lat),
            float(lon),
            candidate["host_latitude"].to_numpy(dtype=float),
            candidate["host_longitude"].to_numpy(dtype=float),
        )
        prevalence, min_dist, n_neighbors, provenance, tested, positive = evaluate_subset(candidate, dists)
        if provenance == "prevalence_missing" and match_quality == "same_host":
            candidate = species_donors
            match_quality = "pathogen_only"
            dists = haversine_km(
                float(lat),
                float(lon),
                candidate["host_latitude"].to_numpy(dtype=float),
                candidate["host_longitude"].to_numpy(dtype=float),
            )
            prevalence, min_dist, n_neighbors, provenance, tested, positive = evaluate_subset(candidate, dists)

        out.at[idx, "arha_prev_local"] = prevalence
        out.at[idx, "arha_prev_distance_km"] = min_dist
        out.at[idx, "arha_prev_neighbor_count"] = n_neighbors
        out.at[idx, "arha_prev_provenance"] = provenance
        out.at[idx, "arha_prev_host_match_quality"] = match_quality if provenance != "prevalence_missing" else pd.NA
        out.at[idx, "arha_prev_tested_total"] = tested
        out.at[idx, "arha_prev_positive_total"] = positive

    out = out.drop(columns=["query_host_norm"])
    return out


def load_sequence_host_traits(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["accession_key"])
    df = pd.read_csv(path, sep="\t", low_memory=False).copy()
    if "Accession" not in df.columns:
        return pd.DataFrame(columns=["accession_key"])
    df["accession_key"] = df["Accession"].map(normalize_accession)
    keep = [
        "accession_key",
        "resolved_host_species",
        "resolved_host_species_norm",
        "resolution_stage",
        "resolution_score",
        "gbif_match_type",
        "gbif_confidence",
        "gbif_status",
        "gbif_accepted_name",
        "gbif_accepted_rank",
        "resolved_has_any_traits",
        "trait_fill_method",
        "trait_fill_donor_count",
        "trait_fill_anchor_name",
        "trait_fill_anchor_ott_id",
        "has_direct_any_traits",
    ]
    keep += [c for c in df.columns if c.startswith("combine_") or c.startswith("elton_")]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    rename_map = {
        col: f"seqhost_{col}"
        for col in out.columns
        if col != "accession_key"
    }
    return out.rename(columns=rename_map).drop_duplicates(subset=["accession_key"])


def load_host_species_traits(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["host_species_norm"])
    df = pd.read_csv(path, sep="\t", low_memory=False).copy()
    if "host_species_norm" not in df.columns:
        return pd.DataFrame(columns=["host_species_norm"])
    df["host_species_norm"] = df["host_species_norm"].map(normalize_species_name)
    keep = [
        "host_species_norm",
        "host_species",
        "has_any_traits",
        "trait_fill_method",
        "trait_fill_donor_count",
        "trait_fill_anchor_name",
        "trait_fill_anchor_ott_id",
        "has_direct_any_traits",
    ]
    keep += [c for c in df.columns if c.startswith("combine_") or c.startswith("elton_")]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    rename_map = {
        col: f"hostspecies_{col}"
        for col in out.columns
        if col != "host_species_norm"
    }
    return out.rename(columns=rename_map).drop_duplicates(subset=["host_species_norm"])


def coalesce_host_trait_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seq_available = pd.to_numeric(series_or_na(df, "seqhost_resolved_has_any_traits"), errors="coerce").fillna(0) > 0
    species_available = pd.to_numeric(series_or_na(df, "hostspecies_has_any_traits"), errors="coerce").fillna(0) > 0
    df["hosttrait_source"] = np.where(
        seq_available,
        "sequence_host_traits",
        np.where(species_available, "host_species_traits", "missing"),
    )
    df["hosttrait_has_any_traits"] = (seq_available | species_available).astype("Int64")
    df["hosttrait_has_direct_any_traits"] = np.where(
        seq_available,
        pd.to_numeric(series_or_na(df, "seqhost_has_direct_any_traits"), errors="coerce"),
        pd.to_numeric(series_or_na(df, "hostspecies_has_direct_any_traits"), errors="coerce"),
    )
    df["hosttrait_fill_method"] = series_or_na(df, "seqhost_trait_fill_method").where(
        seq_available,
        series_or_na(df, "hostspecies_trait_fill_method"),
    )
    df["hosttrait_fill_donor_count"] = pd.to_numeric(
        series_or_na(df, "seqhost_trait_fill_donor_count").where(
            seq_available,
            series_or_na(df, "hostspecies_trait_fill_donor_count"),
        ),
        errors="coerce",
    )
    df["hosttrait_fill_anchor_name"] = series_or_na(df, "seqhost_trait_fill_anchor_name").where(
        seq_available,
        series_or_na(df, "hostspecies_trait_fill_anchor_name"),
    )
    df["hosttrait_fill_anchor_ott_id"] = pd.to_numeric(
        series_or_na(df, "seqhost_trait_fill_anchor_ott_id").where(
            seq_available,
            series_or_na(df, "hostspecies_trait_fill_anchor_ott_id"),
        ),
        errors="coerce",
    )
    df["hosttrait_resolved_host_species"] = series_or_na(df, "seqhost_resolved_host_species").where(
        seq_available,
        series_or_na(df, "host"),
    )
    df["hosttrait_resolution_stage"] = series_or_na(df, "seqhost_resolution_stage").where(
        seq_available,
        np.where(species_available, "host_species_direct_or_imputed", pd.NA),
    )
    df["hosttrait_resolution_score"] = pd.to_numeric(series_or_na(df, "seqhost_resolution_score"), errors="coerce")

    seq_trait_cols = {
        col[len("seqhost_") :]: col
        for col in df.columns
        if col.startswith("seqhost_combine_") or col.startswith("seqhost_elton_")
    }
    species_trait_cols = {
        col[len("hostspecies_") :]: col
        for col in df.columns
        if col.startswith("hostspecies_combine_") or col.startswith("hostspecies_elton_")
    }
    for suffix in sorted(set(seq_trait_cols) | set(species_trait_cols)):
        seq_col = seq_trait_cols.get(suffix)
        species_col = species_trait_cols.get(suffix)
        final_col = f"hosttrait_{suffix}"
        if seq_col and species_col:
            df[final_col] = series_or_na(df, seq_col).where(seq_available, series_or_na(df, species_col))
        elif seq_col:
            df[final_col] = df[seq_col]
        elif species_col:
            df[final_col] = df[species_col]
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    species_feature_cols = [c for c in df.columns if c.startswith("species_feat_")]
    summary: Dict[str, object] = {
        "n_rows_total": int(len(df)),
        "n_sequence_true_labels": int(pd.to_numeric(series_or_na(df, "sequence_true_is_reassortant"), errors="coerce").notna().sum()),
        "n_isolate_true_labels": int(pd.to_numeric(series_or_na(df, "is_reassortant"), errors="coerce").notna().sum()),
        "n_true_labels": int(df["target_source"].isin(["sequence_true_label", "isolate_true_label"]).sum()),
        "n_proxy_labels": int(pd.to_numeric(series_or_na(df, "is_reassortant_proxy"), errors="coerce").notna().sum()),
        "n_final_labels": int(pd.to_numeric(series_or_na(df, "is_reassortant_final"), errors="coerce").notna().sum()),
        "n_main_explanatory_rows": int(pd.to_numeric(series_or_na(df, "target_available_main_explanatory"), errors="coerce").fillna(0).sum()),
        "n_proxy_only_rows": int(pd.to_numeric(series_or_na(df, "proxy_only_label"), errors="coerce").fillna(0).sum()),
        "n_species_feature_columns": int(len(species_feature_cols)),
        "n_rows_with_species_features": int(df[species_feature_cols].notna().any(axis=1).sum()) if species_feature_cols else 0,
        "n_with_accession": int(series_or_na(df, "accession_key").notna().sum()),
        "n_with_arha_sequence_match": int(series_or_na(df, "sequence_record_id").notna().sum()),
        "n_with_arha_pathogen_match": int(series_or_na(df, "arha_pathogen_harmonized_name").notna().sum()),
        "n_with_arha_host_match": int(series_or_na(df, "arha_host_species").notna().sum()),
        "n_with_host_overlap": int(pd.to_numeric(series_or_na(df, "arha_host_sampling_overlap"), errors="coerce").notna().sum()),
        "n_with_overlap_opportunity_local": int(pd.to_numeric(series_or_na(df, "overlap_opportunity_local"), errors="coerce").notna().sum()),
        "n_with_host_prevalence_local": int(pd.to_numeric(series_or_na(df, "arha_prev_local"), errors="coerce").notna().sum()),
        "n_with_prevalence_exact": int(series_or_na(df, "arha_prev_provenance").eq("prevalence_exact").sum()),
        "n_with_prevalence_nearest": int(series_or_na(df, "arha_prev_provenance").eq("prevalence_nearest").sum()),
        "n_with_prevalence_idw": int(series_or_na(df, "arha_prev_provenance").eq("prevalence_idw").sum()),
        "n_with_host_traits": int(pd.to_numeric(series_or_na(df, "hosttrait_has_any_traits"), errors="coerce").fillna(0).sum()),
        "n_with_sequence_host_traits": int(pd.to_numeric(series_or_na(df, "seqhost_resolved_has_any_traits"), errors="coerce").fillna(0).sum()),
        "n_species_match_to_pathogen": int(pd.to_numeric(series_or_na(df, "arha_species_match_to_pathogen"), errors="coerce").fillna(0).sum()),
    }
    return pd.DataFrame([summary])


def drop_fully_empty_seqhost_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sequence-host scaffolding columns that are entirely empty.

    These columns are not part of the modeled hosttrait block and can trigger
    noisy downstream handling when they carry no information at all.
    """
    drop_cols = [
        col for col in df.columns
        if col.startswith("seqhost_") and df[col].isna().all()
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def drop_problematic_hosttrait_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop hosttrait fields that are too sparse to support fold-wise modeling."""
    drop_cols = [
        col
        for col in ["hosttrait_combine_teat_number_n"]
        if col in df.columns
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    isolate_df = load_or_build_isolates(args)
    if isolate_df.empty:
        raise ValueError("Isolate dataset is empty; cannot build final sequence-level dataset.")
    isolate_df = reset_existing_true_label_columns(isolate_df)

    species_features = load_species_feature_table(args.species_feature_table)
    isolate_df = merge_species_features(isolate_df, species_features)
    isolate_df = annotate_target_availability(isolate_df)
    isolate_df = merge_isolate_true_labels(isolate_df, args.isolate_labels)
    isolate_df = annotate_target_availability(isolate_df)
    sequence_df = load_sequence_table(args.arha_sequence)
    final_df = choose_best_sequence_matches(isolate_df, sequence_df)

    sequence_true_labels = load_sequence_true_labels(args.sequence_true_labels)
    if not sequence_true_labels.empty:
        merge_keys = ["tip_name"]
        if "species" in sequence_true_labels.columns and "species" in final_df.columns:
            final_df["species"] = final_df["species"].map(slugify_species)
            merge_keys = ["species", "tip_name"]
        final_df = final_df.merge(sequence_true_labels, on=merge_keys, how="left")
    final_df = choose_reassortment_target(final_df)
    final_df = annotate_sequence_target_availability(final_df)

    pathogen_df = load_pathogen_table(args.arha_pathogen)
    final_df = final_df.merge(
        pathogen_df,
        on="pathogen_record_id",
        how="left",
        suffixes=("", "_pathogen"),
    )

    host_df = load_host_table(args.arha_host, args.host_overlap_column)
    final_df = final_df.merge(
        host_df,
        on="host_record_id",
        how="left",
        suffixes=("", "_host"),
    )

    final_df["accession_key"] = final_df.get("accession", pd.Series(index=final_df.index)).map(normalize_accession)
    final_df["host_species_norm"] = final_df.get("host", pd.Series(index=final_df.index)).map(normalize_species_name)

    seqhost_df = load_sequence_host_traits(args.sequence_host_traits)
    if not seqhost_df.empty:
        final_df = final_df.merge(seqhost_df, on="accession_key", how="left")

    host_species_traits = load_host_species_traits(args.host_species_traits)
    if not host_species_traits.empty:
        final_df = final_df.merge(host_species_traits, on="host_species_norm", how="left")

    final_df = coalesce_host_trait_block(final_df)

    final_df["arha_species_match_to_pathogen"] = (
        final_df["species"].map(slugify_species)
        == final_df["arha_pathogen_harmonized_slug"]
    ).astype("Int64")
    final_df["arha_match_confidence"] = np.where(
        final_df["arha_sequence_match_source"].isin(["accession_primary", "query_accession"]),
        "high",
        np.where(final_df["arha_sequence_match_source"].notna(), "medium", "none"),
    )
    final_df["has_arha_sequence_match"] = final_df["sequence_record_id"].notna().astype("Int64")
    final_df["has_arha_pathogen_match"] = final_df["arha_pathogen_harmonized_name"].notna().astype("Int64")
    final_df["has_arha_host_match"] = final_df["arha_host_species"].notna().astype("Int64")
    final_df["has_arha_host_overlap"] = pd.to_numeric(
        final_df["arha_host_sampling_overlap"],
        errors="coerce",
    ).notna().astype("Int64")
    final_df["has_host_traits"] = pd.to_numeric(
        final_df["hosttrait_has_any_traits"],
        errors="coerce",
    ).fillna(0).astype("Int64")
    prevalence_donors = load_prevalence_donors(args.arha_pathogen, args.arha_host)
    final_df = assign_local_ecology_features(
        final_df,
        prevalence_donors,
        exact_radius_km=args.local_prevalence_exact_radius_km,
        max_radius_km=args.local_prevalence_max_radius_km,
        idw_min_neighbors=args.local_prevalence_idw_min_neighbors,
    )
    final_df = drop_fully_empty_seqhost_columns(final_df)
    final_df = drop_problematic_hosttrait_columns(final_df)

    ensure_parent(args.output_dataset)
    ensure_parent(args.output_summary)
    final_df.to_csv(args.output_dataset, sep="\t", index=False)
    build_summary(final_df).to_csv(args.output_summary, sep="\t", index=False)

    print(f"Wrote final sequence-level dataset: {args.output_dataset.resolve()}")
    print(f"Wrote final dataset summary: {args.output_summary.resolve()}")


if __name__ == "__main__":
    main()
