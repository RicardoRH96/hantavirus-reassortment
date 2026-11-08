#!/usr/bin/env python3
"""
Build sampling-site/event level dataset for reassortment-frequency modeling.

Current default target is a tip-level reassortment proxy inferred from
cross-segment tree discordance (until descendant-based labels are provided).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import Phylo


def slugify_species(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("'", "")
    text = text.replace(".", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def extract_year(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    m = re.search(r"(\d{4})", str(value))
    return float(m.group(1)) if m else np.nan


def first_valid(series: pd.Series) -> object:
    for value in series:
        if pd.notna(value) and str(value).strip() != "":
            return value
    return np.nan


def safe_read(path: Path, sep: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=sep, low_memory=False)


def normalize_feature_token(value: object) -> str:
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def read_case_species(path: Path) -> List[str]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def load_sequence_ld_table(path: Optional[Path], case_species: Iterable[str]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "species" not in df.columns or "sequence_id" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["species"] = df["species"].map(slugify_species)
    keep_species = set(case_species)
    df = df[df["species"].isin(keep_species)].copy()
    if df.empty:
        return pd.DataFrame()

    ld_cols = [c for c in df.columns if c not in {"species", "sequence_id"}]
    rename = {col: f"ld_site_{col}" for col in ld_cols}
    df = df.rename(columns=rename)
    df["ld_sequence_id"] = df["sequence_id"].astype(str)
    drop_cols = ["sequence_id"]
    return df.drop(columns=drop_cols)


def load_species_segment_metadata(
    species: str,
    segment: str,
    espalier_root: Path,
    species_input_root: Path,
) -> pd.DataFrame:
    matched_path = espalier_root / species / "augur" / segment / f"{segment}_matched_metadata.tsv"
    input_path = species_input_root / species / f"{segment}_segment" / "metadata.tsv"
    matched = safe_read(matched_path, sep="\t")
    source = safe_read(input_path, sep="\t")

    if matched.empty or source.empty:
        return pd.DataFrame()

    matched = matched.copy()
    if "strain" in matched.columns and "tip_name" not in matched.columns:
        matched = matched.rename(columns={"strain": "tip_name"})

    source = source.copy()
    source["accession"] = source["Accession"].astype(str)
    keep_cols = [
        "accession",
        "Isolate",
        "Geo_Location",
        "Country",
        "Host",
        "Collection_Date",
        "Nuc_Completeness",
        "Length",
        "GenBank_RefSeq",
    ]
    keep_cols = [c for c in keep_cols if c in source.columns]
    source = source[keep_cols]

    joined = matched.merge(source, on="accession", how="left")
    joined["segment"] = segment
    return joined


def load_isolate_matrix(species: str, espalier_root: Path) -> pd.DataFrame:
    path = espalier_root / species / "reports" / "isolate_matrix.tsv"
    df = safe_read(path, sep="\t")
    if df.empty:
        return pd.DataFrame()
    return df


def _compute_corr(values_a: List[float], values_b: List[float]) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size < 3 or b.size < 3:
        return np.nan
    if np.nanstd(a) == 0 and np.nanstd(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def compute_tree_discordance(
    species: str,
    species_dir: Path,
    quantile: float,
    min_discordance_threshold: float,
) -> pd.DataFrame:
    tree_paths = {
        seg: species_dir / "espalier" / "trees" / f"{seg}_tree_common_binary.tre"
        for seg in ["S", "M", "L"]
    }
    if not all(path.exists() for path in tree_paths.values()):
        return pd.DataFrame(columns=["species", "tip_name", "tree_discordance_score", "is_reassortant_proxy", "proxy_threshold"])

    trees = {seg: Phylo.read(str(path), "newick") for seg, path in tree_paths.items()}
    tips_by_seg = {seg: {tip.name for tip in tree.get_terminals()} for seg, tree in trees.items()}
    common_tips = sorted(set.intersection(*tips_by_seg.values()))
    if len(common_tips) < 4:
        return pd.DataFrame(columns=["species", "tip_name", "tree_discordance_score", "is_reassortant_proxy", "proxy_threshold"])

    pair_defs = [("S", "M"), ("S", "L"), ("M", "L")]
    rows: List[Dict[str, object]] = []

    for tip in common_tips:
        others = [o for o in common_tips if o != tip]
        pair_scores: List[float] = []
        for seg_a, seg_b in pair_defs:
            d_a = [trees[seg_a].distance(tip, other) for other in others]
            d_b = [trees[seg_b].distance(tip, other) for other in others]
            corr = _compute_corr(d_a, d_b)
            if np.isfinite(corr):
                # Higher means more topological/branch-distance discordance for this tip.
                pair_scores.append(1.0 - corr)
        score = float(np.mean(pair_scores)) if pair_scores else np.nan
        rows.append({"species": species, "tip_name": tip, "tree_discordance_score": score})

    scores = pd.Series([r["tree_discordance_score"] for r in rows], dtype=float).dropna()
    if scores.empty:
        threshold = np.nan
    else:
        threshold = max(float(scores.quantile(quantile)), float(min_discordance_threshold))

    for r in rows:
        score = r["tree_discordance_score"]
        r["proxy_threshold"] = threshold
        if np.isnan(score) or np.isnan(threshold):
            r["is_reassortant_proxy"] = np.nan
        else:
            r["is_reassortant_proxy"] = int(score >= threshold)

    return pd.DataFrame(rows)


def build_isolate_table(
    case_species: Iterable[str],
    espalier_root: Path,
    species_input_root: Path,
    proxy_quantile: float,
    proxy_min_discordance: float,
    isolate_labels_path: Optional[Path],
    sequence_ld_path: Optional[Path],
) -> pd.DataFrame:
    all_rows: List[pd.DataFrame] = []
    sequence_ld = load_sequence_ld_table(sequence_ld_path, case_species)

    labels_df = pd.DataFrame()
    if isolate_labels_path and isolate_labels_path.exists():
        sep = "\t" if isolate_labels_path.suffix.lower() in {".tsv", ".txt"} else ","
        labels_df = pd.read_csv(isolate_labels_path, sep=sep)
        labels_df.columns = [c.strip() for c in labels_df.columns]
        if "sequence_id" in labels_df.columns and "canonical_isolate" not in labels_df.columns:
            labels_df["canonical_isolate"] = labels_df["sequence_id"].astype(str).str.split("|").str[0]
        if "is_reassortant" not in labels_df.columns:
            raise ValueError(f"isolate label file must contain 'is_reassortant': {isolate_labels_path}")

    for species in case_species:
        species_dir = espalier_root / species
        isolate_matrix = load_isolate_matrix(species, espalier_root)
        if isolate_matrix.empty:
            continue

        per_seg = []
        for seg in ["S", "M", "L"]:
            seg_df = load_species_segment_metadata(species, seg, espalier_root, species_input_root)
            if not seg_df.empty:
                per_seg.append(seg_df)
        if per_seg:
            meta = pd.concat(per_seg, ignore_index=True, sort=False)
            grouped_meta = (
                meta.groupby("canonical_isolate", dropna=False)
                .agg(
                    tip_name=("tip_name", first_valid),
                    date_meta=("date", first_valid),
                    date_precision_meta=("date_precision", first_valid),
                    accession=("accession", first_valid),
                    isolate_raw=("Isolate", first_valid),
                    geo_location=("Geo_Location", first_valid),
                    country=("Country", first_valid),
                    host=("Host", first_valid),
                    collection_date_meta=("Collection_Date", first_valid),
                    nuc_completeness_meta=("Nuc_Completeness", first_valid),
                )
                .reset_index()
            )
        else:
            grouped_meta = pd.DataFrame(columns=["canonical_isolate"])

        combined = isolate_matrix.merge(grouped_meta, on="canonical_isolate", how="left")
        combined["species"] = species
        if "tip_name_x" in combined.columns:
            combined["tip_name"] = combined["tip_name_x"].fillna(combined.get("tip_name_y"))
            combined = combined.drop(columns=[c for c in ["tip_name_x", "tip_name_y"] if c in combined.columns])
        combined["date"] = combined["date"].fillna(combined.get("date_meta"))
        combined["date_precision"] = combined["date_precision"].fillna(combined.get("date_precision_meta"))
        combined["year"] = combined["date"].map(extract_year)
        combined["country"] = combined["country"].map(normalize_text)
        combined["geo_location"] = combined["geo_location"].map(normalize_text)
        combined["host"] = combined["host"].map(normalize_text)
        combined["site_key"] = (
            combined["species"] + "|" + combined["country"] + "|" + combined["geo_location"]
        )
        combined["event_key"] = (
            combined["site_key"]
            + "|"
            + combined["year"].fillna(-1).astype(int).astype(str)
            + "|"
            + combined["host"]
        )

        proxy = compute_tree_discordance(
            species,
            species_dir,
            quantile=proxy_quantile,
            min_discordance_threshold=proxy_min_discordance,
        )
        if not proxy.empty:
            combined = combined.merge(proxy, on=["species", "tip_name"], how="left")
        else:
            combined["tree_discordance_score"] = np.nan
            combined["is_reassortant_proxy"] = np.nan
            combined["proxy_threshold"] = np.nan

        if not labels_df.empty:
            merge_cols = ["canonical_isolate"]
            if "species" in labels_df.columns:
                labels_df["species"] = labels_df["species"].map(slugify_species)
                merge_cols = ["species", "canonical_isolate"]
            true_labels = labels_df.copy()
            combined = combined.merge(
                true_labels[merge_cols + ["is_reassortant"]],
                on=merge_cols,
                how="left",
            )

        if not sequence_ld.empty:
            seq_ld_sp = sequence_ld[sequence_ld["species"] == species].copy()
            if not seq_ld_sp.empty:
                combined = combined.merge(
                    seq_ld_sp,
                    left_on=["species", "tip_name"],
                    right_on=["species", "ld_sequence_id"],
                    how="left",
                )
                combined["ld_match"] = combined["ld_sequence_id"].notna().astype(int)
            else:
                combined["ld_match"] = 0
        else:
            combined["ld_match"] = 0

        all_rows.append(combined)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True, sort=False)


def load_species_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["species"])
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    if "species" not in df.columns:
        return pd.DataFrame(columns=["species"])
    df = df.copy()
    df["species"] = df["species"].map(slugify_species)

    exclude_exact = {
        "is_reassortant",
        "is_reassortant_proxy",
        "is_reassortant_final",
        "reassortment_proxy_mean_spr",
        "reassortment_proxy_mean_spr_per_isolate",
        "target_available",
        "target_available_true",
        "target_proxy_available",
        "target_available_proxy",
        "target_available_final",
        "target_available_main_explanatory",
    }
    keep = ["species"]
    for col in df.columns:
        if col in keep or col in exclude_exact:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            keep.append(col)
            continue

        normalized = series.astype("string").str.strip().str.lower()
        bool_map = {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "1": 1.0,
            "0": 0.0,
        }
        non_missing = normalized[normalized.notna() & (normalized != "")]
        if not non_missing.empty and non_missing.isin(bool_map).all():
            df[col] = normalized.map(bool_map)
            keep.append(col)
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            df[col] = numeric
            keep.append(col)
    out = df[keep].copy()
    out = out.drop_duplicates(subset=["species"], keep="first")
    rename = {c: f"species_feat_{c}" for c in out.columns if c != "species"}
    return out.rename(columns=rename)


def merge_species_features(df: pd.DataFrame, species_features: pd.DataFrame) -> pd.DataFrame:
    if df.empty or species_features.empty or "species" not in df.columns:
        return df

    feature_cols = [c for c in species_features.columns if c != "species"]
    if not feature_cols:
        return df

    out = df.merge(
        species_features[["species"] + feature_cols],
        on="species",
        how="left",
        suffixes=("", "__species_feat"),
    )
    for col in feature_cols:
        incoming = f"{col}__species_feat"
        if incoming not in out.columns:
            continue
        if col in df.columns:
            out[col] = out[col].where(out[col].notna(), out[incoming])
            out = out.drop(columns=[incoming])
        else:
            out = out.rename(columns={incoming: col})
    return out


def annotate_target_availability(
    df: pd.DataFrame,
    *,
    true_label_col: str = "is_reassortant",
    proxy_label_col: str = "is_reassortant_proxy",
) -> pd.DataFrame:
    out = df.copy()
    true_vals = pd.to_numeric(out.get(true_label_col, pd.Series(np.nan, index=out.index)), errors="coerce")
    proxy_vals = pd.to_numeric(out.get(proxy_label_col, pd.Series(np.nan, index=out.index)), errors="coerce")
    out["target_available_true"] = true_vals.notna().astype(int)
    out["target_available_proxy"] = proxy_vals.notna().astype(int)
    out["target_available_final"] = (true_vals.notna() | proxy_vals.notna()).astype(int)
    out["target_available_main_explanatory"] = out["target_available_true"].astype(int)
    out["proxy_only_label"] = (true_vals.isna() & proxy_vals.notna()).astype(int)
    return out


def _entropy(values: pd.Series) -> float:
    counts = values.value_counts(dropna=True)
    if counts.empty:
        return np.nan
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def build_ld_state_features(
    isolate_df: pd.DataFrame,
    group_key: str,
    min_state_count: int,
) -> pd.DataFrame:
    ld_cols = [col for col in isolate_df.columns if col.startswith("ld_site_")]
    if not ld_cols:
        return pd.DataFrame()

    kept_state_map: Dict[str, set] = {}
    for col in ld_cols:
        global_vals = isolate_df[col].astype("string").dropna().astype(str).str.strip()
        global_vals = global_vals[global_vals != ""]
        if global_vals.empty:
            kept_state_map[col] = set()
            continue
        global_counts = global_vals.value_counts()
        kept_state_map[col] = set(global_counts[global_counts >= int(min_state_count)].index.tolist())

    rows: List[Dict[str, object]] = []
    grouped = isolate_df.groupby(group_key, dropna=False)

    for key, sub in grouped:
        row: Dict[str, object] = {group_key: key}
        for col in ld_cols:
            vals = sub[col].astype("string")
            vals = vals.dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            if vals.empty:
                continue

            # Keep only states that are sufficiently represented globally.
            # This avoids unstable dominant-state frequencies from singleton artifacts.
            keep_states = kept_state_map.get(col, set())
            if keep_states:
                vals = vals[vals.isin(keep_states)]
            if vals.empty:
                continue

            for state in sorted(set(vals.tolist())):
                token = normalize_feature_token(state)
                row[f"{col}_has_state_{token}"] = 1.0
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).fillna(0.0)


def build_site_dataset(
    isolate_df: pd.DataFrame,
    species_features: pd.DataFrame,
    group_mode: str,
    min_isolates_per_site: int,
    ld_min_state_count: int,
) -> pd.DataFrame:
    if isolate_df.empty:
        return pd.DataFrame()

    group_key = "site_key" if group_mode == "site" else "event_key"

    def count_true(series: pd.Series) -> float:
        return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())

    grouped = isolate_df.groupby(group_key, dropna=False)
    site_df = grouped.agg(
        species=("species", first_valid),
        country=("country", first_valid),
        geo_location=("geo_location", first_valid),
        host_mode=("host", first_valid),
        n_isolates_total=("canonical_isolate", "nunique"),
        n_complete_isolates=("segment_completeness", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1.0).sum())),
        mean_segment_completeness=("segment_completeness", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        n_hosts_unique=("host", "nunique"),
        host_entropy=("host", _entropy),
        year_min=("year", "min"),
        year_max=("year", "max"),
        n_years_unique=("year", "nunique"),
        mean_tree_discordance=("tree_discordance_score", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        n_proxy_labeled=("is_reassortant_proxy", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        n_reassortants_proxy=("is_reassortant_proxy", count_true),
    ).reset_index()

    ld_cov_col = "ld_sequence_id" if "ld_sequence_id" in isolate_df.columns else None
    if ld_cov_col is None:
        ld_cov = grouped.size().rename("n_isolates_group").reset_index()
        ld_cov["n_ld_labeled"] = 0
        ld_cov["ld_labeled_fraction"] = 0.0
    else:
        ld_cov = grouped.agg(
            n_isolates_group=("canonical_isolate", "nunique"),
            n_ld_labeled=(ld_cov_col, lambda s: int(pd.Series(s).notna().sum())),
        ).reset_index()
        ld_cov["ld_labeled_fraction"] = np.where(
            ld_cov["n_isolates_group"] > 0,
            ld_cov["n_ld_labeled"] / ld_cov["n_isolates_group"],
            0.0,
        )
    site_df = site_df.merge(
        ld_cov[[group_key, "n_ld_labeled", "ld_labeled_fraction"]],
        on=group_key,
        how="left",
    )

    ld_state = build_ld_state_features(
        isolate_df=isolate_df,
        group_key=group_key,
        min_state_count=ld_min_state_count,
    )
    if not ld_state.empty:
        site_df = site_df.merge(ld_state, on=group_key, how="left")

    site_df["n_reassortants_proxy"] = site_df["n_reassortants_proxy"].astype(float)
    site_df["reassortment_proxy_frequency"] = site_df.apply(
        lambda r: r["n_reassortants_proxy"] / r["n_proxy_labeled"] if r["n_proxy_labeled"] > 0 else np.nan,
        axis=1,
    )

    if "is_reassortant" in isolate_df.columns:
        true_group = grouped.agg(
            n_true_labeled=("is_reassortant", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            n_reassortants=("is_reassortant", count_true),
        ).reset_index()
        site_df = site_df.merge(true_group, on=group_key, how="left")
        site_df["reassortment_frequency"] = site_df.apply(
            lambda r: r["n_reassortants"] / r["n_true_labeled"] if r.get("n_true_labeled", 0) and r["n_true_labeled"] > 0 else np.nan,
            axis=1,
        )

    site_df = site_df[site_df["n_isolates_total"] >= int(min_isolates_per_site)].copy()

    if not species_features.empty:
        site_df = site_df.merge(species_features, on="species", how="left")

    ld_feature_cols = [c for c in site_df.columns if c.startswith("ld_site_")]
    if ld_feature_cols:
        site_df[ld_feature_cols] = site_df[ld_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for c in ["n_ld_labeled", "ld_labeled_fraction"]:
        if c in site_df.columns:
            site_df[c] = pd.to_numeric(site_df[c], errors="coerce").fillna(0.0)

    site_df["target_available_true"] = site_df.get("reassortment_frequency", pd.Series(index=site_df.index)).notna().astype(int)
    site_df["target_available_proxy"] = site_df["reassortment_proxy_frequency"].notna().astype(int)
    return site_df


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build case-study sampling-site dataset.")
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
        "--species-feature-table",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "model" / "datasets" / "case_species_model_dataset_proxy.tsv",
    )
    parser.add_argument(
        "--sequence-ld-features",
        type=Path,
        default=REPO_ROOT / "Data" / "ML" / "ld_features" / "sequence_ld_features.csv",
        help="CSV with per-sequence linked-site nucleotide identities.",
    )
    parser.add_argument(
        "--ld-min-state-count",
        type=int,
        default=1,
        help="Minimum isolate count per linked-site state to keep as a site-level frequency feature.",
    )
    parser.add_argument(
        "--isolate-labels",
        type=Path,
        default=None,
        help="Optional true isolate-level labels with is_reassortant column.",
    )
    parser.add_argument(
        "--group-mode",
        choices=["site", "event"],
        default="site",
        help="site: species+country+geo_location; event adds year+host.",
    )
    parser.add_argument("--proxy-quantile", type=float, default=0.90)
    parser.add_argument("--proxy-min-discordance", type=float, default=0.20)
    parser.add_argument("--min-isolates-per-site", type=int, default=3)
    parser.add_argument(
        "--output-isolates",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "model_site" / "datasets" / "isolate_level_with_proxy.tsv",
    )
    parser.add_argument(
        "--output-sites",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "model_site" / "datasets" / "site_level_dataset_proxy.tsv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "model_site" / "datasets" / "site_dataset_build_summary.tsv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_species = read_case_species(args.case_species)

    isolate_df = build_isolate_table(
        case_species=case_species,
        espalier_root=args.espalier_root,
        species_input_root=args.species_input_root,
        proxy_quantile=args.proxy_quantile,
        proxy_min_discordance=args.proxy_min_discordance,
        isolate_labels_path=args.isolate_labels,
        sequence_ld_path=args.sequence_ld_features,
    )
    if isolate_df.empty:
        raise RuntimeError("No isolate records were assembled.")

    species_features = load_species_feature_table(args.species_feature_table)
    isolate_df = merge_species_features(isolate_df, species_features)
    isolate_df = annotate_target_availability(isolate_df)
    site_df = build_site_dataset(
        isolate_df=isolate_df,
        species_features=species_features,
        group_mode=args.group_mode,
        min_isolates_per_site=args.min_isolates_per_site,
        ld_min_state_count=args.ld_min_state_count,
    )

    args.output_isolates.parent.mkdir(parents=True, exist_ok=True)
    args.output_sites.parent.mkdir(parents=True, exist_ok=True)
    isolate_df.to_csv(args.output_isolates, sep="\t", index=False)
    site_df.to_csv(args.output_sites, sep="\t", index=False)

    summary = pd.DataFrame(
        [
            {
                "group_mode": args.group_mode,
                "n_isolates": len(isolate_df),
                "n_sites_rows": len(site_df),
                "n_sites_proxy_target": int(site_df["target_available_proxy"].sum()) if not site_df.empty else 0,
                "n_sites_true_target": int(site_df["target_available_true"].sum()) if "target_available_true" in site_df.columns and not site_df.empty else 0,
                "min_isolates_per_site": args.min_isolates_per_site,
                "ld_min_state_count": args.ld_min_state_count,
                "n_ld_state_features": int(len([c for c in site_df.columns if c.startswith("ld_site_")])),
                "n_sites_with_ld": int((pd.to_numeric(site_df["n_ld_labeled"], errors="coerce").fillna(0) > 0).sum()) if ("n_ld_labeled" in site_df.columns and not site_df.empty) else 0,
                "isolates_output": str(args.output_isolates.resolve()),
                "sites_output": str(args.output_sites.resolve()),
            }
        ]
    )
    summary.to_csv(args.output_summary, sep="\t", index=False)

    print(f"Built isolate table: {args.output_isolates.resolve()} ({len(isolate_df)} rows)")
    print(f"Built site table: {args.output_sites.resolve()} ({len(site_df)} rows)")
    print(f"Site build summary: {args.output_summary.resolve()}")


if __name__ == "__main__":
    main()
