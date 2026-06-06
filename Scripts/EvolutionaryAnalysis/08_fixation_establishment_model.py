from __future__ import annotations

import json
import math
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib.cm import ScalarMappable
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from evo_utils import (
    COMMON_PRIORS,
    DATASET_DIR,
    DISPLAY_NAMES,
    FIGURE_DIR,
    PASTEL,
    REPORT_DIR,
    SPECIES_LABELS,
    SPECIES_ORDER,
    apply_numeric_reference,
    build_base_dataframe,
    load_species_tree,
    numeric_reference,
    prune_tree_to_tips,
    render_dataframe_markdown,
    set_plot_style,
    stable_hash,
    summarize_bayesian_terms,
    write_markdown,
)


BASE_EVENT_FEATURES = [
    "overlap_opportunity_local",
    "arha_prev_local",
    "species_feat_mean_cramers_v_overall",
    "utr5_structural_distance",
    "utr3_structural_distance",
    "terminal_utr5_length_disparity",
    "terminal_utr3_length_disparity",
    "tree_discordance_score",
    "graph_neighbor_overlap",
    "log_sampling_effort",
]
HURDLE_FEATURES = [
    "ecological_opportunity",
    "molecular_permissiveness",
    "ecology_x_molecular_permissiveness",
    "tree_discordance_score",
    "graph_neighbor_overlap",
    "log_sampling_effort",
]
COUNT_FEATURES = [
    "ecological_opportunity",
    "molecular_permissiveness",
    "ecology_x_molecular_permissiveness",
    "tree_discordance_score",
    "graph_neighbor_overlap",
]

ESTABLISHMENT_DRAWS = int(os.environ.get("HANTA_EGT_ESTABLISHMENT_DRAWS", "900"))
ESTABLISHMENT_TUNE = int(os.environ.get("HANTA_EGT_ESTABLISHMENT_TUNE", "900"))
COUNT_DRAWS = int(os.environ.get("HANTA_EGT_COUNT_DRAWS", "900"))
COUNT_TUNE = int(os.environ.get("HANTA_EGT_COUNT_TUNE", "1600"))
EGT_CHAINS = int(os.environ.get("HANTA_EGT_CHAINS", "4"))
EGT_CORES = int(os.environ.get("HANTA_EGT_CORES", str(EGT_CHAINS)))
FOCAL_FEATURES = [
    "ecological_opportunity",
    "molecular_permissiveness",
    "ecology_x_molecular_permissiveness",
    "tree_discordance_score",
    "graph_neighbor_overlap",
]
COMPACT_COMPONENT_FEATURES = [
    "ecological_opportunity",
    "ld_compatibility",
    "species_utr_compatibility",
    "terminal_utr_regime",
    "molecular_permissiveness",
    "ecology_x_molecular_permissiveness",
]
UTR5_STRUCTDIST_COLUMNS = [
    "species_feat_utr5_S_structdist_mean",
    "species_feat_utr5_M_structdist_mean",
    "species_feat_utr5_L_structdist_mean",
]
UTR3_STRUCTDIST_COLUMNS = [
    "species_feat_utr3_S_structdist_mean",
    "species_feat_utr3_M_structdist_mean",
    "species_feat_utr3_L_structdist_mean",
]
TERMINAL_UTR_LENGTH_COLUMNS = [
    "utrterm_S_5p_length",
    "utrterm_S_3p_length",
    "utrterm_M_5p_length",
    "utrterm_M_3p_length",
    "utrterm_L_5p_length",
    "utrterm_L_3p_length",
]
TERMINAL_UTR5_LENGTH_COLUMNS = [
    "utrterm_S_5p_length",
    "utrterm_M_5p_length",
    "utrterm_L_5p_length",
]
TERMINAL_UTR3_LENGTH_COLUMNS = [
    "utrterm_S_3p_length",
    "utrterm_M_3p_length",
    "utrterm_L_3p_length",
]
TERMINAL_UTR_RECOVERY_COLUMNS = [
    "utrterm_S_5p_recovered",
    "utrterm_S_3p_recovered",
    "utrterm_M_5p_recovered",
    "utrterm_M_3p_recovered",
    "utrterm_L_5p_recovered",
    "utrterm_L_3p_recovered",
]
TERMINAL_UTR5_DISPARITY_COLUMNS = [
    "utrterm_SM_len5_absdiff",
    "utrterm_SL_len5_absdiff",
    "utrterm_ML_len5_absdiff",
]
TERMINAL_UTR3_DISPARITY_COLUMNS = [
    "utrterm_SM_len3_absdiff",
    "utrterm_SL_len3_absdiff",
    "utrterm_ML_len3_absdiff",
]
TERMINAL_UTR_WORKHORSE_COLUMNS = (
    TERMINAL_UTR_LENGTH_COLUMNS
    + TERMINAL_UTR_RECOVERY_COLUMNS
    + TERMINAL_UTR5_DISPARITY_COLUMNS
    + TERMINAL_UTR3_DISPARITY_COLUMNS
)
DISPLAY_NAMES_EXTENDED = {
    **DISPLAY_NAMES,
    "ecological_opportunity": "Ecological opportunity",
    "ld_compatibility": "LD compatibility",
    "species_utr_compatibility": "Species UTR compatibility",
    "terminal_utr_regime": "Terminal UTR regime",
    "molecular_permissiveness": "Molecular permissiveness",
    "ecology_x_molecular_permissiveness": "Ecology x molecular permissiveness",
    "utr5_structural_distance": "UTR5 structural distance",
    "utr3_structural_distance": "UTR3 structural distance",
    "terminal_utr5_length_disparity": "Terminal UTR5 length disparity",
    "terminal_utr3_length_disparity": "Terminal UTR3 length disparity",
    "graph_neighbor_overlap": "Graph neighborhood opportunity",
    "log_sampling_effort": "Sampling effort",
}
N_CONTROLS_PER_ORIGIN = 3
RAREFACTION_ITERATIONS = 200
RAREFACTION_ESTABLISHMENT_THRESHOLD = 2


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if not all(np.isfinite([lat1, lon1, lat2, lon2])):
        return np.nan
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


def max_pairwise_geo_span_km(frame: pd.DataFrame) -> float:
    coords = frame[["local_latitude", "local_longitude"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(coords) < 2:
        return 0.0
    values = coords.to_numpy(dtype=float)
    max_dist = 0.0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            dist = haversine_km(values[i, 0], values[i, 1], values[j, 0], values[j, 1])
            if np.isfinite(dist):
                max_dist = max(max_dist, dist)
    return float(max_dist)


def safe_mode(values: Iterable[str]) -> str:
    counts = Counter(str(v) for v in values if pd.notna(v))
    if not counts:
        return "missing"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def mean_available_columns(df: pd.DataFrame, columns: Sequence[str]) -> float:
    available = [col for col in columns if col in df.columns]
    if not available:
        return np.nan
    values = df[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


def add_workhorse_terminal_utr_features(df: pd.DataFrame) -> pd.DataFrame:
    if set(TERMINAL_UTR_WORKHORSE_COLUMNS).issubset(df.columns):
        return df
    scripts_dir = Path(__file__).resolve().parents[1] / "model_test" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from benchmark_sequence_model_suite import add_direct_utr_terminal_features
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Could not import the direct terminal UTR feature generator used by "
            "ecology_molecular_lineage_plus_context / lightgbm."
        ) from exc
    terminal = add_direct_utr_terminal_features(df)
    overlap = [col for col in terminal.columns if col in df.columns]
    base = df.drop(columns=overlap) if overlap else df
    return pd.concat([base, terminal], axis=1)


def summarize_tip_set(df: pd.DataFrame, event_id: str, event_type: str, full_df: pd.DataFrame) -> dict:
    descendant_tip_n = int(len(df))
    years = pd.to_numeric(df["year"], errors="coerce")
    countries = {str(v) for v in df["country"].dropna() if str(v)}
    hosts = {str(v) for v in df["host"].dropna() if str(v)}
    sites = {str(v) for v in df["site_key"].dropna() if str(v)}
    lineage = safe_mode(df["lineage_id"])
    species = str(df["species"].iloc[0])
    species_label = str(df["species_label"].iloc[0])
    lineage_sampling_effort = int(
        full_df[
            (full_df["species"].astype(str) == species)
            & (full_df["lineage_id"].astype(str) == lineage)
        ].shape[0]
    )
    species_sampling_effort = int(full_df[full_df["species"].astype(str) == species].shape[0])
    if lineage_sampling_effort <= 0:
        lineage_sampling_effort = descendant_tip_n
    if species_sampling_effort <= 0:
        species_sampling_effort = descendant_tip_n
    row = {
        "event_id": event_id,
        "event_type": event_type,
        "species": species,
        "species_label": species_label,
        "dominant_lineage_id": lineage,
        "descendant_tip_n": descendant_tip_n,
        "excess_descendant_tip_n": max(descendant_tip_n - 1, 0),
        "year_min": float(years.min()) if years.notna().any() else np.nan,
        "year_max": float(years.max()) if years.notna().any() else np.nan,
        "year_mean": float(years.mean()) if years.notna().any() else np.nan,
        "temporal_span_years": float(years.max() - years.min()) if years.notna().any() else 0.0,
        "host_spread_n": len(hosts),
        "country_spread_n": len(countries),
        "site_spread_n": len(sites),
        "geo_span_km": max_pairwise_geo_span_km(df),
        "countries_json": json.dumps(sorted(countries)),
        "hosts_json": json.dumps(sorted(hosts)),
        "sites_json": json.dumps(sorted(sites)),
        "tip_set_json": json.dumps(sorted(df["tip_name"].astype(str).tolist())),
        "sampling_effort_n": lineage_sampling_effort,
        "lineage_sampling_effort_n": lineage_sampling_effort,
        "species_sampling_effort_n": species_sampling_effort,
        "log_sampling_effort": math.log(max(lineage_sampling_effort, 1)),
        "log_species_sampling_effort": math.log(max(species_sampling_effort, 1)),
    }
    for feature in [
        "overlap_opportunity_local",
        "arha_prev_local",
        "species_feat_mean_cramers_v_overall",
        "tree_discordance_score",
    ]:
        row[feature] = float(pd.to_numeric(df[feature], errors="coerce").mean())
    row["utr5_structural_distance"] = mean_available_columns(df, UTR5_STRUCTDIST_COLUMNS)
    row["utr3_structural_distance"] = mean_available_columns(df, UTR3_STRUCTDIST_COLUMNS)
    row["terminal_utr_length_mean"] = mean_available_columns(df, TERMINAL_UTR_LENGTH_COLUMNS)
    row["terminal_utr5_length_mean"] = mean_available_columns(df, TERMINAL_UTR5_LENGTH_COLUMNS)
    row["terminal_utr3_length_mean"] = mean_available_columns(df, TERMINAL_UTR3_LENGTH_COLUMNS)
    row["terminal_utr_recovery_fraction"] = mean_available_columns(df, TERMINAL_UTR_RECOVERY_COLUMNS)
    row["terminal_utr5_length_disparity"] = mean_available_columns(df, TERMINAL_UTR5_DISPARITY_COLUMNS)
    row["terminal_utr3_length_disparity"] = mean_available_columns(df, TERMINAL_UTR3_DISPARITY_COLUMNS)
    row["terminal_utr_workhorse_feature_count"] = float(
        sum(col in df.columns for col in TERMINAL_UTR_WORKHORSE_COLUMNS)
    )
    for col in TERMINAL_UTR_WORKHORSE_COLUMNS:
        if col in df.columns:
            row[f"event_mean_{col}"] = float(pd.to_numeric(df[col], errors="coerce").mean())
    return row


def lineage_graph_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows = []
    for lineage_id, sub in df.groupby("lineage_id", sort=False):
        years = pd.to_numeric(sub["year"], errors="coerce")
        node_rows.append(
            {
                "lineage_id": lineage_id,
                "species": str(sub["species"].iloc[0]),
                "species_label": str(sub["species_label"].iloc[0]),
                "n_tips": int(len(sub)),
                "positive_fraction": float(pd.to_numeric(sub["response"], errors="coerce").mean()),
                "mean_overlap": float(pd.to_numeric(sub["overlap_opportunity_local"], errors="coerce").mean()),
                "mean_prevalence": float(pd.to_numeric(sub["arha_prev_local"], errors="coerce").mean()),
                "mean_ld_burden": float(pd.to_numeric(sub["species_feat_mean_cramers_v_overall"], errors="coerce").mean()),
                "mean_utr5_structural_distance": mean_available_columns(sub, UTR5_STRUCTDIST_COLUMNS),
                "mean_utr3_structural_distance": mean_available_columns(sub, UTR3_STRUCTDIST_COLUMNS),
                "mean_terminal_utr5_length_disparity": mean_available_columns(sub, TERMINAL_UTR5_DISPARITY_COLUMNS),
                "mean_terminal_utr3_length_disparity": mean_available_columns(sub, TERMINAL_UTR3_DISPARITY_COLUMNS),
                "mean_terminal_utr_recovery_fraction": mean_available_columns(sub, TERMINAL_UTR_RECOVERY_COLUMNS),
                "mean_tree_discordance": float(pd.to_numeric(sub["tree_discordance_score"], errors="coerce").mean()),
                "year_mean": float(years.mean()) if years.notna().any() else np.nan,
                "lat_mean": float(pd.to_numeric(sub["local_latitude"], errors="coerce").mean()),
                "lon_mean": float(pd.to_numeric(sub["local_longitude"], errors="coerce").mean()),
                "countries": {str(v) for v in sub["country"].dropna() if str(v)},
                "hosts": {str(v) for v in sub["host"].dropna() if str(v)},
            }
        )
    nodes = pd.DataFrame(node_rows)
    edge_rows = []
    graph_rows = []
    for species, sp_nodes in nodes.groupby("species", sort=False):
        sp_nodes = sp_nodes.reset_index(drop=True)
        weighted_overlap = {row.lineage_id: 0.0 for row in sp_nodes.itertuples(index=False)}
        weighted_degree = {row.lineage_id: 0.0 for row in sp_nodes.itertuples(index=False)}
        for i in range(len(sp_nodes)):
            left = sp_nodes.iloc[i]
            for j in range(i + 1, len(sp_nodes)):
                right = sp_nodes.iloc[j]
                country_weight = jaccard(left["countries"], right["countries"])
                host_weight = jaccard(left["hosts"], right["hosts"])
                year_gap = abs(float(left["year_mean"]) - float(right["year_mean"])) if np.isfinite(left["year_mean"]) and np.isfinite(right["year_mean"]) else np.nan
                time_weight = math.exp(-year_gap / 10.0) if np.isfinite(year_gap) else 0.0
                geo_dist = haversine_km(float(left["lat_mean"]), float(left["lon_mean"]), float(right["lat_mean"]), float(right["lon_mean"]))
                geo_weight = math.exp(-geo_dist / 1000.0) if np.isfinite(geo_dist) else 0.0
                weight = (0.35 * country_weight) + (0.25 * host_weight) + (0.20 * time_weight) + (0.20 * geo_weight)
                if weight <= 0:
                    continue
                edge_rows.append(
                    {
                        "species": species,
                        "species_label": SPECIES_LABELS[species],
                        "left_lineage_id": left["lineage_id"],
                        "right_lineage_id": right["lineage_id"],
                        "edge_weight": float(weight),
                        "country_jaccard": float(country_weight),
                        "host_jaccard": float(host_weight),
                        "time_weight": float(time_weight),
                        "geo_weight": float(geo_weight),
                    }
                )
                weighted_degree[left["lineage_id"]] += weight
                weighted_degree[right["lineage_id"]] += weight
                if np.isfinite(right["mean_overlap"]):
                    weighted_overlap[left["lineage_id"]] += weight * float(right["mean_overlap"])
                if np.isfinite(left["mean_overlap"]):
                    weighted_overlap[right["lineage_id"]] += weight * float(left["mean_overlap"])
        for row in sp_nodes.itertuples(index=False):
            degree = weighted_degree[row.lineage_id]
            own_overlap = row.mean_overlap if np.isfinite(row.mean_overlap) else 0.0
            neighbor_overlap = weighted_overlap[row.lineage_id] / degree if degree > 0 else own_overlap
            graph_rows.append(
                {
                    "lineage_id": row.lineage_id,
                    "species": row.species,
                    "species_label": row.species_label,
                    "graph_weighted_degree": float(degree),
                    "graph_neighbor_overlap": float(neighbor_overlap),
                }
            )
    node_export = nodes.drop(columns=["countries", "hosts"]).merge(pd.DataFrame(graph_rows), on=["lineage_id", "species", "species_label"], how="left")
    return node_export, pd.DataFrame(edge_rows)


def denest_origin_summary(origin_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep minimal ARG-supported origin clusters within each species."""
    if origin_summary.empty:
        return origin_summary.copy(), pd.DataFrame()

    records = []
    for row in origin_summary.itertuples(index=False):
        tips = parse_json_set(row.tip_set_json)
        records.append(
            {
                **row._asdict(),
                "tip_set": tips,
                "computed_positive_tip_n": len(tips),
            }
        )

    kept_records: list[dict] = []
    manifest_rows: list[dict] = []
    for species in SPECIES_ORDER:
        species_records = [rec for rec in records if str(rec["species"]) == species]
        if not species_records:
            continue
        species_records = sorted(
            species_records,
            key=lambda rec: (
                int(rec["computed_positive_tip_n"]),
                int(rec.get("all_descendant_n", 0)),
                float(rec.get("arg_time", 0.0)),
                str(rec["origin_cluster_id"]),
            ),
        )
        species_kept: list[dict] = []
        for rec in species_records:
            tips = rec["tip_set"]
            parent = next(
                (
                    kept
                    for kept in species_kept
                    if kept["tip_set"] < tips
                ),
                None,
            )
            keep = parent is None
            if keep:
                species_kept.append(rec)
                kept_records.append(rec)
            manifest_rows.append(
                {
                    "origin_cluster_id": rec["origin_cluster_id"],
                    "species": rec["species"],
                    "species_label": rec["species_label"],
                    "positive_tip_n": int(rec["positive_tip_n"]),
                    "all_descendant_n": int(rec["all_descendant_n"]),
                    "arg_node_id": int(rec["arg_node_id"]),
                    "arg_time": float(rec["arg_time"]),
                    "tip_set_json": json.dumps(sorted(tips)),
                    "denested_keep": int(keep),
                    "denesting_reason": "minimal_arg_supported_cluster" if keep else "nested_superset_removed",
                    "minimal_parent_origin_id": None if parent is None else parent["origin_cluster_id"],
                    "minimal_parent_positive_tip_n": None if parent is None else int(parent["computed_positive_tip_n"]),
                }
            )

    retained = pd.DataFrame(kept_records)
    manifest = pd.DataFrame(manifest_rows)
    if retained.empty:
        return origin_summary.iloc[0:0].copy(), manifest
    retained = retained.drop(columns=["tip_set", "computed_positive_tip_n"])
    return retained.reset_index(drop=True), manifest.sort_values(["species", "denested_keep", "positive_tip_n", "origin_cluster_id"]).reset_index(drop=True)


def positive_origin_events(df: pd.DataFrame, origin_summary: pd.DataFrame, graph_nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin in origin_summary.itertuples(index=False):
        tips = parse_json_set(origin.tip_set_json)
        sub = df[(df["species"] == origin.species) & (df["tip_name"].astype(str).isin(tips))].copy()
        if sub.empty:
            continue
        row = summarize_tip_set(sub, origin.origin_cluster_id, "arg_candidate_reassortment_origin", full_df=df)
        row["arg_node_id"] = int(origin.arg_node_id)
        row["arg_time"] = float(origin.arg_time)
        row["arg_positive_tip_n"] = int(origin.positive_tip_n)
        row["arg_all_descendant_n"] = int(origin.all_descendant_n)
        rows.append(row)
    events = pd.DataFrame(rows)
    return events.merge(
        graph_nodes[["lineage_id", "graph_neighbor_overlap", "graph_weighted_degree"]],
        left_on="dominant_lineage_id",
        right_on="lineage_id",
        how="left",
    ).drop(columns=["lineage_id"])


def nonreassortant_internal_nodes(df: pd.DataFrame, graph_nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for species in SPECIES_ORDER:
        sp = df[df["species"] == species].copy()
        if sp.empty:
            continue
        keep_tips = set(sp["tip_name"].astype(str))
        tree = prune_tree_to_tips(load_species_tree(species), keep_tips)
        for clade in tree.find_clades(order="postorder"):
            descendants = sorted(t.name for t in clade.get_terminals() if t.name in keep_tips)
            if len(descendants) < 2:
                continue
            sub = sp[sp["tip_name"].astype(str).isin(descendants)].copy()
            if sub.empty or int(pd.to_numeric(sub["response"], errors="coerce").fillna(0).sum()) > 0:
                continue
            event_id = f"{species}|background_{stable_hash(descendants)}"
            rows.append(summarize_tip_set(sub, event_id, "matched_background", full_df=df))
    controls = pd.DataFrame(rows)
    if controls.empty:
        return controls
    return controls.merge(
        graph_nodes[["lineage_id", "graph_neighbor_overlap", "graph_weighted_degree"]],
        left_on="dominant_lineage_id",
        right_on="lineage_id",
        how="left",
    ).drop(columns=["lineage_id"])


def parse_json_set(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return set()
    return {str(v) for v in parsed}


def match_controls(origins: pd.DataFrame, controls: pd.DataFrame, n_controls: int = N_CONTROLS_PER_ORIGIN) -> pd.DataFrame:
    matched_rows = []
    controls = controls.copy()
    controls["country_set"] = controls["countries_json"].map(parse_json_set)
    controls["host_set"] = controls["hosts_json"].map(parse_json_set)
    for origin in origins.itertuples(index=False):
        candidates = controls[controls["species"] == origin.species].copy()
        if candidates.empty:
            continue
        origin_countries = parse_json_set(origin.countries_json)
        origin_hosts = parse_json_set(origin.hosts_json)
        candidates["same_lineage"] = (candidates["dominant_lineage_id"] == origin.dominant_lineage_id).astype(int)
        candidates["log_size_distance"] = np.abs(np.log1p(candidates["descendant_tip_n"]) - math.log1p(origin.descendant_tip_n))
        candidates["year_distance"] = np.abs(pd.to_numeric(candidates["year_mean"], errors="coerce") - float(origin.year_mean)).fillna(20.0)
        candidates["country_distance"] = 1.0 - candidates["country_set"].map(lambda s: jaccard(origin_countries, s))
        candidates["host_distance"] = 1.0 - candidates["host_set"].map(lambda s: jaccard(origin_hosts, s))
        candidates["sampling_distance"] = np.abs(
            np.log1p(pd.to_numeric(candidates["sampling_effort_n"], errors="coerce"))
            - math.log1p(float(origin.sampling_effort_n))
        ).fillna(0.0)
        candidates["terminal_utr_recovery_distance"] = np.abs(
            pd.to_numeric(candidates["terminal_utr_recovery_fraction"], errors="coerce")
            - float(origin.terminal_utr_recovery_fraction)
        ).fillna(0.0)
        candidates["match_score"] = (
            (1 - candidates["same_lineage"]) * 2.0
            + candidates["log_size_distance"]
            + 0.05 * candidates["year_distance"]
            + 0.5 * candidates["country_distance"]
            + 0.25 * candidates["host_distance"]
            + 0.25 * candidates["sampling_distance"]
            + 0.5 * candidates["terminal_utr_recovery_distance"]
        )
        chosen = candidates.sort_values(["match_score", "log_size_distance", "year_distance", "event_id"]).head(n_controls).copy()
        chosen["matched_origin_id"] = origin.event_id
        matched_rows.append(chosen.drop(columns=["country_set", "host_set"]))
    if not matched_rows:
        return pd.DataFrame()
    return pd.concat(matched_rows, ignore_index=True)


def standardize_columns(out: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    ref = {}
    for feature in features:
        series = pd.to_numeric(out[feature], errors="coerce")
        median = float(series.median()) if series.notna().any() else 0.0
        filled = series.fillna(median)
        std = float(filled.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        mean = float(filled.mean())
        out[f"{feature}_z"] = (filled - mean) / std
        out[f"{feature}_missing"] = series.isna().astype(int)
        ref[feature] = {"median": median, "mean": mean, "std": std}
    return out, ref


def add_compact_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def zmean(columns: Sequence[str], signs: Sequence[float] | None = None) -> pd.Series:
        signs = list(signs or [1.0] * len(columns))
        values = []
        for col, sign in zip(columns, signs):
            values.append(sign * pd.to_numeric(out[col], errors="coerce"))
        return pd.concat(values, axis=1).mean(axis=1)

    out["ecological_opportunity"] = zmean(
        ["overlap_opportunity_local_z", "arha_prev_local_z"],
    )
    out["ld_compatibility"] = -pd.to_numeric(out["species_feat_mean_cramers_v_overall_z"], errors="coerce")
    out["species_utr_compatibility"] = -zmean(
        ["utr5_structural_distance_z", "utr3_structural_distance_z"],
    )
    out["terminal_utr_regime"] = zmean(
        ["terminal_utr5_length_disparity_z", "terminal_utr3_length_disparity_z"],
        signs=[-1.0, 1.0],
    )
    out["molecular_permissiveness"] = zmean(
        ["ld_compatibility", "species_utr_compatibility", "terminal_utr_regime"],
    )
    out["ecology_x_molecular_permissiveness"] = (
        pd.to_numeric(out["ecological_opportunity"], errors="coerce")
        * pd.to_numeric(out["molecular_permissiveness"], errors="coerce")
    )
    return out


def add_standardized_event_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = df.copy()
    out, base_ref = standardize_columns(out, BASE_EVENT_FEATURES)
    out = add_compact_strategy_features(out)
    out, compact_ref = standardize_columns(out, COMPACT_COMPONENT_FEATURES)
    ref = {
        **{f"base::{key}": value for key, value in base_ref.items()},
        **{f"compact::{key}": value for key, value in compact_ref.items()},
    }
    return out, ref


def fit_establishment_model(df: pd.DataFrame):
    terms = [f"{feature}_z" for feature in HURDLE_FEATURES]
    formula = "established ~ " + " + ".join(terms) + " + (1|species) + (1|dominant_lineage_id)"
    model = bmb.Model(formula, df, family="bernoulli", priors=COMMON_PRIORS)
    idata = model.fit(
        draws=ESTABLISHMENT_DRAWS,
        tune=ESTABLISHMENT_TUNE,
        chains=EGT_CHAINS,
        cores=EGT_CORES,
        target_accept=0.99,
        random_seed=808,
        idata_kwargs={"log_likelihood": True},
    )
    return model, idata, formula, terms


def fit_count_model(df: pd.DataFrame, terms: Sequence[str]):
    count_df = df[df["established"] == 1].copy().reset_index(drop=True)
    X = count_df[list(terms)].to_numpy(dtype=float)
    y = count_df["excess_descendant_tip_n"].astype(int).to_numpy()
    offset = np.log(np.maximum(count_df["lineage_sampling_effort_n"].astype(float).to_numpy(), 1.0))
    with pm.Model(coords={"term": list(terms), "event": np.arange(len(count_df))}) as model:
        intercept = pm.Normal("Intercept", mu=0.0, sigma=1.5)
        beta = pm.Normal("beta", mu=0.0, sigma=0.75, dims="term")
        eta = intercept + pm.math.dot(X, beta) + offset
        mu = pm.math.exp(eta)
        alpha = pm.Exponential("alpha", lam=1.0)
        pm.NegativeBinomial("excess_descendant_tip_n", mu=mu, alpha=alpha, observed=y, dims="event")
        idata = pm.sample(
            draws=COUNT_DRAWS,
            tune=COUNT_TUNE,
            chains=EGT_CHAINS,
            cores=EGT_CORES,
            target_accept=0.995,
            random_seed=809,
            idata_kwargs={"log_likelihood": True},
        )
    return model, idata, count_df


def summarize_count_model(idata, terms: Sequence[str]) -> pd.DataFrame:
    summary = az.summary(idata, var_names=["beta"], kind="stats", hdi_prob=0.94)
    rows = []
    beta = np.asarray(idata.posterior["beta"])
    for idx, term in enumerate(terms):
        label = f"beta[{term}]"
        if label not in summary.index:
            label = summary.index[idx]
        values = beta[:, :, idx].reshape(-1)
        rows.append(
            {
                "model": "count_expansion_negative_binomial",
                "term": term,
                "mean": float(summary.loc[label, "mean"]),
                "sd": float(summary.loc[label, "sd"]),
                "hdi_3%": float(summary.loc[label, "hdi_3%"]),
                "hdi_97%": float(summary.loc[label, "hdi_97%"]),
                "posterior_prob_positive": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def model_diagnostics(idata, model_name: str) -> dict:
    diagnostics = az.summary(idata, kind="diagnostics")
    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    return {
        "model": model_name,
        "max_rhat": float(diagnostics["r_hat"].max()),
        "min_ess_bulk": float(diagnostics["ess_bulk"].min()),
        "divergences": divergences,
    }


def _rarefied_descendant_count(row: pd.Series, target_n: int, rng: np.random.Generator) -> int:
    species_effort = int(max(float(row.get("species_sampling_effort_n", 0)), 0))
    observed = int(max(float(row.get("descendant_tip_n", 0)), 0))
    if species_effort <= 0 or observed <= 0:
        return 0
    observed = min(observed, species_effort)
    sampled_n = min(int(target_n), species_effort)
    if sampled_n <= 0:
        return 0
    return int(rng.hypergeometric(observed, max(species_effort - observed, 0), sampled_n))


def rarefaction_targets(origin_events: pd.DataFrame) -> list[tuple[str, int]]:
    efforts = (
        origin_events[["species", "species_sampling_effort_n"]]
        .drop_duplicates()
        ["species_sampling_effort_n"]
        .astype(int)
        .to_numpy()
    )
    if len(efforts) == 0:
        return []
    strict = int(np.min(efforts))
    moderate = int(np.quantile(efforts, 0.25))
    targets = [("strict_species_min", strict)]
    if moderate != strict:
        targets.append(("species_q25", moderate))
    return targets


def run_rarefaction_sensitivity(
    matched_events: pd.DataFrame,
    origin_events: pd.DataFrame,
    iterations: int = RAREFACTION_ITERATIONS,
    seed: int = 810,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = rarefaction_targets(origin_events)
    if not targets:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    hurdle_terms = [f"{feature}_z" for feature in HURDLE_FEATURES]
    count_terms = [f"{feature}_z" for feature in COUNT_FEATURES]
    rows = []

    for target_label, target_n in targets:
        for iteration in range(iterations):
            rare = matched_events.copy()
            rare["rarefaction_target_n"] = int(target_n)
            rare["rarefied_descendant_tip_n"] = [
                _rarefied_descendant_count(row, target_n, rng)
                if int(row.get("established", 0)) == 1
                else 0
                for _, row in rare.iterrows()
            ]
            rare["rarefied_established"] = (
                (rare["established"].astype(int) == 1)
                & (rare["rarefied_descendant_tip_n"] >= RAREFACTION_ESTABLISHMENT_THRESHOLD)
            ).astype(int)
            rare["rarefied_excess_descendant_tip_n"] = np.maximum(rare["rarefied_descendant_tip_n"] - 1, 0)
            established_origin_n = int(
                rare.loc[rare["event_type"].eq("arg_candidate_reassortment_origin"), "rarefied_established"].sum()
            )
            origin_retention_fraction = float(
                rare.loc[rare["event_type"].eq("arg_candidate_reassortment_origin"), "rarefied_established"].mean()
            )

            rare_hurdle = rare[
                rare["event_type"].eq("matched_background")
                | (
                    rare["event_type"].eq("arg_candidate_reassortment_origin")
                    & rare["rarefied_established"].eq(1)
                )
            ].copy()
            y = rare_hurdle["established"].astype(int).to_numpy()
            if len(np.unique(y)) > 1:
                X = rare_hurdle[hurdle_terms].to_numpy(dtype=float)
                clf = LogisticRegression(
                    penalty="l2",
                    C=0.5,
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    clf.fit(X, y)
                for term, coef in zip(hurdle_terms, clf.coef_[0]):
                    rows.append(
                        {
                            "rarefaction_target": target_label,
                            "target_sample_n": int(target_n),
                            "iteration": iteration,
                            "model": "rarefied_hurdle_logistic_l2",
                            "term": term,
                            "coefficient": float(coef),
                            "established_origin_n": established_origin_n,
                            "origin_retention_fraction": origin_retention_fraction,
                        }
                    )

            count_df = rare[(rare["event_type"].eq("arg_candidate_reassortment_origin")) & (rare["rarefied_established"] == 1)].copy()
            if len(count_df) >= len(count_terms) + 5 and count_df["rarefied_excess_descendant_tip_n"].nunique() > 1:
                Xc = count_df[count_terms].to_numpy(dtype=float)
                yc = count_df["rarefied_excess_descendant_tip_n"].astype(float).to_numpy()
                reg = PoissonRegressor(alpha=1.0, max_iter=5000)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    reg.fit(Xc, yc)
                for term, coef in zip(count_terms, reg.coef_):
                    rows.append(
                        {
                            "rarefaction_target": target_label,
                            "target_sample_n": int(target_n),
                            "iteration": iteration,
                            "model": "rarefied_count_poisson_l2",
                            "term": term,
                            "coefficient": float(coef),
                            "established_origin_n": established_origin_n,
                            "origin_retention_fraction": origin_retention_fraction,
                        }
                    )

    iteration_coefficients = pd.DataFrame(rows)
    if iteration_coefficients.empty:
        return iteration_coefficients, pd.DataFrame()
    summary = (
        iteration_coefficients.groupby(["rarefaction_target", "target_sample_n", "model", "term"], as_index=False)
        .agg(
            coefficient_mean=("coefficient", "mean"),
            coefficient_sd=("coefficient", "std"),
            sign_stability_positive=("coefficient", lambda x: float((pd.Series(x) > 0).mean())),
            iterations_n=("iteration", "nunique"),
            mean_established_origin_n=("established_origin_n", "mean"),
            median_established_origin_n=("established_origin_n", "median"),
            mean_origin_retention_fraction=("origin_retention_fraction", "mean"),
        )
        .sort_values(["rarefaction_target", "model", "term"])
    )
    return iteration_coefficients, summary


def plot_first_pass(origin_events: pd.DataFrame, matched: pd.DataFrame, establishment_summary: pd.DataFrame, count_summary: pd.DataFrame) -> None:
    set_plot_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    species_order = [label for label in [SPECIES_LABELS[s] for s in SPECIES_ORDER] if label in set(origin_events["species_label"])]
    data = [origin_events.loc[origin_events["species_label"] == label, "descendant_tip_n"].to_numpy() for label in species_order]
    violins = axes[0].violinplot(data, showmeans=False, showextrema=False, widths=0.8)
    palette = ["#f2c6d8", "#c6def1", "#d4f0c0", "#f7d6b6", "#fde2e4", "#e2ece9"]
    for body, color in zip(violins["bodies"], palette):
        body.set_facecolor(color)
        body.set_edgecolor("#444444")
        body.set_alpha(0.9)
    axes[0].set_xticks(np.arange(1, len(species_order) + 1))
    axes[0].set_xticklabels(species_order, rotation=25, ha="right")
    axes[0].set_ylabel("Descendant reassortant tips per origin")

    y_terms = [f"{feature}_z" for feature in FOCAL_FEATURES]
    est = establishment_summary[establishment_summary["term"].isin(y_terms)].copy()
    cnt = count_summary[count_summary["term"].isin(y_terms)].copy()
    for ax, table, color, xlabel in [
        (axes[1], est, "#6d597a", "Establishment log-odds"),
        (axes[2], cnt, "#577590", "Expansion log count"),
    ]:
        for _, row in table.iterrows():
            ypos = y_terms.index(row["term"])
            ax.hlines(y=ypos, xmin=row["hdi_3%"], xmax=row["hdi_97%"], color=color, linewidth=2.4)
            ax.scatter(row["mean"], ypos, color=color, edgecolors="#333333", linewidths=0.35, s=36)
        ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
        ax.set_yticks(np.arange(len(y_terms)))
        ax.set_yticklabels([DISPLAY_NAMES_EXTENDED[t.removesuffix("_z")] for t in y_terms])
        ax.set_xlabel(xlabel)
    fig.savefig(FIGURE_DIR / "08_fixation_establishment_first_pass.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _logistic(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _posterior_draws(idata, name: str, max_draws: int = 1400) -> np.ndarray:
    values = np.asarray(idata.posterior[name]).reshape(-1)
    if len(values) <= max_draws:
        return values
    idx = np.linspace(0, len(values) - 1, max_draws).astype(int)
    return values[idx]


def _compact_raw_from_z(z_values: np.ndarray, feature: str, ref: dict[str, dict[str, float]]) -> np.ndarray:
    scale = ref[f"compact::{feature}"]
    return z_values * float(scale["std"]) + float(scale["mean"])


def _compact_z_from_raw(raw_values: np.ndarray, feature: str, ref: dict[str, dict[str, float]]) -> np.ndarray:
    scale = ref[f"compact::{feature}"]
    return (raw_values - float(scale["mean"])) / float(scale["std"])


def predict_establishment_fixed_effects(
    idata,
    ecology_z: np.ndarray,
    molecular_z: np.ndarray,
    ref: dict[str, dict[str, float]],
    tree_z: float = 0.0,
    graph_z: float = 0.0,
    sampling_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ecology_z = np.asarray(ecology_z, dtype=float)
    molecular_z = np.asarray(molecular_z, dtype=float)
    raw_ecology = _compact_raw_from_z(ecology_z, "ecological_opportunity", ref)
    raw_molecular = _compact_raw_from_z(molecular_z, "molecular_permissiveness", ref)
    interaction_z = _compact_z_from_raw(
        raw_ecology * raw_molecular,
        "ecology_x_molecular_permissiveness",
        ref,
    )
    eta = (
        _posterior_draws(idata, "Intercept")[:, None]
        + _posterior_draws(idata, "ecological_opportunity_z")[:, None] * ecology_z.reshape(1, -1)
        + _posterior_draws(idata, "molecular_permissiveness_z")[:, None] * molecular_z.reshape(1, -1)
        + _posterior_draws(idata, "ecology_x_molecular_permissiveness_z")[:, None] * interaction_z.reshape(1, -1)
        + _posterior_draws(idata, "tree_discordance_score_z")[:, None] * float(tree_z)
        + _posterior_draws(idata, "graph_neighbor_overlap_z")[:, None] * float(graph_z)
        + _posterior_draws(idata, "log_sampling_effort_z")[:, None] * float(sampling_z)
    )
    probs = _logistic(eta)
    return (
        np.quantile(probs, 0.50, axis=0),
        np.quantile(probs, 0.03, axis=0),
        np.quantile(probs, 0.97, axis=0),
    )


def plot_strategy_landscape(
    origin_events: pd.DataFrame,
    matched_events: pd.DataFrame,
    denesting_manifest: pd.DataFrame,
    rarefaction_summary: pd.DataFrame,
    establishment_idata,
    event_ref: dict[str, dict[str, float]],
) -> None:
    set_plot_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    surface_cmap = LinearSegmentedColormap.from_list(
        "payoff_surface",
        ["#f4f7f8", "#dbe8f1", "#ded7ee", "#efc2ca", "#d8879c"],
    )
    div_cmap = LinearSegmentedColormap.from_list(
        "soft_diverging",
        ["#8fb8cc", "#f3f4f2", "#d8879c"],
    )
    fig = plt.figure(figsize=(10.8, 7.4), constrained_layout=False)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.15, 1.15, 0.95],
        height_ratios=[1.08, 1.08, 0.86],
        left=0.07,
        right=0.96,
        top=0.96,
        bottom=0.10,
        wspace=0.34,
        hspace=0.42,
    )
    ax_surface = fig.add_subplot(gs[0:2, 0:2])
    ax_curves = fig.add_subplot(gs[0:2, 2])
    ax_denest = fig.add_subplot(gs[2, 0:2])
    ax_rare = fig.add_subplot(gs[2, 2])

    x = pd.to_numeric(matched_events["ecological_opportunity_z"], errors="coerce")
    y = pd.to_numeric(matched_events["molecular_permissiveness_z"], errors="coerce")
    xlim = np.nanpercentile(x, [2, 98])
    ylim = np.nanpercentile(y, [2, 98])
    xpad = max(0.15, 0.08 * float(xlim[1] - xlim[0]))
    ypad = max(0.15, 0.08 * float(ylim[1] - ylim[0]))
    xlim = np.array([xlim[0] - xpad, xlim[1] + xpad])
    ylim = np.array([ylim[0] - ypad, ylim[1] + ypad])

    xx = np.linspace(xlim[0], xlim[1], 90)
    yy = np.linspace(ylim[0], ylim[1], 90)
    grid_x, grid_y = np.meshgrid(xx, yy)
    surface, _, _ = predict_establishment_fixed_effects(
        establishment_idata,
        grid_x.ravel(),
        grid_y.ravel(),
        event_ref,
    )
    surface = surface.reshape(grid_x.shape)
    levels = np.linspace(0.05, 0.95, 19)
    cf = ax_surface.contourf(grid_x, grid_y, surface, levels=levels, cmap=surface_cmap, extend="both")
    ax_surface.contour(grid_x, grid_y, surface, levels=[0.50], colors=["#676767"], linewidths=0.9, alpha=0.9)

    controls = matched_events[matched_events["established"].astype(int).eq(0)].copy()
    origins = matched_events[matched_events["established"].astype(int).eq(1)].copy()
    ax_surface.scatter(
        controls["ecological_opportunity_z"],
        controls["molecular_permissiveness_z"],
        s=18,
        facecolors="none",
        edgecolors="#9c9c9c",
        linewidths=0.45,
        alpha=0.42,
        zorder=3,
    )
    for species_label, sub in origins.groupby("species_label", sort=False):
        color = PASTEL.get(species_label, "#c7c7c7")
        sizes = 28.0 + 15.0 * np.log1p(pd.to_numeric(sub["descendant_tip_n"], errors="coerce").fillna(1.0))
        ax_surface.scatter(
            sub["ecological_opportunity_z"],
            sub["molecular_permissiveness_z"],
            s=sizes,
            color=color,
            edgecolors="#3f3f3f",
            linewidths=0.45,
            alpha=0.95,
            zorder=4,
            label=species_label,
        )
    ax_surface.set_xlabel("Ecological opportunity")
    ax_surface.set_ylabel("Molecular permissiveness")
    ax_surface.set_xlim(xlim)
    ax_surface.set_ylim(ylim)
    ax_surface.text(-0.07, 1.04, "A", transform=ax_surface.transAxes, fontsize=13, fontweight="bold", va="top")
    cbar = fig.colorbar(cf, ax=ax_surface, fraction=0.034, pad=0.018)
    cbar.set_label("Pr(establishment)")
    cbar.outline.set_visible(False)

    curve_x = np.linspace(xlim[0], xlim[1], 140)
    mol_levels = np.nanquantile(y, [0.15, 0.50, 0.85])
    curve_specs = [
        ("restrictive", mol_levels[0], "#8fb8cc"),
        ("median", mol_levels[1], "#8f8f8f"),
        ("permissive", mol_levels[2], "#d8879c"),
    ]
    for label, mol_value, color in curve_specs:
        median, lo, hi = predict_establishment_fixed_effects(
            establishment_idata,
            curve_x,
            np.full_like(curve_x, mol_value),
            event_ref,
        )
        ax_curves.fill_between(curve_x, lo, hi, color=color, alpha=0.16, linewidth=0)
        ax_curves.plot(curve_x, median, color=color, linewidth=2.0, label=label)
    ax_curves.set_xlabel("Ecological opportunity")
    ax_curves.set_ylabel("Pr(establishment)")
    ax_curves.set_xlim(xlim)
    ax_curves.set_ylim(0, min(1.0, max(0.65, ax_curves.get_ylim()[1])))
    ax_curves.legend(frameon=False, loc="upper left", handlelength=1.6, borderpad=0.1)
    ax_curves.text(-0.18, 1.04, "B", transform=ax_curves.transAxes, fontsize=13, fontweight="bold", va="top")

    if not denesting_manifest.empty:
        denest = (
            denesting_manifest.groupby("species_label", as_index=False)
            .agg(
                candidates=("origin_cluster_id", "nunique"),
                retained=("denested_keep", "sum"),
            )
        )
        order = [SPECIES_LABELS[s] for s in SPECIES_ORDER if SPECIES_LABELS[s] in set(denest["species_label"])]
        denest["species_label"] = pd.Categorical(denest["species_label"], categories=order, ordered=True)
        denest = denest.sort_values("species_label").reset_index(drop=True)
        ypos = np.arange(len(denest))
        for idx, row in denest.iterrows():
            color = PASTEL.get(str(row["species_label"]), "#c7c7c7")
            ax_denest.hlines(idx, row["retained"], row["candidates"], color="#b5b5b5", linewidth=1.0, zorder=1)
            ax_denest.scatter(row["candidates"], idx, s=42, facecolors="white", edgecolors="#7e7e7e", linewidths=0.9, zorder=2)
            ax_denest.scatter(row["retained"], idx, s=52, color=color, edgecolors="#3f3f3f", linewidths=0.45, zorder=3)
        ax_denest.set_yticks(ypos)
        ax_denest.set_yticklabels(denest["species_label"])
        ax_denest.invert_yaxis()
        ax_denest.set_xlabel("ARG origin clusters")
        ax_denest.set_xlim(0, max(denest["candidates"].max() + 3, 6))
    ax_denest.text(-0.07, 1.11, "C", transform=ax_denest.transAxes, fontsize=13, fontweight="bold", va="top")

    hurdle_rare = rarefaction_summary[
        rarefaction_summary["model"].eq("rarefied_hurdle_logistic_l2")
        & rarefaction_summary["term"].isin(
            [
                "ecological_opportunity_z",
                "molecular_permissiveness_z",
                "ecology_x_molecular_permissiveness_z",
                "tree_discordance_score_z",
            ]
        )
    ].copy()
    term_order = [
        "ecological_opportunity_z",
        "molecular_permissiveness_z",
        "ecology_x_molecular_permissiveness_z",
        "tree_discordance_score_z",
    ]
    term_labels = ["Ecology", "Molecular", "Interaction", "Discordance"]
    target_order = ["species_q25", "strict_species_min"]
    target_labels = ["q25", "strict"]
    if not hurdle_rare.empty:
        max_abs = max(0.1, float(np.nanmax(np.abs(hurdle_rare["coefficient_mean"]))))
        norm = Normalize(vmin=-max_abs, vmax=max_abs)
        for _, row in hurdle_rare.iterrows():
            xpos = term_order.index(row["term"])
            ypos = target_order.index(row["rarefaction_target"]) if row["rarefaction_target"] in target_order else 0
            stability = float(row["sign_stability_positive"])
            size = 85.0 + 210.0 * abs(stability - 0.5) * 2.0
            ax_rare.scatter(
                xpos,
                ypos,
                s=size,
                color=div_cmap(norm(float(row["coefficient_mean"]))),
                edgecolors="#464646",
                linewidths=0.45,
                alpha=0.95,
                zorder=3,
            )
        ax_rare.set_xticks(np.arange(len(term_order)))
        ax_rare.set_xticklabels(term_labels, rotation=35, ha="right")
        ax_rare.set_yticks(np.arange(len(target_order)))
        ax_rare.set_yticklabels(target_labels)
        ax_rare.set_xlim(-0.5, len(term_order) - 0.5)
        ax_rare.set_ylim(len(target_order) - 0.5, -0.5)
        for xpos in np.arange(-0.5, len(term_order), 1):
            ax_rare.axvline(xpos, color="#ececec", linewidth=0.7, zorder=0)
        for ypos in np.arange(-0.5, len(target_order), 1):
            ax_rare.axhline(ypos, color="#ececec", linewidth=0.7, zorder=0)
        sm = ScalarMappable(norm=norm, cmap=div_cmap)
        sm.set_array([])
        cbar2 = fig.colorbar(sm, ax=ax_rare, fraction=0.08, pad=0.02)
        cbar2.set_label("Coefficient")
        cbar2.outline.set_visible(False)
    ax_rare.text(-0.18, 1.11, "D", transform=ax_rare.transAxes, fontsize=13, fontweight="bold", va="top")

    for ax in [ax_surface, ax_curves, ax_denest, ax_rare]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, color="#777777")
    ax_surface.grid(False)
    ax_curves.grid(axis="y", color="#eeeeee", linewidth=0.7)
    ax_denest.grid(axis="x", color="#eeeeee", linewidth=0.7)

    png = FIGURE_DIR / "08_fixation_establishment_strategy_landscape.png"
    svg = FIGURE_DIR / "08_fixation_establishment_strategy_landscape.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def render_report(
    origin_events: pd.DataFrame,
    denesting_manifest: pd.DataFrame,
    all_controls: pd.DataFrame,
    matched_controls: pd.DataFrame,
    matched_events: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    graph_edges: pd.DataFrame,
    establishment_formula: str,
    establishment_summary: pd.DataFrame,
    count_summary: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    rarefaction_summary: pd.DataFrame,
) -> str:
    species_summary = (
        origin_events.groupby("species_label", as_index=False)
        .agg(
            origin_n=("event_id", "nunique"),
            median_descendant_tips=("descendant_tip_n", "median"),
            max_descendant_tips=("descendant_tip_n", "max"),
            median_temporal_span_years=("temporal_span_years", "median"),
            median_country_spread=("country_spread_n", "median"),
            median_host_spread=("host_spread_n", "median"),
            median_terminal_utr5_disparity=("terminal_utr5_length_disparity", "median"),
            median_terminal_utr3_disparity=("terminal_utr3_length_disparity", "median"),
            median_terminal_utr_recovery=("terminal_utr_recovery_fraction", "median"),
        )
        .sort_values("species_label")
    )
    graph_summary = (
        graph_nodes.groupby("species_label", as_index=False)
        .agg(
            lineage_nodes=("lineage_id", "nunique"),
            mean_weighted_degree=("graph_weighted_degree", "mean"),
            mean_neighbor_overlap=("graph_neighbor_overlap", "mean"),
        )
        .merge(graph_edges.groupby("species_label", as_index=False).agg(edges=("edge_weight", "size")), on="species_label", how="left")
        .fillna({"edges": 0})
        .sort_values("species_label")
    )
    est_display = establishment_summary.copy()
    est_display["raw_term"] = est_display["term"].str.removesuffix("_z")
    count_display = count_summary.copy()
    count_display["raw_term"] = count_display["term"].str.removesuffix("_z")
    rarefaction_display = rarefaction_summary.copy()
    if not rarefaction_display.empty:
        rarefaction_display["raw_term"] = rarefaction_display["term"].str.removesuffix("_z")
        rarefaction_display = rarefaction_display[
            rarefaction_display["raw_term"].isin(FOCAL_FEATURES)
        ][
            [
                "rarefaction_target",
                "target_sample_n",
                "model",
                "raw_term",
                "coefficient_mean",
                "sign_stability_positive",
                "mean_established_origin_n",
                "mean_origin_retention_fraction",
            ]
        ]
    interpretation = [
        "# Fixation-Inspired Establishment Model",
        "",
        "This analysis treats de-nested ARG-derived reassortment-origin clusters as observed reassortment-permissive events and matched non-reassortant internal S-tree clades as local reassortment-resistant background states. It remains an establishment/payoff approximation rather than a full Moran simulation, but it now applies a minimal-origin rule, a compact pre-specified molecular-filter index, and rarefaction sensitivity for sampling bias.",
        "",
        "## Design",
        "",
        f"- Candidate ARG origin clusters before de-nesting: `{denesting_manifest['origin_cluster_id'].nunique() if not denesting_manifest.empty else origin_events['event_id'].nunique()}`",
        f"- Retained minimal ARG origin clusters: `{origin_events['event_id'].nunique()}`",
        f"- Nested candidate clusters removed: `{int((denesting_manifest['denested_keep'] == 0).sum()) if not denesting_manifest.empty else 0}`",
        f"- Eligible non-reassortant internal background nodes: `{all_controls['event_id'].nunique()}`",
        f"- Matched background rows used in hurdle model: `{len(matched_controls)}`",
        f"- Controls per origin requested: `{N_CONTROLS_PER_ORIGIN}`",
        f"- Species-specific lineage graph nodes: `{graph_nodes['lineage_id'].nunique()}`",
        f"- Species-specific lineage graph edges: `{len(graph_edges)}`",
        "",
        "De-nesting keeps the smallest ARG-supported origin cluster when a candidate origin is a strict superset of an already retained origin within the same species. The hurdle component models retained origin versus matched background status with species and lineage random effects. The count component models excess descendant tips among retained origins with a negative-binomial likelihood and a log lineage-level sampling-effort offset.",
        "",
        "The definitive strategy model uses compact, pre-specified predictors rather than the full raw feature block: ecological opportunity, molecular permissiveness, their interaction, tree discordance, graph-neighborhood opportunity, and sampling effort. Molecular permissiveness is an oriented composite of lower LD/coadaptation burden, lower species-level UTR structural distance, and a direct terminal UTR regime that combines lower 5-prime length disparity with higher 3-prime length disparity. The individual `utrterm_*` workhorse feature means are still exported in the dataset for traceability.",
        "",
        f"- Establishment formula: `{establishment_formula}`",
        "",
        render_dataframe_markdown("Origin Clade Summaries By Species", species_summary),
        render_dataframe_markdown("Species Graph Summaries", graph_summary),
        render_dataframe_markdown("Model Diagnostics", diagnostics_df),
        render_dataframe_markdown(
            "Establishment Hurdle Posterior",
            est_display[["raw_term", "mean", "hdi_3%", "hdi_97%", "posterior_prob_positive"]],
        ),
        render_dataframe_markdown(
            "Expansion Count Posterior",
            count_display[["raw_term", "mean", "hdi_3%", "hdi_97%", "posterior_prob_positive"]],
        ),
        render_dataframe_markdown(
            "Rarefaction Sensitivity",
            rarefaction_display if not rarefaction_display.empty else pd.DataFrame({"status": ["No rarefaction coefficients were estimable."]}),
        ),
        "## Interpretation",
        "",
        "This definitive pass is an establishment-oriented test of the ecological-opportunity x molecular-filter synthesis. The important target is the sign and stability of ecological opportunity, molecular permissiveness, and their interaction after minimal-origin de-nesting and rarefaction. A positive interaction means reassortment-prone regimes have their highest payoff when exposure opportunity and molecular permissiveness coincide; weak or negative effects imply reassortment-resistant regimes dominate under restrictive molecular backgrounds or low opportunity.",
        "",
        "Sampling remains a limitation because descendant clade size is sampled expansion rather than true epidemiological expansion. The rarefaction sensitivity is therefore the main guardrail: the result is strongest when the compact terms retain their sign under strict common-sampling downsampling.",
    ]
    return "\n".join(interpretation)


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df, lineage_df, origin_summary = build_base_dataframe()
    df = add_workhorse_terminal_utr_features(df)
    ref = numeric_reference(df)
    analysis = apply_numeric_reference(df, ref)

    graph_nodes, graph_edges = lineage_graph_tables(analysis)
    denested_origin_summary, denesting_manifest = denest_origin_summary(origin_summary)
    origin_events = positive_origin_events(analysis, denested_origin_summary, graph_nodes)
    all_controls = nonreassortant_internal_nodes(analysis, graph_nodes)
    matched_controls = match_controls(origin_events, all_controls)

    origin_events = origin_events.assign(established=1, matched_origin_id=pd.NA, match_score=0.0)
    matched_controls = matched_controls.assign(established=0)
    matched_events = pd.concat([origin_events, matched_controls], ignore_index=True, sort=False)
    matched_events, event_ref = add_standardized_event_features(matched_events)

    graph_nodes.to_csv(REPORT_DIR / "08_species_lineage_graph_nodes.tsv", sep="\t", index=False)
    graph_edges.to_csv(REPORT_DIR / "08_species_lineage_graph_edges.tsv", sep="\t", index=False)
    denesting_manifest.to_csv(REPORT_DIR / "08_arg_origin_denesting_manifest.tsv", sep="\t", index=False)
    origin_events.to_csv(REPORT_DIR / "08_reassortment_origin_establishment_events.tsv", sep="\t", index=False)
    all_controls.to_csv(REPORT_DIR / "08_nonreassortant_internal_node_candidates.tsv", sep="\t", index=False)
    matched_controls.to_csv(REPORT_DIR / "08_matched_nonreassortant_internal_nodes.tsv", sep="\t", index=False)
    matched_events.to_csv(DATASET_DIR / "08_fixation_establishment_model_dataset.tsv", sep="\t", index=False)
    pd.DataFrame(
        [{"feature": key, **value} for key, value in event_ref.items()]
    ).to_csv(REPORT_DIR / "08_event_predictor_standardization.tsv", sep="\t", index=False)

    _, establishment_idata, establishment_formula, establishment_terms = fit_establishment_model(matched_events)
    establishment_summary = summarize_bayesian_terms(
        establishment_idata,
        establishment_terms,
        "hurdle_establishment_bernoulli",
    )
    count_terms = [f"{feature}_z" for feature in COUNT_FEATURES]
    _, count_idata, count_df = fit_count_model(matched_events, count_terms)
    count_summary = summarize_count_model(count_idata, count_terms)
    rarefaction_iteration_coefficients, rarefaction_summary = run_rarefaction_sensitivity(matched_events, origin_events)
    diagnostics_df = pd.DataFrame(
        [
            model_diagnostics(establishment_idata, "hurdle_establishment_bernoulli"),
            model_diagnostics(count_idata, "count_expansion_negative_binomial"),
        ]
    )

    establishment_summary.to_csv(REPORT_DIR / "08_hurdle_establishment_coefficients.tsv", sep="\t", index=False)
    count_summary.to_csv(REPORT_DIR / "08_count_expansion_coefficients.tsv", sep="\t", index=False)
    rarefaction_iteration_coefficients.to_csv(REPORT_DIR / "08_rarefaction_iteration_coefficients.tsv", sep="\t", index=False)
    rarefaction_summary.to_csv(REPORT_DIR / "08_rarefaction_sensitivity.tsv", sep="\t", index=False)
    diagnostics_df.to_csv(REPORT_DIR / "08_model_diagnostics.tsv", sep="\t", index=False)
    az.to_netcdf(establishment_idata, REPORT_DIR / "08_hurdle_establishment_idata.nc")
    az.to_netcdf(count_idata, REPORT_DIR / "08_count_expansion_idata.nc")

    plot_first_pass(origin_events, matched_events, establishment_summary, count_summary)
    plot_strategy_landscape(
        origin_events=origin_events,
        matched_events=matched_events,
        denesting_manifest=denesting_manifest,
        rarefaction_summary=rarefaction_summary,
        establishment_idata=establishment_idata,
        event_ref=event_ref,
    )
    report = render_report(
        origin_events=origin_events,
        denesting_manifest=denesting_manifest,
        all_controls=all_controls,
        matched_controls=matched_controls,
        matched_events=matched_events,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        establishment_formula=establishment_formula,
        establishment_summary=establishment_summary,
        count_summary=count_summary,
        diagnostics_df=diagnostics_df,
        rarefaction_summary=rarefaction_summary,
    )
    write_markdown(REPORT_DIR / "08_fixation_establishment_model_report.md", report)


if __name__ == "__main__":
    main()
