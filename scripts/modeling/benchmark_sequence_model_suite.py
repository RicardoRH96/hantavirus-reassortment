#!/usr/bin/env python3
"""Mechanism-focused explanatory suite for the sequence-level reassortment dataset."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import Entrez, Phylo, SeqIO
from group_lasso import LogisticGroupLasso
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, spearmanr
if not hasattr(np, "int"):  # pragma: no cover - skopt compatibility on newer NumPy
    np.int = int  # type: ignore[attr-defined]
try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real
except ImportError:  # pragma: no cover - optional dependency
    BayesSearchCV = None
    Categorical = None
    Integer = None
    Real = None

from sequence_model_suite_explanatory_utils import (
    apply_ld_encoding,
    FEATURE_SETS,
    FeatureSet,
    build_feature_catalog,
    build_feature_frame,
    detect_feature_blocks,
    summarize_species_effects,
    try_import_bayesian_tooling,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "final_sequence_dataset.tsv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "modeling" / "explanatory_model_suite"
DEFAULT_LOG = REPO_ROOT / "results" / "modeling" / "model_iteration_log_explanatory.tsv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str


MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(name="penalized_logistic_l2", kind="logistic_l2"),
    ModelSpec(name="penalized_logistic_elasticnet", kind="logistic_elasticnet"),
    ModelSpec(name="group_lasso", kind="group_lasso"),
    ModelSpec(name="sparse_group_lasso", kind="sparse_group_lasso"),
    ModelSpec(name="random_forest_shallow", kind="rf"),
    ModelSpec(name="gradient_boosting", kind="gb"),
    ModelSpec(name="catboost", kind="catboost"),
    ModelSpec(name="xgboost", kind="xgboost"),
    ModelSpec(name="lightgbm", kind="lightgbm"),
)

TREE_MODEL_KINDS = {"rf", "gb", "catboost", "xgboost", "lightgbm"}


@dataclass
class TreeTuningConfig:
    enabled: bool
    n_iter: int
    inner_splits: int
    metric: str
    random_state: int
    cv_scheme: str
    species_groups: pd.Series
    clade_groups: pd.Series
    within_species_groups: pd.Series


SPECIES_CLADE_MAP: Dict[str, str] = {
    "orthohantavirus_andesense": "new_world_sigmodontinae",
    "orthohantavirus_sinnombreense": "new_world_sigmodontinae",
    "orthohantavirus_puumalaense": "old_world_arvicolinae",
    "orthohantavirus_tulaense": "old_world_arvicolinae",
    "orthohantavirus_dobravaense": "old_world_murinae",
    "orthohantavirus_hantanense": "old_world_murinae",
    "orthohantavirus_seoulense": "old_world_murinae",
}

TREE_SEARCH_ROOTS: Tuple[Path, ...] = (
    REPO_ROOT / "data" / "processed" / "trees",
)
TREE_SEGMENT_PRIORITY: Tuple[str, ...] = ("S", "M", "L")
UTR_RUN_ROOT = REPO_ROOT / "data" / "processed" / "utr5"

_UTR_METADATA_CACHE: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
_UTR_STRUCT_CACHE: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
_SEGMENT_ACCESSION_CACHE: Dict[Tuple[str, str], Dict[str, str]] = {}
_SEGMENT_TREE_PATH_CACHE: Dict[Tuple[str, str], Optional[Path]] = {}
_SEGMENT_TREE_DISTANCE_CACHE: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
_SEGMENT_ALIGNMENT_PATH_CACHE: Dict[Tuple[str, str], Optional[Path]] = {}
_SEGMENT_SEQUENCE_DISTANCE_CACHE: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
_SEGMENT_MATCHED_FASTA_PATH_CACHE: Dict[Tuple[str, str], Optional[Path]] = {}
_SEGMENT_TIP_SEQUENCE_CACHE: Dict[Tuple[str, str], Optional[Dict[str, str]]] = {}
_SEGMENT_INPUT_CDS_CACHE: Dict[Tuple[str, str], Optional[Dict[str, str]]] = {}
_FILTERED_VIRUS_SEQ_CACHE: Optional[pd.DataFrame] = None
_LOCAL_GENBANK_SEQ_CACHE: Optional[Dict[str, str]] = None
_ENTREZ_FASTA_CACHE: Dict[str, Optional[str]] = {}
_ENTREZ_GFF3_CACHE: Dict[str, Optional[str]] = {}
_UTR_FASTA_CACHE: Dict[str, Optional[str]] = {}
_SEGMENT_TERMINAL_CACHE: Dict[Tuple[str, str, str], Optional[Dict[str, object]]] = {}
_BALTIC_MODULE_CACHE: Dict[str, Any] = {}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iteration-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--target-source",
        choices=["all", "true_label", "sequence_true_label", "isolate_true_label", "proxy_label"],
        default="true_label",
    )
    parser.add_argument("--group-column", choices=["species"], default="species")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-effects", type=int, default=25)
    parser.add_argument(
        "--feature-set",
        action="append",
        dest="feature_sets",
        help="Optional feature set filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Optional model filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap on evaluated grouped folds for smoke tests.",
    )
    parser.add_argument(
        "--cv-scheme",
        action="append",
        dest="cv_schemes",
        choices=["grouped_species", "grouped_clade", "grouped_within_species_clade", "repeated_stratified"],
        help="Validation scheme(s) to run. Defaults to grouped_species, grouped_within_species_clade, and repeated_stratified.",
    )
    parser.add_argument(
        "--repeated-splits",
        type=int,
        default=5,
        help="Number of splits for repeated stratified CV.",
    )
    parser.add_argument(
        "--repeated-repeats",
        type=int,
        default=3,
        help="Number of repeats for repeated stratified CV.",
    )
    parser.add_argument(
        "--within-species-min-clade-size",
        type=int,
        default=3,
        help="Minimum held-out lineage size when defining within-species tree clades.",
    )
    parser.add_argument(
        "--within-species-max-clusters",
        type=int,
        default=4,
        help="Maximum number of lineage clusters to define within a species tree.",
    )
    parser.add_argument(
        "--within-species-max-test-fraction",
        type=float,
        default=0.30,
        help="Maximum fraction of a species that can be held out in a within-species lineage fold.",
    )
    parser.add_argument(
        "--within-species-min-test-fraction",
        type=float,
        default=0.20,
        help="Preferred minimum fraction of a species to hold out when constructing internal-node within-species folds.",
    )
    parser.add_argument(
        "--within-species-grouping-method",
        choices=["capped_pack", "baltic_nodes"],
        default="capped_pack",
        help="Strategy for building grouped_within_species_clade evaluation folds.",
    )
    parser.add_argument(
        "--within-species-exclude-species",
        action="append",
        default=[],
        help="Species label to exclude from grouped_within_species_clade evaluation. Can be passed multiple times.",
    )
    parser.add_argument(
        "--within-species-baltic-path",
        type=Path,
        default=None,
        help="Path to a baltic Python source file used for baltic_nodes within-species grouping.",
    )
    parser.add_argument(
        "--ld-encoding",
        choices=["full", "interaction_only"],
        default="full",
        help="LD genotype encoding for the molecular block. interaction_only removes single-site LD terms and retains pairwise LD terms.",
    )
    parser.add_argument(
        "--tree-bayes-search",
        action="store_true",
        help="Run nested Bayesian hyperparameter search for tree-based classifiers within each outer fold.",
    )
    parser.add_argument(
        "--tree-bayes-iters",
        type=int,
        default=20,
        help="Bayesian search iterations per tree model and outer fold.",
    )
    parser.add_argument(
        "--tree-bayes-inner-splits",
        type=int,
        default=4,
        help="Maximum number of inner CV splits used during tree-model tuning.",
    )
    parser.add_argument(
        "--tree-bayes-metric",
        choices=["average_precision", "roc_auc", "balanced_accuracy", "neg_brier"],
        default="average_precision",
        help="Objective for Bayesian tree-model tuning.",
    )
    parser.add_argument(
        "--tree-bayes-shared-by-cv",
        action="store_true",
        help="Tune tree-model hyperparameters once per feature set/model/CV scheme and reuse them across outer folds.",
    )
    parser.add_argument(
        "--within-species-lineage-molecular-context",
        action="store_true",
        help="Add exploratory leakage-safe lineage-aware molecular context features for grouped_within_species_clade folds.",
    )
    parser.add_argument(
        "--exploratory-real-lineage-context",
        action="store_true",
        help="Add exploratory fold-safe lineage-aware features from real segment trees, alignments, and spatiotemporal co-circulation.",
    )
    parser.add_argument(
        "--exploratory-codon-composition",
        action="store_true",
        help="Add exploratory codon/compositional compatibility features derived from segment CDS sequences.",
    )
    parser.add_argument(
        "--lineage-terminal-min-coverage",
        type=float,
        default=0.85,
        help="Minimum non-missing fraction required for direct terminal features to enter lineage-core gating.",
    )
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
    if df.empty:
        raise SystemExit("No labeled rows remain after filtering.")
    return df


def assign_species_clade(species: pd.Series) -> pd.Series:
    return species.map(SPECIES_CLADE_MAP).fillna("other_or_unknown")


def _normalize_species_exclusion(values: Optional[Sequence[str]]) -> set[str]:
    return {str(value) for value in (values or []) if value is not None and str(value).strip()}


def find_refined_species_tree(species: str) -> Optional[Path]:
    for root in TREE_SEARCH_ROOTS:
        for seg in TREE_SEGMENT_PRIORITY:
            for ext in ("newick", "nwk"):
                candidate = root / species / "augur" / seg / f"{seg}_tree_refined.{ext}"
                if candidate.exists():
                    return candidate
    return None


def cluster_tree_tips_within_species(
    tip_names: Sequence[str],
    tree_path: Path,
    min_clade_size: int,
    max_clusters: int,
) -> Optional[Dict[str, int]]:
    unique_tips = list(dict.fromkeys([str(t) for t in tip_names if pd.notna(t)]))
    if len(unique_tips) < max(2 * min_clade_size, 2):
        return None
    tree = Phylo.read(str(tree_path), "newick")
    available = {term.name for term in tree.get_terminals() if term.name}
    unique_tips = [tip for tip in unique_tips if tip in available]
    if len(unique_tips) < max(2 * min_clade_size, 2):
        return None
    n = len(unique_tips)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(tree.distance(unique_tips[i], unique_tips[j]))
            dist[i, j] = d
            dist[j, i] = d
    condensed = squareform(dist, checks=False)
    if condensed.size == 0 or np.allclose(condensed, 0.0):
        return None
    Z = linkage(condensed, method="average")
    max_k = min(max_clusters, n // max(min_clade_size, 1))
    if max_k < 2:
        return None
    best_labels: Optional[np.ndarray] = None
    best_assigned = -1
    best_cluster_count = -1
    for k in range(2, max_k + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        cluster_ids, counts = np.unique(labels, return_counts=True)
        eligible = cluster_ids[counts >= min_clade_size]
        assigned = int(counts[counts >= min_clade_size].sum())
        cluster_count = int(len(eligible))
        if cluster_count < 2:
            continue
        if (assigned > best_assigned) or (assigned == best_assigned and cluster_count > best_cluster_count):
            keep = np.isin(labels, eligible)
            masked = np.where(keep, labels, -1)
            best_labels = masked
            best_assigned = assigned
            best_cluster_count = cluster_count
    if best_labels is None:
        return None
    valid_old = sorted({int(label) for label in best_labels if int(label) != -1})
    relabeled = {old: idx + 1 for idx, old in enumerate(valid_old)}
    return {
        tip: relabeled[int(label)]
        for tip, label in zip(unique_tips, best_labels)
        if int(label) != -1
    }


def assign_within_species_tree_clades(
    df: pd.DataFrame,
    min_clade_size: int,
    max_clusters: int,
    excluded_species: Optional[set[str]] = None,
) -> tuple[pd.Series, pd.DataFrame]:
    labels = pd.Series(pd.NA, index=df.index, dtype="object")
    records: List[Dict[str, object]] = []
    excluded_species = excluded_species or set()
    for species, group in df.groupby("species", dropna=False):
        species_label = str(species)
        tree_path = find_refined_species_tree(species_label)
        record: Dict[str, object] = {
            "species": species_label,
            "tree_path": str(tree_path) if tree_path else "",
            "n_rows": int(len(group)),
            "n_tip_names": int(group["tip_name"].notna().sum()) if "tip_name" in group.columns else 0,
            "n_assigned_rows": 0,
            "n_lineages": 0,
            "status": "missing_tree" if tree_path is None else "unassigned",
        }
        if species_label in excluded_species:
            record["status"] = "excluded"
            records.append(record)
            continue
        if tree_path is None or "tip_name" not in group.columns:
            records.append(record)
            continue
        assignments = cluster_tree_tips_within_species(
            group["tip_name"].astype(str).tolist(),
            tree_path,
            min_clade_size=min_clade_size,
            max_clusters=max_clusters,
        )
        if not assignments:
            record["status"] = "insufficient_or_unresolved"
            records.append(record)
            continue
        group_labels = group["tip_name"].map(lambda x: f"{species_label}::clade_{assignments[x]}" if x in assignments else pd.NA)
        labels.loc[group.index] = group_labels
        assigned_unique = sorted(set(group_labels.dropna()))
        record["n_assigned_rows"] = int(group_labels.notna().sum())
        record["n_lineages"] = int(len(assigned_unique))
        record["status"] = "assigned"
        record["lineages"] = "|".join(assigned_unique)
        records.append(record)
    return labels, pd.DataFrame(records).sort_values("species").reset_index(drop=True)


def _fallback_partition_indices(
    indices: Sequence[int],
    y: np.ndarray,
    n_parts: int,
    *,
    random_state: int,
) -> List[np.ndarray]:
    idx = np.asarray(indices, dtype=int)
    if len(idx) == 0:
        return []
    n_parts = max(1, min(int(n_parts), len(idx)))
    if n_parts == 1:
        return [idx]
    y_sub = np.asarray(y[idx]).astype(int)
    unique_classes = np.unique(y_sub)
    if len(unique_classes) >= 2:
        class_counts = np.bincount(y_sub, minlength=2)
        if int(class_counts.min()) >= n_parts:
            splitter = StratifiedKFold(n_splits=n_parts, shuffle=True, random_state=random_state)
            return [idx[test] for _, test in splitter.split(np.zeros((len(idx), 1)), y_sub)]
    ordered = np.sort(idx)
    return [np.asarray(part, dtype=int) for part in np.array_split(ordered, n_parts) if len(part)]


def _split_oversized_species_unit(
    df: pd.DataFrame,
    y: np.ndarray,
    indices: Sequence[int],
    *,
    tree_path: Optional[Path],
    min_clade_size: int,
    max_clusters: int,
    max_test_rows: int,
    random_state: int,
    label_prefix: str,
) -> List[Tuple[str, np.ndarray]]:
    pending: List[Tuple[str, np.ndarray]] = [(label_prefix, np.asarray(indices, dtype=int))]
    results: List[Tuple[str, np.ndarray]] = []
    split_round = 0
    while pending:
        label, idx = pending.pop(0)
        if len(idx) <= max_test_rows:
            results.append((label, idx))
            continue
        candidate_parts: List[np.ndarray] = []
        if tree_path is not None and "tip_name" in df.columns:
            requested_clusters = max(2, int(np.ceil(len(idx) / max(max_test_rows, 1))))
            assignments = cluster_tree_tips_within_species(
                df.loc[idx, "tip_name"].astype(str).tolist(),
                tree_path,
                min_clade_size=min_clade_size,
                max_clusters=max(max_clusters, requested_clusters),
            )
            if assignments:
                tip_labels = df.loc[idx, "tip_name"].astype(str).map(assignments)
                parts: List[np.ndarray] = []
                for cluster_id in sorted({int(v) for v in tip_labels.dropna().tolist()}):
                    mask = tip_labels.to_numpy() == cluster_id
                    part = idx[mask]
                    if len(part):
                        parts.append(np.asarray(part, dtype=int))
                remainder = idx[tip_labels.isna().to_numpy()]
                if len(remainder):
                    split_round += 1
                    parts.extend(
                        _fallback_partition_indices(
                            remainder,
                            y,
                            int(np.ceil(len(remainder) / max(max_test_rows, 1))),
                            random_state=random_state + split_round,
                        )
                    )
                if parts and sum(len(part) for part in parts) == len(idx) and max(len(part) for part in parts) < len(idx):
                    candidate_parts = parts
        if not candidate_parts:
            split_round += 1
            candidate_parts = _fallback_partition_indices(
                idx,
                y,
                int(np.ceil(len(idx) / max(max_test_rows, 1))),
                random_state=random_state + 1000 + split_round,
            )
            if not candidate_parts or max(len(part) for part in candidate_parts) >= len(idx):
                results.append((label, idx))
                continue
        for part_num, part_idx in enumerate(candidate_parts, start=1):
            pending.append((f"{label}.part_{part_num}", np.asarray(part_idx, dtype=int)))
    return results


def _pack_species_units_into_eval_groups(
    species: str,
    units: Sequence[Dict[str, object]],
    *,
    max_test_rows: int,
) -> List[Dict[str, object]]:
    species_has_both_classes = any(int(unit["positives"]) > 0 for unit in units) and any(int(unit["negatives"]) > 0 for unit in units)
    bins: List[Dict[str, object]] = []
    ordered_units = sorted(
        units,
        key=lambda unit: (
            int(int(unit["positives"]) > 0 and int(unit["negatives"]) > 0),
            -int(unit["size"]),
            -max(int(unit["positives"]), int(unit["negatives"])),
        ),
    )
    for unit in ordered_units:
        best_idx: Optional[int] = None
        best_score: Optional[Tuple[int, int, int]] = None
        for idx, current in enumerate(bins):
            new_size = int(current["size"]) + int(unit["size"])
            if new_size > max_test_rows:
                continue
            new_pos = int(current["positives"]) + int(unit["positives"])
            new_neg = int(current["negatives"]) + int(unit["negatives"])
            mixed_before = int(current["positives"]) > 0 and int(current["negatives"]) > 0
            mixed_after = new_pos > 0 and new_neg > 0
            closes_single = int(species_has_both_classes and mixed_after and not mixed_before)
            keeps_mixed = int(mixed_after and mixed_before)
            score = (closes_single, keeps_mixed, new_size)
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            bins.append(
                {
                    "species": species,
                    "indices": list(np.asarray(unit["indices"], dtype=int)),
                    "size": int(unit["size"]),
                    "positives": int(unit["positives"]),
                    "negatives": int(unit["negatives"]),
                    "component_groups": [str(unit["component_label"])],
                    "source_groups": [str(unit["source_label"])],
                }
            )
            continue
        current = bins[best_idx]
        current["indices"].extend(list(np.asarray(unit["indices"], dtype=int)))
        current["size"] = int(current["size"]) + int(unit["size"])
        current["positives"] = int(current["positives"]) + int(unit["positives"])
        current["negatives"] = int(current["negatives"]) + int(unit["negatives"])
        current["component_groups"].append(str(unit["component_label"]))
        current["source_groups"].append(str(unit["source_label"]))
    changed = True
    while changed and len(bins) > 1:
        changed = False
        for idx, current in list(enumerate(bins)):
            current_mixed = int(current["positives"]) > 0 and int(current["negatives"]) > 0
            if current_mixed:
                continue
            best_idx = None
            best_score = None
            for other_idx, other in enumerate(bins):
                if idx == other_idx:
                    continue
                new_size = int(current["size"]) + int(other["size"])
                if new_size > max_test_rows:
                    continue
                new_pos = int(current["positives"]) + int(other["positives"])
                new_neg = int(current["negatives"]) + int(other["negatives"])
                if species_has_both_classes and not (new_pos > 0 and new_neg > 0):
                    continue
                score = (int(int(other["positives"]) > 0 and int(other["negatives"]) > 0), new_size)
                if best_score is None or score > best_score:
                    best_idx = other_idx
                    best_score = score
            if best_idx is None:
                continue
            target = bins[best_idx]
            target["indices"].extend(current["indices"])
            target["size"] = int(target["size"]) + int(current["size"])
            target["positives"] = int(target["positives"]) + int(current["positives"])
            target["negatives"] = int(target["negatives"]) + int(current["negatives"])
            target["component_groups"].extend(current["component_groups"])
            target["source_groups"].extend(current["source_groups"])
            bins.pop(idx)
            changed = True
            break
    bins.sort(key=lambda row: (-int(row["size"]), -int(row["positives"]), str(row["species"])))
    return bins


def build_within_species_eval_groups(
    df: pd.DataFrame,
    y: np.ndarray,
    within_species_clades: pd.Series,
    *,
    min_clade_size: int,
    max_clusters: int,
    max_test_fraction: float,
    random_state: int,
    excluded_species: Optional[set[str]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    eval_groups = pd.Series(pd.NA, index=df.index, dtype="object")
    records: List[Dict[str, object]] = []
    excluded_species = excluded_species or set()
    for species, group in df.groupby("species", dropna=False):
        species_label = str(species)
        species_idx = group.index.to_numpy(dtype=int)
        species_n = int(len(species_idx))
        if species_n == 0:
            continue
        if species_label in excluded_species:
            records.append(
                {
                    "species": species_label,
                    "eval_group": pd.NA,
                    "n_rows": species_n,
                    "n_positives": int(y[species_idx].sum()),
                    "n_negatives": int(species_n - int(y[species_idx].sum())),
                    "is_single_class": bool(len(np.unique(y[species_idx])) <= 1),
                    "max_allowed_test_rows": pd.NA,
                    "max_test_fraction": float(max_test_fraction),
                    "component_groups": "",
                    "source_groups": "",
                    "status": "excluded",
                    "grouping_method": "capped_pack",
                }
            )
            continue
        max_test_rows = int(np.floor(species_n * max_test_fraction))
        max_test_rows = max(max_test_rows, min_clade_size, 1)
        if species_n > 1:
            max_test_rows = min(max_test_rows, species_n - 1)
        base_labels = within_species_clades.loc[species_idx]
        tree_path = find_refined_species_tree(species_label)
        units: List[Dict[str, object]] = []
        seen_labels = list(dict.fromkeys(base_labels.dropna().astype(str).tolist()))
        for source_label in seen_labels:
            mask = base_labels.astype("object").eq(source_label).fillna(False).to_numpy()
            unit_idx = species_idx[mask]
            for component_label, component_idx in _split_oversized_species_unit(
                df,
                y,
                unit_idx,
                tree_path=tree_path,
                min_clade_size=min_clade_size,
                max_clusters=max_clusters,
                max_test_rows=max_test_rows,
                random_state=random_state,
                label_prefix=source_label,
            ):
                positives = int(y[component_idx].sum())
                units.append(
                    {
                        "species": species_label,
                        "source_label": source_label,
                        "component_label": component_label,
                        "indices": component_idx,
                        "size": int(len(component_idx)),
                        "positives": positives,
                        "negatives": int(len(component_idx) - positives),
                    }
                )
        missing_idx = species_idx[base_labels.isna().to_numpy()]
        if len(missing_idx):
            remainder_parts = _fallback_partition_indices(
                missing_idx,
                y,
                int(np.ceil(len(missing_idx) / max(max_test_rows, 1))),
                random_state=random_state + species_n,
            )
            for part_num, component_idx in enumerate(remainder_parts, start=1):
                source_label = f"{species_label}::unassigned"
                positives = int(y[component_idx].sum())
                units.append(
                    {
                        "species": species_label,
                        "source_label": source_label,
                        "component_label": f"{source_label}.part_{part_num}",
                        "indices": component_idx,
                        "size": int(len(component_idx)),
                        "positives": positives,
                        "negatives": int(len(component_idx) - positives),
                    }
                )
        if not units:
            positives = int(y[species_idx].sum())
            units.append(
                {
                    "species": species_label,
                    "source_label": f"{species_label}::all_rows",
                    "component_label": f"{species_label}::all_rows",
                    "indices": species_idx,
                    "size": species_n,
                    "positives": positives,
                    "negatives": int(species_n - positives),
                }
            )
        for eval_num, packed in enumerate(_pack_species_units_into_eval_groups(species_label, units, max_test_rows=max_test_rows), start=1):
            eval_label = f"{species_label}::eval_{eval_num}"
            eval_groups.loc[np.asarray(packed["indices"], dtype=int)] = eval_label
            records.append(
                {
                    "species": species_label,
                    "eval_group": eval_label,
                    "n_rows": int(packed["size"]),
                    "n_positives": int(packed["positives"]),
                    "n_negatives": int(packed["negatives"]),
                    "is_single_class": bool(int(packed["positives"]) == 0 or int(packed["negatives"]) == 0),
                    "max_allowed_test_rows": int(max_test_rows),
                    "max_test_fraction": float(max_test_fraction),
                    "component_groups": "|".join(sorted(set(str(v) for v in packed["component_groups"]))),
                    "source_groups": "|".join(sorted(set(str(v) for v in packed["source_groups"]))),
                    "status": "assigned",
                    "grouping_method": "capped_pack",
                }
            )
    manifest = pd.DataFrame(records).sort_values(["species", "eval_group"]).reset_index(drop=True)
    return eval_groups, manifest


def _load_baltic_module(baltic_path: Path):
    resolved = str(baltic_path.resolve())
    cached = _BALTIC_MODULE_CACHE.get(resolved)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("codex_baltic_module", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load baltic module from {baltic_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _BALTIC_MODULE_CACHE[resolved] = module
    return module


def _build_fallback_units_from_indices(
    indices: Sequence[int],
    y: np.ndarray,
    *,
    label_prefix: str,
    max_test_rows: int,
    random_state: int,
    species_label: str,
) -> List[Dict[str, object]]:
    idx = np.asarray(indices, dtype=int)
    if len(idx) == 0:
        return []
    if len(idx) <= max_test_rows:
        parts = [idx]
    else:
        parts = _fallback_partition_indices(
            idx,
            y,
            int(np.ceil(len(idx) / max(max_test_rows, 1))),
            random_state=random_state,
        )
    units: List[Dict[str, object]] = []
    for part_num, part_idx in enumerate(parts, start=1):
        positives = int(y[part_idx].sum())
        label = label_prefix if len(parts) == 1 else f"{label_prefix}.part_{part_num}"
        units.append(
            {
                "species": species_label,
                "source_label": label_prefix,
                "component_label": label,
                "indices": np.asarray(part_idx, dtype=int),
                "size": int(len(part_idx)),
                "positives": positives,
                "negatives": int(len(part_idx) - positives),
            }
        )
    return units


def _build_baltic_species_units(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    species_label: str,
    species_idx: np.ndarray,
    tree_path: Optional[Path],
    baltic_path: Path,
    min_clade_size: int,
    min_test_rows: int,
    max_test_rows: int,
    random_state: int,
) -> List[Dict[str, object]]:
    if tree_path is None or "tip_name" not in df.columns:
        return _build_fallback_units_from_indices(
            species_idx,
            y,
            label_prefix=f"{species_label}::all_rows",
            max_test_rows=max_test_rows,
            random_state=random_state,
            species_label=species_label,
        )
    species_tip_series = df.loc[species_idx, "tip_name"]
    tip_to_indices: Dict[str, List[int]] = {}
    for row_idx, tip_name in zip(species_idx, species_tip_series.astype("object")):
        if pd.isna(tip_name):
            continue
        tip_to_indices.setdefault(str(tip_name), []).append(int(row_idx))
    if not tip_to_indices:
        return _build_fallback_units_from_indices(
            species_idx,
            y,
            label_prefix=f"{species_label}::all_rows",
            max_test_rows=max_test_rows,
            random_state=random_state,
            species_label=species_label,
        )
    baltic = _load_baltic_module(baltic_path)
    tree = baltic.loadNewick(str(tree_path), absoluteTime=False, verbose=False, sortBranches=False)
    tree.traverse_tree()
    row_cache: Dict[Tuple[str, str], np.ndarray] = {}
    species_has_both = len(np.unique(y[species_idx])) >= 2

    def branch_row_indices(branch) -> np.ndarray:
        key = ("node", str(branch.index)) if branch.is_node() else ("leaf", str(getattr(branch, "name", branch.index)))
        cached = row_cache.get(key)
        if cached is not None:
            return cached
        if branch.is_node():
            tips = [tip for tip in branch.leaves if tip in tip_to_indices]
        elif branch.is_leaflike() and getattr(branch, "name", None) in tip_to_indices:
            tips = [str(branch.name)]
        else:
            tips = []
        idx = np.asarray(sorted({row for tip in tips for row in tip_to_indices[tip]}), dtype=int)
        row_cache[key] = idx
        return idx

    def recurse(branch, label_prefix: str, depth_seed: int) -> List[Dict[str, object]]:
        idx = branch_row_indices(branch)
        if len(idx) == 0:
            return []
        positives = int(y[idx].sum())
        negatives = int(len(idx) - positives)
        mixed = positives > 0 and negatives > 0
        if branch.is_node():
            branch_label = f"{species_label}::node_{branch.index}"
        else:
            tip_name = str(getattr(branch, "name", branch.index))
            branch_label = f"{species_label}::tip_{tip_name}"
        if len(idx) > max_test_rows and branch.is_node():
            child_units: List[Dict[str, object]] = []
            for child_num, child in enumerate(branch.children, start=1):
                child_units.extend(recurse(child, f"{branch_label}.child_{child_num}", depth_seed + child_num))
            if child_units:
                covered = np.asarray(
                    sorted({row for unit in child_units for row in np.asarray(unit["indices"], dtype=int)}),
                    dtype=int,
                )
                missing = np.setdiff1d(idx, covered, assume_unique=False)
                if len(missing):
                    child_units.extend(
                        _build_fallback_units_from_indices(
                            missing,
                            y,
                            label_prefix=f"{branch_label}.remainder",
                            max_test_rows=max_test_rows,
                            random_state=random_state + depth_seed + len(idx),
                            species_label=species_label,
                        )
                    )
                return child_units
            return _build_fallback_units_from_indices(
                idx,
                y,
                label_prefix=branch_label,
                max_test_rows=max_test_rows,
                random_state=random_state + depth_seed + 1000,
                species_label=species_label,
            )
        if len(idx) >= max(min_test_rows, min_clade_size) and (mixed or not species_has_both):
            return [
                {
                    "species": species_label,
                    "source_label": branch_label,
                    "component_label": branch_label,
                    "indices": idx,
                    "size": int(len(idx)),
                    "positives": positives,
                    "negatives": negatives,
                }
            ]
        if branch.is_node():
            child_units: List[Dict[str, object]] = []
            for child_num, child in enumerate(branch.children, start=1):
                child_units.extend(recurse(child, f"{branch_label}.child_{child_num}", depth_seed + child_num))
            if child_units:
                covered = np.asarray(
                    sorted({row for unit in child_units for row in np.asarray(unit["indices"], dtype=int)}),
                    dtype=int,
                )
                missing = np.setdiff1d(idx, covered, assume_unique=False)
                if len(missing):
                    child_units.extend(
                        _build_fallback_units_from_indices(
                            missing,
                            y,
                            label_prefix=f"{branch_label}.remainder",
                            max_test_rows=max_test_rows,
                            random_state=random_state + depth_seed + 2000,
                            species_label=species_label,
                        )
                    )
                return child_units
        return _build_fallback_units_from_indices(
            idx,
            y,
            label_prefix=branch_label,
            max_test_rows=max_test_rows,
            random_state=random_state + depth_seed + 3000,
            species_label=species_label,
        )

    units = recurse(tree.root, f"{species_label}::root", 0)
    if not units:
        units = _build_fallback_units_from_indices(
            species_idx,
            y,
            label_prefix=f"{species_label}::all_rows",
            max_test_rows=max_test_rows,
            random_state=random_state,
            species_label=species_label,
        )
    return units


def build_within_species_baltic_eval_groups(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    min_clade_size: int,
    min_test_fraction: float,
    max_test_fraction: float,
    random_state: int,
    baltic_path: Path,
    excluded_species: Optional[set[str]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    eval_groups = pd.Series(pd.NA, index=df.index, dtype="object")
    records: List[Dict[str, object]] = []
    excluded_species = excluded_species or set()
    for species, group in df.groupby("species", dropna=False):
        species_label = str(species)
        species_idx = group.index.to_numpy(dtype=int)
        species_n = int(len(species_idx))
        if species_n == 0:
            continue
        if species_label in excluded_species:
            records.append(
                {
                    "species": species_label,
                    "eval_group": pd.NA,
                    "n_rows": species_n,
                    "n_positives": int(y[species_idx].sum()),
                    "n_negatives": int(species_n - int(y[species_idx].sum())),
                    "is_single_class": bool(len(np.unique(y[species_idx])) <= 1),
                    "max_allowed_test_rows": pd.NA,
                    "max_test_fraction": float(max_test_fraction),
                    "component_groups": "",
                    "source_groups": "",
                    "status": "excluded",
                    "grouping_method": "baltic_nodes",
                }
            )
            continue
        max_test_rows = int(np.floor(species_n * max_test_fraction))
        max_test_rows = max(max_test_rows, min_clade_size, 1)
        if species_n > 1:
            max_test_rows = min(max_test_rows, species_n - 1)
        min_test_rows = int(np.ceil(species_n * min_test_fraction))
        min_test_rows = max(min_test_rows, min_clade_size, 1)
        min_test_rows = min(min_test_rows, max_test_rows)
        tree_path = find_refined_species_tree(species_label)
        units = _build_baltic_species_units(
            df,
            y,
            species_label=species_label,
            species_idx=species_idx,
            tree_path=tree_path,
            baltic_path=baltic_path,
            min_clade_size=min_clade_size,
            min_test_rows=min_test_rows,
            max_test_rows=max_test_rows,
            random_state=random_state + species_n,
        )
        packed_groups = _pack_species_units_into_eval_groups(species_label, units, max_test_rows=max_test_rows)
        for eval_num, packed in enumerate(packed_groups, start=1):
            eval_label = f"{species_label}::eval_{eval_num}"
            eval_groups.loc[np.asarray(packed["indices"], dtype=int)] = eval_label
            records.append(
                {
                    "species": species_label,
                    "eval_group": eval_label,
                    "n_rows": int(packed["size"]),
                    "n_positives": int(packed["positives"]),
                    "n_negatives": int(packed["negatives"]),
                    "is_single_class": bool(int(packed["positives"]) == 0 or int(packed["negatives"]) == 0),
                    "max_allowed_test_rows": int(max_test_rows),
                    "max_test_fraction": float(max_test_fraction),
                    "component_groups": "|".join(sorted(set(str(v) for v in packed["component_groups"]))),
                    "source_groups": "|".join(sorted(set(str(v) for v in packed["source_groups"]))),
                    "status": "assigned",
                    "grouping_method": "baltic_nodes",
                }
            )
    manifest = pd.DataFrame(records).sort_values(["species", "eval_group"]).reset_index(drop=True)
    return eval_groups, manifest


def select_feature_sets(requested: Optional[Sequence[str]]) -> List[FeatureSet]:
    feature_set_lookup = {feature_set.name: feature_set for feature_set in FEATURE_SETS}
    if not requested:
        return list(FEATURE_SETS)
    missing = sorted(set(requested) - set(feature_set_lookup))
    if missing:
        raise SystemExit(f"Unknown feature set(s): {', '.join(missing)}")
    return [feature_set_lookup[name] for name in requested]


def select_models(requested: Optional[Sequence[str]]) -> List[ModelSpec]:
    model_lookup = {model.name: model for model in MODELS}
    if not requested:
        return list(MODELS)
    missing = sorted(set(requested) - set(model_lookup))
    if missing:
        raise SystemExit(f"Unknown model(s): {', '.join(missing)}")
    return [model_lookup[name] for name in requested]


def build_preprocessor(numeric: Iterable[str], categorical: Iterable[str]) -> ColumnTransformer:
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse_output=False)),
                    ]
                ),
                list(categorical),
            ),
        ],
    )


@dataclass
class FittedModelBundle:
    preprocessor: ColumnTransformer
    model: object
    feature_names: np.ndarray
    feature_group_labels: List[str]
    tuning_info: Optional[Dict[str, object]] = None

    @property
    def named_steps(self) -> Mapping[str, object]:
        return {"prep": self.preprocessor, "model": self.model}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_transformed = self.preprocessor.transform(X)
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X_transformed)
            prob = np.asarray(prob)
            if prob.ndim == 1:
                prob = np.column_stack([1.0 - prob, prob])
            return prob
        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(X_transformed)).reshape(-1)
            prob_1 = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - prob_1, prob_1])
        pred = np.asarray(self.model.predict(X_transformed)).reshape(-1)
        return np.column_stack([1.0 - pred, pred])


def infer_feature_group_label(feature_name: str) -> str:
    base = feature_name.split("__", 1)[1] if "__" in feature_name else feature_name
    if base.startswith("hosttrait_"):
        return "host_traits"
    if base.startswith("species_feat_hostdiv_"):
        return "species_hostdiv"
    if base.startswith("species_feat_overlapopp_"):
        return "species_overlapopp"
    if base.startswith("species_feat_utr5_"):
        return "species_utr5"
    if base.startswith("species_feat_utr3_"):
        return "species_utr3"
    if base.startswith("species_feat_spr_") or base.startswith("species_feat_mean_spr_"):
        return "species_spr"
    if base.startswith("species_feat_mean_cramers_v_") or base.startswith("species_feat_median_cramers_v_"):
        return "species_ld_summary"
    if base.startswith("species_feat_mean_nmi_") or base.startswith("species_feat_prop_significant_"):
        return "species_ld_summary"
    if base.startswith("species_feat_n_site_pairs_") or base.startswith("species_feat_n_significant_"):
        return "species_ld_summary"
    if base.startswith("ld_site_"):
        return "ld_site"
    if base.startswith("arha_host_") or base.startswith("has_arha_host_"):
        return "direct_overlap"
    if base.startswith("has_S") or base.startswith("has_M") or base.startswith("has_L"):
        return "segment_presence"
    if base.startswith("segment_completeness"):
        return "segment_completeness"
    if base.startswith("year"):
        return "sampling_time"
    if base.startswith("arha_sequence_match_count") or base.startswith("has_arha_sequence_match"):
        return "arha_match"
    if base.startswith("has_host_traits") or base.startswith("hosttrait_has_") or base.startswith("hosttrait_fill_"):
        return "trait_resolution"
    if base.startswith("species_feat_eco_") or base.startswith("species_feat_gate_") or base.startswith("species_feat_run_"):
        return "species_qc"
    if base.startswith("species_feat_"):
        return "species_summary"
    if base.startswith("species_") or base == "species":
        return "species"
    return base.split("_", 1)[0]


def build_feature_group_assignments(feature_names: Sequence[str]) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    group_labels = [infer_feature_group_label(name) for name in feature_names]
    ordered_labels = list(dict.fromkeys(group_labels))
    label_to_id = {label: idx for idx, label in enumerate(ordered_labels)}
    group_vector = np.asarray([label_to_id[label] for label in group_labels], dtype=int)
    group_slices = [np.flatnonzero(group_vector == idx) for idx in range(len(ordered_labels))]
    return group_vector, group_slices, group_labels


def build_model(
    spec: ModelSpec,
    seed: int,
    y_train: np.ndarray,
    *,
    group_vector: Optional[np.ndarray] = None,
    group_slices: Optional[List[np.ndarray]] = None,
) -> object:
    if spec.kind == "logistic_l2":
        return LogisticRegression(
            C=0.1,
            class_weight="balanced",
            penalty="l2",
            solver="liblinear",
            random_state=seed,
            max_iter=4000,
        )
    if spec.kind == "logistic_elasticnet":
        return LogisticRegression(
            C=0.1,
            class_weight="balanced",
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            random_state=seed,
            max_iter=6000,
        )
    if spec.kind == "group_lasso":
        if group_vector is None:
            raise ValueError("group_lasso requires feature-group assignments.")
        return LogisticGroupLasso(
            groups=group_vector,
            group_reg=0.05,
            l1_reg=0.0,
            n_iter=400,
            tol=1e-4,
            scale_reg="group_size",
            fit_intercept=True,
            random_state=seed,
            supress_warning=True,
        )
    if spec.kind == "sparse_group_lasso":
        if group_vector is None:
            raise ValueError("sparse_group_lasso requires feature-group assignments.")
        return LogisticGroupLasso(
            groups=group_vector,
            group_reg=0.03,
            l1_reg=0.02,
            n_iter=400,
            tol=1e-4,
            scale_reg="group_size",
            fit_intercept=True,
            random_state=seed,
            supress_warning=True,
        )
    if spec.kind == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if spec.kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=seed,
        )
    if spec.kind == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed.")
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            depth=4,
            learning_rate=0.05,
            iterations=400,
            l2_leaf_reg=5.0,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    if spec.kind == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=4.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    if spec.kind == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed.")
        return LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=12,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unsupported model kind: {spec.kind}")


def is_tree_model(spec: ModelSpec) -> bool:
    return spec.kind in TREE_MODEL_KINDS


def build_tree_search_space(spec: ModelSpec) -> Mapping[str, object]:
    if Integer is None or Real is None or Categorical is None:
        raise RuntimeError("scikit-optimize is not installed.")
    if spec.kind == "rf":
        return {
            "n_estimators": Integer(200, 900),
            "max_depth": Integer(3, 10),
            "min_samples_leaf": Integer(1, 20),
            "min_samples_split": Integer(2, 30),
            "max_features": Categorical(["sqrt", "log2", None]),
        }
    if spec.kind == "gb":
        return {
            "n_estimators": Integer(120, 600),
            "learning_rate": Real(0.01, 0.20, prior="log-uniform"),
            "max_depth": Integer(1, 4),
            "subsample": Real(0.6, 1.0),
            "min_samples_leaf": Integer(1, 20),
            "min_samples_split": Integer(2, 30),
        }
    if spec.kind == "catboost":
        return {
            "depth": Integer(3, 8),
            "learning_rate": Real(0.01, 0.20, prior="log-uniform"),
            "iterations": Integer(250, 1000),
            "l2_leaf_reg": Real(1.0, 20.0, prior="log-uniform"),
            "random_strength": Real(1e-3, 5.0, prior="log-uniform"),
        }
    if spec.kind == "xgboost":
        return {
            "n_estimators": Integer(200, 900),
            "max_depth": Integer(2, 8),
            "learning_rate": Real(0.01, 0.20, prior="log-uniform"),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
            "reg_lambda": Real(1e-2, 10.0, prior="log-uniform"),
            "min_child_weight": Real(1.0, 10.0),
            "gamma": Real(0.0, 5.0),
        }
    if spec.kind == "lightgbm":
        return {
            "n_estimators": Integer(200, 900),
            "learning_rate": Real(0.01, 0.20, prior="log-uniform"),
            "max_depth": Integer(2, 8),
            "num_leaves": Integer(7, 63),
            "min_child_samples": Integer(5, 40),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
            "reg_lambda": Real(1e-2, 10.0, prior="log-uniform"),
        }
    raise ValueError(f"No search space defined for model kind: {spec.kind}")


def tuned_metric_scorer(metric_name: str) -> Callable[[object, Any, np.ndarray], float]:
    def _predict_proba_1(estimator: object, X: Any) -> np.ndarray:
        if hasattr(estimator, "predict_proba"):
            prob = np.asarray(estimator.predict_proba(X))
            if prob.ndim == 1:
                return prob.reshape(-1)
            return prob[:, -1]
        if hasattr(estimator, "decision_function"):
            scores = np.asarray(estimator.decision_function(X)).reshape(-1)
            return 1.0 / (1.0 + np.exp(-scores))
        pred = np.asarray(estimator.predict(X)).reshape(-1)
        return pred.astype(float)

    def _score(estimator: object, X: Any, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        y_true = np.asarray(y_true).astype(int)
        y_prob = _predict_proba_1(estimator, X)
        if metric_name == "average_precision":
            return safe_average_precision(y_true, y_prob)
        if metric_name == "roc_auc":
            return safe_roc_auc(y_true, y_prob)
        if metric_name == "neg_brier":
            return -float(brier_score_loss(y_true, y_prob))
        if metric_name == "balanced_accuracy":
            return safe_balanced_accuracy(y_true, (y_prob >= 0.5).astype(int))
        raise ValueError(f"Unsupported tuning metric: {metric_name}")

    return _score


def build_inner_cv_splits(
    cv_scheme: str,
    y_train: np.ndarray,
    *,
    species_groups: pd.Series,
    clade_groups: pd.Series,
    within_species_groups: pd.Series,
    inner_splits: int,
    random_state: int,
) -> Tuple[Optional[List[Tuple[np.ndarray, np.ndarray]]], str]:
    y_train = np.asarray(y_train).astype(int)
    group_series: Optional[pd.Series] = None
    group_label = "none"
    if cv_scheme == "grouped_species":
        group_series = species_groups.astype(str)
        group_label = "species"
    elif cv_scheme == "grouped_clade":
        group_series = clade_groups.astype(str)
        group_label = "clade"
    elif cv_scheme == "grouped_within_species_clade":
        if within_species_groups.notna().any():
            group_series = within_species_groups.astype("object")
            group_label = "within_species_clade"
    if group_series is not None:
        valid = group_series.notna().to_numpy()
        idx = np.flatnonzero(valid)
        groups = group_series[valid].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            n_splits = min(inner_splits, len(unique_groups))
            if n_splits >= 2:
                splitter = GroupKFold(n_splits=n_splits)
                splits = [(idx[tr], idx[te]) for tr, te in splitter.split(np.zeros((len(idx), 1)), y_train[idx], groups=groups)]
                usable = [
                    (tr, te)
                    for tr, te in splits
                    if len(np.unique(y_train[tr])) >= 2 and len(np.unique(y_train[te])) >= 1
                ]
                if len(usable) >= 2:
                    return usable, group_label
    class_counts = np.bincount(y_train, minlength=2)
    min_class = int(class_counts.min()) if class_counts.size else 0
    n_splits = min(inner_splits, min_class)
    if n_splits >= 2:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(np.zeros((len(y_train), 1)), y_train)), "stratified"
    return None, "unavailable"


def fit_tree_model_with_bayes_search(
    spec: ModelSpec,
    base_model: object,
    X_train_transformed: np.ndarray,
    y_train: np.ndarray,
    *,
    sample_weight: np.ndarray,
    tuning_config: TreeTuningConfig,
    seed: int,
) -> Tuple[object, Dict[str, object]]:
    inner_cv_splits, inner_cv_label = build_inner_cv_splits(
        tuning_config.cv_scheme,
        y_train,
        species_groups=tuning_config.species_groups,
        clade_groups=tuning_config.clade_groups,
        within_species_groups=tuning_config.within_species_groups,
        inner_splits=tuning_config.inner_splits,
        random_state=tuning_config.random_state + seed,
    )
    info: Dict[str, object] = {
        "tuning_status": "skipped",
        "tuning_method": "bayes_search",
        "tuning_metric": tuning_config.metric,
        "tuning_inner_cv": inner_cv_label,
        "tuning_n_iter": int(tuning_config.n_iter),
        "tuning_n_inner_splits": 0 if inner_cv_splits is None else int(len(inner_cv_splits)),
        "tuning_best_score": np.nan,
        "tuning_best_params_json": "",
    }
    if BayesSearchCV is None or inner_cv_splits is None:
        model = clone(base_model)
        model.fit(X_train_transformed, y_train, sample_weight=sample_weight)
        return model, info
    search = BayesSearchCV(
        estimator=clone(base_model),
        search_spaces=build_tree_search_space(spec),
        n_iter=tuning_config.n_iter,
        scoring=tuned_metric_scorer(tuning_config.metric),
        cv=inner_cv_splits,
        n_jobs=1,
        refit=True,
        random_state=seed,
        return_train_score=False,
        error_score=np.nan,
    )
    try:
        search.fit(X_train_transformed, y_train, sample_weight=sample_weight)
        info.update(
            {
                "tuning_status": "success",
                "tuning_n_inner_splits": int(len(inner_cv_splits)),
                "tuning_best_score": float(search.best_score_) if search.best_score_ is not None else np.nan,
                "tuning_best_params_json": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        return search.best_estimator_, info
    except Exception as exc:  # pragma: no cover - defensive fallback
        model = clone(base_model)
        model.fit(X_train_transformed, y_train, sample_weight=sample_weight)
        info.update(
            {
                "tuning_status": f"fallback:{exc.__class__.__name__}",
                "tuning_error": str(exc),
            }
        )
        return model, info


def tune_tree_params_shared(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    numeric: List[str],
    categorical: List[str],
    *,
    cv_scheme: str,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_iter: int,
    metric: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, object]]:
    info: Dict[str, object] = {
        "tuning_status": "skipped",
        "tuning_method": "bayes_search_shared_by_cv",
        "tuning_metric": metric,
        "tuning_inner_cv": cv_scheme,
        "tuning_n_iter": int(n_iter),
        "tuning_n_inner_splits": int(len(splits)),
        "tuning_best_score": np.nan,
        "tuning_best_params_json": "",
    }
    if BayesSearchCV is None:
        return None, info
    preprocessor = build_preprocessor(numeric, categorical)
    X_transformed = preprocessor.fit_transform(X)
    base_model = build_model(spec, seed, y)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    search = BayesSearchCV(
        estimator=clone(base_model),
        search_spaces=build_tree_search_space(spec),
        n_iter=n_iter,
        scoring=tuned_metric_scorer(metric),
        cv=splits,
        n_jobs=1,
        refit=False,
        random_state=seed,
        return_train_score=False,
        error_score=np.nan,
    )
    try:
        search.fit(X_transformed, y, sample_weight=sample_weight)
        best_params = dict(search.best_params_)
        info.update(
            {
                "tuning_status": "success",
                "tuning_best_score": float(search.best_score_) if search.best_score_ is not None else np.nan,
                "tuning_best_params_json": json.dumps(best_params, sort_keys=True),
            }
        )
        return best_params, info
    except Exception as exc:  # pragma: no cover - defensive fallback
        info.update(
            {
                "tuning_status": f"fallback:{exc.__class__.__name__}",
                "tuning_error": str(exc),
            }
        )
        return None, info


def fit_pipeline(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric: List[str],
    categorical: List[str],
    seed: int,
    *,
    tree_tuning: Optional[TreeTuningConfig] = None,
    tuned_params: Optional[Mapping[str, Any]] = None,
    tuned_info: Optional[Dict[str, object]] = None,
) -> FittedModelBundle:
    preprocessor = build_preprocessor(numeric, categorical)
    X_train_transformed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    group_vector, group_slices, group_labels = build_feature_group_assignments(feature_names)
    base_model = build_model(
        spec,
        seed,
        y_train,
        group_vector=group_vector,
        group_slices=group_slices,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    tuning_info: Optional[Dict[str, object]] = dict(tuned_info) if tuned_info else None
    if tuned_params is not None and is_tree_model(spec):
        model = clone(base_model).set_params(**dict(tuned_params))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(X_train_transformed, y_train, sample_weight=sample_weight)
    elif tree_tuning is not None and tree_tuning.enabled and is_tree_model(spec):
        model, tuning_info = fit_tree_model_with_bayes_search(
            spec,
            base_model,
            X_train_transformed,
            y_train,
            sample_weight=sample_weight,
            tuning_config=tree_tuning,
            seed=seed,
        )
    else:
        model = base_model
        fit_kwargs = {}
        if spec.kind in TREE_MODEL_KINDS:
            fit_kwargs["sample_weight"] = sample_weight
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(X_train_transformed, y_train, **fit_kwargs)
    return FittedModelBundle(
        preprocessor=preprocessor,
        model=model,
        feature_names=np.asarray(feature_names),
        feature_group_labels=group_labels,
        tuning_info=tuning_info,
    )


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    labels = np.unique(y_true)
    if len(labels) == 1:
        return 1.0 if labels[0] == 1 else 0.0
    return float(average_precision_score(y_true, y_prob))


def safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    if len(labels) < 2:
        return float((y_pred == y_true).mean())
    return float(balanced_accuracy_score(y_true, y_pred))


def metric_row(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "average_precision": safe_average_precision(y_true, y_prob),
        "balanced_accuracy": safe_balanced_accuracy(y_true, y_pred),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def build_cv_plans(
    args: argparse.Namespace,
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    within_species_eval_groups: Optional[pd.Series] = None,
) -> List[Tuple[str, List[Tuple[np.ndarray, np.ndarray]]]]:
    requested = args.cv_schemes or ["grouped_species", "grouped_within_species_clade", "repeated_stratified"]
    plans: List[Tuple[str, List[Tuple[np.ndarray, np.ndarray]]]] = []
    if "grouped_species" in requested:
        grouped_splits = list(LeaveOneGroupOut().split(df, y, groups=groups))
        if args.max_folds is not None:
            grouped_splits = grouped_splits[: args.max_folds]
        if grouped_splits:
            plans.append(("grouped_species", grouped_splits))
    if "grouped_clade" in requested:
        clade_groups = assign_species_clade(df["species"].fillna("missing_species").astype(str)).to_numpy()
        clade_splits = list(LeaveOneGroupOut().split(df, y, groups=clade_groups))
        if args.max_folds is not None:
            clade_splits = clade_splits[: args.max_folds]
        if clade_splits:
            plans.append(("grouped_clade", clade_splits))
    if "grouped_within_species_clade" in requested and within_species_eval_groups is not None:
        valid = within_species_eval_groups.notna().to_numpy()
        unique_groups = list(dict.fromkeys(within_species_eval_groups[valid].astype(str).tolist()))
        split_list: List[Tuple[np.ndarray, np.ndarray]] = []
        for group_label in unique_groups:
            test_mask = within_species_eval_groups.astype(str).eq(group_label).to_numpy() & valid
            if not test_mask.any():
                continue
            train_idx = np.flatnonzero(~test_mask)
            test_idx = np.flatnonzero(test_mask)
            split_list.append((train_idx, test_idx))
        if args.max_folds is not None:
            split_list = split_list[: args.max_folds]
        if split_list:
            plans.append(("grouped_within_species_clade", split_list))
    if "repeated_stratified" in requested:
        repeated = RepeatedStratifiedKFold(
            n_splits=args.repeated_splits,
            n_repeats=args.repeated_repeats,
            random_state=args.random_state,
        )
        repeated_splits = list(repeated.split(df, y))
        if args.max_folds is not None:
            repeated_splits = repeated_splits[: args.max_folds]
        if repeated_splits:
            plans.append(("repeated_stratified", repeated_splits))
    return plans


def _mode_or_nan(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode(dropna=True)
    if modes.empty:
        return np.nan
    return modes.iloc[0]


def _best_accession(row: pd.Series) -> Optional[str]:
    for col in ["accession", "accession_primary", "query_accession"]:
        value = row.get(col, np.nan)
        if pd.notna(value):
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
    return None


def _normalize_join_key(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    text = text.split("|")[0]
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or None


def _load_utr_metadata(species: str, segment: str) -> Optional[pd.DataFrame]:
    key = (species, segment)
    if key in _UTR_METADATA_CACHE:
        return _UTR_METADATA_CACHE[key]
    path = UTR_RUN_ROOT / species / segment / "utr5_metadata.tsv"
    if not path.exists():
        _UTR_METADATA_CACHE[key] = None
        return None
    df = pd.read_csv(path, sep="\t")
    df["accession"] = df["accession"].astype(str)
    _UTR_METADATA_CACHE[key] = df
    return df


def _load_utr_struct_matrix(species: str, segment: str) -> Optional[pd.DataFrame]:
    key = (species, segment)
    if key in _UTR_STRUCT_CACHE:
        return _UTR_STRUCT_CACHE[key]
    path = UTR_RUN_ROOT / species / segment / "utr5_structure_distance.tsv"
    if not path.exists():
        _UTR_STRUCT_CACHE[key] = None
        return None
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        _UTR_STRUCT_CACHE[key] = None
        return None
    df = df.set_index("id")
    _UTR_STRUCT_CACHE[key] = df
    return df


def _load_segment_accession_map(species: str, segment: str) -> Dict[str, str]:
    key = (species, segment)
    if key in _SEGMENT_ACCESSION_CACHE:
        return _SEGMENT_ACCESSION_CACHE[key]

    mapping: Dict[str, str] = {}

    def _store(candidates: Sequence[object], accession: object) -> None:
        if accession is None or pd.isna(accession):
            return
        acc = str(accession).strip()
        if not acc or acc.lower() == "nan":
            return
        mapping.setdefault(acc, acc)
        for candidate in candidates:
            norm = _normalize_join_key(candidate)
            if norm:
                mapping.setdefault(norm, acc)

    for root in TREE_SEARCH_ROOTS:
        augur_path = root / species / "augur" / segment / f"{segment}_matched_metadata.tsv"
        if not augur_path.exists():
            continue
        df = pd.read_csv(augur_path, sep="\t")
        for _, row in df.iterrows():
            _store(
                [
                    row.get("accession"),
                    row.get("strain"),
                    row.get("canonical_isolate"),
                    row.get("isolate_display"),
                ],
                row.get("accession"),
            )

    for base in [
        REPO_ROOT / "data" / "processed" / "trees",
    ]:
        input_path = (
            base
            / "inputs"
            / "species"
            / species
            / f"{segment}_segment"
            / "metadata.tsv"
        )
        if not input_path.exists():
            continue
        df = pd.read_csv(input_path, sep="\t")
        for _, row in df.iterrows():
            _store(
                [
                    row.get("Accession"),
                    row.get("Isolate"),
                    row.get("GenBank_Title"),
                ],
                row.get("Accession"),
            )

    for base in [
        REPO_ROOT / "data" / "processed" / "global_segments",
    ]:
        path = base / segment / f"{segment}_metadata.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        if "species" in df.columns:
            df = df[df["species"].map(_species_slug).eq(species)].copy()
        for _, row in df.iterrows():
            _store(
                [
                    row.get("accession"),
                    row.get("strain"),
                    row.get("canonical_isolate"),
                ],
                row.get("accession"),
            )

    filtered_df = _load_filtered_virus_sequences()
    if filtered_df is not None:
        sub = filtered_df[
            filtered_df["species_slug"].eq(species)
            & filtered_df["segment_norm"].eq(str(segment).upper())
        ].copy()
        for _, row in sub.iterrows():
            candidates: List[object] = [
                row.get("Accession"),
                row.get("Isolate"),
                row.get("GenBank_Title"),
            ]
            candidates.extend(_extract_title_isolate_candidates(row.get("GenBank_Title")))
            _store(candidates, row.get("Accession"))
    _SEGMENT_ACCESSION_CACHE[key] = mapping
    return mapping


def _resolve_segment_accession(
    row: pd.Series,
    species: str,
    segment: str,
    valid_accessions: Optional[set[str]] = None,
) -> Optional[str]:
    candidates: List[str] = []
    for col in [
        "accession",
        "accession_primary",
        "query_accession",
        "tip_name",
        "canonical_isolate",
        "display_name",
    ]:
        value = row.get(col, np.nan)
        if pd.notna(value):
            text = str(value).strip()
            if text and text.lower() != "nan":
                candidates.append(text)

    if valid_accessions:
        for candidate in candidates:
            if candidate in valid_accessions:
                return candidate

    segment_map = _load_segment_accession_map(species, segment)
    for candidate in candidates:
        if candidate in segment_map:
            resolved = segment_map[candidate]
            if valid_accessions is None or resolved in valid_accessions:
                return resolved
        norm = _normalize_join_key(candidate)
        if norm and norm in segment_map:
            resolved = segment_map[norm]
            if valid_accessions is None or resolved in valid_accessions:
                return resolved
    return None


def _find_segment_tree(species: str, segment: str) -> Optional[Path]:
    key = (species, segment)
    if key in _SEGMENT_TREE_PATH_CACHE:
        return _SEGMENT_TREE_PATH_CACHE[key]
    for root in TREE_SEARCH_ROOTS:
        for ext in ("newick", "nwk"):
            candidate = root / species / "augur" / segment / f"{segment}_tree_refined.{ext}"
            if candidate.exists():
                _SEGMENT_TREE_PATH_CACHE[key] = candidate
                return candidate
    _SEGMENT_TREE_PATH_CACHE[key] = None
    return None


def _load_segment_tree_distance_matrix(species: str, segment: str) -> Optional[pd.DataFrame]:
    key = (species, segment)
    if key in _SEGMENT_TREE_DISTANCE_CACHE:
        return _SEGMENT_TREE_DISTANCE_CACHE[key]
    tree_path = _find_segment_tree(species, segment)
    if tree_path is None:
        _SEGMENT_TREE_DISTANCE_CACHE[key] = None
        return None
    tree = Phylo.read(str(tree_path), "newick")
    tips = [t.name for t in tree.get_terminals() if t.name]
    if not tips:
        _SEGMENT_TREE_DISTANCE_CACHE[key] = None
        return None
    mat = np.zeros((len(tips), len(tips)), dtype=float)
    for i, tip_i in enumerate(tips):
        for j in range(i + 1, len(tips)):
            d = float(tree.distance(tip_i, tips[j]))
            mat[i, j] = d
            mat[j, i] = d
    df = pd.DataFrame(mat, index=tips, columns=tips)
    _SEGMENT_TREE_DISTANCE_CACHE[key] = df
    return df


def _find_segment_alignment(species: str, segment: str) -> Optional[Path]:
    key = (species, segment)
    if key in _SEGMENT_ALIGNMENT_PATH_CACHE:
        return _SEGMENT_ALIGNMENT_PATH_CACHE[key]
    for root in TREE_SEARCH_ROOTS:
        candidates = [
            root / species / "augur" / segment / f"{segment}_aligned.fasta",
            root / species / "espalier" / "alignments" / f"{segment}_aligned_common.fasta",
        ]
        for candidate in candidates:
            if candidate.exists():
                _SEGMENT_ALIGNMENT_PATH_CACHE[key] = candidate
                return candidate
    _SEGMENT_ALIGNMENT_PATH_CACHE[key] = None
    return None


def _find_segment_matched_fasta(species: str, segment: str) -> Optional[Path]:
    key = (species, segment)
    if key in _SEGMENT_MATCHED_FASTA_PATH_CACHE:
        return _SEGMENT_MATCHED_FASTA_PATH_CACHE[key]
    for root in TREE_SEARCH_ROOTS:
        candidates = [
            root / species / "augur" / segment / f"{segment}_matched.fasta",
            root / species / "segments" / segment / f"{segment}_matched.fasta",
        ]
        for candidate in candidates:
            if candidate.exists():
                _SEGMENT_MATCHED_FASTA_PATH_CACHE[key] = candidate
                return candidate
    _SEGMENT_MATCHED_FASTA_PATH_CACHE[key] = None
    return None


def _load_segment_tip_sequences(species: str, segment: str) -> Optional[Dict[str, str]]:
    key = (species, segment)
    if key in _SEGMENT_TIP_SEQUENCE_CACHE:
        return _SEGMENT_TIP_SEQUENCE_CACHE[key]
    seqs: Dict[str, str] = {}
    for root in TREE_SEARCH_ROOTS:
        meta_by_strain: Dict[str, Dict[str, object]] = {}
        meta_path = root / species / "augur" / segment / f"{segment}_matched_metadata.tsv"
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path, sep="\t")
            if "strain" in meta_df.columns:
                for _, row in meta_df.iterrows():
                    meta_by_strain[str(row.get("strain"))] = row.to_dict()
        for path in [
            root / species / "augur" / segment / f"{segment}_matched.fasta",
            root / species / "segments" / segment / f"{segment}_matched.fasta",
        ]:
            if not path.exists():
                continue
            for rec in SeqIO.parse(str(path), "fasta"):
                seq = str(rec.seq).upper()
                rec_id = str(rec.id)
                seqs.setdefault(rec_id, seq)
                norm = _normalize_join_key(rec.id)
                if norm:
                    seqs.setdefault(norm, seq)
                meta_row = meta_by_strain.get(rec_id)
                if meta_row:
                    for candidate in [
                        meta_row.get("accession"),
                        meta_row.get("canonical_isolate"),
                        meta_row.get("isolate_display"),
                    ]:
                        cand_text = str(candidate).strip() if pd.notna(candidate) else ""
                        if not cand_text or cand_text.lower() == "nan":
                            continue
                        seqs.setdefault(cand_text, seq)
                        cand_norm = _normalize_join_key(cand_text)
                        if cand_norm:
                            seqs.setdefault(cand_norm, seq)
    if not seqs:
        _SEGMENT_TIP_SEQUENCE_CACHE[key] = None
        return None
    _SEGMENT_TIP_SEQUENCE_CACHE[key] = seqs
    return seqs


def _load_segment_input_cds_sequences(species: str, segment: str) -> Optional[Dict[str, str]]:
    key = (species, segment)
    if key in _SEGMENT_INPUT_CDS_CACHE:
        return _SEGMENT_INPUT_CDS_CACHE[key]
    seqs: Dict[str, str] = {}
    input_roots = [
        REPO_ROOT / "data" / "processed" / "species_inputs",
    ]
    for base in input_roots:
        meta_path = base / species / f"{segment}_segment" / "metadata.tsv"
        fasta_path = base / species / f"{segment}_segment" / "sequences.fasta"
        if not (meta_path.exists() and fasta_path.exists()):
            continue
        meta_df = pd.read_csv(meta_path, sep="\t")
        recs = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not recs:
            continue
        if len(meta_df) != len(recs):
            continue
        for (_, row), rec in zip(meta_df.iterrows(), recs):
            seq = str(rec.seq).upper().replace("-", "")
            candidates: List[object] = [
                row.get("Accession"),
                row.get("Isolate"),
                row.get("GenBank_Title"),
            ]
            candidates.extend(_extract_title_isolate_candidates(row.get("GenBank_Title")))
            for candidate in candidates:
                if candidate is None or pd.isna(candidate):
                    continue
                text = str(candidate).strip()
                if not text or text.lower() == "nan":
                    continue
                seqs.setdefault(text, seq)
                norm = _normalize_join_key(text)
                if norm:
                    seqs.setdefault(norm, seq)
    _SEGMENT_INPUT_CDS_CACHE[key] = seqs or None
    return _SEGMENT_INPUT_CDS_CACHE[key]


def _compute_sequence_distance_matrix(path: Path) -> Optional[pd.DataFrame]:
    recs = list(SeqIO.parse(str(path), "fasta"))
    if not recs:
        return None
    names = [str(r.id) for r in recs]
    seqs = [np.frombuffer(str(r.seq).upper().encode("ascii"), dtype="S1") for r in recs]
    if not seqs:
        return None
    arr = np.vstack(seqs)
    invalid = np.isin(arr, [b"N", b"-", b"?", b"X"])
    n = arr.shape[0]
    dist = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(dist, 0.0)
    for i in range(n):
        ai = arr[i]
        inv_i = invalid[i]
        for j in range(i + 1, n):
            valid = ~(inv_i | invalid[j])
            denom = int(valid.sum())
            if denom <= 0:
                continue
            d = float((ai[valid] != arr[j][valid]).mean())
            dist[i, j] = d
            dist[j, i] = d
    return pd.DataFrame(dist, index=names, columns=names)


def _load_segment_sequence_distance_matrix(species: str, segment: str) -> Optional[pd.DataFrame]:
    key = (species, segment)
    if key in _SEGMENT_SEQUENCE_DISTANCE_CACHE:
        return _SEGMENT_SEQUENCE_DISTANCE_CACHE[key]
    path = _find_segment_alignment(species, segment)
    if path is None:
        _SEGMENT_SEQUENCE_DISTANCE_CACHE[key] = None
        return None
    df = _compute_sequence_distance_matrix(path)
    _SEGMENT_SEQUENCE_DISTANCE_CACHE[key] = df
    return df


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_r = radians(float(lat1))
    lon1_r = radians(float(lon1))
    lat2_r = np.radians(lat2.astype(float))
    lon2_r = np.radians(lon2.astype(float))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + cos(lat1_r) * np.cos(lat2_r) * (np.sin(dlon / 2.0) ** 2)
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(a))


SENSE_CODONS: Tuple[str, ...] = tuple(
    codon
    for codon in (
        a + b + c
        for a in "TCAG"
        for b in "TCAG"
        for c in "TCAG"
    )
    if codon not in {"TAA", "TAG", "TGA"}
)
DINUCS: Tuple[str, ...] = tuple(a + b for a in "ACGT" for b in "ACGT")


def _species_slug(text: object) -> str:
    if text is None or pd.isna(text):
        return ""
    return str(text).strip().lower().replace(" ", "_")


def _extract_title_isolate_candidates(title: object) -> List[str]:
    if title is None or pd.isna(title):
        return []
    text = str(title).strip()
    patterns = [
        r"\bstrain\s+(.+?)\s+segment\b",
        r"\bisolate\s+(.+?)\s+(?:segment\b|putative\b|nucleoprotein\b|glycoprotein\b|complete\b)",
    ]
    candidates: List[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidates.append(match.group(1).strip())
    return candidates


def _load_filtered_virus_sequences() -> Optional[pd.DataFrame]:
    global _FILTERED_VIRUS_SEQ_CACHE
    if _FILTERED_VIRUS_SEQ_CACHE is not None:
        return _FILTERED_VIRUS_SEQ_CACHE
    path = REPO_ROOT / "results" / "intermediate" / "filtered_virus_host_sequences.csv"
    if not path.exists():
        _FILTERED_VIRUS_SEQ_CACHE = None
        return None
    df = pd.read_csv(path, low_memory=False)
    required = {"Species", "Segment", "Accession"}
    if not required.issubset(df.columns):
        _FILTERED_VIRUS_SEQ_CACHE = None
        return None
    df = df.copy()
    df["species_slug"] = df["Species"].map(_species_slug)
    df["segment_norm"] = df["Segment"].astype(str).str.upper().str.strip()
    _FILTERED_VIRUS_SEQ_CACHE = df
    return df


def _load_local_genbank_sequences() -> Dict[str, str]:
    global _LOCAL_GENBANK_SEQ_CACHE
    if _LOCAL_GENBANK_SEQ_CACHE is not None:
        return _LOCAL_GENBANK_SEQ_CACHE
    seqs: Dict[str, str] = {}
    path = REPO_ROOT / "src" / "r" / "analysis" / "NP_NT_Seqs.csv"
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        if {"Accession", "Sequence"}.issubset(df.columns):
            for _, row in df.iterrows():
                accession = str(row.get("Accession", "")).strip()
                seq = str(row.get("Sequence", "")).strip().upper()
                if not accession or accession.lower() == "nan" or not seq or seq.lower() == "nan":
                    continue
                seqs.setdefault(accession, seq)
                seqs.setdefault(accession.split(".")[0], seq)
    _LOCAL_GENBANK_SEQ_CACHE = seqs
    return seqs


def _ensure_entrez_client() -> None:
    if getattr(Entrez, "email", None):
        return
    email = os.getenv("NCBI_EMAIL") or os.getenv("ENTREZ_EMAIL")
    if not email:
        try:
            email = subprocess.check_output(["git", "config", "user.email"], text=True).strip()
        except Exception:
            email = "87875949+RicardoRH96@users.noreply.github.com"
    Entrez.email = email
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def _fetch_entrez_text_cached(accession: str, rettype: str, retmode: str, suffix: str) -> Optional[str]:
    cache_root = REPO_ROOT / "results" / "cache" / "codon_entrez_cache"
    cache_path = cache_root / suffix / f"{accession}.{suffix}"
    if cache_path.exists():
        return cache_path.read_text()
    try:
        _ensure_entrez_client()
        handle = Entrez.efetch(db="nuccore", id=accession, rettype=rettype, retmode=retmode)
        text = handle.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
        time.sleep(0.34)
        return text
    except Exception:
        return None


def _fetch_entrez_fasta_sequence(accession: str) -> Optional[str]:
    if accession in _ENTREZ_FASTA_CACHE:
        return _ENTREZ_FASTA_CACHE[accession]
    local = _load_local_genbank_sequences()
    if accession in local:
        _ENTREZ_FASTA_CACHE[accession] = local[accession]
        return local[accession]
    base = accession.split(".")[0]
    if base in local:
        _ENTREZ_FASTA_CACHE[accession] = local[base]
        return local[base]
    text = _fetch_entrez_text_cached(accession, "fasta", "text", "fasta")
    if not text:
        _ENTREZ_FASTA_CACHE[accession] = None
        return None
    try:
        record = SeqIO.read(io.StringIO(text), "fasta")
        seq = str(record.seq).upper()
    except Exception:
        _ENTREZ_FASTA_CACHE[accession] = None
        return None
    _ENTREZ_FASTA_CACHE[accession] = seq
    return seq


def _load_utr_cached_fasta_sequence(accession: str) -> Optional[str]:
    if accession in _UTR_FASTA_CACHE:
        return _UTR_FASTA_CACHE[accession]
    cache_root = REPO_ROOT / "results" / "cache" / "utr5" / "fasta"
    candidates = [
        cache_root / f"{accession}.fasta",
        cache_root / f"{accession.split('.')[0]}.fasta",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            record = SeqIO.read(str(path), "fasta")
            seq = str(record.seq).upper()
            _UTR_FASTA_CACHE[accession] = seq
            return seq
        except Exception:
            continue
    _UTR_FASTA_CACHE[accession] = None
    return None


def _reverse_complement_dna(seq: str) -> str:
    trans = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return str(seq).translate(trans)[::-1]


def _parse_gff3_cds(text: str) -> List[Tuple[int, int, str]]:
    cds: List[Tuple[int, int, str]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9 or parts[2].strip().lower() != "cds":
            continue
        try:
            start = int(parts[3])
            end = int(parts[4])
        except ValueError:
            continue
        strand = parts[6].strip()
        if strand not in {"+", "-"}:
            continue
        cds.append((start, end, strand))
    return cds


def _choose_primary_cds(candidates: List[Tuple[int, int, str]]) -> Optional[Tuple[int, int, str]]:
    if not candidates:
        return None
    strands = {strand for _, _, strand in candidates}
    if len(strands) != 1:
        return None
    strand = next(iter(strands))
    if strand == "+":
        return min(candidates, key=lambda x: (x[0], x[1]))
    return max(candidates, key=lambda x: (x[1], x[0]))


def _fetch_entrez_gff3_cds(accession: str) -> Optional[Tuple[int, int, str]]:
    if accession in _ENTREZ_GFF3_CACHE:
        text = _ENTREZ_GFF3_CACHE[accession]
    else:
        text = _fetch_entrez_text_cached(accession, "gff3", "text", "gff3")
        _ENTREZ_GFF3_CACHE[accession] = text
    if not text:
        return None
    return _choose_primary_cds(_parse_gff3_cds(text))


def _safe_jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        keep = (a > 0) & (b > 0)
        if not keep.any():
            return 0.0
        return float(np.sum(a[keep] * np.log2(a[keep] / b[keep])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _extract_segment_cds_sequence(row: pd.Series, species: str, segment: str) -> Optional[str]:
    seq_map = _load_segment_tip_sequences(species, segment)
    input_cds_map = _load_segment_input_cds_sequences(species, segment)
    meta = _load_utr_metadata(species, segment)
    valid = set(meta["accession"].astype(str)) if meta is not None else None
    accession = _resolve_segment_accession(row, species, segment, valid)
    fallback_accession = accession or _resolve_segment_accession(row, species, segment, None)
    candidates: List[str] = []
    for col in ["tip_name", "canonical_isolate", "display_name", "accession", "accession_primary", "query_accession"]:
        value = row.get(col, np.nan)
        if pd.notna(value):
            text = str(value).strip()
            if text and text.lower() != "nan":
                candidates.append(text)
    if accession is not None:
        candidates.insert(0, accession)
    elif fallback_accession is not None:
        candidates.insert(0, fallback_accession)

    if input_cds_map:
        for candidate in candidates:
            if candidate in input_cds_map:
                cds = input_cds_map[candidate]
                trim = len(cds) - (len(cds) % 3)
                cds = cds[:trim]
                if len(cds) >= 9:
                    return cds
            norm = _normalize_join_key(candidate)
            if norm and norm in input_cds_map:
                cds = input_cds_map[norm]
                trim = len(cds) - (len(cds) % 3)
                cds = cds[:trim]
                if len(cds) >= 9:
                    return cds

    seq: Optional[str] = None
    if seq_map:
        for candidate in candidates:
            if candidate in seq_map:
                seq = seq_map[candidate]
                break
            norm = _normalize_join_key(candidate)
            if norm and norm in seq_map:
                seq = seq_map[norm]
                break

    cds_info: Optional[Tuple[int, int, str]] = None
    meta_lookup_accession = accession or fallback_accession
    if meta_lookup_accession is not None and meta is not None:
        meta_row = meta[meta["accession"].astype(str) == str(meta_lookup_accession)]
        if not meta_row.empty:
            cds_start = pd.to_numeric(meta_row["cds_start"], errors="coerce").dropna()
            cds_end = pd.to_numeric(meta_row["cds_end"], errors="coerce").dropna()
            if not cds_start.empty and not cds_end.empty:
                strand = str(meta_row["cds_strand"].iloc[0]) if "cds_strand" in meta_row.columns else "+"
                cds_info = (int(cds_start.iloc[0]), int(cds_end.iloc[0]), strand)

    if (seq is None or cds_info is None) and fallback_accession is not None:
        entrez_seq = _fetch_entrez_fasta_sequence(fallback_accession)
        entrez_cds = _fetch_entrez_gff3_cds(fallback_accession)
        if entrez_seq is not None:
            seq = entrez_seq
        if entrez_cds is not None:
            cds_info = entrez_cds

    if seq is None or cds_info is None:
        return None

    start = int(cds_info[0]) - 1
    end = int(cds_info[1])
    seq_clean = seq.upper().replace("-", "")
    expected_len = end - start
    if start >= 0 and end > start and end <= len(seq):
        cds = seq[start:end].upper().replace("-", "")
    elif expected_len > 0 and abs(len(seq_clean) - expected_len) <= 6:
        # Many matched FASTAs are already CDS-only rather than full segments.
        cds = seq_clean
    else:
        return None
    cds = "".join(base for base in cds if base in {"A", "C", "G", "T", "N"})
    trim = len(cds) - (len(cds) % 3)
    cds = cds[:trim]
    if len(cds) < 9:
        return None
    return cds


def _extract_segment_terminal_info(row: pd.Series, species: str, segment: str) -> Optional[Dict[str, object]]:
    meta = _load_utr_metadata(species, segment)
    valid = set(meta["accession"].astype(str)) if meta is not None else None
    accession = _resolve_segment_accession(row, species, segment, valid)
    fallback_accession = accession or _resolve_segment_accession(row, species, segment, None)
    if fallback_accession is None:
        return None
    cache_key = (species, segment, str(fallback_accession))
    if cache_key in _SEGMENT_TERMINAL_CACHE:
        return _SEGMENT_TERMINAL_CACHE[cache_key]

    candidates: List[str] = []
    for col in ["tip_name", "canonical_isolate", "display_name", "accession", "accession_primary", "query_accession"]:
        value = row.get(col, np.nan)
        if pd.notna(value):
            text = str(value).strip()
            if text and text.lower() != "nan":
                candidates.append(text)
    if accession is not None:
        candidates.insert(0, accession)
    elif fallback_accession is not None:
        candidates.insert(0, fallback_accession)

    seq: Optional[str] = None
    seq_map = _load_segment_tip_sequences(species, segment)
    if seq_map:
        for candidate in candidates:
            if candidate in seq_map:
                seq = seq_map[candidate]
                break
            norm = _normalize_join_key(candidate)
            if norm and norm in seq_map:
                seq = seq_map[norm]
                break
    if seq is None and fallback_accession is not None:
        seq = _load_utr_cached_fasta_sequence(fallback_accession)
    if seq is None and fallback_accession is not None:
        seq = _fetch_entrez_fasta_sequence(fallback_accession)

    cds_info: Optional[Tuple[int, int, str]] = None
    meta_lookup_accession = accession or fallback_accession
    if meta_lookup_accession is not None and meta is not None:
        meta_row = meta[meta["accession"].astype(str) == str(meta_lookup_accession)]
        if not meta_row.empty:
            cds_start = pd.to_numeric(meta_row["cds_start"], errors="coerce").dropna()
            cds_end = pd.to_numeric(meta_row["cds_end"], errors="coerce").dropna()
            if not cds_start.empty and not cds_end.empty:
                strand = str(meta_row["cds_strand"].iloc[0]) if "cds_strand" in meta_row.columns else "+"
                cds_info = (int(cds_start.iloc[0]), int(cds_end.iloc[0]), strand)
    if cds_info is None and fallback_accession is not None:
        cds_info = _fetch_entrez_gff3_cds(fallback_accession)

    if seq is None or cds_info is None:
        _SEGMENT_TERMINAL_CACHE[cache_key] = None
        return None

    seq_clean = "".join(base for base in str(seq).upper().replace("-", "") if base in {"A", "C", "G", "T", "N"})
    start = int(cds_info[0]) - 1
    end = int(cds_info[1])
    strand = str(cds_info[2]).strip() if cds_info[2] is not None else "+"
    expected_len = end - start
    if not (start >= 0 and end > start and end <= len(seq_clean)):
        if expected_len > 0 and abs(len(seq_clean) - expected_len) <= 6:
            # Many local segment FASTAs are CDS-only. Keep these as valid
            # zero-UTR observations so terminal coverage is not artificially
            # dropped to missing when the underlying segment is present.
            info = {
                "accession": fallback_accession,
                "segment": segment,
                "sequence_length": float(len(seq_clean)),
                "utr5": "",
                "utr3": "",
                "utr5_length": 0.0,
                "utr3_length": 0.0,
                "utr5_recovered": 0.0,
                "utr3_recovered": 0.0,
                "cds_strand": str(cds_info[2]).strip() if cds_info[2] is not None else "+",
            }
            _SEGMENT_TERMINAL_CACHE[cache_key] = info
            return info
        _SEGMENT_TERMINAL_CACHE[cache_key] = None
        return None

    oriented = seq_clean
    oriented_start = start
    oriented_end = end
    if strand == "-":
        oriented = _reverse_complement_dna(seq_clean)
        oriented_start = len(seq_clean) - end
        oriented_end = len(seq_clean) - start

    if not (0 <= oriented_start < oriented_end <= len(oriented)):
        _SEGMENT_TERMINAL_CACHE[cache_key] = None
        return None

    utr5 = oriented[:oriented_start]
    utr3 = oriented[oriented_end:]
    info = {
        "accession": fallback_accession,
        "segment": segment,
        "sequence_length": float(len(oriented)),
        "utr5": utr5,
        "utr3": utr3,
        "utr5_length": float(len(utr5)),
        "utr3_length": float(len(utr3)),
        "utr5_recovered": float(len(utr5) > 0),
        "utr3_recovered": float(len(utr3) > 0),
        "cds_strand": strand,
    }
    _SEGMENT_TERMINAL_CACHE[cache_key] = info
    return info


def _terminal_pair_stats(five_prime: str, three_prime: str, window: int = 30) -> Dict[str, float]:
    five = "".join(base for base in str(five_prime).upper() if base in {"A", "C", "G", "T"})
    three = "".join(base for base in str(three_prime).upper() if base in {"A", "C", "G", "T"})
    k = min(len(five), len(three), int(window))
    if k < 6:
        return {"comp_frac": np.nan, "mismatch_count": np.nan, "stability_proxy": np.nan}
    a = five[:k]
    b = _reverse_complement_dna(three[-k:])
    matches = np.array([float(x == y) for x, y in zip(a, b)], dtype=float)
    if not np.isfinite(matches).any():
        return {"comp_frac": np.nan, "mismatch_count": np.nan, "stability_proxy": np.nan}
    gc_match = np.array([float(x == y and x in {"G", "C"}) for x, y in zip(a, b)], dtype=float)
    stability = matches + gc_match
    return {
        "comp_frac": float(np.mean(matches)),
        "mismatch_count": float(k - float(matches.sum())),
        "stability_proxy": float(np.mean(stability)),
    }


def add_direct_utr_terminal_features(df_part: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pair_defs = [("S", "M"), ("S", "L"), ("M", "L")]
    for _, row in df_part.iterrows():
        species = str(row.get("species", "missing_species"))
        seg_info: Dict[str, Dict[str, object]] = {}
        out: Dict[str, object] = {}
        for seg in ("S", "M", "L"):
            info = _extract_segment_terminal_info(row, species, seg)
            if info is not None:
                seg_info[seg] = info
            out[f"utrterm_{seg}_5p_length"] = float(info["utr5_length"]) if info is not None else np.nan
            out[f"utrterm_{seg}_3p_length"] = float(info["utr3_length"]) if info is not None else np.nan
            out[f"utrterm_{seg}_5p_recovered"] = float(info["utr5_recovered"]) if info is not None else 0.0
            out[f"utrterm_{seg}_3p_recovered"] = float(info["utr3_recovered"]) if info is not None else 0.0
        for a, b in pair_defs:
            out[f"utrterm_{a}{b}_comp_frac_mean"] = np.nan
            out[f"utrterm_{a}{b}_mismatch_mean"] = np.nan
            out[f"utrterm_{a}{b}_stability_mean"] = np.nan
            out[f"utrterm_{a}{b}_comp_asymmetry"] = np.nan
            out[f"utrterm_{a}{b}_len5_absdiff"] = np.nan
            out[f"utrterm_{a}{b}_len3_absdiff"] = np.nan
            if a not in seg_info or b not in seg_info:
                continue
            stats_ab = _terminal_pair_stats(str(seg_info[a]["utr5"]), str(seg_info[b]["utr3"]))
            stats_ba = _terminal_pair_stats(str(seg_info[b]["utr5"]), str(seg_info[a]["utr3"]))
            comp_vals = [v for v in [stats_ab["comp_frac"], stats_ba["comp_frac"]] if np.isfinite(v)]
            mismatch_vals = [v for v in [stats_ab["mismatch_count"], stats_ba["mismatch_count"]] if np.isfinite(v)]
            stability_vals = [v for v in [stats_ab["stability_proxy"], stats_ba["stability_proxy"]] if np.isfinite(v)]
            if comp_vals:
                out[f"utrterm_{a}{b}_comp_frac_mean"] = float(np.mean(comp_vals))
            if mismatch_vals:
                out[f"utrterm_{a}{b}_mismatch_mean"] = float(np.mean(mismatch_vals))
            if stability_vals:
                out[f"utrterm_{a}{b}_stability_mean"] = float(np.mean(stability_vals))
            if np.isfinite(stats_ab["comp_frac"]) and np.isfinite(stats_ba["comp_frac"]):
                out[f"utrterm_{a}{b}_comp_asymmetry"] = abs(float(stats_ab["comp_frac"]) - float(stats_ba["comp_frac"]))
            out[f"utrterm_{a}{b}_len5_absdiff"] = abs(float(seg_info[a]["utr5_length"]) - float(seg_info[b]["utr5_length"]))
            out[f"utrterm_{a}{b}_len3_absdiff"] = abs(float(seg_info[a]["utr3_length"]) - float(seg_info[b]["utr3_length"]))
        rows.append(out)
    return pd.DataFrame(rows, index=df_part.index)


def _codon_freqs(cds: str) -> np.ndarray:
    counts = {codon: 0 for codon in SENSE_CODONS}
    total = 0
    for i in range(0, len(cds) - 2, 3):
        codon = cds[i : i + 3]
        if "N" in codon or len(codon) != 3:
            continue
        if codon in counts:
            counts[codon] += 1
            total += 1
    if total <= 0:
        return np.full(len(SENSE_CODONS), np.nan)
    return np.array([counts[c] / total for c in SENSE_CODONS], dtype=float)


def _dinuc_freqs(seq: str) -> np.ndarray:
    counts = {d: 0 for d in DINUCS}
    total = 0
    for i in range(len(seq) - 1):
        d = seq[i : i + 2]
        if "N" in d or any(base not in "ACGT" for base in d):
            continue
        counts[d] += 1
        total += 1
    if total <= 0:
        return np.full(len(DINUCS), np.nan)
    return np.array([counts[d] / total for d in DINUCS], dtype=float)


def _gc_fraction(seq: str) -> float:
    bases = [b for b in seq if b in "ACGT"]
    if not bases:
        return np.nan
    return float(sum(b in "GC" for b in bases) / len(bases))


def _gc3_fraction(cds: str) -> float:
    third = [cds[i + 2] for i in range(0, len(cds) - 2, 3) if cds[i + 2] in "ACGT"]
    if not third:
        return np.nan
    return float(sum(b in "GC" for b in third) / len(third))


def add_codon_composition_features(df_part: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in df_part.iterrows():
        species = str(row.get("species", "missing_species"))
        seg_data: Dict[str, Dict[str, object]] = {}
        for seg in ("S", "M", "L"):
            cds = _extract_segment_cds_sequence(row, species, seg)
            if cds is None:
                continue
            seg_data[seg] = {
                "cds": cds,
                "codon": _codon_freqs(cds),
                "dinuc": _dinuc_freqs(cds),
                "gc": _gc_fraction(cds),
                "gc3": _gc3_fraction(cds),
            }
        out: Dict[str, object] = {
            "codoncomp_n_segments": float(len(seg_data)),
            "codoncomp_n_pairs": 0.0,
            "codoncomp_jsd_codon_mean": np.nan,
            "codoncomp_jsd_codon_std": np.nan,
            "codoncomp_jsd_dinuc_mean": np.nan,
            "codoncomp_jsd_dinuc_std": np.nan,
            "codoncomp_gc_absdiff_mean": np.nan,
            "codoncomp_gc3_absdiff_mean": np.nan,
        }
        codon_jsds: List[float] = []
        dinuc_jsds: List[float] = []
        gc_diffs: List[float] = []
        gc3_diffs: List[float] = []
        for a, b in [("S", "M"), ("S", "L"), ("M", "L")]:
            key_codon = f"codoncomp_jsd_codon_{a}{b}"
            key_dinuc = f"codoncomp_jsd_dinuc_{a}{b}"
            key_gc = f"codoncomp_gc_absdiff_{a}{b}"
            key_gc3 = f"codoncomp_gc3_absdiff_{a}{b}"
            out[key_codon] = np.nan
            out[key_dinuc] = np.nan
            out[key_gc] = np.nan
            out[key_gc3] = np.nan
            if a not in seg_data or b not in seg_data:
                continue
            codon_jsd = _safe_jsd(seg_data[a]["codon"], seg_data[b]["codon"])
            dinuc_jsd = _safe_jsd(seg_data[a]["dinuc"], seg_data[b]["dinuc"])
            gc_diff = abs(float(seg_data[a]["gc"]) - float(seg_data[b]["gc"])) if np.isfinite(seg_data[a]["gc"]) and np.isfinite(seg_data[b]["gc"]) else np.nan
            gc3_diff = abs(float(seg_data[a]["gc3"]) - float(seg_data[b]["gc3"])) if np.isfinite(seg_data[a]["gc3"]) and np.isfinite(seg_data[b]["gc3"]) else np.nan
            out[key_codon] = codon_jsd
            out[key_dinuc] = dinuc_jsd
            out[key_gc] = gc_diff
            out[key_gc3] = gc3_diff
            if np.isfinite(codon_jsd):
                codon_jsds.append(float(codon_jsd))
            if np.isfinite(dinuc_jsd):
                dinuc_jsds.append(float(dinuc_jsd))
            if np.isfinite(gc_diff):
                gc_diffs.append(float(gc_diff))
            if np.isfinite(gc3_diff):
                gc3_diffs.append(float(gc3_diff))
        pair_count = len(codon_jsds) or len(dinuc_jsds) or len(gc_diffs) or len(gc3_diffs)
        out["codoncomp_n_pairs"] = float(pair_count)
        if codon_jsds:
            out["codoncomp_jsd_codon_mean"] = float(np.mean(codon_jsds))
            out["codoncomp_jsd_codon_std"] = float(np.std(codon_jsds))
        if dinuc_jsds:
            out["codoncomp_jsd_dinuc_mean"] = float(np.mean(dinuc_jsds))
            out["codoncomp_jsd_dinuc_std"] = float(np.std(dinuc_jsds))
        if gc_diffs:
            out["codoncomp_gc_absdiff_mean"] = float(np.mean(gc_diffs))
        if gc3_diffs:
            out["codoncomp_gc3_absdiff_mean"] = float(np.mean(gc3_diffs))
        rows.append(out)
    return pd.DataFrame(rows, index=df_part.index)


def add_direct_discordance_features(df_part: pd.DataFrame) -> pd.DataFrame:
    species_discordance_stats = (
        df_part.groupby("species", dropna=False)["tree_discordance_score"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "species_tree_mean", "std": "species_tree_std"})
    )
    rows: List[Dict[str, object]] = []
    pair_defs = [("S", "M"), ("S", "L"), ("M", "L")]
    for _, row in df_part.iterrows():
        species = str(row.get("species", "missing_species"))
        tip_name = str(row.get("tip_name", ""))
        out: Dict[str, object] = {
            "discord_nn_tip_agreement_frac": np.nan,
            "discord_nn_patristic_mean": np.nan,
            "discord_nn_patristic_std": np.nan,
            "discord_tree_resid_species_z": np.nan,
            "discord_tree_resid_species_abs": np.nan,
        }
        nn_tips: Dict[str, str] = {}
        nn_dists: Dict[str, float] = {}
        for seg in ("S", "M", "L"):
            dm = _load_segment_tree_distance_matrix(species, seg)
            out[f"discord_nn_patristic_{seg}"] = np.nan
            if dm is None or tip_name not in dm.index:
                continue
            series = dm.loc[tip_name].drop(labels=[tip_name], errors="ignore").astype(float)
            series = series[series.notna()]
            if series.empty:
                continue
            nn_tips[seg] = str(series.idxmin())
            nn_dists[seg] = float(series.min())
            out[f"discord_nn_patristic_{seg}"] = float(series.min())
        if nn_dists:
            vals = np.array(list(nn_dists.values()), dtype=float)
            out["discord_nn_patristic_mean"] = float(np.mean(vals))
            out["discord_nn_patristic_std"] = float(np.std(vals))
        pair_tip_agree: List[float] = []
        for a, b in pair_defs:
            out[f"discord_{a}{b}_profile_corr"] = np.nan
            out[f"discord_{a}{b}_profile_absz"] = np.nan
            out[f"discord_{a}{b}_nn_same_tip"] = np.nan
            out[f"discord_{a}{b}_nn_patristic_absdiff"] = np.nan
            dm_a = _load_segment_tree_distance_matrix(species, a)
            dm_b = _load_segment_tree_distance_matrix(species, b)
            if a in nn_tips and b in nn_tips:
                same_tip = float(nn_tips[a] == nn_tips[b])
                out[f"discord_{a}{b}_nn_same_tip"] = same_tip
                pair_tip_agree.append(same_tip)
                out[f"discord_{a}{b}_nn_patristic_absdiff"] = abs(float(nn_dists[a]) - float(nn_dists[b]))
            if dm_a is None or dm_b is None or tip_name not in dm_a.index or tip_name not in dm_b.index:
                continue
            common = [
                tip
                for tip in dm_a.columns
                if tip in dm_b.columns and tip != tip_name
            ]
            if len(common) < 4:
                continue
            da = dm_a.loc[tip_name, common].astype(float)
            db = dm_b.loc[tip_name, common].astype(float)
            keep = da.notna() & db.notna()
            if int(keep.sum()) < 4:
                continue
            va = da[keep].to_numpy(dtype=float)
            vb = db[keep].to_numpy(dtype=float)
            corr, _ = spearmanr(va, vb)
            if np.isfinite(corr):
                out[f"discord_{a}{b}_profile_corr"] = float(corr)
            if np.nanstd(va) > 0 and np.nanstd(vb) > 0:
                za = (va - np.nanmean(va)) / np.nanstd(va)
                zb = (vb - np.nanmean(vb)) / np.nanstd(vb)
                out[f"discord_{a}{b}_profile_absz"] = float(np.nanmean(np.abs(za - zb)))
        if pair_tip_agree:
            out["discord_nn_tip_agreement_frac"] = float(np.mean(pair_tip_agree))
        species_mean = float(species_discordance_stats.loc[species, "species_tree_mean"]) if species in species_discordance_stats.index else np.nan
        species_std = float(species_discordance_stats.loc[species, "species_tree_std"]) if species in species_discordance_stats.index else np.nan
        row_tree = pd.to_numeric(pd.Series([row.get("tree_discordance_score")]), errors="coerce").iloc[0]
        if np.isfinite(row_tree) and np.isfinite(species_mean):
            out["discord_tree_resid_species_abs"] = abs(float(row_tree) - float(species_mean))
            if np.isfinite(species_std) and species_std > 0:
                out["discord_tree_resid_species_z"] = float((float(row_tree) - float(species_mean)) / float(species_std))
        rows.append(out)
    return pd.DataFrame(rows, index=df_part.index)


def _upper_triangle_stats(matrix: pd.DataFrame) -> Tuple[float, float]:
    values = matrix.to_numpy(dtype=float)
    if values.shape[0] < 2:
        return np.nan, np.nan
    tri = values[np.triu_indices(values.shape[0], k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return np.nan, np.nan
    return float(np.mean(tri)), float(np.std(tri))


def _cramers_v_and_p(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    keep = x.notna() & y.notna()
    if int(keep.sum()) < 4:
        return np.nan, np.nan
    table = pd.crosstab(x[keep], y[keep])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan
    try:
        chi2, pval, _, _ = chi2_contingency(table, correction=False)
    except Exception:
        return np.nan, np.nan
    n = float(table.to_numpy().sum())
    if n <= 0:
        return np.nan, np.nan
    phi2 = chi2 / n
    r, k = table.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return np.nan, np.nan
    return float(np.sqrt(phi2 / denom)), float(pval)


def add_lineage_molecular_context(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    molecular_numeric: Sequence[str],
    molecular_categorical: Sequence[str],
    within_species_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Build exploratory within-species lineage-aware molecular context features.

    Uses only training rows to define lineage molecular profiles. Test rows are
    scored against same-species training lineages, avoiding leakage from the
    held-out lineage itself.
    """
    num_cols = [
        col
        for col in molecular_numeric
        if col in X_train.columns
        and not col.startswith("species_feat_")
        and col not in {"segment_completeness", "has_S", "has_M", "has_L"}
    ]
    cat_cols = [
        col
        for col in molecular_categorical
        if col in X_train.columns and col.startswith("ld_site_")
    ]
    main_site_cols = [col for col in cat_cols if len(col.split("_")) == 4]
    if not num_cols and not cat_cols:
        return X_train, X_test, [], []

    def _ld_nonmissing_fraction(frame: pd.DataFrame) -> pd.Series:
        if not cat_cols:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        vals = frame[cat_cols]
        return vals.notna().mean(axis=1).astype(float)

    train_species = df_train["species"].fillna("missing_species").astype(str)
    test_species = df_test["species"].fillna("missing_species").astype(str)
    species_scale: Dict[str, pd.Series] = {}
    train_numeric_frame = X_train[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame(index=X_train.index)
    test_numeric_frame = X_test[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame(index=X_test.index)
    train_ld_fraction = _ld_nonmissing_fraction(X_train)
    test_ld_fraction = _ld_nonmissing_fraction(X_test)

    profile_records: List[Dict[str, object]] = []
    profile_lookup: Dict[Tuple[str, str], Dict[str, object]] = {}

    if num_cols:
        for species, idx in train_species.groupby(train_species).groups.items():
            sub = train_numeric_frame.loc[idx, num_cols]
            scale = sub.std(axis=0, ddof=0).replace(0.0, np.nan)
            scale = scale.fillna(1.0)
            species_scale[str(species)] = scale

    lineages = within_species_train.fillna(pd.NA).astype("object")
    lineage_labels = lineages.dropna().astype(str)
    for lineage_label, row_idx in lineage_labels.groupby(lineage_labels).groups.items():
        idx = list(row_idx)
        species_values = train_species.loc[idx].dropna().astype(str).unique().tolist()
        if len(species_values) != 1:
            continue
        species = species_values[0]
        record: Dict[str, object] = {
            "species": species,
            "lineage": str(lineage_label),
            "lineage_n": int(len(idx)),
            "tree_discordance_mean": np.nan,
            "ld_nonmissing_fraction_mean": float(train_ld_fraction.loc[idx].mean()) if len(idx) else np.nan,
            "lineage_ld_selected_mean_cramers": np.nan,
            "lineage_ld_selected_prop_significant": np.nan,
            "lineage_ld_selected_n_pairs": np.nan,
            "lineage_utr5_L_retention_rate": np.nan,
            "lineage_utr5_M_retention_rate": np.nan,
            "lineage_utr5_L_structdist_mean": np.nan,
            "lineage_utr5_L_structdist_std": np.nan,
            "lineage_utr5_M_structdist_mean": np.nan,
            "lineage_utr5_M_structdist_std": np.nan,
            "lineage_utr5_overall_retention_rate": np.nan,
            "lineage_utr5_overall_structdist_mean": np.nan,
        }
        if "tree_discordance_score" in train_numeric_frame.columns:
            record["tree_discordance_mean"] = float(train_numeric_frame.loc[idx, "tree_discordance_score"].mean())
        if num_cols:
            means = train_numeric_frame.loc[idx, num_cols].mean(axis=0, numeric_only=True)
            for col in num_cols:
                record[f"mean__{col}"] = float(means.get(col, np.nan))
        for col in cat_cols:
            record[f"mode__{col}"] = _mode_or_nan(X_train.loc[idx, col])

        if main_site_cols:
            seg_pairs: List[Tuple[str, str]] = []
            for i, a in enumerate(main_site_cols):
                seg_a = a.split("_")[2]
                for b in main_site_cols[i + 1 :]:
                    seg_b = b.split("_")[2]
                    if seg_a == seg_b:
                        continue
                    seg_pairs.append((a, b))
            cramers_vals: List[float] = []
            sig_flags: List[float] = []
            subset = X_train.loc[idx, main_site_cols]
            for a, b in seg_pairs:
                v, pval = _cramers_v_and_p(subset[a], subset[b])
                if np.isfinite(v):
                    cramers_vals.append(float(v))
                    if np.isfinite(pval):
                        sig_flags.append(float(pval < 0.05))
            if cramers_vals:
                record["lineage_ld_selected_mean_cramers"] = float(np.mean(cramers_vals))
                record["lineage_ld_selected_n_pairs"] = float(len(cramers_vals))
            if sig_flags:
                record["lineage_ld_selected_prop_significant"] = float(np.mean(sig_flags))

        seg_ret_rates: List[float] = []
        seg_struct_means: List[float] = []
        for seg in ["L", "M"]:
            meta = _load_utr_metadata(species, seg)
            if meta is None:
                continue
            valid_accessions = set(meta["accession"].astype(str))
            accessions = {
                acc
                for acc in (
                    _resolve_segment_accession(df_train.loc[i], species, seg, valid_accessions)
                    for i in idx
                )
                if acc is not None
            }
            if not accessions:
                continue
            matched = meta[meta["accession"].isin(accessions)].copy()
            if matched.empty:
                continue
            retained = matched[matched["status"].astype(str).eq("retained")].copy()
            ret_rate = float(len(retained) / len(matched)) if len(matched) else np.nan
            record[f"lineage_utr5_{seg}_retention_rate"] = ret_rate
            if np.isfinite(ret_rate):
                seg_ret_rates.append(ret_rate)
            if len(retained) >= 2:
                matrix = _load_utr_struct_matrix(species, seg)
                if matrix is not None:
                    kept = [acc for acc in retained["accession"].astype(str).tolist() if acc in matrix.index and acc in matrix.columns]
                    if len(kept) >= 2:
                        subm = matrix.loc[kept, kept]
                        mean_val, std_val = _upper_triangle_stats(subm)
                        record[f"lineage_utr5_{seg}_structdist_mean"] = mean_val
                        record[f"lineage_utr5_{seg}_structdist_std"] = std_val
                        if np.isfinite(mean_val):
                            seg_struct_means.append(mean_val)
        if seg_ret_rates:
            record["lineage_utr5_overall_retention_rate"] = float(np.mean(seg_ret_rates))
        if seg_struct_means:
            record["lineage_utr5_overall_structdist_mean"] = float(np.mean(seg_struct_means))
        profile_records.append(record)
        profile_lookup[(species, str(lineage_label))] = record

    def _score_rows(
        X_part: pd.DataFrame,
        df_part: pd.DataFrame,
        numeric_frame: pd.DataFrame,
        ld_fraction: pd.Series,
    ) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        part_species = df_part["species"].fillna("missing_species").astype(str)
        for row_id in X_part.index:
            species = str(part_species.loc[row_id])
            candidates = [
                rec for (sp, _), rec in profile_lookup.items()
                if sp == species
            ]
            out: Dict[str, object] = {
                "lineage_ctx_best_match_score": np.nan,
                "lineage_ctx_best_ld_match_fraction": np.nan,
                "lineage_ctx_best_numeric_similarity": np.nan,
                "lineage_ctx_second_best_gap": np.nan,
                "lineage_ctx_best_lineage_n": np.nan,
                "lineage_ctx_candidate_lineages": float(len(candidates)),
                "lineage_ctx_best_tree_discordance_mean": np.nan,
                "lineage_ctx_tree_discordance_delta": np.nan,
                "lineage_ctx_best_ld_nonmissing_fraction": np.nan,
                "lineage_ctx_ld_selected_mean_cramers": np.nan,
                "lineage_ctx_ld_selected_prop_significant": np.nan,
                "lineage_ctx_ld_selected_n_pairs": np.nan,
                "lineage_ctx_utr5_L_retention_rate": np.nan,
                "lineage_ctx_utr5_M_retention_rate": np.nan,
                "lineage_ctx_utr5_L_structdist_mean": np.nan,
                "lineage_ctx_utr5_L_structdist_std": np.nan,
                "lineage_ctx_utr5_M_structdist_mean": np.nan,
                "lineage_ctx_utr5_M_structdist_std": np.nan,
                "lineage_ctx_utr5_overall_retention_rate": np.nan,
                "lineage_ctx_utr5_overall_structdist_mean": np.nan,
            }
            if not candidates:
                rows.append(out)
                continue

            row_num = numeric_frame.loc[row_id] if num_cols else pd.Series(dtype=float)
            row_cat = X_part.loc[row_id, cat_cols] if cat_cols else pd.Series(dtype=object)
            row_tree = float(row_num.get("tree_discordance_score", np.nan)) if "tree_discordance_score" in row_num.index else np.nan
            species_sigma = species_scale.get(species, pd.Series(1.0, index=num_cols, dtype=float))
            scores: List[Tuple[float, float, float, Dict[str, object]]] = []
            for rec in candidates:
                cat_match = np.nan
                if cat_cols:
                    comp_vals = []
                    for col in cat_cols:
                        mode_val = rec.get(f"mode__{col}", np.nan)
                        row_val = row_cat.get(col, np.nan)
                        if pd.isna(mode_val) or pd.isna(row_val):
                            continue
                        comp_vals.append(float(str(row_val) == str(mode_val)))
                    if comp_vals:
                        cat_match = float(np.mean(comp_vals))
                num_sim = np.nan
                if num_cols:
                    dists = []
                    for col in num_cols:
                        mean_val = rec.get(f"mean__{col}", np.nan)
                        row_val = row_num.get(col, np.nan)
                        if pd.isna(mean_val) or pd.isna(row_val):
                            continue
                        sigma = float(species_sigma.get(col, 1.0))
                        sigma = 1.0 if not np.isfinite(sigma) or sigma <= 0 else sigma
                        dists.append(abs(float(row_val) - float(mean_val)) / sigma)
                    if dists:
                        num_sim = float(np.exp(-np.mean(dists)))
                parts = [v for v in [cat_match, num_sim] if np.isfinite(v)]
                combined = float(np.mean(parts)) if parts else np.nan
                scores.append((combined, cat_match, num_sim, rec))

            scores = [s for s in scores if np.isfinite(s[0])]
            if not scores:
                rows.append(out)
                continue
            scores.sort(key=lambda item: item[0], reverse=True)
            best_score, best_cat, best_num, best_rec = scores[0]
            second_score = scores[1][0] if len(scores) > 1 else np.nan
            out.update(
                {
                    "lineage_ctx_best_match_score": float(best_score),
                    "lineage_ctx_best_ld_match_fraction": float(best_cat) if np.isfinite(best_cat) else np.nan,
                    "lineage_ctx_best_numeric_similarity": float(best_num) if np.isfinite(best_num) else np.nan,
                    "lineage_ctx_second_best_gap": float(best_score - second_score) if np.isfinite(second_score) else np.nan,
                    "lineage_ctx_best_lineage_n": float(best_rec.get("lineage_n", np.nan)),
                    "lineage_ctx_best_tree_discordance_mean": float(best_rec.get("tree_discordance_mean", np.nan))
                    if np.isfinite(best_rec.get("tree_discordance_mean", np.nan))
                    else np.nan,
                    "lineage_ctx_tree_discordance_delta": (
                        float(row_tree - float(best_rec.get("tree_discordance_mean", np.nan)))
                        if np.isfinite(row_tree) and np.isfinite(best_rec.get("tree_discordance_mean", np.nan))
                        else np.nan
                    ),
                    "lineage_ctx_best_ld_nonmissing_fraction": float(best_rec.get("ld_nonmissing_fraction_mean", np.nan))
                    if np.isfinite(best_rec.get("ld_nonmissing_fraction_mean", np.nan))
                    else np.nan,
                    "lineage_ctx_ld_selected_mean_cramers": float(best_rec.get("lineage_ld_selected_mean_cramers", np.nan))
                    if np.isfinite(best_rec.get("lineage_ld_selected_mean_cramers", np.nan))
                    else np.nan,
                    "lineage_ctx_ld_selected_prop_significant": float(best_rec.get("lineage_ld_selected_prop_significant", np.nan))
                    if np.isfinite(best_rec.get("lineage_ld_selected_prop_significant", np.nan))
                    else np.nan,
                    "lineage_ctx_ld_selected_n_pairs": float(best_rec.get("lineage_ld_selected_n_pairs", np.nan))
                    if np.isfinite(best_rec.get("lineage_ld_selected_n_pairs", np.nan))
                    else np.nan,
                    "lineage_ctx_utr5_L_retention_rate": float(best_rec.get("lineage_utr5_L_retention_rate", np.nan))
                    if np.isfinite(best_rec.get("lineage_utr5_L_retention_rate", np.nan))
                    else np.nan,
                    "lineage_ctx_utr5_M_retention_rate": float(best_rec.get("lineage_utr5_M_retention_rate", np.nan))
                    if np.isfinite(best_rec.get("lineage_utr5_M_retention_rate", np.nan))
                    else np.nan,
                    "lineage_ctx_utr5_L_structdist_mean": float(best_rec.get("lineage_utr5_L_structdist_mean", np.nan))
                    if np.isfinite(best_rec.get("lineage_utr5_L_structdist_mean", np.nan))
                    else np.nan,
                    "lineage_ctx_utr5_L_structdist_std": float(best_rec.get("lineage_utr5_L_structdist_std", np.nan))
                    if np.isfinite(best_rec.get("lineage_utr5_L_structdist_std", np.nan))
                    else np.nan,
                    "lineage_ctx_utr5_M_structdist_mean": float(best_rec.get("lineage_utr5_M_structdist_mean", np.nan))
                    if np.isfinite(best_rec.get("lineage_utr5_M_structdist_mean", np.nan))
                    else np.nan,
                    "lineage_ctx_utr5_M_structdist_std": float(best_rec.get("lineage_utr5_M_structdist_std", np.nan))
                    if np.isfinite(best_rec.get("lineage_utr5_M_structdist_std", np.nan))
                    else np.nan,
                    "lineage_ctx_utr5_overall_retention_rate": float(best_rec.get("lineage_utr5_overall_retention_rate", np.nan))
                    if np.isfinite(best_rec.get("lineage_utr5_overall_retention_rate", np.nan))
                    else np.nan,
                    "lineage_ctx_utr5_overall_structdist_mean": float(best_rec.get("lineage_utr5_overall_structdist_mean", np.nan))
                    if np.isfinite(best_rec.get("lineage_utr5_overall_structdist_mean", np.nan))
                    else np.nan,
                }
            )
            rows.append(out)
        return pd.DataFrame(rows, index=X_part.index)

    train_aug = _score_rows(X_train, df_train, train_numeric_frame, train_ld_fraction)
    test_aug = _score_rows(X_test, df_test, test_numeric_frame, test_ld_fraction)
    new_numeric = list(train_aug.columns)
    X_train_aug = pd.concat([X_train, train_aug], axis=1)
    X_test_aug = pd.concat([X_test, test_aug], axis=1)
    return X_train_aug, X_test_aug, new_numeric, []


def add_real_lineage_context(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    within_species_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Exploratory fold-safe lineage context from real trees, alignments, and spatiotemporal data."""

    lineage_labels = within_species_train.fillna(pd.NA).astype("object")
    train_species = df_train["species"].fillna("missing_species").astype(str)

    train_records: Dict[str, Dict[str, object]] = {}
    for idx in X_train.index:
        species = str(train_species.loc[idx])
        lineage = lineage_labels.loc[idx]
        if pd.isna(lineage):
            continue
        row = df_train.loc[idx]
        train_records[idx] = {
            "species": species,
            "lineage": str(lineage),
            "tip_name": str(row.get("tip_name", "")),
            "lat": pd.to_numeric(pd.Series([row.get("local_latitude", np.nan)]), errors="coerce").iloc[0],
            "lon": pd.to_numeric(pd.Series([row.get("local_longitude", np.nan)]), errors="coerce").iloc[0],
            "year": pd.to_numeric(pd.Series([row.get("year", np.nan)]), errors="coerce").iloc[0],
        }

    by_species: Dict[str, List[int]] = {}
    for idx, rec in train_records.items():
        by_species.setdefault(str(rec["species"]), []).append(idx)

    feature_names = [
        "realctx_tree_nn_S",
        "realctx_tree_nn_M",
        "realctx_tree_nn_L",
        "realctx_seq_nn_S",
        "realctx_seq_nn_M",
        "realctx_seq_nn_L",
        "realctx_tree_nn_mean",
        "realctx_tree_nn_std",
        "realctx_seq_nn_mean",
        "realctx_seq_nn_std",
        "realctx_nn_lineage_agreement_frac",
        "realctx_nn_unique_lineages",
        "realctx_nn_same_lineage_all_segments",
        "realctx_profile_corr_SM",
        "realctx_profile_corr_SL",
        "realctx_profile_corr_ML",
        "realctx_profile_absz_SM",
        "realctx_profile_absz_SL",
        "realctx_profile_absz_ML",
        "realctx_cocirc_kernel_total",
        "realctx_cocirc_lineage_count",
        "realctx_cocirc_max_lineage_share",
        "realctx_cocirc_entropy",
        "realctx_cocirc_nearest_km",
        "realctx_cocirc_nearest_years",
        "realctx_cocirc_nearest_combined",
        "realctx_cocirc_neighbor_count_500km_3y",
    ]

    def _score_partition(X_part: pd.DataFrame, df_part: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for idx in X_part.index:
            row = df_part.loc[idx]
            species = str(row.get("species", "missing_species"))
            candidate_idx = list(by_species.get(species, []))
            if is_train and idx in candidate_idx:
                candidate_idx = [j for j in candidate_idx if j != idx]
            out = {name: np.nan for name in feature_names}
            if not candidate_idx:
                rows.append(out)
                continue

            row_tip = str(row.get("tip_name", ""))
            nearest_lineages: Dict[str, str] = {}
            tree_vals: List[float] = []
            seq_vals: List[float] = []

            candidate_lineages = {j: str(train_records[j]["lineage"]) for j in candidate_idx}
            candidate_tips = {j: str(train_records[j]["tip_name"]) for j in candidate_idx}

            for seg in ("S", "M", "L"):
                tree_dm = _load_segment_tree_distance_matrix(species, seg)
                if tree_dm is not None and row_tip in tree_dm.index:
                    seg_tips = [candidate_tips[j] for j in candidate_idx if candidate_tips[j] in tree_dm.columns]
                    if seg_tips:
                        d = tree_dm.loc[row_tip, seg_tips].astype(float)
                        if d.notna().any():
                            best_tip = str(d.idxmin())
                            best_j = next((j for j in candidate_idx if candidate_tips[j] == best_tip), None)
                            out[f"realctx_tree_nn_{seg}"] = float(d.min())
                            tree_vals.append(float(d.min()))
                            if best_j is not None:
                                nearest_lineages[f"tree_{seg}"] = candidate_lineages[best_j]

                seq_dm = _load_segment_sequence_distance_matrix(species, seg)
                if seq_dm is not None and row_tip in seq_dm.index:
                    seg_tips = [candidate_tips[j] for j in candidate_idx if candidate_tips[j] in seq_dm.columns]
                    if seg_tips:
                        d = seq_dm.loc[row_tip, seg_tips].astype(float)
                        if d.notna().any():
                            best_tip = str(d.idxmin())
                            best_j = next((j for j in candidate_idx if candidate_tips[j] == best_tip), None)
                            out[f"realctx_seq_nn_{seg}"] = float(d.min())
                            seq_vals.append(float(d.min()))
                            if best_j is not None:
                                nearest_lineages.setdefault(f"seq_{seg}", candidate_lineages[best_j])

            if tree_vals:
                out["realctx_tree_nn_mean"] = float(np.mean(tree_vals))
                out["realctx_tree_nn_std"] = float(np.std(tree_vals))
            if seq_vals:
                out["realctx_seq_nn_mean"] = float(np.mean(seq_vals))
                out["realctx_seq_nn_std"] = float(np.std(seq_vals))

            tree_seg_lineages = [nearest_lineages[k] for k in sorted(nearest_lineages) if k.startswith("tree_")]
            if len(tree_seg_lineages) >= 2:
                pair_matches = []
                for i in range(len(tree_seg_lineages)):
                    for j in range(i + 1, len(tree_seg_lineages)):
                        pair_matches.append(float(tree_seg_lineages[i] == tree_seg_lineages[j]))
                out["realctx_nn_lineage_agreement_frac"] = float(np.mean(pair_matches))
                out["realctx_nn_unique_lineages"] = float(len(set(tree_seg_lineages)))
                out["realctx_nn_same_lineage_all_segments"] = float(len(set(tree_seg_lineages)) == 1)
            elif len(tree_seg_lineages) == 1:
                out["realctx_nn_unique_lineages"] = 1.0
                out["realctx_nn_same_lineage_all_segments"] = 1.0

            for seg_a, seg_b in [("S", "M"), ("S", "L"), ("M", "L")]:
                key_corr = f"realctx_profile_corr_{seg_a}{seg_b}"
                key_z = f"realctx_profile_absz_{seg_a}{seg_b}"
                dm_a = _load_segment_tree_distance_matrix(species, seg_a)
                dm_b = _load_segment_tree_distance_matrix(species, seg_b)
                if dm_a is None or dm_b is None or row_tip not in dm_a.index or row_tip not in dm_b.index:
                    continue
                common_tips = [
                    candidate_tips[j]
                    for j in candidate_idx
                    if candidate_tips[j] in dm_a.columns and candidate_tips[j] in dm_b.columns
                ]
                common_tips = list(dict.fromkeys(common_tips))
                if len(common_tips) < 4:
                    continue
                da = dm_a.loc[row_tip, common_tips].astype(float)
                db = dm_b.loc[row_tip, common_tips].astype(float)
                keep = da.notna() & db.notna()
                if int(keep.sum()) < 4:
                    continue
                va = da[keep].to_numpy(dtype=float)
                vb = db[keep].to_numpy(dtype=float)
                corr, _ = spearmanr(va, vb)
                if np.isfinite(corr):
                    out[key_corr] = float(corr)
                if np.nanstd(va) > 0 and np.nanstd(vb) > 0:
                    za = (va - np.nanmean(va)) / np.nanstd(va)
                    zb = (vb - np.nanmean(vb)) / np.nanstd(vb)
                    out[key_z] = float(np.nanmean(np.abs(za - zb)))

            lat = pd.to_numeric(pd.Series([row.get("local_latitude", np.nan)]), errors="coerce").iloc[0]
            lon = pd.to_numeric(pd.Series([row.get("local_longitude", np.nan)]), errors="coerce").iloc[0]
            year = pd.to_numeric(pd.Series([row.get("year", np.nan)]), errors="coerce").iloc[0]
            coord_candidates = [
                train_records[j]
                for j in candidate_idx
                if np.isfinite(train_records[j]["lat"])
                and np.isfinite(train_records[j]["lon"])
                and np.isfinite(train_records[j]["year"])
            ]
            if np.isfinite(lat) and np.isfinite(lon) and np.isfinite(year) and coord_candidates:
                lats = np.array([float(rec["lat"]) for rec in coord_candidates], dtype=float)
                lons = np.array([float(rec["lon"]) for rec in coord_candidates], dtype=float)
                years = np.array([float(rec["year"]) for rec in coord_candidates], dtype=float)
                lineages = np.array([str(rec["lineage"]) for rec in coord_candidates], dtype=object)
                km = _haversine_km(float(lat), float(lon), lats, lons)
                dy = np.abs(years - float(year))
                combo = np.sqrt((km / 500.0) ** 2 + (dy / 3.0) ** 2)
                weights = np.exp(-0.5 * combo**2)
                if np.isfinite(weights).any():
                    out["realctx_cocirc_kernel_total"] = float(np.nansum(weights))
                    out["realctx_cocirc_nearest_km"] = float(np.nanmin(km))
                    out["realctx_cocirc_nearest_years"] = float(np.nanmin(dy))
                    out["realctx_cocirc_nearest_combined"] = float(np.nanmin(combo))
                    out["realctx_cocirc_neighbor_count_500km_3y"] = float(np.sum((km <= 500.0) & (dy <= 3.0)))
                    lineage_weight: Dict[str, float] = {}
                    for lineage, w in zip(lineages, weights):
                        lineage_weight[str(lineage)] = lineage_weight.get(str(lineage), 0.0) + float(w)
                    if lineage_weight:
                        vals = np.array(list(lineage_weight.values()), dtype=float)
                        total = float(vals.sum())
                        probs = vals / total if total > 0 else np.full_like(vals, np.nan)
                        out["realctx_cocirc_lineage_count"] = float(np.sum(vals > 0))
                        out["realctx_cocirc_max_lineage_share"] = float(np.nanmax(probs)) if np.isfinite(probs).any() else np.nan
                        out["realctx_cocirc_entropy"] = (
                            float(-np.nansum(probs * np.log(probs + 1e-12))) if np.isfinite(probs).any() else np.nan
                        )
            rows.append(out)
        return pd.DataFrame(rows, index=X_part.index)

    train_aug = _score_partition(X_train, df_train, is_train=True)
    test_aug = _score_partition(X_test, df_test, is_train=False)
    X_train_aug = pd.concat([X_train, train_aug], axis=1)
    X_test_aug = pd.concat([X_test, test_aug], axis=1)
    return X_train_aug, X_test_aug, list(train_aug.columns), []


def heldout_label(cv_scheme: str, species_values: pd.Series) -> str:
    species_values = species_values.fillna("missing_species").astype(str)
    if cv_scheme == "grouped_species":
        return "|".join(sorted(set(species_values)))
    if cv_scheme == "grouped_clade":
        return "|".join(sorted(set(assign_species_clade(species_values))))
    if cv_scheme == "grouped_within_species_clade":
        return "|".join(sorted(set(species_values)))
    return "resampled_fold"


def summarize_prediction_flatness(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (cv_scheme, feature_set, model, heldout_group, species), group in pred_df.groupby(
        ["cv_scheme", "feature_set", "model", "heldout_group", "species"],
        dropna=False,
    ):
        probs = pd.to_numeric(group["probability"], errors="coerce")
        rows.append(
            {
                "cv_scheme": cv_scheme,
                "feature_set": feature_set,
                "model": model,
                "heldout_group": heldout_group,
                "species": species,
                "n_rows": int(len(group)),
                "target_rate": float(pd.to_numeric(group["target"], errors="coerce").mean()),
                "probability_min": float(probs.min()),
                "probability_max": float(probs.max()),
                "probability_std": float(probs.std(ddof=0)),
                "n_unique_probabilities_6dp": int(probs.round(6).nunique()),
                "is_flat_probability": int(float(probs.std(ddof=0)) < 1e-6),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["cv_scheme", "heldout_group", "is_flat_probability", "probability_std", "n_rows"],
        ascending=[True, True, False, True, True],
    )


def summarize_grouped_holdout_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_df[
        fold_df["cv_scheme"].isin(["grouped_species", "grouped_clade", "grouped_within_species_clade"])
    ].copy()
    if grouped.empty:
        return pd.DataFrame()
    grouped["train_positive_rate"] = grouped["train_positives"] / grouped["train_rows"]
    grouped["test_positive_rate"] = grouped["test_positives"] / grouped["test_rows"]
    return grouped.sort_values(
        ["cv_scheme", "feature_set", "model", "heldout_group"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def summarize_repeated_fold_context(fold_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    repeated = fold_df[fold_df["cv_scheme"] == "repeated_stratified"].copy()
    if repeated.empty:
        return pd.DataFrame()
    repeated["train_positive_rate"] = repeated["train_positives"] / repeated["train_rows"]
    repeated["test_positive_rate"] = repeated["test_positives"] / repeated["test_rows"]
    species_mix = (
        pred_df[pred_df["cv_scheme"] == "repeated_stratified"]
        .groupby(["feature_set", "model", "fold_id", "species"], dropna=False)
        .size()
        .reset_index(name="species_rows")
    )
    top_species = (
        species_mix.sort_values(["feature_set", "model", "fold_id", "species_rows"], ascending=[True, True, True, False])
        .groupby(["feature_set", "model", "fold_id"], as_index=False)
        .first()
        .rename(columns={"species": "dominant_species"})
    )
    repeated = repeated.merge(top_species, on=["feature_set", "model", "fold_id"], how="left")
    repeated["dominant_species_fraction"] = repeated["species_rows"] / repeated["test_rows"]
    return repeated.sort_values(["feature_set", "model", "fold_id"]).reset_index(drop=True)


def summarize_within_species_feature_variation(
    df: pd.DataFrame,
    feature_blocks: Mapping[str, Mapping[str, Sequence[str]]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    block_columns = {
        name: list(dict.fromkeys(list(block["numeric"]) + list(block["categorical"])))
        for name, block in feature_blocks.items()
    }
    for species, group in df.groupby("species", dropna=False):
        row: Dict[str, object] = {"species": species, "n_rows": int(len(group))}
        for block_name, cols in block_columns.items():
            available = [col for col in cols if col in group.columns]
            varied = 0
            for col in available:
                series = group[col]
                numeric = pd.to_numeric(series, errors="coerce")
                if numeric.notna().any():
                    values = numeric.dropna()
                else:
                    values = series.dropna().astype(str)
                if values.nunique() > 1:
                    varied += 1
            row[f"{block_name}_n_columns"] = int(len(available))
            row[f"{block_name}_n_varied"] = int(varied)
            row[f"{block_name}_variation_fraction"] = np.nan if not available else float(varied / len(available))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("species").reset_index(drop=True)


def summarize_lineage_prediction_behavior(
    pred_df: pd.DataFrame,
    threshold: float = 0.55,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = pred_df[pred_df["cv_scheme"] == "grouped_within_species_clade"].copy()
    if grouped.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    grouped["pred_label"] = (pd.to_numeric(grouped["probability"], errors="coerce") >= float(threshold)).astype(int)
    grouped["correct"] = (pd.to_numeric(grouped["target"], errors="coerce").astype(int) == grouped["pred_label"]).astype(int)

    tip_summary = (
        grouped.groupby(["feature_set", "model"], dropna=False)
        .agg(
            n_tips=("target", "size"),
            tip_accuracy_overall=("correct", "mean"),
        )
        .reset_index()
    )
    fold_tip = (
        grouped.groupby(["feature_set", "model", "fold_id"], dropna=False)["correct"]
        .mean()
        .reset_index(name="tip_accuracy_fold")
    )
    tip_mean = (
        fold_tip.groupby(["feature_set", "model"], dropna=False)["tip_accuracy_fold"]
        .mean()
        .reset_index(name="tip_accuracy_mean_fold")
    )
    tip_summary = tip_summary.merge(tip_mean, on=["feature_set", "model"], how="left")

    clade_rows = []
    for (feature_set, model, species, clade), group in grouped.groupby(
        ["feature_set", "model", "species", "within_species_clade"],
        dropna=False,
    ):
        observed = float(pd.to_numeric(group["target"], errors="coerce").mean())
        pred_prob = float(pd.to_numeric(group["probability"], errors="coerce").mean())
        pred_call = float(pd.to_numeric(group["pred_label"], errors="coerce").mean())
        clade_rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "species": species,
                "within_species_clade": clade,
                "n": int(len(group)),
                "observed_rate": observed,
                "predicted_mean_prob": pred_prob,
                "predicted_call_rate": pred_call,
                "abs_error_pred_prob": abs(pred_prob - observed),
                "sq_error_pred_prob": (pred_prob - observed) ** 2,
                "abs_error_pred_call": abs(pred_call - observed),
                "sq_error_pred_call": (pred_call - observed) ** 2,
            }
        )
    clade_detail = pd.DataFrame(clade_rows)

    rank_rows = []
    calib_rows = []
    for (feature_set, model, species), group in clade_detail.groupby(["feature_set", "model", "species"], dropna=False):
        rank_rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "species": species,
                "n_clades": int(len(group)),
                "spearman_obs_vs_pred_prob": (
                    float(spearmanr(group["observed_rate"], group["predicted_mean_prob"]).statistic)
                    if len(group) >= 2 and group["observed_rate"].nunique() > 1 and group["predicted_mean_prob"].nunique() > 1
                    else np.nan
                ),
                "spearman_obs_vs_pred_call": (
                    float(spearmanr(group["observed_rate"], group["predicted_call_rate"]).statistic)
                    if len(group) >= 2 and group["observed_rate"].nunique() > 1 and group["predicted_call_rate"].nunique() > 1
                    else np.nan
                ),
                "exact_rank_match_pred_prob": int(
                    group.sort_values("observed_rate")["within_species_clade"].tolist()
                    == group.sort_values("predicted_mean_prob")["within_species_clade"].tolist()
                ),
                "exact_rank_match_pred_call": int(
                    group.sort_values("observed_rate")["within_species_clade"].tolist()
                    == group.sort_values("predicted_call_rate")["within_species_clade"].tolist()
                ),
            }
        )
        calib_rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "species": species,
                "n_clades": int(len(group)),
                "mae_pred_prob": float(group["abs_error_pred_prob"].mean()),
                "rmse_pred_prob": float(np.sqrt(group["sq_error_pred_prob"].mean())),
                "mae_pred_call": float(group["abs_error_pred_call"].mean()),
                "rmse_pred_call": float(np.sqrt(group["sq_error_pred_call"].mean())),
            }
        )
    rank_df = pd.DataFrame(rank_rows)
    calib_df = pd.DataFrame(calib_rows)

    summary = tip_summary.copy()
    if not rank_df.empty:
        rank_mean = (
            rank_df.groupby(["feature_set", "model"], dropna=False)
            .agg(
                rank_species_n=("species", "nunique"),
                rank_spearman_prob_mean=("spearman_obs_vs_pred_prob", "mean"),
                rank_spearman_call_mean=("spearman_obs_vs_pred_call", "mean"),
                rank_exact_prob_fraction=("exact_rank_match_pred_prob", "mean"),
                rank_exact_call_fraction=("exact_rank_match_pred_call", "mean"),
            )
            .reset_index()
        )
        summary = summary.merge(rank_mean, on=["feature_set", "model"], how="left")
    if not calib_df.empty:
        calib_mean = (
            calib_df.groupby(["feature_set", "model"], dropna=False)
            .agg(
                clade_mae_pred_prob=("mae_pred_prob", "mean"),
                clade_rmse_pred_prob=("rmse_pred_prob", "mean"),
                clade_mae_pred_call=("mae_pred_call", "mean"),
                clade_rmse_pred_call=("rmse_pred_call", "mean"),
            )
            .reset_index()
        )
        summary = summary.merge(calib_mean, on=["feature_set", "model"], how="left")
    return summary, clade_detail, rank_df, calib_df


def summarize_fold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cv_scheme, feature_set, model), group in df.groupby(["cv_scheme", "feature_set", "model"], dropna=False):
        row = {"cv_scheme": cv_scheme, "feature_set": feature_set, "model": model, "n_folds": int(len(group))}
        for metric in ["roc_auc", "average_precision", "balanced_accuracy", "brier"]:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_oof(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cv_scheme, feature_set, model), group in pred_df.groupby(["cv_scheme", "feature_set", "model"], dropna=False):
        y_true = group["target"].to_numpy(dtype=int)
        y_prob = group["probability"].to_numpy(dtype=float)
        row = {"cv_scheme": cv_scheme, "feature_set": feature_set, "model": model, "n_rows": int(len(group))}
        row.update({f"oof_{k}": v for k, v in metric_row(y_true, y_prob).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def merge_summaries(fold_summary: pd.DataFrame, oof_summary: pd.DataFrame, prevalence: float) -> pd.DataFrame:
    merged = fold_summary.merge(oof_summary, on=["cv_scheme", "feature_set", "model"], how="left")
    merged["average_precision_lift_vs_prevalence"] = merged["oof_average_precision"] / prevalence
    return (
        merged.sort_values(["cv_scheme", "oof_average_precision", "oof_roc_auc"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def extract_feature_effects(pipeline: FittedModelBundle, model_spec: ModelSpec, top_n: int) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    feature_names = pipeline.feature_names
    if model_spec.kind in {"logistic_l2", "logistic_elasticnet", "group_lasso", "sparse_group_lasso"} and hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            values = coef
        elif coef.shape[0] == len(feature_names):
            values = coef[:, -1]
        elif coef.shape[-1] == len(feature_names):
            values = coef.reshape(coef.shape[0], -1)[-1]
        else:
            values = coef.reshape(-1)[: len(feature_names)]
        out = pd.DataFrame({"feature": feature_names, "effect": values})
        out["abs_effect"] = out["effect"].abs()
        out["direction"] = np.where(out["effect"] >= 0, "positive", "negative")
        return out.sort_values("abs_effect", ascending=False).head(top_n).reset_index(drop=True)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        out = pd.DataFrame({"feature": feature_names, "effect": values})
        out["abs_effect"] = out["effect"].abs()
        out["direction"] = "positive"
        return out.sort_values("abs_effect", ascending=False).head(top_n).reset_index(drop=True)
    return pd.DataFrame(columns=["feature", "effect", "abs_effect", "direction"])


def _map_transformed_feature_name(transformed_name: str, original_columns: Sequence[str]) -> str:
    base = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    if transformed_name.startswith("cat__"):
        for candidate in sorted(set(str(col) for col in original_columns), key=len, reverse=True):
            if base == candidate or base.startswith(f"{candidate}_"):
                return candidate
    return base


def aggregate_feature_importances(
    pipeline: FittedModelBundle,
    model_spec: ModelSpec,
    original_columns: Sequence[str],
) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature_name", "importance"])
    values = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
    rows: List[Dict[str, object]] = []
    for transformed, value in zip(pipeline.feature_names, values):
        if not np.isfinite(value) or float(value) <= 0.0:
            continue
        rows.append(
            {
                "feature_name": _map_transformed_feature_name(str(transformed), original_columns),
                "importance": float(value),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature_name", "importance"])
    out = pd.DataFrame(rows)
    out = out.groupby("feature_name", as_index=False)["importance"].sum()
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def _nonmissing_fraction(series: pd.Series) -> float:
    return float(series.notna().mean()) if len(series) else np.nan


def _within_species_varied(group: pd.DataFrame, col: str) -> bool:
    series = group[col]
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.dropna() if numeric.notna().any() else series.dropna().astype(str)
    return bool(values.nunique() > 1)


def _feature_within_species_variation_stats(df: pd.DataFrame, feature: str) -> Dict[str, object]:
    varied_species = 0
    eligible_species = 0
    for _, group in df.groupby("species", dropna=False):
        series = group[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        values = numeric.dropna() if numeric.notna().any() else series.dropna().astype(str)
        if values.shape[0] < 2:
            continue
        eligible_species += 1
        if values.nunique() > 1:
            varied_species += 1
    return {
        "within_species_varied_species": int(varied_species),
        "within_species_eligible_species": int(eligible_species),
        "within_species_variation_fraction": (
            float(varied_species / eligible_species) if eligible_species > 0 else np.nan
        ),
        "has_within_species_variation": int(varied_species > 0),
    }


def compute_ld_feature_stability(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_blocks: Mapping[str, Mapping[str, Sequence[str]]],
    within_species_clades: pd.Series,
    seed: int,
) -> pd.DataFrame:
    lineage_feature_set = next(
        feature_set for feature_set in FEATURE_SETS if feature_set.name == "ecology_molecular_lineage_plus_context"
    )
    X, numeric, categorical = build_feature_frame(df, lineage_feature_set, feature_blocks)
    original_columns = list(numeric) + list(categorical)
    ld_candidates = [col for col in original_columns if str(col).startswith("ld_site_")]
    if not ld_candidates:
        return pd.DataFrame(columns=["feature_name", "model", "top_tranche_folds", "n_folds", "stability_fraction"])

    fold_list = build_cv_plans(
        argparse.Namespace(
            cv_schemes=["grouped_within_species_clade"],
            max_folds=None,
            repeated_splits=0,
            repeated_repeats=0,
            random_state=seed,
        ),
        df,
        y,
        df["species"].fillna("missing_species").astype(str).to_numpy(),
        within_species_eval_groups=within_species_clades,
    )[0][1]
    rows: List[Dict[str, object]] = []
    for model_name in ["xgboost", "lightgbm"]:
        model_spec = next(spec for spec in MODELS if spec.name == model_name)
        counts = {feature: 0 for feature in ld_candidates}
        n_folds = 0
        for fold_id, (train_idx, test_idx) in enumerate(fold_list, start=1):
            del test_idx
            X_train = X.iloc[train_idx].copy()
            y_train = y[train_idx]
            try:
                pipeline = fit_pipeline(
                    model_spec,
                    X_train,
                    y_train,
                    list(numeric),
                    list(categorical),
                    seed=seed + fold_id,
                )
            except Exception:
                continue
            imp = aggregate_feature_importances(pipeline, model_spec, original_columns)
            imp = imp[imp["feature_name"].isin(ld_candidates)].copy()
            if imp.empty:
                continue
            n_folds += 1
            tranche_n = max(10, int(np.ceil(0.2 * len(imp))))
            top = imp.head(tranche_n)["feature_name"].astype(str).tolist()
            for feature in top:
                counts[feature] += 1
        for feature, count in counts.items():
            rows.append(
                {
                    "feature_name": feature,
                    "model": model_name,
                    "top_tranche_folds": int(count),
                    "n_folds": int(n_folds),
                    "stability_fraction": float(count / n_folds) if n_folds > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_lineage_experimental_gate(
    df: pd.DataFrame,
    feature_blocks: Mapping[str, Mapping[str, Sequence[str]]],
    ld_stability: pd.DataFrame,
    terminal_min_coverage: float = 0.85,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ecology_rows: List[Dict[str, object]] = []
    molecular_rows: List[Dict[str, object]] = []
    gate_rows: List[Dict[str, object]] = []

    stability_lookup: Dict[str, Dict[str, float]] = {}
    if not ld_stability.empty:
        for feature, group in ld_stability.groupby("feature_name", dropna=False):
            stability_lookup[str(feature)] = {
                str(row["model"]): float(row["stability_fraction"])
                for _, row in group.iterrows()
            }

    def _add_gate_row(feature: str, block_name: str, include: bool, reason: str, audit: Dict[str, object]) -> None:
        gate_rows.append(
            {
                "feature_name": feature,
                "experimental_block": block_name,
                "include": int(include),
                "reason": reason,
                **audit,
            }
        )

    ecology_candidates = list(dict.fromkeys(feature_blocks["ecology_core"]["numeric"] + feature_blocks["ecology_core"]["categorical"]))
    for feature in ecology_candidates:
        audit = {
            "nonmissing_fraction": _nonmissing_fraction(df[feature]),
            **_feature_within_species_variation_stats(df, feature),
            "is_species_level_summary": int(str(feature).startswith("species_feat_")),
        }
        ecology_rows.append({"feature_name": feature, **audit})
        include = bool(audit["has_within_species_variation"])
        if str(feature).startswith("hosttrait_"):
            include = include and float(audit["nonmissing_fraction"]) >= 0.5
        if feature in {"overlap_opportunity_local", "arha_prev_local"}:
            include = include and float(audit["nonmissing_fraction"]) > 0
        reason = "pass"
        if not include:
            if not audit["has_within_species_variation"]:
                reason = "no_within_species_variation"
            elif str(feature).startswith("hosttrait_") and float(audit["nonmissing_fraction"]) < 0.5:
                reason = "host_trait_low_coverage"
            else:
                reason = "excluded"
        _add_gate_row(feature, "ecology_core", include, reason, audit)

    molecular_candidates = list(dict.fromkeys(feature_blocks["molecular_core"]["numeric"] + feature_blocks["molecular_core"]["categorical"]))
    for feature in molecular_candidates:
        family = (
            "ld_site" if str(feature).startswith("ld_site_")
            else "terminal" if str(feature).startswith("utrterm_")
            else "discord" if str(feature).startswith("discord_") or feature == "tree_discordance_score"
            else "other"
        )
        audit = {
            "family": family,
            "nonmissing_fraction": _nonmissing_fraction(df[feature]),
            **_feature_within_species_variation_stats(df, feature),
            "is_species_level_summary": int(str(feature).startswith("species_feat_")),
            "xgboost_stability_fraction": stability_lookup.get(str(feature), {}).get("xgboost", np.nan),
            "lightgbm_stability_fraction": stability_lookup.get(str(feature), {}).get("lightgbm", np.nan),
        }
        molecular_rows.append({"feature_name": feature, **audit})
        include = bool(audit["has_within_species_variation"]) and not bool(audit["is_species_level_summary"])
        reason = "pass"
        if family == "terminal":
            include = include and float(audit["nonmissing_fraction"]) >= float(terminal_min_coverage)
            if not include and float(audit["nonmissing_fraction"]) < float(terminal_min_coverage):
                reason = "terminal_low_coverage"
        elif family == "ld_site":
            stable = max(
                float(audit["xgboost_stability_fraction"]) if np.isfinite(audit["xgboost_stability_fraction"]) else 0.0,
                float(audit["lightgbm_stability_fraction"]) if np.isfinite(audit["lightgbm_stability_fraction"]) else 0.0,
            )
            include = include and float(audit["nonmissing_fraction"]) >= 0.9 and stable >= 0.3
            if not include:
                if float(audit["nonmissing_fraction"]) < 0.9:
                    reason = "ld_low_coverage"
                elif stable < 0.3:
                    reason = "ld_unstable_across_folds"
                elif not audit["has_within_species_variation"]:
                    reason = "no_within_species_variation"
                else:
                    reason = "excluded"
        else:
            if not include:
                reason = "no_within_species_variation"
        _add_gate_row(feature, "molecular_core", include, reason, audit)

    for block_name in ["context", "ecology_species", "molecular_species"]:
        block = feature_blocks[block_name]
        for feature in list(dict.fromkeys(block["numeric"] + block["categorical"])):
            audit = {
                "nonmissing_fraction": _nonmissing_fraction(df[feature]),
                **_feature_within_species_variation_stats(df, feature),
                "is_species_level_summary": int(str(feature).startswith("species_feat_")),
            }
            include = float(audit["nonmissing_fraction"]) > 0 and (
                audit["has_within_species_variation"] or audit["is_species_level_summary"]
            )
            reason = "pass" if include else "excluded"
            _add_gate_row(feature, block_name, include, reason, audit)

    return (
        pd.DataFrame(ecology_rows).sort_values("feature_name").reset_index(drop=True),
        pd.DataFrame(molecular_rows).sort_values("feature_name").reset_index(drop=True),
        pd.DataFrame(gate_rows).sort_values(["experimental_block", "include", "feature_name"], ascending=[True, False, True]).reset_index(drop=True),
    )


def apply_lineage_gate_to_blocks(
    feature_blocks: Mapping[str, Mapping[str, Sequence[str]]],
    gate_df: pd.DataFrame,
) -> Dict[str, Dict[str, List[str]]]:
    gated: Dict[str, Dict[str, List[str]]] = {
        name: {"numeric": list(block["numeric"]), "categorical": list(block["categorical"])}
        for name, block in feature_blocks.items()
    }
    if gate_df.empty:
        return gated
    include_lookup = {
        (str(row["experimental_block"]), str(row["feature_name"])): bool(row["include"])
        for _, row in gate_df.iterrows()
    }
    for block_name in ["ecology_core", "molecular_core", "context", "ecology_species", "molecular_species"]:
        if block_name not in gated:
            continue
        gated[block_name]["numeric"] = [
            col for col in gated[block_name]["numeric"]
            if include_lookup.get((block_name, str(col)), True)
        ]
        gated[block_name]["categorical"] = [
            col for col in gated[block_name]["categorical"]
            if include_lookup.get((block_name, str(col)), True)
        ]
    return gated


def append_iteration_log(path: Path, summary_df: pd.DataFrame, dataset: Path, run_info: Dict[str, object]) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_df = summary_df.copy()
    log_df.insert(0, "timestamp_utc", timestamp)
    log_df.insert(1, "dataset", str(dataset))
    log_df.insert(2, "target_source_filter", str(run_info["target_source_filter"]))
    log_df.insert(3, "group_column", str(run_info.get("group_column", "")))
    log_df.insert(4, "labeled_rows", int(run_info["labeled_rows"]))
    log_df.insert(5, "positive_rows", int(run_info["positive_rows"]))
    log_df.insert(6, "positive_rate", float(run_info["positive_rate"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, sep="\t")
        log_df = pd.concat([old, log_df], ignore_index=True, sort=False)
    log_df.to_csv(path, sep="\t", index=False)


def build_interpretation_markdown(
    summary_df: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    species_effect_summary: pd.DataFrame,
    flatness_df: pd.DataFrame,
    run_info: Dict[str, object],
    top_effects: Dict[Tuple[str, str], pd.DataFrame],
) -> str:
    best_row = summary_df.sort_values(["oof_average_precision", "oof_roc_auc"], ascending=[False, False]).iloc[0]
    per_feature = (
        summary_df.sort_values(["cv_scheme", "feature_set", "oof_average_precision"], ascending=[True, True, False])
        .groupby(["cv_scheme", "feature_set"], as_index=False)
        .first()
    )
    per_scheme = (
        summary_df.sort_values(["cv_scheme", "oof_average_precision", "oof_roc_auc"], ascending=[True, False, False])
        .groupby("cv_scheme", as_index=False)
        .first()
    )
    flat_rows = flatness_df[flatness_df["is_flat_probability"] == 1].copy()
    lines = [
        "# Sequence Explanatory Model Suite",
        "",
        "## Dataset",
        f"- Dataset: {run_info['dataset']}",
        f"- Labeled rows: {run_info['labeled_rows']}",
        f"- Positives: {run_info['positive_rows']}",
        f"- Positive rate: {run_info['positive_rate']:.3f}",
        f"- Validation schemes: {', '.join(run_info['cv_schemes'])}",
        f"- Species groups evaluated: {run_info['n_groups']}",
        f"- Virus clades evaluated: {run_info['n_clades']}",
        f"- Target filter: {run_info['target_source_filter']}",
        f"- LD encoding: {run_info.get('ld_encoding', 'full')}",
        "",
        "## Bayesian Status",
        f"- PyMC: {run_info['bayesian_tooling']['pymc']}",
        f"- Bambi: {run_info['bayesian_tooling']['bambi']}",
        f"- ArviZ: {run_info['bayesian_tooling']['arviz']}",
        f"- statsmodels: {run_info['bayesian_tooling']['statsmodels']}",
        "- Fallback: species-effect reduction is reported with regularized fixed-effect logistic models.",
        "",
        "## Primary Grouped Report",
        f"- Best overall model: {best_row['cv_scheme']} / {best_row['feature_set']} / {best_row['model']}",
        f"- OOF average precision: {best_row['oof_average_precision']:.3f}",
        f"- OOF ROC AUC: {best_row['oof_roc_auc']:.3f}" if pd.notna(best_row["oof_roc_auc"]) else "- OOF ROC AUC: NA",
        f"- OOF balanced accuracy: {best_row['oof_balanced_accuracy']:.3f}",
        f"- AP lift vs prevalence: {best_row['average_precision_lift_vs_prevalence']:.2f}x",
        "",
        "## Best Model Per CV Scheme",
    ]
    for _, row in per_scheme.iterrows():
        lines.append(
            f"- {row['cv_scheme']}: {row['feature_set']} / {row['model']} with AP={row['oof_average_precision']:.3f}, ROC AUC={row['oof_roc_auc']:.3f}"
            if pd.notna(row["oof_roc_auc"])
            else f"- {row['cv_scheme']}: {row['feature_set']} / {row['model']} with AP={row['oof_average_precision']:.3f}, ROC AUC=NA"
        )
    lines.extend([
        "",
        "## Feature Stacks",
    ])
    for _, row in feature_catalog.sort_values("feature_set").iterrows():
        lines.append(
            f"- {row['feature_set']}: {row['description']} ({row['n_numeric']} numeric, {row['n_categorical']} categorical)"
        )
    lines.extend(["", "## Best Model Per Stack"])
    for _, row in per_feature.iterrows():
        lines.append(
            f"- {row['cv_scheme']} / {row['feature_set']}: {row['model']} with AP={row['oof_average_precision']:.3f}, ROC AUC={row['oof_roc_auc']:.3f}"
            if pd.notna(row["oof_roc_auc"])
            else f"- {row['cv_scheme']} / {row['feature_set']}: {row['model']} with AP={row['oof_average_precision']:.3f}, ROC AUC=NA"
        )
    lines.extend(["", "## Flat Probability Diagnostics"])
    if flat_rows.empty:
        lines.append("- No species/model combinations were perfectly flat.")
    else:
        for _, row in flat_rows.head(12).iterrows():
            lines.append(
                f"- {row['cv_scheme']} / {row['feature_set']} / {row['model']} / {row['heldout_group']} / {row['species']}: flat predictions across {int(row['n_rows'])} held-out rows"
            )
    lines.extend(["", "## Species-Effect Reduction Fallback"])
    for _, row in species_effect_summary.sort_values("model").iterrows():
        l2_reduction = row["l2_reduction_fraction"]
        mean_reduction = row["mean_abs_reduction_fraction"]
        lines.append(
            f"- {row['model']}: L2 reduction={l2_reduction:.3f}, mean-abs reduction={mean_reduction:.3f}"
            if pd.notna(l2_reduction) and pd.notna(mean_reduction)
            else f"- {row['model']}: insufficient variation to quantify reduction"
        )
    lines.extend(["", "## Top Effects"])
    for key, effects in top_effects.items():
        feature_set, model = key
        if effects.empty:
            continue
        lines.append(f"- {feature_set} / {model}")
        preview = effects.head(8)
        for _, row in preview.iterrows():
            lines.append(f"  - {row['feature']}: {row['effect']:.4f} ({row['direction']})")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    excluded_within_species = _normalize_species_exclusion(args.within_species_exclude_species)

    selected_feature_sets = select_feature_sets(args.feature_sets)
    selected_models = select_models(args.models)
    requested_lineage_experimental = any(
        feature_set.name.startswith("ecology_molecular_lineage_") for feature_set in selected_feature_sets
    )

    df = load_dataset(args.dataset, args.target_source)
    y = df["target"].to_numpy()
    groups = df[args.group_column].fillna("missing_group").astype(str).to_numpy()
    clade_groups = assign_species_clade(df["species"].fillna("missing_species").astype(str))
    within_species_clades, within_species_clade_manifest = assign_within_species_tree_clades(
        df,
        min_clade_size=args.within_species_min_clade_size,
        max_clusters=args.within_species_max_clusters,
        excluded_species=excluded_within_species,
    )
    if args.within_species_grouping_method == "baltic_nodes":
        if args.within_species_baltic_path is None:
            raise SystemExit("--within-species-baltic-path is required when using --within-species-grouping-method baltic_nodes.")
        within_species_eval_groups, within_species_eval_manifest = build_within_species_baltic_eval_groups(
            df,
            y,
            min_clade_size=args.within_species_min_clade_size,
            min_test_fraction=float(args.within_species_min_test_fraction),
            max_test_fraction=float(args.within_species_max_test_fraction),
            random_state=int(args.random_state),
            baltic_path=args.within_species_baltic_path,
            excluded_species=excluded_within_species,
        )
    else:
        within_species_eval_groups, within_species_eval_manifest = build_within_species_eval_groups(
            df,
            y,
            within_species_clades,
            min_clade_size=args.within_species_min_clade_size,
            max_clusters=args.within_species_max_clusters,
            max_test_fraction=float(args.within_species_max_test_fraction),
            random_state=int(args.random_state),
            excluded_species=excluded_within_species,
        )
    cv_plans = build_cv_plans(args, df, y, groups, within_species_eval_groups=within_species_eval_groups)
    if not cv_plans:
        raise SystemExit("No grouped folds available for evaluation.")

    df_model = df.copy()
    ecology_audit_df = pd.DataFrame()
    molecular_audit_df = pd.DataFrame()
    lineage_gate_df = pd.DataFrame()
    ld_stability_df = pd.DataFrame()
    if requested_lineage_experimental:
        df_model = pd.concat(
            [
                df_model,
                add_direct_discordance_features(df_model),
                add_direct_utr_terminal_features(df_model),
            ],
            axis=1,
        )
    feature_blocks = apply_ld_encoding(detect_feature_blocks(df_model), args.ld_encoding)
    if requested_lineage_experimental:
        ld_stability_df = compute_ld_feature_stability(
            df_model,
            y,
            feature_blocks,
            within_species_clades,
            seed=args.random_state + 9000,
        )
        ecology_audit_df, molecular_audit_df, lineage_gate_df = build_lineage_experimental_gate(
            df_model,
            feature_blocks,
            ld_stability_df,
            terminal_min_coverage=float(args.lineage_terminal_min_coverage),
        )
        feature_blocks = apply_lineage_gate_to_blocks(feature_blocks, lineage_gate_df)
    feature_catalog = build_feature_catalog(selected_feature_sets, feature_blocks)
    within_species_variation = summarize_within_species_feature_variation(df_model, feature_blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        species_effect_summary, species_effect_coefficients = summarize_species_effects(
            df_model,
            y,
            feature_blocks,
            seed=args.random_state + 5000,
        )
    bayesian_tooling = try_import_bayesian_tooling()

    run_info = {
        "dataset": str(args.dataset),
        "dataset_role": "final explanatory dataset",
        "labeled_rows": int(len(df)),
        "positive_rows": int(y.sum()),
        "negative_rows": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
        "target_source_filter": args.target_source,
        "group_column": args.group_column,
        "n_groups": int(pd.Series(groups).nunique()),
        "group_sizes": pd.Series(groups).value_counts().to_dict(),
        "n_clades": int(clade_groups.nunique()),
        "clade_sizes": clade_groups.value_counts().to_dict(),
        "within_species_lineage_groups": int(within_species_clades.dropna().nunique()),
        "within_species_eval_groups": int(within_species_eval_groups.dropna().nunique()),
        "within_species_min_clade_size": int(args.within_species_min_clade_size),
        "within_species_max_clusters": int(args.within_species_max_clusters),
        "within_species_min_test_fraction": float(args.within_species_min_test_fraction),
        "within_species_max_test_fraction": float(args.within_species_max_test_fraction),
        "within_species_grouping_method": str(args.within_species_grouping_method),
        "within_species_excluded_species": sorted(excluded_within_species),
        "within_species_baltic_path": str(args.within_species_baltic_path) if args.within_species_baltic_path else "",
        "ld_encoding": args.ld_encoding,
        "cv_schemes": [scheme for scheme, _ in cv_plans],
        "evaluated_folds": {scheme: int(len(splits)) for scheme, splits in cv_plans},
        "feature_sets": [feature_set.name for feature_set in selected_feature_sets],
        "models": [model.name for model in selected_models],
        "requested_lineage_experimental": bool(requested_lineage_experimental),
        "tree_bayes_search": bool(args.tree_bayes_search),
        "tree_bayes_iters": int(args.tree_bayes_iters),
        "tree_bayes_inner_splits": int(args.tree_bayes_inner_splits),
        "tree_bayes_metric": str(args.tree_bayes_metric),
        "tree_bayes_shared_by_cv": bool(args.tree_bayes_shared_by_cv),
        "within_species_lineage_molecular_context": bool(args.within_species_lineage_molecular_context),
        "exploratory_real_lineage_context": bool(args.exploratory_real_lineage_context),
        "exploratory_codon_composition": bool(args.exploratory_codon_composition),
        "bayesian_tooling": bayesian_tooling,
        "species_effect_fallback": "regularized_fixed_effect_logistic",
        "ld_encoding": args.ld_encoding,
    }

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    fitted_models: Dict[Tuple[str, str], FittedModelBundle] = {}
    fitted_specs: Dict[Tuple[str, str], ModelSpec] = {}

    for feature_set in selected_feature_sets:
        X, numeric, categorical = build_feature_frame(df_model, feature_set, feature_blocks)
        if args.exploratory_codon_composition and "molecular" in feature_set.block_names:
            codon_aug = add_codon_composition_features(df_model)
            X = pd.concat([X, codon_aug], axis=1)
            numeric = list(dict.fromkeys(list(numeric) + list(codon_aug.columns)))
        molecular_numeric = list(feature_blocks["molecular"]["numeric"])
        molecular_categorical = list(feature_blocks["molecular"]["categorical"])
        for model_spec in selected_models:
            shared_params_by_scheme: Dict[str, Optional[Dict[str, Any]]] = {}
            shared_info_by_scheme: Dict[str, Dict[str, object]] = {}
            if args.tree_bayes_search and args.tree_bayes_shared_by_cv and is_tree_model(model_spec):
                for cv_scheme, split_iter in cv_plans:
                    print(f"[tune:{cv_scheme}] {feature_set.name} / {model_spec.name}", flush=True)
                    params, info = tune_tree_params_shared(
                        model_spec,
                        X,
                        y,
                        numeric,
                        categorical,
                        cv_scheme=cv_scheme,
                        splits=split_iter,
                        seed=args.random_state + 7000,
                        n_iter=int(args.tree_bayes_iters),
                        metric=str(args.tree_bayes_metric),
                    )
                    shared_params_by_scheme[cv_scheme] = params
                    shared_info_by_scheme[cv_scheme] = dict(info)
                    tuning_rows.append(
                        {
                            "cv_scheme": cv_scheme,
                            "feature_set": feature_set.name,
                            "model": model_spec.name,
                            "fold_id": -1,
                            "fit_scope": "shared_cv_tune",
                            **info,
                        }
                    )
            for cv_scheme, split_iter in cv_plans:
                print(f"[{cv_scheme}] {feature_set.name} / {model_spec.name}", flush=True)
                for fold_id, (train_idx, test_idx) in enumerate(split_iter, start=1):
                    X_train = X.iloc[train_idx].copy()
                    X_test = X.iloc[test_idx].copy()
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                    fold_numeric = list(numeric)
                    fold_categorical = list(categorical)
                    lineage_context_active = bool(
                        args.within_species_lineage_molecular_context
                        and cv_scheme == "grouped_within_species_clade"
                        and "molecular" in feature_set.block_names
                    )
                    real_lineage_context_active = bool(
                        args.exploratory_real_lineage_context
                        and "molecular" in feature_set.block_names
                    )
                    if lineage_context_active:
                        X_train, X_test, extra_numeric, extra_categorical = add_lineage_molecular_context(
                            X_train,
                            X_test,
                            df_model.iloc[train_idx],
                            df_model.iloc[test_idx],
                            molecular_numeric=molecular_numeric,
                            molecular_categorical=molecular_categorical,
                            within_species_train=within_species_clades.iloc[train_idx],
                        )
                        fold_numeric = list(dict.fromkeys(fold_numeric + extra_numeric))
                        fold_categorical = list(dict.fromkeys(fold_categorical + extra_categorical))
                    if real_lineage_context_active:
                        X_train, X_test, extra_numeric, extra_categorical = add_real_lineage_context(
                            X_train,
                            X_test,
                            df_model.iloc[train_idx],
                            df_model.iloc[test_idx],
                            within_species_clades.iloc[train_idx],
                        )
                        fold_numeric = list(dict.fromkeys(fold_numeric + extra_numeric))
                        fold_categorical = list(dict.fromkeys(fold_categorical + extra_categorical))
                    seed = args.random_state + fold_id
                    tree_tuning = TreeTuningConfig(
                        enabled=bool(args.tree_bayes_search and not args.tree_bayes_shared_by_cv and is_tree_model(model_spec)),
                        n_iter=int(args.tree_bayes_iters),
                        inner_splits=int(args.tree_bayes_inner_splits),
                        metric=str(args.tree_bayes_metric),
                        random_state=int(args.random_state),
                        cv_scheme=str(cv_scheme),
                        species_groups=df.iloc[train_idx][args.group_column].fillna("missing_group").astype(str),
                        clade_groups=assign_species_clade(df_model.iloc[train_idx]["species"].fillna("missing_species").astype(str)),
                        within_species_groups=within_species_eval_groups.iloc[train_idx],
                    )

                    pipeline = fit_pipeline(
                        model_spec,
                        X_train,
                        y_train,
                        fold_numeric,
                        fold_categorical,
                        seed,
                        tree_tuning=tree_tuning,
                        tuned_params=shared_params_by_scheme.get(cv_scheme),
                        tuned_info=shared_info_by_scheme.get(cv_scheme),
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        y_prob = pipeline.predict_proba(X_test)[:, 1]

                    row = {
                        "cv_scheme": cv_scheme,
                        "feature_set": feature_set.name,
                        "model": model_spec.name,
                        "fold_id": fold_id,
                        "train_rows": int(len(train_idx)),
                        "test_rows": int(len(test_idx)),
                        "train_positives": int(y_train.sum()),
                        "test_positives": int(y_test.sum()),
                        "test_groups": "|".join(sorted(set(groups[test_idx]))),
                        "heldout_group": (
                            "|".join(sorted(set(within_species_eval_groups.iloc[test_idx].dropna().astype(str))))
                            if cv_scheme == "grouped_within_species_clade"
                            else heldout_label(cv_scheme, df_model.iloc[test_idx]["species"])
                        ),
                    }
                    row.update(metric_row(y_test, y_prob))
                    if pipeline.tuning_info:
                        row.update(pipeline.tuning_info)
                    fold_rows.append(row)
                    if pipeline.tuning_info:
                        tuning_rows.append(
                            {
                                "cv_scheme": cv_scheme,
                                "feature_set": feature_set.name,
                                "model": model_spec.name,
                                "fold_id": fold_id,
                                "fit_scope": "outer_fold",
                                **pipeline.tuning_info,
                            }
                        )

                    for pos, row_idx in enumerate(test_idx):
                        pred_rows.append(
                            {
                                "cv_scheme": cv_scheme,
                                "feature_set": feature_set.name,
                                "model": model_spec.name,
                                "fold_id": fold_id,
                                "row_index": int(row_idx),
                                "tip_name": df_model.iloc[row_idx]["tip_name"] if "tip_name" in df_model.columns else str(row_idx),
                                "species": df_model.iloc[row_idx]["species"] if "species" in df_model.columns else pd.NA,
                                "clade": assign_species_clade(pd.Series([df_model.iloc[row_idx]["species"]])).iloc[0],
                                "within_species_clade": within_species_clades.iloc[row_idx],
                                "within_species_eval_group": within_species_eval_groups.iloc[row_idx],
                                "heldout_group": row["heldout_group"],
                                "target": int(y_test[pos]),
                                "probability": float(y_prob[pos]),
                                "lineage_molecular_context": lineage_context_active,
                                "real_lineage_context": real_lineage_context_active,
                            }
                        )

            full_seed = args.random_state + 1000
            X_full = X.copy()
            full_numeric = list(numeric)
            full_categorical = list(categorical)
            if (
                args.within_species_lineage_molecular_context
                and "molecular" in feature_set.block_names
                and within_species_clades.notna().any()
            ):
                X_full, _, extra_numeric, extra_categorical = add_lineage_molecular_context(
                    X_full,
                    X_full.iloc[0:0].copy(),
                    df_model,
                    df_model.iloc[0:0].copy(),
                    molecular_numeric=molecular_numeric,
                    molecular_categorical=molecular_categorical,
                    within_species_train=within_species_clades,
                )
                full_numeric = list(dict.fromkeys(full_numeric + extra_numeric))
                full_categorical = list(dict.fromkeys(full_categorical + extra_categorical))
            if args.exploratory_real_lineage_context and "molecular" in feature_set.block_names:
                X_full, _, extra_numeric, extra_categorical = add_real_lineage_context(
                    X_full,
                    X_full.iloc[0:0].copy(),
                    df_model,
                    df_model.iloc[0:0].copy(),
                    within_species_clades,
                )
                full_numeric = list(dict.fromkeys(full_numeric + extra_numeric))
                full_categorical = list(dict.fromkeys(full_categorical + extra_categorical))
            full_tree_tuning = TreeTuningConfig(
                enabled=bool(args.tree_bayes_search and not args.tree_bayes_shared_by_cv and is_tree_model(model_spec)),
                n_iter=int(args.tree_bayes_iters),
                inner_splits=int(args.tree_bayes_inner_splits),
                metric=str(args.tree_bayes_metric),
                random_state=int(args.random_state + 1000),
                cv_scheme="grouped_within_species_clade" if within_species_clades.notna().any() else "repeated_stratified",
                species_groups=df_model[args.group_column].fillna("missing_group").astype(str),
                clade_groups=assign_species_clade(df_model["species"].fillna("missing_species").astype(str)),
                within_species_groups=within_species_clades,
            )
            full_pipeline = fit_pipeline(
                model_spec,
                X_full,
                y,
                full_numeric,
                full_categorical,
                full_seed,
                tree_tuning=full_tree_tuning,
                tuned_params=shared_params_by_scheme.get("repeated_stratified"),
                tuned_info=shared_info_by_scheme.get("repeated_stratified"),
            )
            fitted_models[(feature_set.name, model_spec.name)] = full_pipeline
            fitted_specs[(feature_set.name, model_spec.name)] = model_spec
            if full_pipeline.tuning_info:
                tuning_rows.append(
                    {
                        "cv_scheme": full_tree_tuning.cv_scheme,
                        "feature_set": feature_set.name,
                        "model": model_spec.name,
                        "fold_id": 0,
                        "fit_scope": "full_fit",
                        **full_pipeline.tuning_info,
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    tuning_df = pd.DataFrame(tuning_rows)
    fold_summary = summarize_fold_metrics(fold_df)
    oof_summary = summarize_oof(pred_df)
    summary_df = merge_summaries(fold_summary, oof_summary, run_info["positive_rate"])
    flatness_df = summarize_prediction_flatness(pred_df)
    grouped_holdout_df = summarize_grouped_holdout_metrics(fold_df)
    repeated_context_df = summarize_repeated_fold_context(fold_df, pred_df)
    lineage_eval_summary_df, lineage_clade_detail_df, lineage_rank_df, lineage_calib_df = summarize_lineage_prediction_behavior(pred_df)

    top_effects: Dict[Tuple[str, str], pd.DataFrame] = {}
    effect_frames = []
    for key, pipeline in fitted_models.items():
        effects = extract_feature_effects(pipeline, fitted_specs[key], top_n=args.top_effects)
        effects.insert(0, "model", key[1])
        effects.insert(0, "feature_set", key[0])
        effect_frames.append(effects)
        top_effects[key] = effects
    effect_df = pd.concat(effect_frames, ignore_index=True) if effect_frames else pd.DataFrame()

    grouped_summary_path = args.output_dir / "grouped_species_summary.tsv"
    fold_path = args.output_dir / "grouped_species_cv_fold_metrics.tsv"
    pred_path = args.output_dir / "grouped_species_cv_predictions.tsv"
    repeated_summary_path = args.output_dir / "repeated_stratified_summary.tsv"
    repeated_fold_path = args.output_dir / "repeated_stratified_cv_fold_metrics.tsv"
    repeated_pred_path = args.output_dir / "repeated_stratified_cv_predictions.tsv"
    summary_path = args.output_dir / "summary.tsv"
    all_fold_path = args.output_dir / "cv_fold_metrics.tsv"
    all_pred_path = args.output_dir / "oof_predictions.tsv"
    feature_path = args.output_dir / "feature_catalog.tsv"
    effect_path = args.output_dir / "model_effects.tsv"
    species_effect_path = args.output_dir / "species_effect_reduction.tsv"
    species_coeff_path = args.output_dir / "species_effect_coefficients.tsv"
    flatness_path = args.output_dir / "prediction_flatness_by_species.tsv"
    variation_path = args.output_dir / "within_species_feature_variation.tsv"
    grouped_holdout_path = args.output_dir / "grouped_holdout_metrics.tsv"
    repeated_context_path = args.output_dir / "repeated_stratified_fold_context.tsv"
    lineage_eval_summary_path = args.output_dir / "grouped_within_species_clade_lineage_eval_summary.tsv"
    lineage_clade_detail_path = args.output_dir / "grouped_within_species_clade_clade_observed_vs_predicted.tsv"
    lineage_rank_path = args.output_dir / "grouped_within_species_clade_clade_rank_agreement.tsv"
    lineage_calib_path = args.output_dir / "grouped_within_species_clade_clade_calibration_by_species.tsv"
    tuning_path = args.output_dir / "tree_model_tuning.tsv"
    lineage_ecology_audit_path = args.output_dir / "lineage_core_ecology_audit.tsv"
    lineage_molecular_audit_path = args.output_dir / "lineage_core_molecular_audit.tsv"
    lineage_gate_path = args.output_dir / "lineage_core_feature_gate.tsv"
    lineage_ld_stability_path = args.output_dir / "lineage_core_ld_stability.tsv"
    within_species_manifest_path = args.output_dir / "within_species_tree_clade_manifest.tsv"
    within_species_eval_manifest_path = args.output_dir / "within_species_tree_eval_group_manifest.tsv"
    info_path = args.output_dir / "run_info.json"
    md_path = args.output_dir / "interpretation.md"

    summary_df.to_csv(summary_path, sep="\t", index=False)
    fold_df.to_csv(all_fold_path, sep="\t", index=False)
    pred_df.to_csv(all_pred_path, sep="\t", index=False)
    summary_df[summary_df["cv_scheme"] == "grouped_species"].to_csv(grouped_summary_path, sep="\t", index=False)
    fold_df[fold_df["cv_scheme"] == "grouped_species"].to_csv(fold_path, sep="\t", index=False)
    pred_df[pred_df["cv_scheme"] == "grouped_species"].to_csv(pred_path, sep="\t", index=False)
    summary_df[summary_df["cv_scheme"] == "grouped_clade"].to_csv(
        args.output_dir / "grouped_clade_summary.tsv",
        sep="\t",
        index=False,
    )
    fold_df[fold_df["cv_scheme"] == "grouped_clade"].to_csv(
        args.output_dir / "grouped_clade_cv_fold_metrics.tsv",
        sep="\t",
        index=False,
    )
    pred_df[pred_df["cv_scheme"] == "grouped_clade"].to_csv(
        args.output_dir / "grouped_clade_cv_predictions.tsv",
        sep="\t",
        index=False,
    )
    summary_df[summary_df["cv_scheme"] == "grouped_within_species_clade"].to_csv(
        args.output_dir / "grouped_within_species_clade_summary.tsv",
        sep="\t",
        index=False,
    )
    fold_df[fold_df["cv_scheme"] == "grouped_within_species_clade"].to_csv(
        args.output_dir / "grouped_within_species_clade_cv_fold_metrics.tsv",
        sep="\t",
        index=False,
    )
    pred_df[pred_df["cv_scheme"] == "grouped_within_species_clade"].to_csv(
        args.output_dir / "grouped_within_species_clade_cv_predictions.tsv",
        sep="\t",
        index=False,
    )
    summary_df[summary_df["cv_scheme"] == "repeated_stratified"].to_csv(repeated_summary_path, sep="\t", index=False)
    fold_df[fold_df["cv_scheme"] == "repeated_stratified"].to_csv(repeated_fold_path, sep="\t", index=False)
    pred_df[pred_df["cv_scheme"] == "repeated_stratified"].to_csv(repeated_pred_path, sep="\t", index=False)
    feature_catalog.to_csv(feature_path, sep="\t", index=False)
    effect_df.to_csv(effect_path, sep="\t", index=False)
    species_effect_summary.to_csv(species_effect_path, sep="\t", index=False)
    species_effect_coefficients.to_csv(species_coeff_path, sep="\t", index=False)
    flatness_df.to_csv(flatness_path, sep="\t", index=False)
    within_species_variation.to_csv(variation_path, sep="\t", index=False)
    grouped_holdout_df.to_csv(grouped_holdout_path, sep="\t", index=False)
    repeated_context_df.to_csv(repeated_context_path, sep="\t", index=False)
    lineage_eval_summary_df.to_csv(lineage_eval_summary_path, sep="\t", index=False)
    lineage_clade_detail_df.to_csv(lineage_clade_detail_path, sep="\t", index=False)
    lineage_rank_df.to_csv(lineage_rank_path, sep="\t", index=False)
    lineage_calib_df.to_csv(lineage_calib_path, sep="\t", index=False)
    tuning_df.to_csv(tuning_path, sep="\t", index=False)
    ecology_audit_df.to_csv(lineage_ecology_audit_path, sep="\t", index=False)
    molecular_audit_df.to_csv(lineage_molecular_audit_path, sep="\t", index=False)
    lineage_gate_df.to_csv(lineage_gate_path, sep="\t", index=False)
    ld_stability_df.to_csv(lineage_ld_stability_path, sep="\t", index=False)
    within_species_clade_manifest.to_csv(within_species_manifest_path, sep="\t", index=False)
    within_species_eval_manifest.to_csv(within_species_eval_manifest_path, sep="\t", index=False)
    info_path.write_text(json.dumps(run_info, indent=2, sort_keys=True))
    md_path.write_text(
        build_interpretation_markdown(
            summary_df,
            feature_catalog,
            species_effect_summary,
            flatness_df,
            run_info,
            top_effects,
        )
    )
    append_iteration_log(args.iteration_log, summary_df, args.dataset, run_info)

    print(summary_df.to_string(index=False))
    print(json.dumps(run_info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
