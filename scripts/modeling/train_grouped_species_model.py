#!/usr/bin/env python3
"""Dedicated grouped-species final-model workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_sequence_model_suite as bench
from sequence_model_suite_explanatory_utils import (
    FeatureSet,
    apply_ld_encoding,
    build_feature_catalog,
    build_feature_frame,
    detect_feature_blocks,
)


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
    / "grouped_species_ecology_molecular_rf"
)

LOCAL_ECOLOGY_KEEP = {
    "has_arha_host_match",
    "overlap_opportunity_local",
    "local_latitude",
    "local_longitude",
    "arha_prev_local",
    "hosttrait_has_any_traits",
    "hosttrait_has_direct_any_traits",
}
LOCAL_CONTROL_KEEP = {
    "year",
    "arha_sequence_match_count",
}
REDUCED_NO_COORDS_ECOLOGY_KEEP = {
    "has_arha_host_match",
    "overlap_opportunity_local",
    "arha_prev_local",
    "hosttrait_has_any_traits",
    "hosttrait_has_direct_any_traits",
}
REDUCED_NO_YEAR_CONTROL_KEEP = {
    "arha_sequence_match_count",
}
MOLECULAR_SUMMARY_PREFIXES = (
    "species_feat_spr_",
    "species_feat_mean_spr_",
    "species_feat_max_spr_",
    "species_feat_min_spr_",
    "species_feat_std_spr_",
    "species_feat_utr5_",
    "species_feat_utr3_",
)
OVERLAP_SUMMARY_PREFIX = "species_feat_overlapopp_"
TREE_DISCORDANCE_FEATURE = "tree_discordance_score"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    feature_set: FeatureSet


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-source", choices=["true_label"], default="true_label")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tree-bayes-iters", type=int, default=30)
    parser.add_argument("--tree-bayes-inner-splits", type=int, default=4)
    parser.add_argument(
        "--tree-bayes-metric",
        choices=["average_precision", "roc_auc", "balanced_accuracy", "neg_brier"],
        default="average_precision",
    )
    parser.add_argument("--threshold-grid-points", type=int, default=181)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--disable-tuning",
        action="store_true",
        help="Use the base untuned random_forest_shallow model.",
    )
    parser.add_argument(
        "--exclude-species",
        action="append",
        default=[],
        help="Exclude one or more species labels before grouped-species CV.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Optional variant filter. Can be passed multiple times.",
    )
    return parser.parse_args(argv)


def _collect_reduced_feature_lists(
    out: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    ecology_keep: Sequence[str],
    control_keep: Sequence[str],
    keep_tree_discordance: bool,
) -> Tuple[List[str], List[str]]:
    reduced_numeric: List[str] = []
    reduced_categorical: List[str] = []

    ecology_keep_set = set(ecology_keep)
    control_keep_set = set(control_keep)

    for col in out["ecology"]["numeric"]:
        if col in ecology_keep_set or col.startswith(OVERLAP_SUMMARY_PREFIX):
            reduced_numeric.append(col)
    for col in out["ecology"]["categorical"]:
        if col in ecology_keep_set or col.startswith(OVERLAP_SUMMARY_PREFIX):
            reduced_categorical.append(col)

    for col in out["molecular"]["numeric"]:
        if any(col.startswith(prefix) for prefix in MOLECULAR_SUMMARY_PREFIXES):
            reduced_numeric.append(col)
        elif keep_tree_discordance and col == TREE_DISCORDANCE_FEATURE:
            reduced_numeric.append(col)
    for col in out["molecular"]["categorical"]:
        if any(col.startswith(prefix) for prefix in MOLECULAR_SUMMARY_PREFIXES):
            reduced_categorical.append(col)

    for col in out["control"]["numeric"]:
        if col in control_keep_set:
            reduced_numeric.append(col)
    for col in out["control"]["categorical"]:
        if col in control_keep_set:
            reduced_categorical.append(col)

    reduced_numeric = list(dict.fromkeys(reduced_numeric))
    reduced_categorical = [col for col in dict.fromkeys(reduced_categorical) if col not in reduced_numeric]
    return reduced_numeric, reduced_categorical


def build_reduced_blocks(df: pd.DataFrame, base_blocks: Mapping[str, Mapping[str, Sequence[str]]]) -> Dict[str, Dict[str, List[str]]]:
    out = {
        name: {
            "numeric": list(block["numeric"]),
            "categorical": list(block["categorical"]),
        }
        for name, block in base_blocks.items()
    }

    reduced_numeric, reduced_categorical = _collect_reduced_feature_lists(
        out,
        ecology_keep=LOCAL_ECOLOGY_KEEP,
        control_keep=LOCAL_CONTROL_KEEP,
        keep_tree_discordance=False,
    )
    out["grouped_species_reduced"] = {
        "numeric": reduced_numeric,
        "categorical": reduced_categorical,
    }

    noyear_keepdiscord_num, noyear_keepdiscord_cat = _collect_reduced_feature_lists(
        out,
        ecology_keep=REDUCED_NO_COORDS_ECOLOGY_KEEP,
        control_keep=REDUCED_NO_YEAR_CONTROL_KEEP,
        keep_tree_discordance=True,
    )
    out["grouped_species_reduced_noyear_nocoords_keepdiscord"] = {
        "numeric": noyear_keepdiscord_num,
        "categorical": noyear_keepdiscord_cat,
    }

    noyear_nodiscord_num, noyear_nodiscord_cat = _collect_reduced_feature_lists(
        out,
        ecology_keep=REDUCED_NO_COORDS_ECOLOGY_KEEP,
        control_keep=REDUCED_NO_YEAR_CONTROL_KEEP,
        keep_tree_discordance=False,
    )
    out["grouped_species_reduced_noyear_nocoords_nodiscord"] = {
        "numeric": noyear_nodiscord_num,
        "categorical": noyear_nodiscord_cat,
    }
    return out


def grouped_species_splits(df: pd.DataFrame, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = df["species"].fillna("missing_species").astype(str).to_numpy()
    return list(LeaveOneGroupOut().split(df, y, groups=groups))


def build_variant_specs() -> List[VariantSpec]:
    return [
        VariantSpec(
            name="ecology_molecular_full",
            description="Full ecology+molecular grouped-species model with tuned shallow RF.",
            feature_set=FeatureSet(
                name="ecology_molecular",
                description="Combined ecology, molecular, and control signals without species identity.",
                block_names=("ecology", "molecular", "control"),
            ),
        ),
        VariantSpec(
            name="ecology_molecular_reduced",
            description="Reduced grouped-species model keeping local ecology, overlap, SPR, and UTR summaries.",
            feature_set=FeatureSet(
                name="grouped_species_reduced",
                description="Reduced grouped-species feature set with local ecology plus overlap, SPR, and UTR summaries.",
                block_names=("grouped_species_reduced",),
            ),
        ),
        VariantSpec(
            name="ecology_molecular_reduced_noyear_nocoords_keepdiscord",
            description="Reduced grouped-species model without year/coordinates, keeping tree discordance.",
            feature_set=FeatureSet(
                name="grouped_species_reduced_noyear_nocoords_keepdiscord",
                description="Reduced grouped-species feature set without year/coordinates, with tree discordance kept.",
                block_names=("grouped_species_reduced_noyear_nocoords_keepdiscord",),
            ),
        ),
        VariantSpec(
            name="ecology_molecular_reduced_noyear_nocoords_nodiscord",
            description="Reduced grouped-species model without year/coordinates or tree discordance.",
            feature_set=FeatureSet(
                name="grouped_species_reduced_noyear_nocoords_nodiscord",
                description="Reduced grouped-species feature set without year/coordinates and without tree discordance.",
                block_names=("grouped_species_reduced_noyear_nocoords_nodiscord",),
            ),
        ),
    ]


def select_variants(requested: Optional[Sequence[str]]) -> List[VariantSpec]:
    variants = build_variant_specs()
    if not requested:
        return variants
    lookup = {variant.name: variant for variant in variants}
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise SystemExit(f"Unknown grouped-species variant(s): {', '.join(missing)}")
    return [lookup[name] for name in requested]


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid_points: int) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    lo = float(np.nanmin(y_prob))
    hi = float(np.nanmax(y_prob))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.5
    candidates = np.linspace(max(0.0, lo), min(1.0, hi), max(grid_points, 3))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        score = bench.safe_balanced_accuracy(y_true, (y_prob >= threshold).astype(int))
        tie_break = -abs(float(threshold) - 0.5)
        if (score > best_score) or (np.isclose(score, best_score) and tie_break > -abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def inner_species_threshold(
    spec: bench.ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric: List[str],
    categorical: List[str],
    species_train: pd.Series,
    tuned_params: Mapping[str, Any],
    seed: int,
    grid_points: int,
) -> Tuple[float, str]:
    groups = species_train.astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3 or len(np.unique(y_train)) < 2:
        return 0.5, "fallback_insufficient_groups"
    n_splits = min(4, len(unique_groups))
    if n_splits < 2:
        return 0.5, "fallback_insufficient_groups"
    splitter = GroupKFold(n_splits=n_splits)
    oof_prob = np.full(len(y_train), np.nan, dtype=float)
    for fold_id, (inner_tr, inner_te) in enumerate(splitter.split(X_train, y_train, groups=groups), start=1):
        if len(np.unique(y_train[inner_tr])) < 2:
            continue
        bundle = bench.fit_pipeline(
            spec,
            X_train.iloc[inner_tr],
            y_train[inner_tr],
            numeric,
            categorical,
            seed=seed + fold_id,
            tuned_params=tuned_params,
        )
        prob = bundle.predict_proba(X_train.iloc[inner_te])[:, 1]
        oof_prob[inner_te] = prob
    mask = np.isfinite(oof_prob)
    if mask.sum() < 3 or len(np.unique(y_train[mask])) < 2:
        return 0.5, "fallback_no_inner_oof"
    return choose_threshold(y_train[mask], oof_prob[mask], grid_points), "inner_groupkfold"


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int,
    species: Optional[pd.Series] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def _bin_frame(y_local: np.ndarray, prob_local: np.ndarray, label: str) -> None:
        if len(prob_local) == 0:
            return
        frame = pd.DataFrame({"y": y_local, "p": prob_local}).sort_values("p").reset_index(drop=True)
        frame["bin"] = pd.qcut(frame.index, q=min(n_bins, len(frame)), labels=False, duplicates="drop")
        grouped = frame.groupby("bin", dropna=False)
        for bin_id, group in grouped:
            rows.append(
                {
                    "scope": label,
                    "bin": int(bin_id),
                    "n_rows": int(len(group)),
                    "mean_predicted": float(group["p"].mean()),
                    "observed_rate": float(group["y"].mean()),
                    "abs_gap": float(abs(group["p"].mean() - group["y"].mean())),
                }
            )

    _bin_frame(y_true, y_prob, "overall")
    if species is not None:
        for species_label, idx in species.groupby(species).groups.items():
            arr = np.asarray(list(idx), dtype=int)
            _bin_frame(y_true[arr], y_prob[arr], str(species_label))
    return pd.DataFrame(rows)


def summarize_species_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    grouped = pred_df.groupby("species", dropna=False)
    rows: List[Dict[str, object]] = []
    for species, group in grouped:
        y_true = group["target"].to_numpy(dtype=int)
        y_prob = group["probability"].to_numpy(dtype=float)
        y_pred_default = (y_prob >= 0.5).astype(int)
        y_pred_threshold = (y_prob >= group["threshold_inner"].iloc[0]).astype(int)
        rows.append(
            {
                "species": str(species),
                "n_rows": int(group["row_index"].nunique()),
                "positives": int(y_true.sum()),
                "mean_probability": float(y_prob.mean()),
                "std_probability": float(group["probability"].std(ddof=0)),
                "roc_auc": bench.safe_roc_auc(y_true, y_prob),
                "average_precision": bench.safe_average_precision(y_true, y_prob),
                "balanced_accuracy_default": bench.safe_balanced_accuracy(y_true, y_pred_default),
                "balanced_accuracy_inner_threshold": bench.safe_balanced_accuracy(y_true, y_pred_threshold),
                "brier": float(np.mean((y_prob - y_true) ** 2)),
            }
        )
    return pd.DataFrame(rows).sort_values("species").reset_index(drop=True)


def compute_summary(pred_df: pd.DataFrame, fold_df: pd.DataFrame) -> Dict[str, object]:
    y_true = pred_df["target"].to_numpy(dtype=int)
    y_prob = pred_df["probability"].to_numpy(dtype=float)
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_threshold = (y_prob >= pred_df["threshold_inner"].to_numpy(dtype=float)).astype(int)
    return {
        "n_folds": int(fold_df["fold_id"].nunique()),
        "roc_auc_mean": float(fold_df["roc_auc"].mean(skipna=True)),
        "roc_auc_std": float(fold_df["roc_auc"].std(ddof=0, skipna=True)),
        "average_precision_mean": float(fold_df["average_precision"].mean(skipna=True)),
        "average_precision_std": float(fold_df["average_precision"].std(ddof=0, skipna=True)),
        "balanced_accuracy_mean_default": float(fold_df["balanced_accuracy_default"].mean(skipna=True)),
        "balanced_accuracy_std_default": float(fold_df["balanced_accuracy_default"].std(ddof=0, skipna=True)),
        "balanced_accuracy_mean_inner_threshold": float(fold_df["balanced_accuracy_inner_threshold"].mean(skipna=True)),
        "balanced_accuracy_std_inner_threshold": float(fold_df["balanced_accuracy_inner_threshold"].std(ddof=0, skipna=True)),
        "brier_mean": float(fold_df["brier"].mean(skipna=True)),
        "brier_std": float(fold_df["brier"].std(ddof=0, skipna=True)),
        "n_rows": int(pred_df["row_index"].nunique()),
        "oof_roc_auc": bench.safe_roc_auc(y_true, y_prob),
        "oof_average_precision": bench.safe_average_precision(y_true, y_prob),
        "oof_balanced_accuracy_default": bench.safe_balanced_accuracy(y_true, y_pred_default),
        "oof_balanced_accuracy_inner_threshold": bench.safe_balanced_accuracy(y_true, y_pred_threshold),
        "oof_brier": float(np.mean((y_prob - y_true) ** 2)),
        "global_oof_opt_threshold": choose_threshold(y_true, y_prob, 181),
    }


def run_variant(
    df: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    species: pd.Series,
    variant: VariantSpec,
    blocks: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    seed: int,
    bayes_iters: int,
    bayes_inner_splits: int,
    bayes_metric: str,
    threshold_grid_points: int,
    disable_tuning: bool,
) -> Dict[str, object]:
    spec = next(model for model in bench.MODELS if model.name == "random_forest_shallow")
    X, numeric, categorical = build_feature_frame(df, variant.feature_set, blocks)
    pred_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]
        train_species = species.iloc[train_idx].astype(str)
        unique_species = np.unique(train_species.to_numpy())
        n_inner = min(bayes_inner_splits, len(unique_species))
        inner_splitter = GroupKFold(n_splits=n_inner)
        inner_splits = list(inner_splitter.split(X_train, y_train, groups=train_species.to_numpy()))
        if disable_tuning:
            tuned_params = {}
            tuned_info = {
                "tuning_status": "disabled",
                "tuning_method": "disabled",
                "tuning_metric": bayes_metric,
                "tuning_inner_cv": "grouped_species",
                "tuning_n_iter": 0,
                "tuning_n_inner_splits": int(len(inner_splits)),
                "tuning_best_score": np.nan,
                "tuning_best_params_json": "",
            }
        else:
            tuned_params, tuned_info = bench.tune_tree_params_shared(
                spec,
                X_train,
                y_train,
                numeric,
                categorical,
                cv_scheme="grouped_species",
                splits=inner_splits,
                seed=seed + fold_id,
                n_iter=bayes_iters,
                metric=bayes_metric,
            )
        bundle = bench.fit_pipeline(
            spec,
            X_train,
            y_train,
            numeric,
            categorical,
            seed=seed + 1000 + fold_id,
            tuned_params=tuned_params,
            tuned_info=tuned_info,
        )
        threshold_inner, threshold_status = inner_species_threshold(
            spec,
            X_train,
            y_train,
            numeric,
            categorical,
            train_species,
            tuned_params=tuned_params or {},
            seed=seed + 2000 + fold_id,
            grid_points=threshold_grid_points,
        )
        prob = bundle.predict_proba(X_test)[:, 1]
        pred_default = (prob >= 0.5).astype(int)
        pred_threshold = (prob >= threshold_inner).astype(int)
        heldout_species = species.iloc[test_idx].astype(str)
        fold_rows.append(
            {
                "variant": variant.name,
                "fold_id": int(fold_id),
                "heldout_group": "|".join(sorted(set(heldout_species.tolist()))),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_positives": int(y_train.sum()),
                "test_positives": int(y_test.sum()),
                "roc_auc": bench.safe_roc_auc(y_test, prob),
                "average_precision": bench.safe_average_precision(y_test, prob),
                "balanced_accuracy_default": bench.safe_balanced_accuracy(y_test, pred_default),
                "balanced_accuracy_inner_threshold": bench.safe_balanced_accuracy(y_test, pred_threshold),
                "brier": float(np.mean((prob - y_test) ** 2)),
                "threshold_inner": float(threshold_inner),
                "threshold_status": threshold_status,
            }
        )
        info = bench.aggregate_feature_importances(bundle, spec, list(numeric) + list(categorical))
        if not info.empty:
            info = info.assign(variant=variant.name, fold_id=fold_id)
            importance_rows.extend(info.to_dict("records"))
        tuning_rows.append(
            {
                "variant": variant.name,
                "fold_id": int(fold_id),
                **(tuned_info or {}),
            }
        )
        for pos, row_idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "variant": variant.name,
                    "fold_id": int(fold_id),
                    "row_index": int(row_idx),
                    "species": str(species.iloc[row_idx]),
                    "target": int(y[row_idx]),
                    "probability": float(prob[pos]),
                    "pred_default": int(pred_default[pos]),
                    "pred_inner_threshold": int(pred_threshold[pos]),
                    "threshold_inner": float(threshold_inner),
                    "threshold_status": threshold_status,
                }
            )

    pred_df = pd.DataFrame(pred_rows).sort_values(["variant", "fold_id", "row_index"]).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).sort_values(["variant", "fold_id"]).reset_index(drop=True)
    importance_df = pd.DataFrame(importance_rows)
    if not importance_df.empty:
        importance_df = (
            importance_df.groupby(["variant", "feature_name"], as_index=False)["importance"]
            .mean()
            .sort_values(["variant", "importance"], ascending=[True, False])
            .reset_index(drop=True)
        )
    tuning_df = pd.DataFrame(tuning_rows).sort_values(["variant", "fold_id"]).reset_index(drop=True)
    species_diag_df = summarize_species_predictions(pred_df)
    calibration_df = calibration_bins(
        pred_df["target"].to_numpy(dtype=int),
        pred_df["probability"].to_numpy(dtype=float),
        n_bins=10,
        species=pred_df["species"],
    )
    threshold_rows: List[Dict[str, object]] = []
    y_true = pred_df["target"].to_numpy(dtype=int)
    y_prob = pred_df["probability"].to_numpy(dtype=float)
    global_opt = choose_threshold(y_true, y_prob, threshold_grid_points)
    threshold_rows.append(
        {
            "variant": variant.name,
            "threshold_type": "default_0.5",
            "threshold": 0.5,
            "oof_balanced_accuracy": bench.safe_balanced_accuracy(y_true, (y_prob >= 0.5).astype(int)),
        }
    )
    threshold_rows.append(
        {
            "variant": variant.name,
            "threshold_type": "inner_species_cv",
            "threshold": float(pred_df["threshold_inner"].mean()),
            "oof_balanced_accuracy": bench.safe_balanced_accuracy(
                y_true, pred_df["pred_inner_threshold"].to_numpy(dtype=int)
            ),
        }
    )
    threshold_rows.append(
        {
            "variant": variant.name,
            "threshold_type": "global_oof_opt",
            "threshold": float(global_opt),
            "oof_balanced_accuracy": bench.safe_balanced_accuracy(y_true, (y_prob >= global_opt).astype(int)),
        }
    )
    summary = compute_summary(pred_df, fold_df)
    summary.update(
        {
            "variant": variant.name,
            "description": variant.description,
            "n_numeric": int(len(numeric)),
            "n_categorical": int(len(categorical)),
            "numeric_features": "|".join(numeric),
            "categorical_features": "|".join(categorical),
        }
    )

    if disable_tuning:
        final_tuned_params = {}
        final_tuning_info = {
            "tuning_status": "disabled",
            "tuning_method": "disabled",
            "tuning_metric": bayes_metric,
            "tuning_inner_cv": "grouped_species",
            "tuning_n_iter": 0,
            "tuning_n_inner_splits": int(len(splits)),
            "tuning_best_score": np.nan,
            "tuning_best_params_json": "",
        }
    else:
        final_tuned_params, final_tuning_info = bench.tune_tree_params_shared(
            spec,
            X,
            y,
            numeric,
            categorical,
            cv_scheme="grouped_species",
            splits=splits,
            seed=seed + 5000,
            n_iter=bayes_iters,
            metric=bayes_metric,
        )
    final_bundle = bench.fit_pipeline(
        spec,
        X,
        y,
        numeric,
        categorical,
        seed=seed + 6000,
        tuned_params=final_tuned_params,
        tuned_info=final_tuning_info,
    )

    return {
        "summary": summary,
        "predictions": pred_df,
        "fold_metrics": fold_df,
        "feature_importance": importance_df,
        "tuning": tuning_df,
        "species_diagnostics": species_diag_df.assign(variant=variant.name),
        "calibration": calibration_df.assign(variant=variant.name),
        "threshold_diagnostics": pd.DataFrame(threshold_rows),
        "final_bundle": final_bundle,
        "final_tuned_params": final_tuned_params or {},
        "final_tuning_info": final_tuning_info,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = bench.load_dataset(args.dataset, args.target_source)
    if args.exclude_species:
        exclude = set(args.exclude_species)
        df = df.loc[~df["species"].fillna("missing_species").astype(str).isin(exclude)].reset_index(drop=True)
        if df.empty:
            raise SystemExit("No rows remain after --exclude-species filtering.")
    y = df["target"].to_numpy(dtype=int)
    species = df["species"].fillna("missing_species").astype(str)
    splits = grouped_species_splits(df, y)

    blocks = apply_ld_encoding(detect_feature_blocks(df), "full")
    blocks = build_reduced_blocks(df, blocks)
    variants = select_variants(args.variants)
    feature_catalog = build_feature_catalog([variant.feature_set for variant in variants], blocks)

    summary_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []
    fold_frames: List[pd.DataFrame] = []
    importance_frames: List[pd.DataFrame] = []
    tuning_frames: List[pd.DataFrame] = []
    species_frames: List[pd.DataFrame] = []
    calibration_frames: List[pd.DataFrame] = []
    threshold_frames: List[pd.DataFrame] = []
    fitted_model_index: List[Dict[str, object]] = []

    for variant_num, variant in enumerate(variants, start=1):
        result = run_variant(
            df,
            y,
            splits,
            species,
            variant,
            blocks,
            seed=args.random_state + variant_num * 100,
            bayes_iters=args.tree_bayes_iters,
            bayes_inner_splits=args.tree_bayes_inner_splits,
            bayes_metric=args.tree_bayes_metric,
            threshold_grid_points=args.threshold_grid_points,
            disable_tuning=bool(args.disable_tuning),
        )
        summary_rows.append(result["summary"])
        pred_frames.append(result["predictions"])
        fold_frames.append(result["fold_metrics"])
        if not result["feature_importance"].empty:
            importance_frames.append(result["feature_importance"])
        tuning_frames.append(result["tuning"])
        species_frames.append(result["species_diagnostics"])
        calibration_frames.append(result["calibration"])
        threshold_frames.append(result["threshold_diagnostics"])
        model_path = args.output_dir / f"{variant.name}_trained_model.joblib"
        joblib.dump(
            {
                "variant": variant.name,
                "bundle": result["final_bundle"],
                "tuned_params": result["final_tuned_params"],
                "tuning_info": result["final_tuning_info"],
                "dataset": str(args.dataset),
            },
            model_path,
        )
        fitted_model_index.append(
            {
                "variant": variant.name,
                "model_path": str(model_path),
                "tuned_params_json": json.dumps(result["final_tuned_params"], sort_keys=True),
                "tuning_status": (result["final_tuning_info"] or {}).get("tuning_status", "unknown"),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("average_precision_mean", ascending=False).reset_index(drop=True)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    fold_df = pd.concat(fold_frames, ignore_index=True)
    tuning_df = pd.concat(tuning_frames, ignore_index=True)
    species_df = pd.concat(species_frames, ignore_index=True)
    calibration_df = pd.concat(calibration_frames, ignore_index=True)
    threshold_df = pd.concat(threshold_frames, ignore_index=True)
    importance_df = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame(columns=["variant", "feature_name", "importance"])
    fitted_model_df = pd.DataFrame(fitted_model_index).sort_values("variant").reset_index(drop=True)

    summary_df.to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)
    pred_df.to_csv(args.output_dir / "grouped_species_oof_predictions.tsv", sep="\t", index=False)
    fold_df.to_csv(args.output_dir / "grouped_species_fold_metrics.tsv", sep="\t", index=False)
    tuning_df.to_csv(args.output_dir / "tree_model_tuning.tsv", sep="\t", index=False)
    species_df.to_csv(args.output_dir / "species_diagnostics.tsv", sep="\t", index=False)
    calibration_df.to_csv(args.output_dir / "calibration_bins.tsv", sep="\t", index=False)
    threshold_df.to_csv(args.output_dir / "threshold_diagnostics.tsv", sep="\t", index=False)
    feature_catalog.to_csv(args.output_dir / "feature_catalog.tsv", sep="\t", index=False)
    importance_df.to_csv(args.output_dir / "feature_importance.tsv", sep="\t", index=False)
    fitted_model_df.to_csv(args.output_dir / "trained_model_index.tsv", sep="\t", index=False)

    run_info = {
        "dataset": str(args.dataset),
        "target_source_filter": args.target_source,
        "labeled_rows": int(len(df)),
        "positive_rows": int(y.sum()),
        "negative_rows": int((1 - y).sum()),
        "species_counts": species.value_counts().to_dict(),
        "outer_cv_scheme": "grouped_species",
        "outer_folds": int(len(splits)),
        "model": "random_forest_shallow",
        "variants": [variant.name for variant in variants],
        "tree_bayes_iters": int(args.tree_bayes_iters),
        "tree_bayes_inner_splits": int(args.tree_bayes_inner_splits),
        "tree_bayes_metric": str(args.tree_bayes_metric),
        "disable_tuning": bool(args.disable_tuning),
        "excluded_species": list(args.exclude_species),
        "threshold_grid_points": int(args.threshold_grid_points),
        "calibration_bins": int(args.calibration_bins),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n")

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
