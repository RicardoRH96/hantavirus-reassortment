#!/usr/bin/env python3
"""Run a biology-only repeated-stratified explanatory model audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_SCRIPT_DIR = REPO_ROOT / "scripts" / "figures"
for path in (SCRIPT_DIR, FIGURE_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_sequence_model_suite as bench
from plot_integrated_explanation_figure import (
    build_module_matrix,
    class_rank_shift,
    cluster_top_features,
    mean_feature_ranks,
)
from sequence_model_suite_explanatory_utils import FeatureSet, FEATURE_SETS, apply_ld_encoding, build_feature_frame, detect_feature_blocks


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
    / "biology_only_lineage_core"
)
DEFAULT_OUTPUT_DIR_BIOLOGY_WITH_UTR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "biology_with_utr"
)
WORKHORSE_SUMMARY = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "final_repeated_stratified_suite"
    / "summary.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_BIOLOGY_WITH_UTR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--repeated-splits", type=int, default=5)
    parser.add_argument("--repeated-repeats", type=int, default=3)
    parser.add_argument("--tree-bayes-iters", type=int, default=30)
    parser.add_argument("--tree-bayes-metric", choices=["average_precision", "roc_auc", "balanced_accuracy", "neg_brier"], default="average_precision")
    parser.add_argument("--ld-encoding", choices=["full", "interaction_only"], default="full")
    parser.add_argument("--disable-tuning", action="store_true")
    parser.add_argument(
        "--feature-mode",
        choices=["lineage_core", "biology_with_utr"],
        default="biology_with_utr",
    )
    parser.add_argument("--drop-utr-coverage-metrics", action="store_true")
    parser.add_argument("--top-features", type=int, default=28)
    parser.add_argument("--top-clusters", type=int, default=10)
    return parser.parse_args()


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    candidates = np.linspace(0.0, 1.0, 1001)
    best_score = -np.inf
    best_threshold = 0.5
    for threshold in candidates:
        score = bench.safe_balanced_accuracy(y_true, (y_prob >= threshold).astype(int))
        tie_break = -abs(float(threshold) - 0.5)
        if (score > best_score) or (np.isclose(score, best_score) and tie_break > -abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)
    return {"threshold": best_threshold, "balanced_accuracy": float(best_score)}


def family_for_feature(feature_name: str) -> str:
    if feature_name == "tree_discordance_score" or feature_name.startswith("discord_"):
        return "discordance"
    if feature_name.startswith("utrterm_"):
        return "utr_terminal"
    if feature_name.startswith("ld_site_"):
        return "ld"
    if feature_name.startswith("hosttrait_"):
        return "host_traits"
    if feature_name in {"overlap_opportunity_local", "arha_prev_local"}:
        return "local_ecology"
    return "other"


def resolve_feature_set(mode: str) -> FeatureSet:
    if mode == "lineage_core":
        return next(fs for fs in FEATURE_SETS if fs.name == "ecology_molecular_lineage_core")
    if mode == "biology_with_utr":
        return FeatureSet(
            name="ecology_molecular_biology_with_utr",
            description=(
                "Biology-only repeated-stratified explanatory model with local ecology, "
                "direct molecular features, and species-level molecular biology summaries "
                "(including UTR and SPR), excluding context and control."
            ),
            block_names=("ecology_core", "molecular_core", "molecular_species"),
        )
    raise ValueError(f"Unsupported feature mode: {mode}")


def is_utr_coverage_metric(column: str) -> bool:
    name = column.replace("num__", "").replace("cat__", "")
    if "species_feat_utr" not in name:
        return False
    suffixes = (
        "_manifest_n",
        "_retained_n",
        "_retention_rate",
        "_seqdist_matrix_n",
        "_structdist_matrix_n",
    )
    return any(name.endswith(suffix) for suffix in suffixes)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = bench.load_dataset(args.dataset, "true_label")
    y = df["target"].to_numpy(dtype=int)
    feature_set = resolve_feature_set(args.feature_mode)
    feature_blocks = apply_ld_encoding(detect_feature_blocks(df), args.ld_encoding)
    X, numeric, categorical = build_feature_frame(df, feature_set, feature_blocks)
    if args.drop_utr_coverage_metrics:
        numeric = [col for col in numeric if not is_utr_coverage_metric(col)]
        categorical = [col for col in categorical if not is_utr_coverage_metric(col)]
        X = X[numeric + categorical].copy()
    original_columns = list(numeric) + list(categorical)

    splitter = RepeatedStratifiedKFold(
        n_splits=args.repeated_splits,
        n_repeats=args.repeated_repeats,
        random_state=args.random_state,
    )
    splits = list(splitter.split(X, y))
    model_spec = next(spec for spec in bench.MODELS if spec.name == "lightgbm")

    tuned_params = None
    tuned_info = {"mode": "disabled"} if args.disable_tuning else None
    if not args.disable_tuning:
        tuned_params, tuned_info = bench.tune_tree_params_shared(
            model_spec,
            X,
            y,
            list(numeric),
            list(categorical),
            cv_scheme="repeated_stratified",
            splits=splits,
            seed=args.random_state,
            n_iter=args.tree_bayes_iters,
            metric=args.tree_bayes_metric,
        )

    pred_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]
        bundle = bench.fit_pipeline(
            model_spec,
            X_train,
            y_train,
            list(numeric),
            list(categorical),
            seed=args.random_state + fold_id,
            tuned_params=tuned_params,
            tuned_info=tuned_info,
        )
        prob = bundle.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_positives": int(y_train.sum()),
                "test_positives": int(y_test.sum()),
                "roc_auc": bench.safe_roc_auc(y_test, prob),
                "average_precision": bench.safe_average_precision(y_test, prob),
                "balanced_accuracy": bench.safe_balanced_accuracy(y_test, pred),
                "brier": float(np.mean((prob - y_test) ** 2)),
            }
        )
        importances = bench.aggregate_feature_importances(bundle, model_spec, original_columns)
        if not importances.empty:
            importances["fold_id"] = fold_id
            importance_rows.extend(importances.to_dict("records"))
        for pos, row_idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "fold_id": int(fold_id),
                    "row_index": int(row_idx),
                    "tip_name": str(df.iloc[row_idx]["tip_name"]),
                    "species": str(df.iloc[row_idx]["species"]),
                    "target": int(y[row_idx]),
                    "probability": float(prob[pos]),
                }
            )

    pred_df = pd.DataFrame(pred_rows).sort_values(["fold_id", "row_index"]).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).sort_values("fold_id").reset_index(drop=True)
    y_rep = pred_df["target"].to_numpy(dtype=int)
    p_rep = pred_df["probability"].to_numpy(dtype=float)
    rep_summary = {
        "roc_auc_mean": float(fold_df["roc_auc"].mean(skipna=True)),
        "average_precision_mean": float(fold_df["average_precision"].mean(skipna=True)),
        "balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean(skipna=True)),
        "brier_mean": float(fold_df["brier"].mean(skipna=True)),
        "oof_roc_auc": bench.safe_roc_auc(y_rep, p_rep),
        "oof_average_precision": bench.safe_average_precision(y_rep, p_rep),
        "oof_balanced_accuracy": bench.safe_balanced_accuracy(y_rep, (p_rep >= 0.5).astype(int)),
        "oof_brier": float(np.mean((p_rep - y_rep) ** 2)),
    }

    unique_df = pred_df.groupby("row_index", as_index=False).agg(
        target=("target", "first"),
        species=("species", "first"),
        tip_name=("tip_name", "first"),
        probability=("probability", "mean"),
    )
    y_unique = unique_df["target"].to_numpy(dtype=int)
    p_unique = unique_df["probability"].to_numpy(dtype=float)
    unique_threshold = choose_threshold(y_unique, p_unique)
    unique_summary = {
        "unique_oof_roc_auc": float(roc_auc_score(y_unique, p_unique)),
        "unique_oof_average_precision": float(average_precision_score(y_unique, p_unique)),
        "unique_oof_balanced_accuracy_default": float(balanced_accuracy_score(y_unique, (p_unique >= 0.5).astype(int))),
        "unique_oof_balanced_accuracy_opt": unique_threshold["balanced_accuracy"],
        "unique_oof_balanced_accuracy_opt_threshold": unique_threshold["threshold"],
        "unique_oof_brier": float(brier_score_loss(y_unique, p_unique)),
    }

    summary_row = {
        "feature_set": feature_set.name,
        "model": model_spec.name,
        "n_folds": int(len(splits)),
        "n_rows_repeated": int(len(pred_df)),
        "n_rows_unique": int(len(unique_df)),
        **rep_summary,
        **unique_summary,
    }
    summary_df = pd.DataFrame([summary_row])

    importance_df = pd.DataFrame(importance_rows)
    if not importance_df.empty:
        importance_df = (
            importance_df.groupby("feature_name", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importance_df["family"] = importance_df["feature_name"].map(family_for_feature)
        importance_df["rank"] = np.arange(1, len(importance_df) + 1)
        family_df = (
            importance_df.groupby("family", as_index=False)
            .agg(total_importance=("importance", "sum"), n_features=("feature_name", "size"))
            .sort_values("total_importance", ascending=False)
            .reset_index(drop=True)
        )
    else:
        family_df = pd.DataFrame(columns=["family", "total_importance", "n_features"])

    workhorse = pd.read_csv(WORKHORSE_SUMMARY, sep="\t").iloc[0]
    compare_df = pd.DataFrame(
        [
            {"metric": "roc_auc_mean", "workhorse": workhorse["roc_auc_mean"], "biology_only": summary_row["roc_auc_mean"]},
            {"metric": "average_precision_mean", "workhorse": workhorse["average_precision_mean"], "biology_only": summary_row["average_precision_mean"]},
            {"metric": "oof_roc_auc", "workhorse": workhorse["oof_roc_auc"], "biology_only": summary_row["oof_roc_auc"]},
            {"metric": "oof_average_precision", "workhorse": workhorse["oof_average_precision"], "biology_only": summary_row["oof_average_precision"]},
            {"metric": "oof_balanced_accuracy", "workhorse": workhorse["oof_balanced_accuracy"], "biology_only": summary_row["oof_balanced_accuracy"]},
            {"metric": "oof_brier", "workhorse": workhorse["oof_brier"], "biology_only": summary_row["oof_brier"]},
            {"metric": "unique_oof_roc_auc", "workhorse": np.nan, "biology_only": summary_row["unique_oof_roc_auc"]},
            {"metric": "unique_oof_average_precision", "workhorse": np.nan, "biology_only": summary_row["unique_oof_average_precision"]},
        ]
    )
    compare_df["delta_biology_minus_workhorse"] = compare_df["biology_only"] - compare_df["workhorse"]

    full_bundle = bench.fit_pipeline(
        model_spec,
        X,
        y,
        list(numeric),
        list(categorical),
        seed=args.random_state,
        tuned_params=tuned_params,
        tuned_info=tuned_info,
    )
    x_t = full_bundle.preprocessor.transform(X)
    full_model = full_bundle.model
    try:
        shap_matrix = np.asarray(full_model.predict(x_t, pred_contrib=True))
    except TypeError:
        shap_matrix = np.asarray(full_model.booster_.predict(x_t, pred_contrib=True))
    contributions = shap_matrix[:, :-1]
    bias = shap_matrix[:, -1]
    logits = contributions.sum(axis=1) + bias
    probabilities = full_bundle.predict_proba(X)[:, 1]
    feature_names = np.asarray(full_bundle.feature_names)

    contrib_df = pd.DataFrame(contributions, columns=feature_names)
    contrib_df.insert(0, "probability", probabilities)
    contrib_df.insert(0, "logit", logits)
    contrib_df.insert(0, "target", y)
    contrib_df.insert(0, "species", df["species"].to_numpy())
    contrib_df.insert(0, "tip_name", df["tip_name"].to_numpy())

    module_df = build_module_matrix(feature_names, contributions)
    module_out = module_df.copy()
    module_out.insert(0, "probability", probabilities)
    module_out.insert(0, "target", y)
    module_out.insert(0, "species", df["species"].to_numpy())
    module_out.insert(0, "tip_name", df["tip_name"].to_numpy())

    rank_df = mean_feature_ranks(feature_names, contributions)
    shift_df = class_rank_shift(feature_names, contributions, y)
    cluster_df = cluster_top_features(feature_names, contributions, args.top_features, args.top_clusters)

    summary_df.to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)
    pred_df.to_csv(args.output_dir / "repeated_stratified_cv_predictions.tsv", sep="\t", index=False)
    unique_df.to_csv(args.output_dir / "unique_row_oof_predictions.tsv", sep="\t", index=False)
    fold_df.to_csv(args.output_dir / "repeated_stratified_cv_fold_metrics.tsv", sep="\t", index=False)
    importance_df.to_csv(args.output_dir / "biology_feature_importance.tsv", sep="\t", index=False)
    family_df.to_csv(args.output_dir / "biology_family_importance.tsv", sep="\t", index=False)
    compare_df.to_csv(args.output_dir / "workhorse_comparison.tsv", sep="\t", index=False)
    contrib_df.to_csv(args.output_dir / "integrated_exact_feature_contributions.tsv", sep="\t", index=False)
    module_out.to_csv(args.output_dir / "integrated_module_contributions.tsv", sep="\t", index=False)
    rank_df.to_csv(args.output_dir / "integrated_feature_rank_summary.tsv", sep="\t", index=False)
    shift_df.to_csv(args.output_dir / "integrated_feature_rank_shift.tsv", sep="\t", index=False)
    cluster_df.to_csv(args.output_dir / "integrated_top_feature_clusters.tsv", sep="\t", index=False)
    (args.output_dir / "run_info.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "feature_set": feature_set.name,
                "model": model_spec.name,
                "drop_utr_coverage_metrics": bool(args.drop_utr_coverage_metrics),
                "random_state": args.random_state,
                "repeated_splits": args.repeated_splits,
                "repeated_repeats": args.repeated_repeats,
                "tree_bayes_iters": args.tree_bayes_iters,
                "tree_bayes_metric": args.tree_bayes_metric,
                "ld_encoding": args.ld_encoding,
                "top_features": args.top_features,
                "top_clusters": args.top_clusters,
                "tuned_params": tuned_params,
                "tuning_info": tuned_info,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(summary_df.to_string(index=False))
    print()
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
