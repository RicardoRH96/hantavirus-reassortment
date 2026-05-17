#!/usr/bin/env python3
"""SHAP-style explanation figure for the integrated explanatory model.

Supports two explanation backends:
- exact additive logit-scale contributions for the integrated L2-logistic model
- exact TreeSHAP contributions from tuned LightGBM via ``pred_contrib=True``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELING_SCRIPT_DIR = REPO_ROOT / "scripts" / "modeling"
if str(MODELING_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_SCRIPT_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist

from benchmark_sequence_model_suite import ModelSpec, fit_pipeline, infer_feature_group_label
from sequence_model_suite_explanatory_utils import FEATURE_SETS, build_feature_frame, detect_feature_blocks


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
    / "final_hierarchical_block_comparison"
    / "figures"
    / "manuscript_utr3_integrated"
)
DEFAULT_TUNED_SUITE_DIR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "final_repeated_stratified_suite"
)

SPECIES_COLORS = {
    "orthohantavirus_andesense": "#1b9e77",
    "orthohantavirus_dobravaense": "#d95f02",
    "orthohantavirus_hantanense": "#7570b3",
    "orthohantavirus_puumalaense": "#66a61e",
    "orthohantavirus_seoulense": "#e7298a",
    "orthohantavirus_sinnombreense": "#a6761d",
    "orthohantavirus_tulaense": "#1f78b4",
}

FAMILY_COLORS = {
    "Species identity": "#6c8ebf",
    "Host life history": "#4daf8b",
    "Host phylogenetic diversity": "#88c999",
    "Overlap opportunity": "#e3b34c",
    "Local host prevalence": "#d9a441",
    "LD / coadaptation": "#d95f5f",
    "UTR structure": "#9b6fb6",
    "Tree incongruence": "#708090",
    "Geographic / sampling context": "#7fa4b7",
    "Controls / sampling": "#b7b7b7",
}

MODULE_ORDER = [
    "Species identity",
    "Host life history",
    "Host phylogenetic diversity",
    "Overlap opportunity",
    "Local host prevalence",
    "Geographic / sampling context",
    "LD / coadaptation",
    "UTR structure",
    "Tree incongruence",
    "Controls / sampling",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tuned-suite-dir", type=Path, default=DEFAULT_TUNED_SUITE_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-source", choices=["true_label"], default="true_label")
    parser.add_argument("--top-features", type=int, default=28)
    parser.add_argument("--top-clusters", type=int, default=20)
    parser.add_argument(
        "--explanation-model",
        choices=["lightgbm_tree_shap", "penalized_logistic_l2"],
        default="lightgbm_tree_shap",
    )
    parser.add_argument("--tuning-cv-scheme", choices=["repeated_stratified", "grouped_within_species_clade", "grouped_species"], default="repeated_stratified")
    return parser.parse_args(argv)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.03, label.lower(), transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def species_short(name: str) -> str:
    return name.replace("orthohantavirus_", "").replace("ense", ".")


def load_labeled_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["is_reassortant_final"].notna()].copy()
    df = df[df["target_source"].isin(["sequence_true_label", "isolate_true_label"])].copy()
    df["target"] = pd.to_numeric(df["is_reassortant_final"], errors="coerce").astype(int)
    return df.reset_index(drop=True)


def feature_family(feature_name: str) -> str:
    name = feature_name.replace("num__", "").replace("cat__", "")
    group = infer_feature_group_label(feature_name)
    if name == "species_feat_hostdiv_n_labels_missing":
        return "Controls / sampling"
    if group == "species":
        return "Species identity"
    if name.startswith("hosttrait_combine_"):
        return "Host life history"
    if name.startswith("species_feat_hostdiv_"):
        return "Host phylogenetic diversity"
    if (
        group in {"species_overlapopp", "direct_overlap"}
        or name.startswith("species_feat_overlapopp_")
        or "overlap_opportunity_local" in name
    ):
        return "Overlap opportunity"
    if "arha_prev_local" in name:
        return "Local host prevalence"
    if (
        "local_latitude" in name
        or "local_longitude" in name
        or "arha_host_latitude" in name
        or "arha_host_longitude" in name
        or "arha_host_country" in name
        or name == "year"
    ):
        return "Geographic / sampling context"
    if group in {"species_ld_summary", "ld_site"}:
        return "LD / coadaptation"
    if group in {"species_utr5", "species_utr3"}:
        return "UTR structure"
    if group == "species_spr" or "tree_discordance_score" in name:
        return "Tree incongruence"
    return "Controls / sampling"


def feature_module(feature_name: str) -> str:
    name = feature_name.replace("num__", "").replace("cat__", "")
    group = infer_feature_group_label(feature_name)
    if name == "species_feat_hostdiv_n_labels_missing":
        return "Controls / sampling"
    if group == "species" or name.startswith("species_orthohantavirus_"):
        return "Species identity"
    if (
        "hosttrait_combine_gestation_length_d" in name
        or "hosttrait_combine_litter_size_n" in name
        or "hosttrait_combine_dispersal_km" in name
        or "hosttrait_combine_adult_mass_g" in name
        or "hosttrait_combine_adult_body_length_mm" in name
    ):
        return "Host life history"
    if "species_feat_hostdiv_faith_pd" in name:
        return "Host phylogenetic diversity"
    if (
        "species_feat_overlapopp_overlap_mean" in name
        or "overlap_opportunity_local" in name
        or "arha_host_sampling_overlap" in name
        or group in {"species_overlapopp", "direct_overlap"}
    ):
        return "Overlap opportunity"
    if "arha_prev_local" in name:
        return "Local host prevalence"
    if (
        "local_latitude" in name
        or "local_longitude" in name
        or
        "arha_host_latitude" in name
        or "arha_host_longitude" in name
        or "arha_host_country" in name
        or name == "year"
    ):
        return "Geographic / sampling context"
    if (
        "species_feat_mean_cramers" in name
        or "species_feat_median_cramers" in name
        or group == "ld_site"
    ):
        return "LD burden"
    if "species_feat_prop_significant_overall" in name:
        return "Significant LD fraction"
    if "species_feat_prop_significant_" in name:
        return "Significant LD fraction"
    if "species_feat_utr5_S_structdist_mean" in name or name.startswith("species_feat_utr5_S_"):
        return "UTR5 S structure"
    if "species_feat_utr5_M_structdist_mean" in name or name.startswith("species_feat_utr5_M_"):
        return "UTR5 M structure"
    if "species_feat_utr5_L_structdist_mean" in name or name.startswith("species_feat_utr5_L_"):
        return "UTR5 L structure"
    if "species_feat_utr3_S_structdist_mean" in name or name.startswith("species_feat_utr3_S_"):
        return "UTR3 S structure"
    if "species_feat_utr3_M_structdist_mean" in name or name.startswith("species_feat_utr3_M_"):
        return "UTR3 M structure"
    if "species_feat_utr3_L_structdist_mean" in name or name.startswith("species_feat_utr3_L_"):
        return "UTR3 L structure"
    if "tree_discordance_score" in name or group == "species_spr":
        return "Tree discordance"
    return "Controls / sampling"


def humanize_feature(feature_name: str) -> str:
    name = feature_name.replace("num__", "").replace("cat__", "")
    name = name.replace("_z", "")
    replacements = {
        "hosttrait_combine_gestation_length_d": "Host gestation length",
        "hosttrait_combine_litter_size_n": "Host litter size",
        "hosttrait_combine_dispersal_km": "Host dispersal",
        "hosttrait_combine_adult_mass_g": "Host body mass",
        "hosttrait_combine_adult_body_length_mm": "Host adult body length",
        "species_feat_hostdiv_faith_pd": "Host phylogenetic diversity",
        "species_feat_hostdiv_n_labels_missing": "Host-diversity labels unresolved",
        "species_feat_overlapopp_overlap_mean": "Overlap opportunity",
        "overlap_opportunity_local": "Overlap opportunity",
        "arha_prev_local": "Local host prevalence",
        "local_latitude": "local latitude",
        "local_longitude": "local longitude",
        "species_feat_eco_year_span": "Species sampling-year span",
        "species_feat_eco_n_records": "Species ecology record count",
        "species_feat_matched_isolates": "Species matched isolate count",
        "arha_sequence_match_count": "ArHa sequence match count",
        "species_feat_mean_cramers_v_overall": "Mean LD burden",
        "species_feat_prop_significant_overall": "Fraction significant LD pairs",
        "species_feat_utr5_S_structdist_mean": "UTR5 S structural distance",
        "species_feat_utr5_M_structdist_mean": "UTR5 M structural distance",
        "species_feat_utr5_L_structdist_mean": "UTR5 L structural distance",
        "species_feat_utr3_S_structdist_mean": "UTR3 S structural distance",
        "species_feat_utr3_M_structdist_mean": "UTR3 M structural distance",
        "species_feat_utr3_L_structdist_mean": "UTR3 L structural distance",
        "species_feat_utr5_S_structdist_std": "UTR5 S structural distance SD",
        "species_feat_utr5_M_structdist_std": "UTR5 M structural distance SD",
        "species_feat_utr5_L_structdist_std": "UTR5 L structural distance SD",
        "species_feat_utr3_S_structdist_std": "UTR3 S structural distance SD",
        "species_feat_utr3_M_structdist_std": "UTR3 M structural distance SD",
        "species_feat_utr3_L_structdist_std": "UTR3 L structural distance SD",
        "species_feat_utr5_S_retention_rate": "UTR5 S retention rate",
        "species_feat_utr5_M_retention_rate": "UTR5 M retention rate",
        "species_feat_utr5_L_retention_rate": "UTR5 L retention rate",
        "species_feat_utr3_S_retention_rate": "UTR3 S retention rate",
        "species_feat_utr3_M_retention_rate": "UTR3 M retention rate",
        "species_feat_utr3_L_retention_rate": "UTR3 L retention rate",
        "tree_discordance_score": "Tree discordance score",
        "arha_host_country": "Sampling country",
        "arha_host_latitude": "Sampling latitude",
        "arha_host_longitude": "Sampling longitude",
        "arha_host_sampling_overlap": "Sampling overlap opportunity",
        "segment_completeness": "Segment completeness",
        "year": "Sampling year",
    }
    for key, label in replacements.items():
        if name.startswith(key):
            suffix = name[len(key):]
            if suffix and not suffix.startswith("_missingindicator"):
                suffix = suffix.strip("_")
                if suffix:
                    return f"{label}: {suffix.replace('_', ' ')}"
            return label
    if name.startswith("species_orthohantavirus_"):
        return f"Species: {species_short(name.replace('species_', ''))}"
    if "missingindicator_" in name:
        base = name.split("missingindicator_", 1)[1]
        return f"Missingness: {humanize_feature(base)}"
    return name.replace("_", " ")


def row_order(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 2:
        return np.arange(matrix.shape[0])
    dist = pdist(matrix, metric="euclidean")
    if np.allclose(dist, 0):
        return np.arange(matrix.shape[0])
    return leaves_list(linkage(dist, method="average"))


def feature_order(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] <= 2:
        return np.arange(matrix.shape[1])
    dist = pdist(matrix.T, metric="correlation")
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    if np.allclose(dist, 0):
        return np.arange(matrix.shape[1])
    return leaves_list(linkage(dist, method="average"))


def load_tuned_params(
    suite_dir: Path,
    *,
    cv_scheme: str,
    feature_set: str,
    model: str,
) -> Dict[str, object]:
    tune_df = pd.read_csv(suite_dir / "tree_model_tuning.tsv", sep="\t")
    row = tune_df[
        (tune_df["fit_scope"] == "shared_cv_tune")
        & (tune_df["cv_scheme"] == cv_scheme)
        & (tune_df["feature_set"] == feature_set)
        & (tune_df["model"] == model)
    ]
    if row.empty:
        raise ValueError(
            f"No tuned params found for cv_scheme={cv_scheme}, feature_set={feature_set}, model={model}"
        )
    return json.loads(row.iloc[0]["tuning_best_params_json"])


def fit_integrated_model(df: pd.DataFrame, seed: int):
    feature_set = next(fs for fs in FEATURE_SETS if fs.name == "integrated_taxon_mechanism")
    blocks = detect_feature_blocks(df)
    X, numeric, categorical = build_feature_frame(df, feature_set, blocks)
    y = df["target"].to_numpy(dtype=int)
    model_spec = ModelSpec(name="penalized_logistic_l2", kind="logistic_l2")
    bundle = fit_pipeline(model_spec, X, y, numeric, categorical, seed)
    x_t = bundle.preprocessor.transform(X)
    coef = np.asarray(bundle.model.coef_).reshape(-1)
    intercept = float(np.asarray(bundle.model.intercept_).reshape(-1)[0])
    contributions = x_t * coef[np.newaxis, :]
    logits = intercept + contributions.sum(axis=1)
    probabilities = sigmoid(logits)
    feature_names = np.asarray(bundle.feature_names)
    return X, y, feature_names, contributions, logits, probabilities, intercept


def fit_integrated_lightgbm_tree_shap(
    df: pd.DataFrame,
    seed: int,
    tuned_suite_dir: Path,
    tuning_cv_scheme: str,
):
    feature_set = next(fs for fs in FEATURE_SETS if fs.name == "integrated_taxon_mechanism")
    blocks = detect_feature_blocks(df)
    X, numeric, categorical = build_feature_frame(df, feature_set, blocks)
    y = df["target"].to_numpy(dtype=int)
    model_spec = ModelSpec(name="lightgbm", kind="lightgbm")
    tuned_params = load_tuned_params(
        tuned_suite_dir,
        cv_scheme=tuning_cv_scheme,
        feature_set="integrated_taxon_mechanism",
        model="lightgbm",
    )
    bundle = fit_pipeline(
        model_spec,
        X,
        y,
        numeric,
        categorical,
        seed,
        tuned_params=tuned_params,
        tuned_info={
            "tuning_status": "loaded_shared_cv_tune",
            "tuning_source_cv_scheme": tuning_cv_scheme,
            "tuning_best_params_json": json.dumps(tuned_params, sort_keys=True),
        },
    )
    x_t = bundle.preprocessor.transform(X)
    model = bundle.model
    try:
        shap_matrix = np.asarray(model.predict(x_t, pred_contrib=True))
    except TypeError:
        shap_matrix = np.asarray(model.booster_.predict(x_t, pred_contrib=True))
    if shap_matrix.shape[1] != (len(bundle.feature_names) + 1):
        raise ValueError(
            f"Unexpected LightGBM pred_contrib shape {shap_matrix.shape}; expected n x ({len(bundle.feature_names)} + 1)"
        )
    contributions = shap_matrix[:, :-1]
    bias = shap_matrix[:, -1]
    logits = contributions.sum(axis=1) + bias
    probabilities = bundle.predict_proba(X)[:, 1]
    intercept = float(np.mean(bias))
    feature_names = np.asarray(bundle.feature_names)
    return X, y, feature_names, contributions, logits, probabilities, intercept, tuned_params


def build_module_matrix(feature_names: np.ndarray, contributions: np.ndarray) -> pd.DataFrame:
    modules = pd.Series([feature_module(name) for name in feature_names], index=feature_names)
    cols = []
    data = {}
    for module in dict.fromkeys(modules.tolist()):
        idx = np.where(modules.to_numpy() == module)[0]
        data[module] = contributions[:, idx].sum(axis=1)
        cols.append(module)
    frame = pd.DataFrame(data)
    preferred = [
        "Species identity",
        "Host life history",
        "Host phylogenetic diversity",
        "Overlap opportunity",
        "Local host prevalence",
        "Geographic / sampling context",
        "LD burden",
        "Significant LD fraction",
        "UTR5 S structure",
        "UTR5 M structure",
        "UTR5 L structure",
        "UTR3 S structure",
        "UTR3 M structure",
        "UTR3 L structure",
        "Tree discordance",
        "Controls / sampling",
    ]
    cols = [c for c in preferred if c in frame.columns] + [c for c in frame.columns if c not in preferred]
    return frame[cols]


def mean_feature_ranks(feature_names: np.ndarray, contributions: np.ndarray) -> pd.DataFrame:
    abs_c = np.abs(contributions)
    ranks = np.argsort(np.argsort(-abs_c, axis=1), axis=1) + 1
    rows = []
    for j, feature in enumerate(feature_names):
        rows.append(
            {
                "feature": feature,
                "feature_label": humanize_feature(feature),
                "family": feature_family(feature),
                "mean_rank": float(ranks[:, j].mean()),
                "median_rank": float(np.median(ranks[:, j])),
                "mean_abs_contribution": float(abs_c[:, j].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)


def class_rank_shift(feature_names: np.ndarray, contributions: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    abs_c = np.abs(contributions)
    rows = []
    for label, mask in [("non_reassortant", y == 0), ("reassortant", y == 1)]:
        ranks = np.argsort(np.argsort(-abs_c[mask], axis=1), axis=1) + 1
        rows.append(pd.DataFrame({"feature": feature_names, f"{label}_mean_rank": ranks.mean(axis=0)}))
    out = rows[0].merge(rows[1], on="feature")
    out["rank_shift"] = out["reassortant_mean_rank"] - out["non_reassortant_mean_rank"]
    out["abs_rank_shift"] = out["rank_shift"].abs()
    out["feature_label"] = out["feature"].map(humanize_feature)
    out["family"] = out["feature"].map(feature_family)
    return out.sort_values("abs_rank_shift", ascending=False).reset_index(drop=True)


def cluster_top_features(feature_names: np.ndarray, contributions: np.ndarray, top_n: int, top_clusters: int) -> pd.DataFrame:
    abs_mean = np.abs(contributions).mean(axis=0)
    keep_idx = np.argsort(abs_mean)[::-1][:top_n]
    sel_names = feature_names[keep_idx]
    sel_contrib = contributions[:, keep_idx]
    if sel_contrib.shape[1] <= 1:
        cluster_ids = np.ones(sel_contrib.shape[1], dtype=int)
    else:
        corr = np.corrcoef(sel_contrib.T)
        corr = np.nan_to_num(corr, nan=0.0)
        dist = 1.0 - np.abs(corr)
        tri = dist[np.triu_indices_from(dist, k=1)]
        if np.allclose(tri, 0):
            cluster_ids = np.ones(sel_contrib.shape[1], dtype=int)
        else:
            Z = linkage(tri, method="average")
            cluster_ids = fcluster(Z, t=0.4, criterion="distance")
    rows = []
    for cluster in sorted(np.unique(cluster_ids)):
        members = np.where(cluster_ids == cluster)[0]
        member_names = sel_names[members]
        member_abs = abs_mean[keep_idx[members]]
        member_families = [feature_family(name) for name in member_names]
        exemplar = member_names[np.argmax(member_abs)]
        row = {
            "cluster_id": int(cluster),
            "n_features": int(len(members)),
            "combined_effect": float(member_abs.sum()),
            "exemplar_feature": exemplar,
            "exemplar_label": humanize_feature(exemplar),
        }
        for family in FAMILY_COLORS:
            family_mask = [fam == family for fam in member_families]
            row[f"family_{family}"] = float(member_abs[np.array(family_mask)].sum()) if any(family_mask) else 0.0
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("combined_effect", ascending=False).head(top_clusters).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def species_color_strip(species: Sequence[str]) -> np.ndarray:
    colors = [mpl.colors.to_rgb(SPECIES_COLORS.get(s, "#bbbbbb")) for s in species]
    return np.asarray(colors)[np.newaxis, :, :]


def plot_panel_a(fig: plt.Figure, gs, df: pd.DataFrame, module_df: pd.DataFrame, probabilities: np.ndarray) -> None:
    contrib = module_df.to_numpy()
    modules = module_df.columns.tolist()
    clip = np.quantile(np.abs(contrib), 0.98)
    cmap = mpl.colormaps["coolwarm"]

    pos_mask = df["target"].to_numpy() == 1
    neg_mask = ~pos_mask
    groups = [
        ("Reassortant", pos_mask, "#c44e52"),
        ("Non-reassortant", neg_mask, "#4c72b0"),
    ]
    for col_idx, (title, mask, accent) in enumerate(groups):
        sub = contrib[mask]
        sub_probs = probabilities[mask]
        sub_species = df.loc[mask, "species"].to_list()
        order = row_order(sub)
        sub = sub[order]
        sub_probs = sub_probs[order]
        sub_species = [sub_species[i] for i in order]

        heat_ax = fig.add_subplot(gs[0, col_idx])
        strip_ax = fig.add_subplot(gs[1, col_idx], sharex=heat_ax)
        prob_ax = fig.add_subplot(gs[2, col_idx], sharex=heat_ax)

        heat_ax.imshow(sub.T, aspect="auto", cmap=cmap, vmin=-clip, vmax=clip, interpolation="nearest")
        heat_ax.set_yticks(np.arange(len(modules)), modules)
        heat_ax.set_xticks([])
        if col_idx == 0:
            panel_label(heat_ax, "A")

        strip_ax.imshow(species_color_strip(sub_species), aspect="auto", interpolation="nearest")
        strip_ax.set_yticks([])
        strip_ax.set_xticks([])
        strip_ax.spines[:].set_visible(False)

        prob_ax.bar(np.arange(len(sub_probs)), sub_probs, color=accent, width=1.0)
        prob_ax.axhline(df["target"].mean(), color="#666666", linestyle=(0, (4, 4)), linewidth=1.0)
        prob_ax.set_ylim(0, 1)
        prob_ax.set_yticks([0, 0.5, 1.0] if col_idx == 0 else [])
        prob_ax.set_ylabel("Predicted\nprobability" if col_idx == 0 else "")
        prob_ax.set_xticks([])
        prob_ax.set_xlabel("Isolates (sorted by explanation similarity)" if col_idx == 0 else "")
        prob_ax.spines["top"].set_visible(False)
        prob_ax.spines["right"].set_visible(False)


def plot_panel_b(ax: plt.Axes, rank_df: pd.DataFrame) -> None:
    panel_label(ax, "B")
    families = [fam for fam in FAMILY_COLORS if fam in rank_df["family"].values]
    values = [rank_df.loc[rank_df["family"] == fam, "mean_rank"].to_numpy() for fam in families]
    bp = ax.boxplot(
        values,
        vert=False,
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.0},
        whiskerprops={"color": "#666666", "linewidth": 0.8},
        capprops={"color": "#666666", "linewidth": 0.8},
        boxprops={"edgecolor": "#666666", "linewidth": 0.8},
    )
    for patch, fam in zip(bp["boxes"], families):
        patch.set_facecolor(mpl.colors.to_rgba(FAMILY_COLORS[fam], 0.18))
        patch.set_edgecolor(FAMILY_COLORS[fam])
    for idx, fam in enumerate(families):
        vals = values[idx]
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(vals, np.full(len(vals), idx + 1) + jitter, s=11, color=FAMILY_COLORS[fam], alpha=0.55, edgecolor="white", linewidth=0.2)
    ax.set_yticks(np.arange(1, len(families) + 1), [fam.replace(" / ", "\n") for fam in families])
    ax.set_xlabel("Mean feature rank")
    ax.invert_xaxis()


def plot_panel_c(ax: plt.Axes, shift_df: pd.DataFrame) -> None:
    panel_label(ax, "C")
    plot_df = (
        shift_df[~shift_df["feature_label"].str.startswith("Missingness:")]
        .head(10)
        .sort_values("abs_rank_shift", ascending=True)
        .reset_index(drop=True)
    )
    y = np.arange(len(plot_df))
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        fam = row["family"]
        color = FAMILY_COLORS.get(fam, "#777777")
        ax.plot(
            [row["non_reassortant_mean_rank"], row["reassortant_mean_rank"]],
            [idx, idx],
            color=color,
            alpha=0.75,
            linewidth=1.2,
        )
        ax.scatter(row["non_reassortant_mean_rank"], idx, color="#777777", s=14, zorder=3)
        ax.scatter(row["reassortant_mean_rank"], idx, color=color, s=18, zorder=3)
    ax.set_yticks(y, plot_df["feature_label"])
    ax.set_xlabel("Mean feature rank")
    ax.invert_xaxis()


def plot_panel_d(ax: plt.Axes, cluster_df: pd.DataFrame) -> None:
    panel_label(ax, "D")
    plot_df = cluster_df.sort_values("combined_effect", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    left = np.zeros(len(plot_df))
    for family, color in FAMILY_COLORS.items():
        vals = plot_df[f"family_{family}"].to_numpy(dtype=float)
        if np.allclose(vals, 0):
            continue
        ax.barh(y, vals, left=left, color=color, edgecolor="none", label=family)
        left = left + vals
    ax.set_yticks(y, plot_df["rank"].astype(str))
    ax.set_xlabel("Combined mean absolute contribution")
    ax.set_ylabel("Feature cluster")
    xmax = float(plot_df["combined_effect"].max()) * 1.38
    ax.set_xlim(0, xmax)
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        ax.text(row["combined_effect"] + xmax * 0.02, yi, row["exemplar_label"], va="center", fontsize=6.0)
    ax.legend(
        frameon=False,
        fontsize=5.6,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.2,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_labeled_dataset(args.dataset)
    if args.explanation_model == "penalized_logistic_l2":
        X, y, feature_names, contributions, logits, probabilities, intercept = fit_integrated_model(df, args.random_state)
        model_label = "integrated_taxon_mechanism / penalized_logistic_l2"
        tuned_params = None
    else:
        X, y, feature_names, contributions, logits, probabilities, intercept, tuned_params = fit_integrated_lightgbm_tree_shap(
            df,
            args.random_state,
            args.tuned_suite_dir,
            args.tuning_cv_scheme,
        )
        model_label = "integrated_taxon_mechanism / lightgbm (TreeSHAP)"

    module_df = build_module_matrix(feature_names, contributions)
    rank_df = mean_feature_ranks(feature_names, contributions)
    shift_df = class_rank_shift(feature_names, contributions, y)
    cluster_df = cluster_top_features(feature_names, contributions, args.top_features, args.top_clusters)

    contrib_df = pd.DataFrame(contributions, columns=feature_names)
    contrib_df.insert(0, "probability", probabilities)
    contrib_df.insert(0, "logit", logits)
    contrib_df.insert(0, "target", y)
    contrib_df.insert(0, "species", df["species"].to_numpy())
    contrib_df.insert(0, "tip_name", df["tip_name"].to_numpy())
    contrib_df.to_csv(args.output_dir / "integrated_exact_feature_contributions.tsv", sep="\t", index=False)
    module_out = module_df.copy()
    module_out.insert(0, "probability", probabilities)
    module_out.insert(0, "target", y)
    module_out.insert(0, "species", df["species"].to_numpy())
    module_out.insert(0, "tip_name", df["tip_name"].to_numpy())
    module_out.to_csv(args.output_dir / "integrated_module_contributions.tsv", sep="\t", index=False)
    rank_df.to_csv(args.output_dir / "integrated_feature_rank_summary.tsv", sep="\t", index=False)
    shift_df.to_csv(args.output_dir / "integrated_feature_rank_shift.tsv", sep="\t", index=False)
    cluster_df.to_csv(args.output_dir / "integrated_top_feature_clusters.tsv", sep="\t", index=False)
    (args.output_dir / "integrated_explanation_manifest.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "rows": int(len(df)),
                "positives": int(y.sum()),
                "feature_count": int(len(feature_names)),
                "intercept": intercept,
                "model": model_label,
                "explanation_model": args.explanation_model,
                "tuning_cv_scheme": args.tuning_cv_scheme,
                "tuned_suite_dir": str(args.tuned_suite_dir),
                "tuned_params": tuned_params,
                "top_features": args.top_features,
                "top_clusters": args.top_clusters,
            },
            indent=2,
        )
    )

    fig = plt.figure(figsize=(7.35, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.25, 0.10, 0.42, 1.1],
        width_ratios=[1, 1],
    )
    upper = gs[:3, :].subgridspec(3, 2, height_ratios=[1.25, 0.10, 0.42], width_ratios=[1, 1])
    plot_panel_a(fig, upper, df, module_df, probabilities)
    axB = fig.add_subplot(gs[3, 0])
    axC = fig.add_subplot(gs[3, 1])

    # Split lower-right cell for C and D.
    axB.remove()
    axC.remove()
    lower = gs[3, :].subgridspec(1, 3, width_ratios=[1.0, 1.15, 1.35], wspace=0.18)
    axB = fig.add_subplot(lower[0, 0])
    axC = fig.add_subplot(lower[0, 1])
    axD = fig.add_subplot(lower[0, 2])
    plot_panel_b(axB, rank_df)
    plot_panel_c(axC, shift_df)
    plot_panel_d(axD, cluster_df)

    outpath = args.output_dir / "figure3_integrated_explanation_architecture.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(outpath)


if __name__ == "__main__":
    main()
