#!/usr/bin/env python3
"""Render manuscript-style multi-panel figures for reassortment results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D
from sklearn.metrics import balanced_accuracy_score


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARE_SUITE_DIR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "figure1_compare_latest_winners"
)
DEFAULT_FOCAL_SUITE_DIR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "final_repeated_stratified_suite"
)
DEFAULT_HIER_DIR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "final_hierarchical_block_comparison"
)
DEFAULT_DATASET = (
    REPO_ROOT
    / "data"
    / "processed"
    / "final_sequence_dataset.tsv"
)


SCHEME_LABELS = {
    "grouped_species": "Leave-one-species-out",
    "grouped_within_species_clade": "Leave-one-lineage-out",
    "repeated_stratified": "Repeated stratified CV",
}

MODEL_DISPLAY = {
    ("integrated_taxon_mechanism", "catboost"): "Int. + taxon\nCatBoost",
    ("ecology_molecular", "catboost"): "Eco. + mol.\nCatBoost",
    ("integrated_taxon_mechanism", "xgboost"): "Int. + taxon\nXGBoost",
    ("ecology_molecular", "xgboost"): "Eco. + mol.\nXGBoost",
    ("integrated_taxon_mechanism", "lightgbm"): "Int. + taxon\nLightGBM",
    ("ecology_molecular", "lightgbm"): "Eco. + mol.\nLightGBM",
}

MODEL_COLORS = {
    ("integrated_taxon_mechanism", "catboost"): "#7a3e9d",
    ("ecology_molecular", "catboost"): "#c06c84",
    ("integrated_taxon_mechanism", "xgboost"): "#1f4e79",
    ("ecology_molecular", "xgboost"): "#2a9d8f",
    ("integrated_taxon_mechanism", "lightgbm"): "#355070",
    ("ecology_molecular", "lightgbm"): "#f4a261",
}

starfish = ['#24492e', '#015b58', '#2c6184', '#59629b', '#89689d', '#ba7999', '#e69b99', '#d9b24c']
pal = [starfish[i - 1] for i in [7, 4, 5, 3, 1, 6, 2]]

SPECIES_LABELS = {
    "orthohantavirus_andesense": "ANDV",
    "orthohantavirus_dobravaense": "DOBV",
    "orthohantavirus_hantanense": "HTNV",
    "orthohantavirus_puumalaense": "PUUV",
    "orthohantavirus_seoulense": "SEOV",
    "orthohantavirus_sinnombreense": "SNV",
    "orthohantavirus_tulaense": "TULV",
}

SPECIES_COLORS = {
    "orthohantavirus_andesense": pal[5],
    "orthohantavirus_dobravaense": pal[0],
    "orthohantavirus_hantanense": pal[1],
    "orthohantavirus_puumalaense": pal[3],
    "orthohantavirus_seoulense": pal[4],
    "orthohantavirus_sinnombreense": pal[6],
    "orthohantavirus_tulaense": pal[2],
}

TERM_DISPLAY = {
    "hosttrait_combine_gestation_length_d_z": "Host gestation length",
    "hosttrait_combine_litter_size_n_z": "Host litter size",
    "hosttrait_combine_dispersal_km_z": "Host dispersal",
    "species_feat_hostdiv_faith_pd_z": "Host Faith's PD",
    "overlap_opportunity_local_z": "Overlap opportunity",
    "arha_prev_local_z": "Local host prevalence",
    "overlap_opportunity_local_z:species_feat_hostdiv_faith_pd_z": "Overlap x host PD",
    "species_feat_mean_cramers_v_overall_z": "LD burden",
    "species_feat_prop_significant_overall_z": "LD significant-pair fraction",
    "species_feat_utr5_M_structdist_mean_z": "UTR5 M structure distance",
    "species_feat_utr5_L_structdist_mean_z": "UTR5 L structure distance",
    "tree_discordance_score_z": "Tree discordance score",
}

HEATMAP_FEATURES = [
    "observed_rate",
    "overlap_opportunity_local",
    "arha_prev_local",
    "species_feat_hostdiv_faith_pd",
    "species_feat_mean_cramers_v_overall",
    "species_feat_prop_significant_overall",
    "species_feat_utr5_M_structdist_mean",
    "species_feat_utr5_L_structdist_mean",
]

HEATMAP_LABELS = {
    "observed_rate": "Observed rate",
    "overlap_opportunity_local": "Overlap",
    "arha_prev_local": "Local prevalence",
    "species_feat_hostdiv_faith_pd": "Faith's PD",
    "species_feat_mean_cramers_v_overall": "LD burden",
    "species_feat_prop_significant_overall": "LD sig. frac.",
    "species_feat_utr5_M_structdist_mean": "UTR5 M struct.",
    "species_feat_utr5_L_structdist_mean": "UTR5 L struct.",
}

BLOCK_DISPLAY = {
    "ecology_interaction_block": "Ecology +\ninteraction",
    "molecular_block": "Molecular",
    "integrated_block": "Integrated",
    "integrated_tree_sensitivity": "Integrated\n+ tree",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-suite-dir", type=Path, default=DEFAULT_COMPARE_SUITE_DIR)
    parser.add_argument("--focal-suite-dir", type=Path, default=DEFAULT_FOCAL_SUITE_DIR)
    parser.add_argument("--hier-dir", type=Path, default=DEFAULT_HIER_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
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


def _species_short(name: str) -> str:
    return SPECIES_LABELS.get(name, name.replace("orthohantavirus_", "").replace("ense", "."))


def _lighten_color(color: str, amount: float = 0.45) -> tuple[float, float, float]:
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.03) -> None:
    ax.text(x, y, label.lower(), transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def make_figure1(
    compare_suite_dir: Path,
    focal_suite_dir: Path,
    hier_dir: Path,
    outpath: Path,
) -> None:
    fold_df = pd.read_csv(compare_suite_dir / "cv_fold_metrics.tsv", sep="\t")
    summary_df = pd.read_csv(compare_suite_dir / "summary.tsv", sep="\t")
    focal_oof_df = pd.read_csv(focal_suite_dir / "grouped_species_cv_predictions.tsv", sep="\t")
    focal_repeated_pred_df = pd.read_csv(focal_suite_dir / "repeated_stratified_cv_predictions.tsv", sep="\t")
    ppc_df = pd.read_csv(hier_dir / "integrated_block" / "ppc_by_species.tsv", sep="\t")

    selected_pairs = list(MODEL_DISPLAY.keys())
    fold_df = fold_df[
        fold_df.apply(lambda r: (r["feature_set"], r["model"]) in selected_pairs, axis=1)
    ].copy()
    summary_df = summary_df[
        summary_df.apply(lambda r: (r["feature_set"], r["model"]) in selected_pairs, axis=1)
    ].copy()

    fig = plt.figure(figsize=(10.6, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.15])
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[1, 2])

    panel_label(axA, "A")
    panel_label(axB, "B", y=1.09)
    panel_label(axC, "C", y=1.09)
    panel_label(axD, "D", y=1.09)

    for ax in (axB, axC, axD):
        ax.set_box_aspect(0.95)

    prevalence = (
        focal_oof_df[["row_index", "target"]]
        .drop_duplicates()["target"]
        .mean()
    )
    spacing = 1.3
    positions = []
    labels = []
    scheme_order = ["repeated_stratified", "grouped_species", "grouped_within_species_clade"]
    scheme_bg = {
        "repeated_stratified": "#fafaf8",
        "grouped_species": "#ffffff",
        "grouped_within_species_clade": "#fafaf8",
    }
    scheme_bases = [scheme_idx * (len(selected_pairs) + 1) * spacing for scheme_idx, _ in enumerate(scheme_order)]
    scheme_spans = []
    for scheme_idx, base in enumerate(scheme_bases):
        left = base - 0.6 if scheme_idx == 0 else 0.5 * (
            scheme_bases[scheme_idx - 1] + (base + (len(selected_pairs) - 1) * spacing)
        )
        right = (
            (base + (len(selected_pairs) - 1) * spacing) + 0.6
            if scheme_idx == len(scheme_bases) - 1
            else 0.5 * ((base + (len(selected_pairs) - 1) * spacing) + scheme_bases[scheme_idx + 1])
        )
        scheme_spans.append((left, right))

    for scheme_idx, scheme in enumerate(scheme_order):
        scheme_df = fold_df[fold_df["cv_scheme"] == scheme]
        base = scheme_bases[scheme_idx]
        start, end = scheme_spans[scheme_idx]
        axA.axvspan(start, end, color=scheme_bg.get(scheme, "#ffffff"), zorder=0)
        axA.text((start + end) / 2, 0.97, SCHEME_LABELS[scheme], transform=axA.get_xaxis_transform(), ha="center", va="top", fontsize=7.4, color="#4d4d4d")
        for j, pair in enumerate(selected_pairs):
            sub = scheme_df[(scheme_df["feature_set"] == pair[0]) & (scheme_df["model"] == pair[1])]
            pos = base + j * spacing
            positions.append(pos)
            labels.append(MODEL_DISPLAY[pair])
            vals = sub["average_precision"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            vp = axA.violinplot(vals, positions=[pos], widths=0.9, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(MODEL_COLORS[pair])
                body.set_edgecolor("none")
                body.set_alpha(0.28)
            bp = axA.boxplot(
                vals,
                positions=[pos],
                widths=0.45,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "#3b3b3b", "linewidth": 1.0},
                whiskerprops={"color": "#555555", "linewidth": 0.8},
                capprops={"color": "#555555", "linewidth": 0.8},
                boxprops={"facecolor": "#f7f7f7", "edgecolor": "#555555", "linewidth": 0.8},
                flierprops={"marker": "o", "markersize": 2.8, "markerfacecolor": "#555555", "markeredgecolor": "#555555", "alpha": 0.65},
            )
            for patch in bp["boxes"]:
                patch.set_alpha(0.9)
            jitter = np.linspace(-0.16, 0.16, len(vals)) if len(vals) > 1 else np.array([0.0])
            axA.scatter(np.full(len(vals), pos) + jitter, vals, s=18, color=MODEL_COLORS[pair], alpha=0.65, edgecolor="white", linewidth=0.3, zorder=3)

            fold_mean = float(np.mean(vals))
            label_y = float(np.clip(fold_mean - 0.045, 0.04, 0.96))
            axA.text(
                pos,
                label_y,
                f"{fold_mean:.2f}",
                ha="center",
                va="top",
                fontsize=6.2,
                color="#333333",
            )

    axA.axhline(prevalence, color="#555555", linestyle=(0, (4, 4)), linewidth=1.3)
    axA.text(positions[-1] + 0.42, prevalence + 0.005, "prevalence", fontsize=6.3, color="#555555")
    axA.set_xlim(min(positions) - 0.8, max(positions) + 0.8)
    axA.set_ylim(0.0, 1.0)
    axA.set_ylabel("Average precision")
    axA.set_xticks(positions, labels)
    plt.setp(axA.get_xticklabels(), rotation=26, ha="right", rotation_mode="anchor")
    axA.tick_params(axis="x", labelsize=5.8, pad=2)

    best_oof = focal_oof_df[
        (focal_oof_df["feature_set"] == "integrated_taxon_mechanism")
        & (focal_oof_df["model"] == "lightgbm")
    ].sort_values("probability", ascending=False).reset_index(drop=True)
    x = np.arange(len(best_oof))
    probs = best_oof["probability"].to_numpy()
    target = best_oof["target"].to_numpy()
    axB.fill_between(x, 0, probs, color="#fee8c8", alpha=0.8)
    axB.plot(x, probs, color="#2c3e50", linewidth=2)
    axB.axhline(prevalence, color="#555555", linestyle=(0, (4, 4)), linewidth=1.2)
    pos_idx = np.where(target == 1)[0]
    neg_idx = np.where(target == 0)[0]
    axB.scatter(pos_idx, np.full(pos_idx.shape, 1.01), marker="|", s=140, color="#bc4749", clip_on=False, transform=axB.get_xaxis_transform())
    neg_sub = neg_idx[:: max(1, len(neg_idx) // 120)]
    axB.scatter(neg_sub, np.full(len(neg_sub), -0.045), marker="|", s=50, color="#b8b8b8", clip_on=False, transform=axB.get_xaxis_transform())
    for i, species in enumerate(best_oof["species"]):
        axB.vlines(i, -0.12, -0.07, color=SPECIES_COLORS.get(species, "#777777"), linewidth=1)
    axB.set_ylim(-0.16, 1.05)
    axB.set_xlim(0, len(best_oof) - 1)
    axB.set_ylabel("Predicted probability")
    axB.set_xlabel("Sequences ranked by out-of-fold probability (high to low)")
    axB.text(0.02, 0.98, "top ticks: reassortant", transform=axB.transAxes, va="top", fontsize=5.9, color="#555555")
    axB.text(0.02, 0.90, "bottom strip: species", transform=axB.transAxes, va="top", fontsize=5.9, color="#555555")

    cm_df = focal_repeated_pred_df[
        (focal_repeated_pred_df["feature_set"] == "integrated_taxon_mechanism")
        & (focal_repeated_pred_df["model"] == "lightgbm")
    ].copy()
    threshold_grid = np.linspace(0.02, 0.98, 97)
    cm_df["threshold_balacc_opt"] = np.nan
    for fold in sorted(cm_df["fold_id"].unique()):
        train = cm_df[cm_df["fold_id"] != fold]
        y_train = train["target"].to_numpy()
        scores_train = train["probability"].to_numpy()
        best_t = 0.5
        best_v = -1.0
        for t in threshold_grid:
            pred_train = (scores_train >= t).astype(int)
            v = balanced_accuracy_score(y_train, pred_train)
            if v > best_v:
                best_t = float(t)
                best_v = float(v)
        cm_df.loc[cm_df["fold_id"] == fold, "threshold_balacc_opt"] = best_t
    cm_df["predicted"] = (cm_df["probability"] >= cm_df["threshold_balacc_opt"]).astype(int)
    actual_yes = int((cm_df["target"] == 1).sum())
    actual_no = int((cm_df["target"] == 0).sum())
    cm_cells = {
        (0, 0): int(((cm_df["predicted"] == 0) & (cm_df["target"] == 1)).sum()),
        (0, 1): int(((cm_df["predicted"] == 0) & (cm_df["target"] == 0)).sum()),
        (1, 0): int(((cm_df["predicted"] == 1) & (cm_df["target"] == 1)).sum()),
        (1, 1): int(((cm_df["predicted"] == 1) & (cm_df["target"] == 0)).sum()),
    }
    cm_mat = np.array(
        [
            [cm_cells[(0, 0)] / actual_yes if actual_yes else 0.0, cm_cells[(0, 1)] / actual_no if actual_no else 0.0],
            [cm_cells[(1, 0)] / actual_yes if actual_yes else 0.0, cm_cells[(1, 1)] / actual_no if actual_no else 0.0],
        ]
    )
    axC.imshow(cm_mat, cmap=mpl.cm.Greens, vmin=0, vmax=1)
    for spine in axC.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("#4a4a4a")
    axC.set_xticks([0, 1], ["Yes", "No"])
    axC.set_yticks([0, 1], ["No", "Yes"])
    axC.set_xlabel("Known reassortant")
    axC.set_ylabel("Predicted reassortant")
    axC.tick_params(length=7, width=1.6, colors="#4a4a4a")
    axC.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    axC.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    axC.grid(which="minor", color="#c7c7c7", linestyle="-", linewidth=0.8)
    axC.tick_params(which="minor", bottom=False, left=False)
    for (r, c), count in cm_cells.items():
        total = actual_yes if c == 0 else actual_no
        txt_color = "white" if cm_mat[r, c] > 0.5 else "#4a4a4a"
        axC.text(c, r - 0.03, f"{count}/{total}", ha="center", va="center", fontsize=16, color=txt_color, fontweight=500)
        axC.text(c, r + 0.18, f"{count / total:.1%}", ha="center", va="center", fontsize=8.5, color=txt_color)

    plot_df = ppc_df.sort_values("observed_rate").reset_index(drop=True)
    obs = plot_df["observed_rate"].to_numpy()
    pred = plot_df["predicted_rate_mean"].to_numpy()
    err_low = pred - plot_df["predicted_rate_hdi_3"].to_numpy()
    err_high = plot_df["predicted_rate_hdi_97"].to_numpy() - pred
    axD.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.1)
    texts = []
    for _, row in plot_df.iterrows():
        species_color = SPECIES_COLORS.get(row["species"], "#6c757d")
        axD.errorbar(
            row["observed_rate"],
            row["predicted_rate_mean"],
            yerr=[[row["predicted_rate_mean"] - row["predicted_rate_hdi_3"]], [row["predicted_rate_hdi_97"] - row["predicted_rate_mean"]]],
            fmt="o",
            color=species_color,
            ecolor=_lighten_color(species_color, 0.55),
            elinewidth=1.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.5,
            capsize=3,
            zorder=3,
        )
        txt = axD.text(
            row["observed_rate"],
            row["predicted_rate_mean"],
            _species_short(row["species"]),
            fontsize=6.1,
            color=species_color,
        )
        texts.append(txt)
    axD.set_xlim(-0.02, 1.0)
    axD.set_ylim(-0.02, 1.0)
    axD.set_xlabel("Observed reassortment rate")
    axD.set_ylabel("Posterior predictive mean")
    adjust_text(
        texts,
        x=obs,
        y=pred,
        ax=axD,
        avoid_self=True,
        force_text=(0.8, 0.8),
        force_static=(1.2, 1.2),
        force_pull=(0.15, 0.15),
        expand=(1.15, 1.25),
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5),
    )

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure2(
    hier_dir: Path,
    dataset_path: Path,
    outpath: Path,
) -> None:
    fx = pd.read_csv(hier_dir / "fixed_effect_comparison.tsv", sep="\t")
    species_re = pd.read_csv(hier_dir / "species_random_effects.tsv", sep="\t")
    raw = pd.read_csv(dataset_path, sep="\t", low_memory=False)
    raw = raw[raw["is_reassortant_final"].notna()].copy()
    raw = raw[raw["target_source"].isin(["sequence_true_label", "isolate_true_label"])].copy()
    raw["target"] = pd.to_numeric(raw["is_reassortant_final"], errors="coerce")

    fig = plt.figure(figsize=(7.2, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.2, 1.0])
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    panel_label(axA, "a)")
    panel_label(axB, "b)")
    panel_label(axC, "c)")

    forest_models = ["ecology_interaction_block", "molecular_block", "integrated_block", "integrated_tree_sensitivity"]
    forest_terms = list(TERM_DISPLAY.keys())
    forest = fx[fx["model"].isin(forest_models) & fx["term"].isin(forest_terms)].copy()
    forest["term_label"] = forest["term"].map(TERM_DISPLAY)
    forest["model_label"] = forest["model"].map(
        {
            "ecology_interaction_block": "Ecology + interaction",
            "molecular_block": "Molecular",
            "integrated_block": "Integrated",
            "integrated_tree_sensitivity": "Integrated + tree",
        }
    )
    forest["y_label"] = forest["term_label"] + " | " + forest["model_label"]
    forest = forest.sort_values(["model_label", "term_label"]).reset_index(drop=True)
    y = np.arange(len(forest))
    colors = {
        "Ecology + interaction": "#2a9d8f",
        "Molecular": "#577590",
        "Integrated": "#bc4749",
        "Integrated + tree": "#8d99ae",
    }
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5.5, markerfacecolor=color, markeredgecolor=color, label=label)
        for label, color in colors.items()
    ]
    for block_idx, (_, block) in enumerate(forest.groupby("model_label", sort=False)):
        if block_idx % 2 == 0:
            start = block.index.min() - 0.5
            end = block.index.max() + 0.5
            axA.axhspan(start, end, color="#fafafa", zorder=0)
    axA.hlines(y, forest["hdi_3%"], forest["hdi_97%"], color=[colors[m] for m in forest["model_label"]], linewidth=2)
    axA.scatter(forest["mean"], y, color=[colors[m] for m in forest["model_label"]], s=28, zorder=3)
    axA.axvline(0.0, linestyle="--", color="#555555", linewidth=1.0)
    axA.set_yticks(y, forest["y_label"])
    axA.invert_yaxis()
    axA.set_xlabel("Posterior coefficient")
    axA.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=6.4,
        ncol=4,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.08),
        mode="expand",
        borderaxespad=0.0,
        handletextpad=0.35,
        columnspacing=1.0,
    )

    shrink = species_re[species_re["model"].isin(["species_only", "integrated_block"])].copy()
    species_order = (
        shrink[shrink["model"] == "species_only"]
        .sort_values("mean")["species"]
        .tolist()
    )
    ypos = np.arange(len(species_order))
    offsets = {"species_only": -0.12, "integrated_block": 0.12}
    point_colors = {"species_only": "#8d99ae", "integrated_block": "#bc4749"}
    legend_labels = {"species_only": "Species only", "integrated_block": "Integrated block"}
    for model in ["species_only", "integrated_block"]:
        sub = shrink[shrink["model"] == model].set_index("species").loc[species_order].reset_index()
        yy = ypos + offsets[model]
        axB.hlines(yy, sub["hdi_3%"], sub["hdi_97%"], color=point_colors[model], linewidth=2, alpha=0.85)
        axB.scatter(sub["mean"], yy, color=point_colors[model], s=32, label=legend_labels[model])
    axB.axvline(0.0, linestyle="--", color="#555555", linewidth=1.0)
    axB.set_yticks(ypos, [_species_short(s) for s in species_order])
    axB.set_xlabel("Species random effect")
    axB.legend(frameon=False, fontsize=6.1, loc="lower right", handletextpad=0.4)

    species_feature = raw.groupby("species").agg(observed_rate=("target", "mean")).reset_index()
    first_cols = [c for c in HEATMAP_FEATURES if c != "observed_rate"]
    feature_table = raw.groupby("species")[first_cols].first().reset_index()
    heat = species_feature.merge(feature_table, on="species", how="left")
    heat = heat.sort_values("observed_rate").reset_index(drop=True)
    heat_mat = heat[HEATMAP_FEATURES].copy()
    for col in heat_mat.columns:
        if col == "observed_rate":
            continue
        series = pd.to_numeric(heat_mat[col], errors="coerce")
        std = series.std(ddof=0)
        if pd.isna(std) or std == 0:
            heat_mat[col] = 0.0
        else:
            heat_mat[col] = (series - series.mean()) / std
    heat_mat["observed_rate"] = heat["observed_rate"]
    data = heat_mat.to_numpy(dtype=float)
    cmap = mpl.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("#e5e5e5")
    masked = np.ma.masked_invalid(data)
    im = axC.imshow(masked, aspect="auto", cmap=cmap, vmin=-2.2, vmax=2.2)
    axC.set_yticks(np.arange(len(heat)), [_species_short(s) for s in heat["species"]])
    axC.set_xticks(np.arange(len(HEATMAP_FEATURES)), [HEATMAP_LABELS[c] for c in HEATMAP_FEATURES], rotation=35, ha="right")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = heat.iloc[i]["observed_rate"] if HEATMAP_FEATURES[j] == "observed_rate" else data[i, j]
            label = "" if pd.isna(value) else f"{value:.2f}"
            axC.text(j, i, label, ha="center", va="center", fontsize=5.8, color="#1b1b1b")
    cbar = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
    cbar.set_label("Standardized feature value")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_style()
    output_dir = args.output_dir or (args.hier_dir / "figures" / "manuscript")
    output_dir.mkdir(parents=True, exist_ok=True)

    make_figure1(args.compare_suite_dir, args.focal_suite_dir, args.hier_dir, output_dir / "figure1_model_comparison_and_calibration.png")
    make_figure2(args.hier_dir, args.dataset, output_dir / "figure2_mechanism_inference.png")


if __name__ == "__main__":
    main()
