from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evo_utils import (
    DATASET_DIR,
    DISPLAY_NAMES,
    FIGURE_DIR,
    KEY_FEATURES,
    REPORT_DIR,
    apply_numeric_reference,
    build_base_dataframe,
    fit_bayesian_logistic,
    numeric_reference,
    render_dataframe_markdown,
    set_plot_style,
    summarize_bayesian_terms,
    write_markdown,
)


FOCAL_RAW_TERMS = [
    "overlap_opportunity_local",
    "arha_prev_local",
    "species_feat_hostdiv_faith_pd",
    "species_feat_mean_cramers_v_overall",
    "tree_discordance_score",
]
FOCAL_TERMS = [f"{term}_z" for term in FOCAL_RAW_TERMS]
DISPLAY_TERM = dict(zip(FOCAL_TERMS, FOCAL_RAW_TERMS))
N_REPLICATES = 8


def one_tip_per_origin(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positive_groups = []
    for _, sub in df[df["response"] == 1].groupby("origin_cluster_id", sort=False):
        idx = int(rng.integers(len(sub)))
        positive_groups.append(sub.iloc[[idx]])
    positives = pd.concat(positive_groups, ignore_index=True)
    negatives = df[df["response"] == 0].copy()
    return pd.concat([positives, negatives], ignore_index=True)


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    df, lineage_df, origin_summary = build_base_dataframe()
    ref = numeric_reference(df)
    analysis = apply_numeric_reference(df, ref)
    analysis.to_csv(DATASET_DIR / "analysis_base_dataframe.tsv", sep="\t", index=False)
    lineage_df.to_csv(REPORT_DIR / "lineage_assignments.tsv", sep="\t", index=False)
    origin_summary.to_csv(REPORT_DIR / "origin_cluster_summary.tsv", sep="\t", index=False)

    _, full_idata, full_formula, full_terms = fit_bayesian_logistic(
        analysis,
        features=KEY_FEATURES,
        group_effects=["species"],
        draws=350,
        tune=350,
        chains=2,
        cores=2,
        target_accept=0.95,
        random_seed=42,
    )
    full_summary = summarize_bayesian_terms(full_idata, [term for term in FOCAL_TERMS if term in full_terms], "full_tip_model")
    full_summary["replicate"] = 0

    replicate_rows = []
    replicate_summaries = []
    for replicate in range(1, N_REPLICATES + 1):
        sub = one_tip_per_origin(analysis, seed=100 + replicate)
        _, idata, formula, terms = fit_bayesian_logistic(
            sub,
            features=KEY_FEATURES,
            group_effects=["species"],
            draws=250,
            tune=250,
            chains=2,
            cores=2,
            target_accept=0.95,
            random_seed=100 + replicate,
        )
        summary = summarize_bayesian_terms(idata, [term for term in FOCAL_TERMS if term in terms], "one_tip_per_origin")
        summary["replicate"] = replicate
        summary["n_positive"] = int(sub["response"].sum())
        replicate_summaries.append(summary)
        replicate_rows.append({"replicate": replicate, "formula": formula, "n": len(sub), "n_positive": int(sub["response"].sum())})

    replicate_df = pd.DataFrame(replicate_rows)
    one_tip_summary = pd.concat(replicate_summaries, ignore_index=True)
    aggregate_rows = []
    for term, sub in one_tip_summary.groupby("term", sort=False):
        aggregate_rows.append(
            {
                "model": "one_tip_per_origin",
                "term": term,
                "mean": float(sub["mean"].mean()),
                "sd": float(sub["mean"].std(ddof=0)),
                "hdi_3%": float(sub["mean"].quantile(0.03)),
                "hdi_97%": float(sub["mean"].quantile(0.97)),
                "posterior_prob_positive": float((sub["mean"] > 0).mean()),
                "replicate_n": len(sub),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows)
    summary_df = pd.concat(
        [
            full_summary.assign(replicate_n=1)[["model", "term", "mean", "sd", "hdi_3%", "hdi_97%", "posterior_prob_positive", "replicate_n"]],
            aggregate_df,
        ],
        ignore_index=True,
    )
    summary_df["raw_term"] = summary_df["term"].map(DISPLAY_TERM)

    full_summary.to_csv(REPORT_DIR / "01_origin_cluster_full_posterior.tsv", sep="\t", index=False)
    one_tip_summary.to_csv(REPORT_DIR / "01_origin_cluster_one_tip_replicates.tsv", sep="\t", index=False)
    replicate_df.to_csv(REPORT_DIR / "01_origin_cluster_one_tip_runs.tsv", sep="\t", index=False)
    summary_df.to_csv(REPORT_DIR / "01_origin_cluster_bayesian_summary.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    origin_sizes = (
        analysis.loc[analysis["response"] == 1, ["origin_cluster_id", "species_label", "tip_name"]]
        .groupby(["origin_cluster_id", "species_label"], as_index=False)
        .agg(cluster_size=("tip_name", "size"))
        .sort_values(["species_label", "cluster_size"], ascending=[True, False])
    )
    species_order = sorted(origin_sizes["species_label"].unique())
    data = [origin_sizes.loc[origin_sizes["species_label"] == species, "cluster_size"].to_numpy() for species in species_order]
    violin = axes[0].violinplot(data, showmeans=False, showextrema=False, widths=0.85)
    palette = ["#d8c6f2", "#f2c6d8", "#c6def1", "#d4f0c0", "#f7d6b6", "#fde2e4", "#e2ece9"]
    for body, color in zip(violin["bodies"], palette[: len(violin["bodies"])]):
        body.set_facecolor(color)
        body.set_edgecolor("#444444")
        body.set_alpha(0.9)
    axes[0].set_xticks(np.arange(1, len(species_order) + 1))
    axes[0].set_xticklabels(species_order, rotation=25, ha="right")
    axes[0].set_ylabel("Positive tips per inferred origin cluster")

    model_order = ["full_tip_model", "one_tip_per_origin"]
    offsets = {"full_tip_model": -0.12, "one_tip_per_origin": 0.12}
    colors = {"full_tip_model": "#6d597a", "one_tip_per_origin": "#577590"}
    y_positions = np.arange(len(FOCAL_TERMS))
    plot_df = summary_df[summary_df["term"].isin(FOCAL_TERMS)].copy()
    for _, row in plot_df.iterrows():
        ypos = y_positions[FOCAL_TERMS.index(row["term"])] + offsets[row["model"]]
        axes[1].hlines(y=ypos, xmin=row["hdi_3%"], xmax=row["hdi_97%"], color=colors[row["model"]], linewidth=2.5)
        axes[1].scatter(row["mean"], ypos, s=38, color=colors[row["model"]], edgecolors="#333333", linewidths=0.35, label=row["model"])
    axes[1].axvline(0, color="#888888", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels([DISPLAY_NAMES[term] for term in FOCAL_RAW_TERMS])
    axes[1].set_xlabel("Posterior mean and 94% HDI")
    handles, labels = axes[1].get_legend_handles_labels()
    legend_map = {"full_tip_model": "Full tip Bayesian", "one_tip_per_origin": "One tip per origin Bayesian"}
    dedup = {}
    for label, handle in zip(labels, handles):
        dedup[legend_map[label]] = handle
    axes[1].legend(dedup.values(), dedup.keys(), frameon=False, loc="best")

    fig.savefig(FIGURE_DIR / "01_origin_cluster_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = (
        "# Origin-Cluster Robustness\n\n"
        f"- Tips analyzed: `{len(analysis)}`\n"
        f"- Reassortant tips: `{int(analysis['response'].sum())}`\n"
        f"- Inferred positive origin clusters: `{origin_sizes['origin_cluster_id'].nunique()}`\n"
        f"- Median positive tips per origin cluster: `{origin_sizes['cluster_size'].median():.1f}`\n"
        f"- One-tip-per-origin Bayesian replicates: `{N_REPLICATES}`\n\n"
        f"- Full-tip formula: `{full_formula}`\n\n"
        + render_dataframe_markdown("Posterior Stability Summary", summary_df[["model", "raw_term", "mean", "hdi_3%", "hdi_97%", "posterior_prob_positive", "replicate_n"]])
    )
    write_markdown(REPORT_DIR / "01_origin_cluster_robustness_report.md", report)


if __name__ == "__main__":
    main()
