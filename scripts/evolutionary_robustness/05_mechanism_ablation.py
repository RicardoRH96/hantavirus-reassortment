from __future__ import annotations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evo_utils import (
    DISPLAY_NAMES,
    ECOLOGY_FEATURES,
    FIGURE_DIR,
    KEY_FEATURES,
    KEY_FEATURES_PLUS_UTR3,
    MOLECULAR_FEATURES,
    MOLECULAR_FEATURES_PLUS_UTR3,
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
    "species_feat_utr5_S_structdist_mean",
    "species_feat_utr5_M_structdist_mean",
    "species_feat_utr5_L_structdist_mean",
    "species_feat_utr3_S_structdist_mean",
    "species_feat_utr3_M_structdist_mean",
    "species_feat_utr3_L_structdist_mean",
    "tree_discordance_score",
]
FOCAL_TERMS = [f"{term}_z" for term in FOCAL_RAW_TERMS]
DISPLAY_TERM = dict(zip(FOCAL_TERMS, FOCAL_RAW_TERMS))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    df, _, _ = build_base_dataframe()
    ref = numeric_reference(df)
    analysis = apply_numeric_reference(df, ref)

    model_specs = [
        ("intercept_only", [], []),
        ("lineage_only", [], ["lineage_id"]),
        ("ecology_over_lineage", ECOLOGY_FEATURES, ["lineage_id"]),
        ("molecular_over_lineage", MOLECULAR_FEATURES, ["lineage_id"]),
        ("molecular_plus_utr3_over_lineage", MOLECULAR_FEATURES_PLUS_UTR3, ["lineage_id"]),
        ("integrated_over_lineage", KEY_FEATURES, ["lineage_id"]),
        ("integrated_plus_utr3_over_lineage", KEY_FEATURES_PLUS_UTR3, ["lineage_id"]),
    ]

    model_rows = []
    coef_rows = []
    baseline_elpd = None
    for idx, (name, features, groups) in enumerate(model_specs):
        _, idata, formula, terms = fit_bayesian_logistic(
            analysis,
            features=features,
            group_effects=groups,
            draws=300,
            tune=300,
            chains=2,
            cores=2,
            target_accept=0.95,
            random_seed=500 + idx,
        )
        loo = az.loo(idata)
        if name == "lineage_only":
            baseline_elpd = float(loo.elpd_loo)
        model_rows.append(
            {
                "model": name,
                "formula": formula,
                "elpd_loo": float(loo.elpd_loo),
                "p_loo": float(loo.p_loo),
                "se": float(loo.se),
                "delta_elpd_vs_lineage": float(loo.elpd_loo - baseline_elpd) if baseline_elpd is not None else np.nan,
            }
        )
        terms_to_keep = [term for term in FOCAL_TERMS if term in terms]
        summary = summarize_bayesian_terms(idata, terms_to_keep, name)
        summary["raw_term"] = summary["term"].map(DISPLAY_TERM)
        coef_rows.append(summary)

    model_df = pd.DataFrame(model_rows)
    coef_df = pd.concat(coef_rows, ignore_index=True)
    model_df.to_csv(REPORT_DIR / "05_mechanism_ablation_models.tsv", sep="\t", index=False)
    coef_df.to_csv(REPORT_DIR / "05_mechanism_ablation_coefficients.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    plot_df = model_df[
        model_df["model"].isin(
            [
                "lineage_only",
                "ecology_over_lineage",
                "molecular_over_lineage",
                "molecular_plus_utr3_over_lineage",
                "integrated_over_lineage",
                "integrated_plus_utr3_over_lineage",
            ]
        )
    ].copy()
    label_map = {
        "lineage_only": "Lineage only",
        "ecology_over_lineage": "Ecology + lineage",
        "molecular_over_lineage": "Molecular + lineage",
        "molecular_plus_utr3_over_lineage": "Molecular + UTR3 + lineage",
        "integrated_over_lineage": "Integrated + lineage",
        "integrated_plus_utr3_over_lineage": "Integrated + UTR3 + lineage",
    }
    axes[0].bar(
        [label_map[m] for m in plot_df["model"]],
        plot_df["delta_elpd_vs_lineage"].fillna(0),
        color=["#e2ece9", "#d4f0c0", "#f7d6b6", "#f9e2cc", "#d8c6f2", "#cfe8f7"],
        edgecolor="#444444",
    )
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Delta ELPD-LOO vs lineage-only")

    integrated = coef_df[coef_df["model"] == "integrated_plus_utr3_over_lineage"].copy()
    y_positions = np.arange(len(FOCAL_RAW_TERMS))
    for _, row in integrated.iterrows():
        ypos = y_positions[FOCAL_RAW_TERMS.index(row["raw_term"])]
        axes[1].hlines(y=ypos, xmin=row["hdi_3%"], xmax=row["hdi_97%"], color="#6d597a", linewidth=2.5)
        axes[1].scatter(row["mean"], ypos, color="#6d597a", s=42, edgecolors="#333333", linewidths=0.35)
    axes[1].axvline(0, color="#888888", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels([DISPLAY_NAMES[term] for term in FOCAL_RAW_TERMS])
    axes[1].set_xlabel("Posterior mean and 94% HDI")

    fig.savefig(FIGURE_DIR / "05_mechanism_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = (
        "# Mechanism Ablation\n\n"
        f"- Tips analyzed: `{len(analysis)}`\n"
        f"- Unique lineage groups: `{analysis['lineage_id'].nunique()}`\n\n"
        + render_dataframe_markdown("Model Comparison", model_df)
        + "\n"
        + render_dataframe_markdown("Integrated-Over-Lineage Posterior", integrated[["model", "raw_term", "mean", "hdi_3%", "hdi_97%", "posterior_prob_positive"]])
    )
    write_markdown(REPORT_DIR / "05_mechanism_ablation_report.md", report)


if __name__ == "__main__":
    main()
