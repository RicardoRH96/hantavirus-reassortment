from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evo_utils import (
    DISPLAY_NAMES,
    FIGURE_DIR,
    KEY_FEATURES,
    REPORT_DIR,
    alternative_label_sets,
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


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    base_df, _, _ = build_base_dataframe()
    ref = numeric_reference(base_df)
    label_sets = alternative_label_sets()
    current = label_sets["current_main"].rename(columns={"response": "current_response"})

    summary_rows = []
    agreement_rows = []
    for idx, (label_name, labels) in enumerate(label_sets.items()):
        df = base_df.drop(columns=["response"]).merge(labels, on=["species", "tip_name"], how="left")
        df["response"] = df["response"].fillna(0).astype(int)
        df = apply_numeric_reference(df, ref)
        _, idata, formula, terms = fit_bayesian_logistic(
            df,
            features=KEY_FEATURES,
            group_effects=["species"],
            draws=300,
            tune=300,
            chains=2,
            cores=2,
            target_accept=0.95,
            random_seed=200 + idx,
        )
        summary = summarize_bayesian_terms(idata, [term for term in FOCAL_TERMS if term in terms], label_name)
        summary["label_set"] = label_name
        summary["raw_term"] = summary["term"].map(DISPLAY_TERM)
        summary["formula"] = formula
        summary["positive_n"] = int(df["response"].sum())
        summary_rows.append(summary)

        merged = labels.merge(current, on=["species", "tip_name"], how="left")
        agreement_rows.append(
            {
                "label_set": label_name,
                "positive_n": int(merged["response"].sum()),
                "agreement_with_current": float((merged["response"] == merged["current_response"]).mean()),
                "positive_delta_vs_current": int(merged["response"].sum() - merged["current_response"].sum()),
            }
        )

    coef_df = pd.concat(summary_rows, ignore_index=True)
    agreement_df = pd.DataFrame(agreement_rows).sort_values("label_set").reset_index(drop=True)
    range_rows = []
    for term, sub in coef_df.groupby("raw_term", sort=False):
        range_rows.append(
            {
                "term": term,
                "posterior_mean_min": float(sub["mean"].min()),
                "posterior_mean_max": float(sub["mean"].max()),
                "posterior_mean_median": float(sub["mean"].median()),
                "posterior_prob_positive_min": float(sub["posterior_prob_positive"].min()),
                "posterior_prob_positive_max": float(sub["posterior_prob_positive"].max()),
            }
        )
    range_df = pd.DataFrame(range_rows)

    coef_df.to_csv(REPORT_DIR / "02_label_uncertainty_coefficients.tsv", sep="\t", index=False)
    agreement_df.to_csv(REPORT_DIR / "02_label_set_agreement.tsv", sep="\t", index=False)
    range_df.to_csv(REPORT_DIR / "02_label_uncertainty_summary.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(agreement_df["label_set"], agreement_df["agreement_with_current"], color="#c6def1", edgecolor="#444444")
    axes[0].set_xlabel("Agreement with current main labels")
    axes[0].set_xlim(0, 1)

    label_order = ["current_main", "backfill_full", "backfill_fast", "distance_leq_1", "distance_leq_3"]
    colors = {
        "current_main": "#6d597a",
        "backfill_full": "#577590",
        "backfill_fast": "#90be6d",
        "distance_leq_1": "#bc6c25",
        "distance_leq_3": "#b56576",
    }
    y_positions = np.arange(len(FOCAL_RAW_TERMS))
    for idx, label_name in enumerate(label_order):
        sub = coef_df[coef_df["label_set"] == label_name]
        for _, row in sub.iterrows():
            ypos = y_positions[FOCAL_RAW_TERMS.index(DISPLAY_TERM[row["term"]])] + (idx - (len(label_order) - 1) / 2) * 0.09
            axes[1].hlines(y=ypos, xmin=row["hdi_3%"], xmax=row["hdi_97%"], color=colors[label_name], linewidth=1.8)
            axes[1].scatter(row["mean"], ypos, color=colors[label_name], s=26, edgecolors="#333333", linewidths=0.3, label=label_name)
    axes[1].axvline(0, color="#888888", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels([DISPLAY_NAMES[t] for t in FOCAL_RAW_TERMS])
    axes[1].set_xlabel("Posterior mean and 94% HDI")
    handles, labels = axes[1].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    legend_map = {
        "current_main": "Current main",
        "backfill_full": "Backfill full",
        "backfill_fast": "Backfill fast",
        "distance_leq_1": "Distance <= 1",
        "distance_leq_3": "Distance <= 3",
    }
    axes[1].legend([dedup[k] for k in legend_map if k in dedup], [legend_map[k] for k in legend_map if k in dedup], frameon=False, loc="best")

    fig.savefig(FIGURE_DIR / "02_label_uncertainty_propagation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = (
        "# Label Uncertainty Propagation\n\n"
        f"- Label sets evaluated: `{len(label_sets)}`\n\n"
        + render_dataframe_markdown("Label-Set Agreement", agreement_df)
        + "\n"
        + render_dataframe_markdown("Posterior Coefficient Ranges", range_df)
    )
    write_markdown(REPORT_DIR / "02_label_uncertainty_propagation_report.md", report)


if __name__ == "__main__":
    main()
