from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evo_utils import (
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
    top_countries,
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


def one_tip_per_origin(df: pd.DataFrame) -> pd.DataFrame:
    positives = (
        df[df["response"] == 1]
        .sort_values(["origin_cluster_id", "tip_name"])
        .groupby("origin_cluster_id", as_index=False)
        .head(1)
    )
    negatives = df[df["response"] == 0].copy()
    return pd.concat([positives, negatives], ignore_index=True)


def cap_lineage_size(df: pd.DataFrame, cap: int = 5) -> pd.DataFrame:
    parts = []
    for _, sub in df.groupby(["lineage_id", "response"], sort=False):
        parts.append(sub.sort_values("tip_name").head(cap))
    return pd.concat(parts, ignore_index=True)


def run_scenario(name: str, df: pd.DataFrame, ref, seed: int) -> pd.DataFrame:
    analysis = apply_numeric_reference(df, ref)
    _, idata, formula, terms = fit_bayesian_logistic(
        analysis,
        features=KEY_FEATURES,
        group_effects=["species"],
        draws=220,
        tune=220,
        chains=2,
        cores=2,
        target_accept=0.95,
        random_seed=seed,
    )
    summary = summarize_bayesian_terms(idata, [term for term in FOCAL_TERMS if term in terms], name)
    summary["scenario"] = name
    summary["raw_term"] = summary["term"].map(DISPLAY_TERM)
    summary["n"] = len(df)
    summary["positive_n"] = int(df["response"].sum())
    summary["formula"] = formula
    return summary


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    base_df, _, _ = build_base_dataframe()
    ref = numeric_reference(base_df)
    summaries = []

    scenario_defs = [
        ("full", base_df),
        ("one_per_site_year_host", base_df.sort_values("tip_name").groupby("site_year_host_key", as_index=False).head(1)),
        ("one_per_origin_positive", one_tip_per_origin(base_df)),
        ("cap_5_per_lineage_response", cap_lineage_size(base_df, cap=5)),
    ]
    for species in sorted(base_df["species"].unique()):
        scenario_defs.append((f"leave_species_out::{species}", base_df[base_df["species"] != species].copy()))
    for country in top_countries(base_df, min_n=20, top_n=5):
        scenario_defs.append((f"leave_country_out::{country}", base_df[base_df["country"] != country].copy()))

    for idx, (name, frame) in enumerate(scenario_defs):
        summaries.append(run_scenario(name, frame, ref, seed=800 + idx))

    scenario_df = pd.concat(summaries, ignore_index=True)
    summary_rows = []
    for term, sub in scenario_df.groupby("raw_term", sort=False):
        signs = np.sign(sub["mean"].fillna(0))
        median_sign = np.sign(np.median(sub["mean"].dropna()))
        summary_rows.append(
            {
                "term": term,
                "posterior_mean_min": float(sub["mean"].min()),
                "posterior_mean_max": float(sub["mean"].max()),
                "posterior_mean_median": float(sub["mean"].median()),
                "posterior_prob_positive_min": float(sub["posterior_prob_positive"].min()),
                "posterior_prob_positive_max": float(sub["posterior_prob_positive"].max()),
                "scenario_n": int(sub["scenario"].nunique()),
                "sign_consistency": float(np.mean(signs == median_sign)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    scenario_df.to_csv(REPORT_DIR / "06_sampling_sensitivity_coefficients.tsv", sep="\t", index=False)
    summary_df.to_csv(REPORT_DIR / "06_sampling_sensitivity_summary.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    scenarios = scenario_df["scenario"].drop_duplicates().tolist()
    y_positions = np.arange(len(scenarios))
    offsets = np.linspace(-0.25, 0.25, num=len(FOCAL_RAW_TERMS))
    colors = ["#6d597a", "#577590", "#90be6d", "#bc6c25", "#b56576"]
    for offset, raw_term, color in zip(offsets, FOCAL_RAW_TERMS, colors):
        sub = scenario_df[scenario_df["raw_term"] == raw_term].set_index("scenario").reindex(scenarios).reset_index()
        ax.scatter(sub["mean"], y_positions + offset, color=color, s=28, edgecolors="#333333", linewidths=0.3, label=DISPLAY_NAMES[raw_term])
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(scenarios)
    ax.set_xlabel("Posterior mean across sampling scenarios")
    ax.legend(frameon=False, loc="best")
    fig.savefig(FIGURE_DIR / "06_sampling_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = (
        "# Sampling Sensitivity\n\n"
        f"- Scenarios evaluated: `{scenario_df['scenario'].nunique()}`\n\n"
        + render_dataframe_markdown("Posterior Stability Summary", summary_df)
    )
    write_markdown(REPORT_DIR / "06_sampling_sensitivity_report.md", report)


if __name__ == "__main__":
    main()
