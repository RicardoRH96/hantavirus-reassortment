from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evo_utils import (
    DISPLAY_NAMES,
    FIGURE_DIR,
    REPORT_DIR,
    build_base_dataframe,
    ensure_output_dirs,
    load_species_tree,
    prune_tree_to_tips,
    set_plot_style,
    write_markdown,
    render_dataframe_markdown,
)


FOCAL_TERMS = [
    "overlap_opportunity_local",
    "arha_prev_local",
    "tree_discordance_score",
]
N_NULL = 150


def candidate_controls(negative: pd.DataFrame, row) -> tuple[pd.DataFrame, str]:
    same_lineage = negative[negative["lineage_id"] == row.lineage_id].copy()
    same_lineage_country = same_lineage[same_lineage["country"] == row.country].copy()
    same_country = negative[negative["country"] == row.country].copy()
    if not same_lineage_country.empty:
        return same_lineage_country, "same_lineage_country"
    if not same_lineage.empty:
        return same_lineage, "same_lineage"
    if not same_country.empty:
        return same_country, "same_country"
    return negative, "species_fallback"


def build_observed_and_null(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_rows = []
    null_rows = []
    rng = np.random.default_rng(42)

    for species, sub in df.groupby("species", sort=False):
        tree = prune_tree_to_tips(load_species_tree(species), set(sub["tip_name"].astype(str)))
        positive = sub[sub["response"] == 1].copy()
        negative = sub[sub["response"] == 0].copy()
        if positive.empty or negative.empty:
            continue

        positives = list(positive.itertuples(index=False))
        for pos in positives:
            pool, scope = candidate_controls(negative, pos)
            distances = []
            for cand in pool.itertuples(index=False):
                year_diff = abs(float(pos.year) - float(cand.year)) if pd.notna(pos.year) and pd.notna(cand.year) else 999.0
                distances.append((year_diff, tree.distance(pos.tip_name, cand.tip_name), cand.tip_name, cand))
            _, _, _, nearest = min(distances, key=lambda x: (x[0], x[1], x[2]))
            for feature in FOCAL_TERMS:
                observed_rows.append(
                    {
                        "species": species,
                        "term": feature,
                        "difference": getattr(pos, feature) - getattr(nearest, feature),
                        "matching_scope": scope,
                    }
                )

        for iteration in range(N_NULL):
            sampled_controls = []
            for pos in positives:
                pool, scope = candidate_controls(negative, pos)
                sampled_controls.append(pool.sample(n=1, random_state=int(rng.integers(1_000_000))).iloc[0])
            sampled_controls = pd.DataFrame(sampled_controls)
            for feature in FOCAL_TERMS:
                diff = pd.to_numeric(positive[feature], errors="coerce").reset_index(drop=True) - pd.to_numeric(sampled_controls[feature], errors="coerce").reset_index(drop=True)
                null_rows.append(
                    {
                        "iteration": iteration,
                        "species": species,
                        "term": feature,
                        "mean_difference": float(diff.mean()),
                    }
                )
    return pd.DataFrame(observed_rows), pd.DataFrame(null_rows)


def main() -> None:
    ensure_output_dirs()
    set_plot_style()

    df, _, _ = build_base_dataframe()
    observed, null = build_observed_and_null(df)
    observed_summary = observed.groupby("term", as_index=False)["difference"].mean().rename(columns={"difference": "observed_mean_difference"})
    null_summary = (
        null.groupby("term", as_index=False)
        .agg(
            null_mean=("mean_difference", "mean"),
            null_q05=("mean_difference", lambda s: s.quantile(0.05)),
            null_q95=("mean_difference", lambda s: s.quantile(0.95)),
        )
    )
    summary = observed_summary.merge(null_summary, on="term", how="left")
    summary["outside_null_interval"] = ~summary["observed_mean_difference"].between(summary["null_q05"], summary["null_q95"])
    empirical_p = []
    for term in summary["term"]:
        observed_value = float(summary.loc[summary["term"] == term, "observed_mean_difference"].iloc[0])
        null_values = null.loc[null["term"] == term, "mean_difference"].to_numpy()
        empirical_p.append(float(np.mean(np.abs(null_values) >= abs(observed_value))))
    summary["empirical_p_two_sided"] = empirical_p

    observed.to_csv(REPORT_DIR / "04_phylogenetic_null_observed.tsv", sep="\t", index=False)
    null.to_csv(REPORT_DIR / "04_phylogenetic_null_distribution.tsv", sep="\t", index=False)
    summary.to_csv(REPORT_DIR / "04_phylogenetic_null_summary.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    y_positions = np.arange(len(summary))
    ax.hlines(y=y_positions, xmin=summary["null_q05"], xmax=summary["null_q95"], color="#c6def1", linewidth=6)
    ax.scatter(summary["null_mean"], y_positions, color="#577590", s=35, edgecolors="#333333", linewidths=0.3, label="Restricted null mean")
    ax.scatter(summary["observed_mean_difference"], y_positions, color="#bc6c25", s=45, edgecolors="#333333", linewidths=0.4, label="Observed nearest-control mean")
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([DISPLAY_NAMES[term] for term in summary["term"]])
    ax.set_xlabel("Mean positive-control difference")
    ax.legend(frameon=False, loc="best")
    fig.savefig(FIGURE_DIR / "04_phylogenetic_nulls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = (
        "# Phylogenetic Nulls\n\n"
        f"- Null iterations per species: `{N_NULL}`\n\n"
        + render_dataframe_markdown("Observed vs Restricted Null", summary)
    )
    write_markdown(REPORT_DIR / "04_phylogenetic_nulls_report.md", report)


if __name__ == "__main__":
    main()
