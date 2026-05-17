from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evo_utils import (
    DISPLAY_NAMES,
    FIGURE_DIR,
    REPORT_DIR,
    build_base_dataframe,
    ensure_output_dirs,
    load_species_tree,
    prune_tree_to_tips,
    render_dataframe_markdown,
    set_plot_style,
    write_markdown,
)


FOCAL_TERMS = [
    "overlap_opportunity_local",
    "arha_prev_local",
    "tree_discordance_score",
]


def pick_best_control(row, negative: pd.DataFrame, tree) -> tuple[object, str, float]:
    same_lineage = negative[negative["lineage_id"] == row.lineage_id].copy()
    same_lineage_country = same_lineage[same_lineage["country"] == row.country].copy()
    same_country = negative[negative["country"] == row.country].copy()
    if not same_lineage_country.empty:
        pool = same_lineage_country
        scope = "same_lineage_country"
    elif not same_lineage.empty:
        pool = same_lineage
        scope = "same_lineage"
    elif not same_country.empty:
        pool = same_country
        scope = "same_country"
    else:
        pool = negative
        scope = "species_fallback"

    distances = []
    for cand in pool.itertuples(index=False):
        year_diff = abs(float(row.year) - float(cand.year)) if pd.notna(row.year) and pd.notna(cand.year) else 999.0
        distances.append((year_diff, tree.distance(row.tip_name, cand.tip_name), cand.tip_name, cand))
    year_diff, distance, _, match = min(distances, key=lambda x: (x[0], x[1], x[2]))
    return match, scope, float(distance)


def nearest_nonreassortant_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for species, sub in df.groupby("species", sort=False):
        tree = prune_tree_to_tips(load_species_tree(species), set(sub["tip_name"].astype(str)))
        positive = sub[sub["response"] == 1].copy()
        negative = sub[sub["response"] == 0].copy()
        if positive.empty or negative.empty:
            continue
        for row in positive.itertuples(index=False):
            match, scope, distance = pick_best_control(row, negative, tree)
            out = {
                "species": species,
                "species_label": row.species_label,
                "positive_tip": row.tip_name,
                "control_tip": match.tip_name,
                "matching_scope": scope,
                "tree_distance": float(distance),
                "year_difference": abs(float(row.year) - float(match.year)) if pd.notna(row.year) and pd.notna(match.year) else np.nan,
                "lineage_id": row.lineage_id,
            }
            for feature in FOCAL_TERMS:
                out[f"{feature}_positive"] = getattr(row, feature)
                out[f"{feature}_control"] = getattr(match, feature)
                out[f"{feature}_diff"] = getattr(row, feature) - getattr(match, feature)
            rows.append(out)
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    species_rows = []
    for feature in FOCAL_TERMS:
        diffs = pd.to_numeric(pairs[f"{feature}_diff"], errors="coerce").dropna()
        if len(diffs) > 0:
            try:
                stat, p_value = wilcoxon(diffs)
            except ValueError:
                stat, p_value = np.nan, np.nan
        else:
            stat, p_value = np.nan, np.nan
        overall_rows.append(
            {
                "term": feature,
                "mean_difference": float(diffs.mean()),
                "median_difference": float(diffs.median()),
                "q25": float(diffs.quantile(0.25)),
                "q75": float(diffs.quantile(0.75)),
                "fraction_positive": float((diffs > 0).mean()) if len(diffs) else np.nan,
                "wilcoxon_p": float(p_value) if pd.notna(p_value) else np.nan,
                "n_pairs": len(diffs),
            }
        )
        for species, sub in pairs.groupby("species_label", sort=False):
            x = pd.to_numeric(sub[f"{feature}_diff"], errors="coerce").dropna()
            species_rows.append(
                {
                    "species_label": species,
                    "term": feature,
                    "mean_difference": float(x.mean()) if len(x) else np.nan,
                    "median_difference": float(x.median()) if len(x) else np.nan,
                    "fraction_positive": float((x > 0).mean()) if len(x) else np.nan,
                    "n_pairs": len(x),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(species_rows)


def main() -> None:
    ensure_output_dirs()
    set_plot_style()

    df, _, _ = build_base_dataframe()
    pairs = nearest_nonreassortant_pairs(df)
    overall, by_species = summarize_pairs(pairs)

    pairs.to_csv(REPORT_DIR / "03_within_lineage_pairs.tsv", sep="\t", index=False)
    overall.to_csv(REPORT_DIR / "03_within_lineage_summary.tsv", sep="\t", index=False)
    by_species.to_csv(REPORT_DIR / "03_within_lineage_by_species.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    scope_order = ["same_lineage_country", "same_lineage", "same_country", "species_fallback"]
    scope_counts = pairs["matching_scope"].value_counts().reindex(scope_order).fillna(0)
    axes[0].bar(scope_counts.index, scope_counts.values, color=["#d4f0c0", "#c6def1", "#f7d6b6", "#fde2e4"], edgecolor="#444444")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Matched reassortant tips")

    species_order = list(by_species["species_label"].drop_duplicates())
    y_positions = np.arange(len(species_order))
    offsets = np.linspace(-0.25, 0.25, num=len(FOCAL_TERMS))
    colors = ["#6d597a", "#577590", "#90be6d"]
    for offset, term, color in zip(offsets, FOCAL_TERMS, colors):
        sub = by_species[by_species["term"] == term].set_index("species_label").reindex(species_order).reset_index()
        axes[1].scatter(sub["mean_difference"], y_positions + offset, color=color, s=35, edgecolors="#333333", linewidths=0.3, label=DISPLAY_NAMES[term])
    axes[1].axvline(0, color="#888888", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(species_order)
    axes[1].set_xlabel("Positive minus matched-control difference")
    axes[1].legend(frameon=False, loc="best")

    fig.savefig(FIGURE_DIR / "03_within_lineage_contrasts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = (
        "# Within-Lineage Contrasts\n\n"
        f"- Matched reassortant tips: `{len(pairs)}`\n"
        f"- Same-lineage-country matches: `{int((pairs['matching_scope'] == 'same_lineage_country').sum())}`\n"
        f"- Same-lineage matches: `{int((pairs['matching_scope'] == 'same_lineage').sum())}`\n"
        f"- Same-country matches: `{int((pairs['matching_scope'] == 'same_country').sum())}`\n"
        f"- Species-fallback matches: `{int((pairs['matching_scope'] == 'species_fallback').sum())}`\n\n"
        + render_dataframe_markdown("Overall Pairwise Differences", overall)
        + "\n"
        + render_dataframe_markdown("Species-Level Mean Differences", by_species)
    )
    write_markdown(REPORT_DIR / "03_within_lineage_contrasts_report.md", report)


if __name__ == "__main__":
    main()
