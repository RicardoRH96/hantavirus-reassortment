from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evo_utils import (
    DISPLAY_NAMES,
    FIGURE_DIR,
    REPORT_DIR,
    render_dataframe_markdown,
    set_plot_style,
    write_markdown,
)


TERM_ORDER = [
    "overlap_opportunity_local_z",
    "arha_prev_local_z",
    "species_feat_hostdiv_faith_pd_z",
    "species_feat_mean_cramers_v_overall_z",
    "tree_discordance_score_z",
]
LOCAL_TERM_ORDER = [
    "overlap_opportunity_local",
    "arha_prev_local",
    "tree_discordance_score",
]
LABEL_SET_LABELS = {
    "current_main": "Current main",
    "backfill_full": "Backfill full",
    "backfill_fast": "Backfill fast",
    "distance_leq_1": "Distance <= 1",
    "distance_leq_3": "Distance <= 3",
}
TERM_LABELS = {f"{raw}_z": DISPLAY_NAMES[raw] for raw in ["overlap_opportunity_local", "arha_prev_local", "species_feat_hostdiv_faith_pd", "species_feat_mean_cramers_v_overall", "tree_discordance_score"]}


def panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")


def build_figure() -> None:
    summary01 = pd.read_csv(REPORT_DIR / "01_origin_cluster_bayesian_summary.tsv", sep="\t")
    coef02 = pd.read_csv(REPORT_DIR / "02_label_uncertainty_coefficients.tsv", sep="\t")
    pairs03 = pd.read_csv(REPORT_DIR / "03_within_lineage_pairs.tsv", sep="\t")
    summary04 = pd.read_csv(REPORT_DIR / "04_phylogenetic_null_summary.tsv", sep="\t")
    models05 = pd.read_csv(REPORT_DIR / "05_mechanism_ablation_models.tsv", sep="\t")
    summary06 = pd.read_csv(REPORT_DIR / "06_sampling_sensitivity_summary.tsv", sep="\t")

    fig, axes = plt.subplots(3, 2, figsize=(15.5, 13.0), constrained_layout=True)

    ax = axes[0, 0]
    panel_label(ax, "A")
    scheme_order = ["full_tip_model", "one_tip_per_origin"]
    offsets = {"full_tip_model": -0.12, "one_tip_per_origin": 0.12}
    colors = {"full_tip_model": "#6d597a", "one_tip_per_origin": "#577590"}
    y_positions = np.arange(len(TERM_ORDER))
    for scheme in scheme_order:
        sub = summary01[(summary01["model"] == scheme) & (summary01["term"].isin(TERM_ORDER))]
        for _, row in sub.iterrows():
            ypos = y_positions[TERM_ORDER.index(row["term"])] + offsets[scheme]
            ax.hlines(y=ypos, xmin=row["hdi_3%"], xmax=row["hdi_97%"], color=colors[scheme], linewidth=2.5)
            ax.scatter(row["mean"], ypos, color=colors[scheme], s=38, edgecolors="#333333", linewidths=0.35, label="Full tip Bayesian" if scheme == "full_tip_model" else "One tip per origin Bayesian")
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([TERM_LABELS[t] for t in TERM_ORDER])
    ax.set_xlabel("Posterior mean and 94% HDI")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="best")

    ax = axes[0, 1]
    panel_label(ax, "B")
    label_order = ["current_main", "backfill_full", "backfill_fast", "distance_leq_1", "distance_leq_3"]
    label_colors = {
        "current_main": "#6d597a",
        "backfill_full": "#577590",
        "backfill_fast": "#90be6d",
        "distance_leq_1": "#bc6c25",
        "distance_leq_3": "#b56576",
    }
    y_positions = np.arange(len(TERM_ORDER))
    for idx, label_name in enumerate(label_order):
        sub = coef02[(coef02["label_set"] == label_name) & (coef02["term"].isin(TERM_ORDER))]
        for _, row in sub.iterrows():
            ypos = y_positions[TERM_ORDER.index(row["term"])] + (idx - (len(label_order) - 1) / 2) * 0.09
            ax.hlines(y=ypos, xmin=row["hdi_3%"], xmax=row["hdi_97%"], color=label_colors[label_name], linewidth=1.6)
            ax.scatter(row["mean"], ypos, color=label_colors[label_name], s=28, edgecolors="#333333", linewidths=0.3, label=LABEL_SET_LABELS[label_name])
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([TERM_LABELS[t] for t in TERM_ORDER])
    ax.set_xlabel("Posterior mean and 94% HDI across label sets")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="best")

    ax = axes[1, 0]
    panel_label(ax, "C")
    box_data = [pd.to_numeric(pairs03[f"{term}_diff"], errors="coerce").dropna().to_numpy() for term in LOCAL_TERM_ORDER]
    bp = ax.boxplot(box_data, vert=False, patch_artist=True, widths=0.65, showfliers=False)
    box_colors = ["#d4f0c0", "#c6def1", "#f7d6b6"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color("#333333")
        median.set_linewidth(1.4)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(1, len(LOCAL_TERM_ORDER) + 1))
    ax.set_yticklabels([DISPLAY_NAMES[t] for t in LOCAL_TERM_ORDER])
    ax.set_xlabel("Reassortant minus matched-control difference")

    ax = axes[1, 1]
    panel_label(ax, "D")
    sub04 = summary04[summary04["term"].isin(LOCAL_TERM_ORDER)].set_index("term").loc[LOCAL_TERM_ORDER].reset_index()
    y_positions = np.arange(len(LOCAL_TERM_ORDER))
    ax.hlines(y=y_positions, xmin=sub04["null_q05"], xmax=sub04["null_q95"], color="#c6def1", linewidth=7)
    ax.scatter(sub04["null_mean"], y_positions, color="#577590", s=36, edgecolors="#333333", linewidths=0.3, label="Restricted null mean")
    ax.scatter(sub04["observed_mean_difference"], y_positions, color="#bc6c25", s=44, edgecolors="#333333", linewidths=0.35, label="Observed")
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([DISPLAY_NAMES[t] for t in LOCAL_TERM_ORDER])
    ax.set_xlabel("Mean matched difference")
    ax.legend(frameon=False, loc="best")

    ax = axes[2, 0]
    panel_label(ax, "E")
    model_order = ["lineage_only", "ecology_over_lineage", "molecular_over_lineage", "integrated_over_lineage"]
    label_map = {
        "lineage_only": "Lineage only",
        "ecology_over_lineage": "Ecology + lineage",
        "molecular_over_lineage": "Molecular + lineage",
        "integrated_over_lineage": "Integrated + lineage",
    }
    model_colors = ["#e2ece9", "#d4f0c0", "#f7d6b6", "#d8c6f2"]
    plot_df = models05[models05["model"].isin(model_order)].set_index("model").loc[model_order].reset_index()
    ax.bar([label_map[m] for m in plot_df["model"]], plot_df["delta_elpd_vs_lineage"].fillna(0), color=model_colors, edgecolor="#444444")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("Delta ELPD-LOO vs lineage-only")

    ax = axes[2, 1]
    panel_label(ax, "F")
    sub06 = summary06.copy()
    sub06["term_z"] = sub06["term"].map(lambda x: f"{x}_z")
    sub06 = sub06[sub06["term_z"].isin(TERM_ORDER)].set_index("term_z").loc[TERM_ORDER].reset_index()
    y_positions = np.arange(len(TERM_ORDER))
    ax.hlines(y=y_positions, xmin=sub06["posterior_mean_min"], xmax=sub06["posterior_mean_max"], color="#f2c6d8", linewidth=6)
    ax.scatter(sub06["posterior_mean_median"], y_positions, color="#6d597a", s=42, edgecolors="#333333", linewidths=0.35)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([TERM_LABELS[t] for t in TERM_ORDER])
    ax.set_xlabel("Posterior mean range across sampling scenarios")

    fig.savefig(FIGURE_DIR / "robustness_mechanism_summary_multi_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_report() -> None:
    summary01 = pd.read_csv(REPORT_DIR / "01_origin_cluster_bayesian_summary.tsv", sep="\t")
    agree02 = pd.read_csv(REPORT_DIR / "02_label_set_agreement.tsv", sep="\t")
    summary02 = pd.read_csv(REPORT_DIR / "02_label_uncertainty_summary.tsv", sep="\t")
    summary03 = pd.read_csv(REPORT_DIR / "03_within_lineage_summary.tsv", sep="\t")
    summary04 = pd.read_csv(REPORT_DIR / "04_phylogenetic_null_summary.tsv", sep="\t")
    models05 = pd.read_csv(REPORT_DIR / "05_mechanism_ablation_models.tsv", sep="\t")
    coef05 = pd.read_csv(REPORT_DIR / "05_mechanism_ablation_coefficients.tsv", sep="\t")
    summary06 = pd.read_csv(REPORT_DIR / "06_sampling_sensitivity_summary.tsv", sep="\t")

    text = textwrap.dedent(
        """
        # Robustness Synthesis

        This report consolidates the standalone robustness analyses in `specific_evolutionary_analyses`. The purpose of these analyses is not to replace the main tip-level hierarchical model in `model_test`, but to test whether the mechanistic interpretation survives three specific objections: non-independence among descendant tips, sensitivity to reassortment-label definition, and sensitivity to lineage structure or uneven sampling.

        ## Inputs

        The analyses read the direct 516-row mechanism dataset from `model_test/runs/model_site/datasets/sequence_level_final_dataset_combined_relaxed_andesense_all_backfills_full_geooverlap_manual_plus_tula_coords_direct516.tsv`, the current sequence-level true labels from `model_test/runs/model_sequence/datasets/sequence_true_labels.tsv`, alternative true-label builds from the species-specific and multi-species backfill runs, and the existing S-segment refined trees plus ARG outputs under `model_test/runs/*true_reassortment*`.

        ## Shared Preprocessing

        1. The working table was restricted to the 516 sequence-level observations already used in the direct mechanism dataset.
        2. The response variable was rebuilt from the current sequence-level true-label file and merged back onto the direct516 table by `species` and `tip_name`.
        3. Numeric predictors were standardized with global means and standard deviations estimated from the full 516-row table, and missingness indicators were added where a predictor contained missing values.
        4. The model-based robustness analyses were fit as Bayesian logistic models with Bambi/PyMC (`family="bernoulli"`). Fixed effects used weakly informative Normal priors (`common ~ Normal(0, 1.0)` and `Intercept ~ Normal(0, 1.5)`). Random intercepts were included for species or lineage where appropriate.
        5. No files under `model_test` were edited. All derived outputs were written under `specific_evolutionary_analyses/outputs`.

        ## Lineage Assignment

        Lineage IDs were defined on the pruned S-segment refined tree for each species. For each tip, the algorithm climbed from the terminal node toward the root and selected the first ancestral clade containing between 8 and 40 labeled descendant tips. If no such clade existed, it fell back to the nearest internal clade with more than one descendant tip. Each lineage was hashed from its descendant tip set to produce a stable `lineage_id`.

        ## Origin-Cluster Assignment

        Potential reassortment-origin clusters were derived from the current ARG reconstructions, not from the ecological `event_key`. For each recombination node in `arg_em.trees`, descendant tip sets were extracted across the S, M, and L local trees. Any recombination node with at least one positively labeled descendant contributed a candidate origin cluster keyed by the sorted set of positive descendants. Positive tips were then assigned to the smallest such positive-descendant cluster containing them; non-reassortant tips were kept as singleton non-event clusters. This clustering is used only as a dependence-robustness device.

        ## Analysis 1: Origin-Cluster Robustness

        The first analysis asked whether the key mechanistic coefficients change when reassortant descendants are not treated as fully independent.

        1. A Bayesian logistic model with a species random intercept and the key ecology/molecular predictors was fit to the full 516-tip table.
        2. A one-tip-per-origin Bayesian analysis sampled one positive representative from each inferred origin cluster and retained all non-reassortant tips.
        3. That one-tip-per-origin model was repeated across 8 representative-tip resamples.

        This analysis addresses pseudoreplication directly. If coefficient signs are preserved in the full-tip posterior and across representative-tip refits, then the main story is not being driven only by multi-tip descendant expansions.

        ## Analysis 2: Label-Uncertainty Propagation

        The second analysis asked whether the direction of the mechanistic effects depends strongly on the exact reassortment label definition.

        Five label sets were evaluated:

        1. The current main sequence-level labels.
        2. The multi-species backfill full label set plus species-specific Andes and Sin Nombre label files.
        3. The multi-species backfill fast label set plus species-specific Andes and Sin Nombre label files.
        4. A stricter distance-based recoding using `descriptive_distance_to_reassort_node <= 1`.
        5. A more permissive distance-based recoding using `descriptive_distance_to_reassort_node <= 3`.

        The same Bayesian logistic design, with a species random intercept, was refit for each label set. The goal here is not to declare one label set correct, but to ask whether the ecological and filtering terms keep the same qualitative direction as labels shift.

        ## Analysis 3: Within-Lineage Contrasts

        This analysis was rewritten to focus only on predictors that can vary within a lineage neighborhood: local overlap opportunity, local prevalence, and tree discordance. Species-level terms such as host phylogenetic diversity and LD burden do not vary meaningfully within a lineage-matched comparison and were therefore excluded from the refined lineage-local analysis.

        For each reassortant tip, the best non-reassortant control was selected under the following priority order:

        1. Same lineage and same country.
        2. Same lineage.
        3. Same country within species.
        4. Any non-reassortant tip within species.

        Within the chosen candidate pool, ties were broken by smallest absolute year difference and then smallest tree distance on the pruned S tree. Paired differences were summarized overall and by species, and Wilcoxon signed-rank tests were applied to the overall paired differences.

        ## Analysis 4: Restricted Phylogenetic Nulls

        The null analysis uses the same restricted candidate pools as Analysis 3. For each reassortant tip, the observed matched-control difference was compared with a null in which one random control was repeatedly drawn from the same eligible pool. This preserves species composition, lineage neighborhood, and country restriction where available. The resulting null distribution therefore asks whether the nearest matched control looks more extreme than expected under random control draws from the same local phylogenetic context.

        ## Analysis 5: Mechanism Ablation After Lineage Absorption

        This analysis asked whether ecology and molecular blocks still improve fit once lineage structure is already encoded. Five Bayesian logistic models were fit:

        1. Intercept only.
        2. Lineage only.
        3. Ecology plus lineage.
        4. Molecular plus lineage.
        5. Integrated ecology and molecular predictors plus lineage.

        Improvement was summarized as delta ELPD-LOO relative to the lineage-only model.

        ## Analysis 6: Sampling Sensitivity

        The final analysis tested whether the main coefficients were stable under common sampling criticisms. The same Bayesian logistic design with a species random intercept was refit under 16 scenarios:

        1. Full dataset.
        2. One sequence per `site_year_host_key`.
        3. One positive tip per inferred origin cluster.
        4. At most five tips per lineage-by-response cell.
        5. Leave-one-species-out for each species.
        6. Leave-one-country-out for the five most represented countries.

        Posterior mean ranges and sign consistency were then summarized across scenarios.

        ## Results
        """
    ).strip() + "\n\n"

    text += render_dataframe_markdown(
        "Origin-Cluster Robustness",
        summary01[summary01["term"].isin(TERM_ORDER)][["model", "term", "mean", "hdi_3%", "hdi_97%", "posterior_prob_positive", "replicate_n"]],
    )
    text += "\n"
    text += render_dataframe_markdown("Label-Set Agreement", agree02)
    text += "\n"
    text += render_dataframe_markdown("Label-Uncertainty Coefficient Ranges", summary02)
    text += "\n"
    text += render_dataframe_markdown("Within-Lineage Paired Differences", summary03[summary03["term"].isin(LOCAL_TERM_ORDER)])
    text += "\n"
    text += render_dataframe_markdown("Restricted Phylogenetic Nulls", summary04[summary04["term"].isin(LOCAL_TERM_ORDER)])
    text += "\n"
    text += render_dataframe_markdown("Mechanism Ablation", models05)
    text += "\n"
    text += render_dataframe_markdown(
        "Integrated-Over-Lineage Coefficients",
        coef05[(coef05["model"] == "integrated_over_lineage") & (coef05["term"].isin(TERM_ORDER))][["model", "term", "mean", "hdi_3%", "hdi_97%", "posterior_prob_positive"]],
    )
    text += "\n"
    text += render_dataframe_markdown("Sampling Sensitivity Summary", summary06)

    interpretation = textwrap.dedent(
        """

        ## Interpretation

        The combined robustness picture is strongest for three points.

        1. Local overlap remains positive under origin clustering, across label-set perturbations, after lineage absorption, and across sampling scenarios.
        2. Local prevalence remains negative in the global Bayesian robustness analyses and stays negative after origin clustering and after lineage absorption.
        3. Molecular filtering terms, especially LD burden, remain negative when the design still allows them to vary meaningfully, and the integrated-over-lineage model outperforms ecology-only and molecular-only alternatives.

        The refined within-lineage and restricted-null analyses are intentionally narrower. They are designed to stress-test only lineage-local variables. They are not appropriate for species-level predictors whose variation is largely between species rather than within matched lineage neighborhoods. In other words, those analyses are best interpreted as local-opportunity checks, not as global mechanism replacements.

        ## Files Produced

        - Figure: `specific_evolutionary_analyses/outputs/figures/robustness_mechanism_summary_multi_panel.png`
        - Report: `specific_evolutionary_analyses/outputs/reports/robustness_synthesis_report.md`
        """
    ).strip() + "\n"
    text += interpretation

    write_markdown(REPORT_DIR / "robustness_synthesis_report.md", text)


def main() -> None:
    set_plot_style()
    build_figure()
    build_report()


if __name__ == "__main__":
    main()
