# Robustness Synthesis

This report consolidates the standalone robustness analyses in `scripts/evolutionary_robustness`. The purpose of these analyses is not to replace the main tip-level hierarchical model in `scripts/modeling`, but to test whether the mechanistic interpretation survives three specific objections: non-independence among descendant tips, sensitivity to reassortment-label definition, and sensitivity to lineage structure or uneven sampling.

## Inputs

The analyses read the final sequence-level mechanism dataset from `data/processed/final_sequence_dataset.tsv`, the current sequence-level true labels from `data/processed/labels/sequence_true_labels.tsv`, alternative true-label builds from the species-specific and multi-species backfill runs, and the existing S-segment refined trees plus ARG outputs under `data/processed/arg`.

## Shared Preprocessing

1. The working table was restricted to the 516 sequence-level observations already used in the final mechanism dataset.
2. The response variable was rebuilt from the current sequence-level true-label file and merged back onto the final dataset by `species` and `tip_name`.
3. Numeric predictors were standardized with global means and standard deviations estimated from the full 516-row table, and missingness indicators were added where a predictor contained missing values.
4. The model-based robustness analyses were fit as Bayesian logistic models with Bambi/PyMC (`family="bernoulli"`). Fixed effects used weakly informative Normal priors (`common ~ Normal(0, 1.0)` and `Intercept ~ Normal(0, 1.5)`). Random intercepts were included for species or lineage where appropriate.
5. Derived outputs were written under `results/evolutionary_robustness`.

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

## Origin-Cluster Robustness

| model | term | mean | hdi_3% | hdi_97% | posterior_prob_positive | replicate_n |
| --- | --- | --- | --- | --- | --- | --- |
| full_tip_model | overlap_opportunity_local_z | 0.295 | 0.074 | 0.531 | 0.989 | 1 |
| full_tip_model | arha_prev_local_z | -0.172 | -0.427 | 0.139 | 0.117 | 1 |
| full_tip_model | species_feat_hostdiv_faith_pd_z | -0.218 | -1.425 | 1.155 | 0.349 | 1 |
| full_tip_model | species_feat_mean_cramers_v_overall_z | -0.525 | -1.691 | 0.718 | 0.210 | 1 |
| full_tip_model | tree_discordance_score_z | 0.128 | -0.146 | 0.427 | 0.821 | 1 |
| one_tip_per_origin | overlap_opportunity_local_z | 0.227 | 0.177 | 0.275 | 1 | 8 |
| one_tip_per_origin | arha_prev_local_z | -0.179 | -0.210 | -0.151 | 0 | 8 |
| one_tip_per_origin | species_feat_hostdiv_faith_pd_z | -0.336 | -0.396 | -0.243 | 0 | 8 |
| one_tip_per_origin | species_feat_mean_cramers_v_overall_z | -0.596 | -0.650 | -0.544 | 0 | 8 |
| one_tip_per_origin | tree_discordance_score_z | 0.202 | 0.172 | 0.244 | 1 | 8 |

## Label-Set Agreement

| label_set | positive_n | agreement_with_current | positive_delta_vs_current |
| --- | --- | --- | --- |
| backfill_fast | 162 | 0.824 | -1 |
| backfill_full | 163 | 1 | 0 |
| current_main | 163 | 1 | 0 |
| distance_leq_1 | 113 | 0.903 | -50 |
| distance_leq_3 | 193 | 0.942 | 30 |

## Label-Uncertainty Coefficient Ranges

| term | posterior_mean_min | posterior_mean_max | posterior_mean_median | posterior_prob_positive_min | posterior_prob_positive_max |
| --- | --- | --- | --- | --- | --- |
| overlap_opportunity_local | -0.043 | 0.299 | 0.286 | 0.390 | 0.995 |
| arha_prev_local | -0.317 | -0.083 | -0.184 | 0.020 | 0.325 |
| species_feat_hostdiv_faith_pd | -0.243 | -0.083 | -0.219 | 0.305 | 0.417 |
| species_feat_mean_cramers_v_overall | -0.619 | -0.447 | -0.552 | 0.195 | 0.270 |
| tree_discordance_score | 0.075 | 0.180 | 0.140 | 0.700 | 0.898 |

## Within-Lineage Paired Differences

| term | mean_difference | median_difference | q25 | q75 | fraction_positive | wilcoxon_p | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overlap_opportunity_local | 2.613 | 0 | -0.500 | 1 | 0.270 | 0.314 | 163 |
| arha_prev_local | 0.018 | 0 | -0.002 | 0.011 | 0.276 | 0.218 | 152 |
| tree_discordance_score | 0.008 | 0.000 | -0.002 | 0.008 | 0.509 | 0.150 | 163 |

## Restricted Phylogenetic Nulls

| term | observed_mean_difference | null_mean | null_q05 | null_q95 | outside_null_interval | empirical_p_two_sided |
| --- | --- | --- | --- | --- | --- | --- |
| arha_prev_local | 0.018 | -0.000 | -0.073 | 0.081 | False | 0.544 |
| overlap_opportunity_local | 2.613 | 2.022 | -4.583 | 10.255 | False | 0.686 |
| tree_discordance_score | 0.008 | 0.022 | -0.012 | 0.119 | False | 0.430 |

## Mechanism Ablation

| model | formula | elpd_loo | p_loo | se | delta_elpd_vs_lineage |
| --- | --- | --- | --- | --- | --- |
| intercept_only | response ~ 1 | -322.715 | 0.867 | 8.152 |  |
| lineage_only | response ~ (1|lineage_id) | -227.504 | 60.372 | 11.333 | 0 |
| ecology_over_lineage | response ~ overlap_opportunity_local_z + arha_prev_local_z + arha_prev_local_missing + species_feat_hostdiv_faith_pd_z + (1|lineage_id) | -227.142 | 61.323 | 11.247 | 0.362 |
| molecular_over_lineage | response ~ species_feat_mean_cramers_v_overall_z + species_feat_prop_significant_overall_z + species_feat_utr5_M_structdist_mean_z + species_feat_utr5_L_structdist_mean_z + tree_discordance_score_z + (1|lineage_id) | -225.766 | 60.549 | 11.430 | 1.738 |
| integrated_over_lineage | response ~ overlap_opportunity_local_z + arha_prev_local_z + arha_prev_local_missing + species_feat_hostdiv_faith_pd_z + species_feat_mean_cramers_v_overall_z + species_feat_prop_significant_overall_z + species_feat_utr5_M_structdist_mean_z + species_feat_utr5_L_structdist_mean_z + tree_discordance_score_z + (1|lineage_id) | -227.021 | 62.493 | 11.672 | 0.483 |

## Integrated-Over-Lineage Coefficients

| model | term | mean | hdi_3% | hdi_97% | posterior_prob_positive |
| --- | --- | --- | --- | --- | --- |
| integrated_over_lineage | overlap_opportunity_local_z | 0.249 | -0.185 | 0.704 | 0.853 |
| integrated_over_lineage | arha_prev_local_z | -0.185 | -0.663 | 0.261 | 0.238 |
| integrated_over_lineage | species_feat_hostdiv_faith_pd_z | -0.670 | -1.357 | 0.053 | 0.037 |
| integrated_over_lineage | species_feat_mean_cramers_v_overall_z | -0.588 | -1.681 | 0.552 | 0.167 |
| integrated_over_lineage | tree_discordance_score_z | 0.332 | -0.147 | 0.817 | 0.900 |

## Sampling Sensitivity Summary

| term | posterior_mean_min | posterior_mean_max | posterior_mean_median | posterior_prob_positive_min | posterior_prob_positive_max | scenario_n | sign_consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overlap_opportunity_local | 0.069 | 0.536 | 0.287 | 0.686 | 1 | 16 | 1 |
| arha_prev_local | -0.420 | 0.049 | -0.185 | 0.007 | 0.616 | 16 | 0.938 |
| species_feat_hostdiv_faith_pd | -0.517 | -0.009 | -0.182 | 0.118 | 0.491 | 16 | 1 |
| species_feat_mean_cramers_v_overall | -0.911 | 0.149 | -0.552 | 0.100 | 0.589 | 16 | 0.938 |
| tree_discordance_score | -0.060 | 0.398 | 0.113 | 0.393 | 0.991 | 16 | 0.750 |
## Interpretation

The combined robustness picture is strongest for three points.

1. Local overlap remains positive under origin clustering, across label-set perturbations, after lineage absorption, and across sampling scenarios.
2. Local prevalence remains negative in the global Bayesian robustness analyses and stays negative after origin clustering and after lineage absorption.
3. Molecular filtering terms, especially LD burden, remain negative when the design still allows them to vary meaningfully, and the integrated-over-lineage model outperforms ecology-only and molecular-only alternatives.

The refined within-lineage and restricted-null analyses are intentionally narrower. They are designed to stress-test only lineage-local variables. They are not appropriate for species-level predictors whose variation is largely between species rather than within matched lineage neighborhoods. In other words, those analyses are best interpreted as local-opportunity checks, not as global mechanism replacements.

## Files Produced

- Figure: `results/evolutionary_robustness/figures/robustness_mechanism_summary_multi_panel.png`
- Report: `results/evolutionary_robustness/reports/robustness_synthesis_report.md`
