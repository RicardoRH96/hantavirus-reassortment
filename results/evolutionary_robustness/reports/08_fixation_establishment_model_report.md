# Fixation-Inspired Establishment Model

This analysis treats de-nested ARG-derived reassortment-origin clusters as observed reassortment-permissive events and matched non-reassortant internal S-tree clades as local reassortment-resistant background states. It remains an establishment/payoff approximation rather than a full Moran simulation, but it now applies a minimal-origin rule, a compact pre-specified molecular-filter index, and rarefaction sensitivity for sampling bias.

## Design

- Candidate ARG origin clusters before de-nesting: `131`
- Retained minimal ARG origin clusters: `73`
- Nested candidate clusters removed: `58`
- Eligible non-reassortant internal background nodes: `188`
- Matched background rows used in hurdle model: `202`
- Controls per origin requested: `3`
- Species-specific lineage graph nodes: `143`
- Species-specific lineage graph edges: `2380`

De-nesting keeps the smallest ARG-supported origin cluster when a candidate origin is a strict superset of an already retained origin within the same species. The hurdle component models retained origin versus matched background status with species and lineage random effects. The count component models excess descendant tips among retained origins with a negative-binomial likelihood and a log lineage-level sampling-effort offset.

The definitive strategy model uses compact, pre-specified predictors rather than the full raw feature block: ecological opportunity, molecular permissiveness, their interaction, tree discordance, graph-neighborhood opportunity, and sampling effort. Molecular permissiveness is an oriented composite of lower LD/coadaptation burden, lower species-level UTR structural distance, and a direct terminal UTR regime that combines lower 5-prime length disparity with higher 3-prime length disparity. The individual `utrterm_*` workhorse feature means are still exported in the dataset for traceability.

- Establishment formula: `established ~ ecological_opportunity_z + molecular_permissiveness_z + ecology_x_molecular_permissiveness_z + tree_discordance_score_z + graph_neighbor_overlap_z + log_sampling_effort_z + (1|species) + (1|dominant_lineage_id)`

## Origin Clade Summaries By Species

| species_label | origin_n | median_descendant_tips | max_descendant_tips | median_temporal_span_years | median_country_spread | median_host_spread | median_terminal_utr5_disparity | median_terminal_utr3_disparity | median_terminal_utr_recovery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dobrava | 4 | 2.500 | 3 | 1 | 1 | 1 | 13.333 | 113 | 0.333 |
| Hantaan | 14 | 2 | 3 | 0 | 1 | 1 | 2.667 | 222 | 1 |
| Puumala | 14 | 1 | 3 | 0 | 1 | 1 | 10.333 | 293.333 | 0.917 |
| Seoul | 18 | 2 | 4 | 0 | 1 | 1 | 6.667 | 267.667 | 1 |
| Sin Nombre | 9 | 2 | 4 | 7 | 2 | 2 | 0 | 0 | 0 |
| Tula | 14 | 2 | 2 | 0 | 1 | 1 | 12.667 | 300.500 | 1 |

## Species Graph Summaries

| species_label | lineage_nodes | mean_weighted_degree | mean_neighbor_overlap | edges |
| --- | --- | --- | --- | --- |
| Andes | 4 | 2.095 | 26.572 | 6 |
| Dobrava | 4 | 0.689 | 22.618 | 6 |
| Hantaan | 51 | 29.708 | 30.418 | 1275 |
| Puumala | 19 | 9.683 | 27.770 | 171 |
| Seoul | 39 | 25.437 | 17.903 | 741 |
| Sin Nombre | 8 | 3.124 | 13.406 | 28 |
| Tula | 18 | 10.836 | 48.280 | 153 |

## Model Diagnostics

| model | max_rhat | min_ess_bulk | divergences |
| --- | --- | --- | --- |
| hurdle_establishment_bernoulli | 1.010 | 848 | 0 |
| count_expansion_negative_binomial | 1 | 2691 | 0 |

## Establishment Hurdle Posterior

| raw_term | mean | hdi_3% | hdi_97% | posterior_prob_positive |
| --- | --- | --- | --- | --- |
| ecological_opportunity | 0.509 | -0.334 | 1.430 | 0.866 |
| molecular_permissiveness | 0.631 | -0.577 | 1.867 | 0.833 |
| ecology_x_molecular_permissiveness | 0.866 | 0.004 | 1.826 | 0.966 |
| tree_discordance_score | 0.741 | -0.080 | 1.544 | 0.960 |
| graph_neighbor_overlap | -0.482 | -1.719 | 0.753 | 0.218 |
| log_sampling_effort | -0.670 | -1.440 | 0.109 | 0.054 |

## Expansion Count Posterior

| raw_term | mean | hdi_3% | hdi_97% | posterior_prob_positive |
| --- | --- | --- | --- | --- |
| ecological_opportunity | -0.165 | -0.529 | 0.235 | 0.209 |
| molecular_permissiveness | -0.165 | -0.480 | 0.138 | 0.152 |
| ecology_x_molecular_permissiveness | 0.192 | -0.161 | 0.520 | 0.856 |
| tree_discordance_score | -0.065 | -0.246 | 0.097 | 0.247 |
| graph_neighbor_overlap | -0.076 | -0.437 | 0.321 | 0.362 |

## Rarefaction Sensitivity

| rarefaction_target | target_sample_n | model | raw_term | coefficient_mean | sign_stability_positive | mean_established_origin_n | mean_origin_retention_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| species_q25 | 42 | rarefied_count_poisson_l2 | ecological_opportunity | 0.008 | 0.685 | 21.830 | 0.299 |
| species_q25 | 42 | rarefied_count_poisson_l2 | ecology_x_molecular_permissiveness | -0.006 | 0.485 | 21.830 | 0.299 |
| species_q25 | 42 | rarefied_count_poisson_l2 | graph_neighbor_overlap | -0.070 | 0 | 21.830 | 0.299 |
| species_q25 | 42 | rarefied_count_poisson_l2 | molecular_permissiveness | -0.062 | 0.030 | 21.830 | 0.299 |
| species_q25 | 42 | rarefied_count_poisson_l2 | tree_discordance_score | 0.010 | 0.990 | 21.830 | 0.299 |
| species_q25 | 42 | rarefied_hurdle_logistic_l2 | ecological_opportunity | 0.185 | 0.675 | 21.830 | 0.299 |
| species_q25 | 42 | rarefied_hurdle_logistic_l2 | ecology_x_molecular_permissiveness | 0.303 | 0.865 | 21.830 | 0.299 |
| species_q25 | 42 | rarefied_hurdle_logistic_l2 | graph_neighbor_overlap | -0.649 | 0.015 | 21.830 | 0.299 |
| species_q25 | 42 | rarefied_hurdle_logistic_l2 | molecular_permissiveness | -0.672 | 0 | 21.830 | 0.299 |
| species_q25 | 42 | rarefied_hurdle_logistic_l2 | tree_discordance_score | 0.429 | 1 | 21.830 | 0.299 |
| strict_species_min | 18 | rarefied_count_poisson_l2 | ecological_opportunity | 0.011 | 0.687 | 11.111 | 0.152 |
| strict_species_min | 18 | rarefied_count_poisson_l2 | ecology_x_molecular_permissiveness | -0.013 | 0.283 | 11.111 | 0.152 |
| strict_species_min | 18 | rarefied_count_poisson_l2 | graph_neighbor_overlap | -0.011 | 0.404 | 11.111 | 0.152 |
| strict_species_min | 18 | rarefied_count_poisson_l2 | molecular_permissiveness | -0.031 | 0.071 | 11.111 | 0.152 |
| strict_species_min | 18 | rarefied_count_poisson_l2 | tree_discordance_score | 0.025 | 0.960 | 11.111 | 0.152 |
| strict_species_min | 18 | rarefied_hurdle_logistic_l2 | ecological_opportunity | 0.275 | 0.660 | 9.615 | 0.132 |
| strict_species_min | 18 | rarefied_hurdle_logistic_l2 | ecology_x_molecular_permissiveness | 0.298 | 0.775 | 9.615 | 0.132 |
| strict_species_min | 18 | rarefied_hurdle_logistic_l2 | graph_neighbor_overlap | -1.245 | 0.050 | 9.615 | 0.132 |
| strict_species_min | 18 | rarefied_hurdle_logistic_l2 | molecular_permissiveness | -1.043 | 0.030 | 9.615 | 0.132 |
| strict_species_min | 18 | rarefied_hurdle_logistic_l2 | tree_discordance_score | 0.624 | 1 | 9.615 | 0.132 |

## Interpretation

This definitive pass is an establishment-oriented test of the ecological-opportunity x molecular-filter synthesis. The important target is the sign and stability of ecological opportunity, molecular permissiveness, and their interaction after minimal-origin de-nesting and rarefaction. A positive interaction means reassortment-prone regimes have their highest payoff when exposure opportunity and molecular permissiveness coincide; weak or negative effects imply reassortment-resistant regimes dominate under restrictive molecular backgrounds or low opportunity.

Sampling remains a limitation because descendant clade size is sampled expansion rather than true epidemiological expansion. The rarefaction sensitivity is therefore the main guardrail: the result is strongest when the compact terms retain their sign under strict common-sampling downsampling.
