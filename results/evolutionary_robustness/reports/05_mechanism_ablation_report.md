# Mechanism Ablation

- Tips analyzed: `516`
- Unique lineage groups: `143`

## Model Comparison

| model | formula | elpd_loo | p_loo | se | delta_elpd_vs_lineage |
| --- | --- | --- | --- | --- | --- |
| intercept_only | response ~ 1 | -322.715 | 0.867 | 8.152 |  |
| lineage_only | response ~ (1|lineage_id) | -227.504 | 60.372 | 11.333 | 0 |
| ecology_over_lineage | response ~ overlap_opportunity_local_z + arha_prev_local_z + arha_prev_local_missing + species_feat_hostdiv_faith_pd_z + (1|lineage_id) | -227.142 | 61.323 | 11.247 | 0.362 |
| molecular_over_lineage | response ~ species_feat_mean_cramers_v_overall_z + species_feat_prop_significant_overall_z + species_feat_utr5_M_structdist_mean_z + species_feat_utr5_L_structdist_mean_z + tree_discordance_score_z + (1|lineage_id) | -225.766 | 60.549 | 11.430 | 1.738 |
| integrated_over_lineage | response ~ overlap_opportunity_local_z + arha_prev_local_z + arha_prev_local_missing + species_feat_hostdiv_faith_pd_z + species_feat_mean_cramers_v_overall_z + species_feat_prop_significant_overall_z + species_feat_utr5_M_structdist_mean_z + species_feat_utr5_L_structdist_mean_z + tree_discordance_score_z + (1|lineage_id) | -227.021 | 62.493 | 11.672 | 0.483 |

## Integrated-Over-Lineage Posterior

| model | raw_term | mean | hdi_3% | hdi_97% | posterior_prob_positive |
| --- | --- | --- | --- | --- | --- |
| integrated_over_lineage | overlap_opportunity_local | 0.249 | -0.185 | 0.704 | 0.853 |
| integrated_over_lineage | arha_prev_local | -0.185 | -0.663 | 0.261 | 0.238 |
| integrated_over_lineage | species_feat_hostdiv_faith_pd | -0.670 | -1.357 | 0.053 | 0.037 |
| integrated_over_lineage | species_feat_mean_cramers_v_overall | -0.588 | -1.681 | 0.552 | 0.167 |
| integrated_over_lineage | tree_discordance_score | 0.332 | -0.147 | 0.817 | 0.900 |
