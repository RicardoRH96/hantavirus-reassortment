# Origin-Cluster Robustness

- Tips analyzed: `516`
- Reassortant tips: `163`
- Inferred positive origin clusters: `104`
- Median positive tips per origin cluster: `1.0`
- One-tip-per-origin Bayesian replicates: `8`

- Full-tip formula: `response ~ overlap_opportunity_local_z + arha_prev_local_z + arha_prev_local_missing + species_feat_hostdiv_faith_pd_z + species_feat_mean_cramers_v_overall_z + species_feat_prop_significant_overall_z + species_feat_utr5_M_structdist_mean_z + species_feat_utr5_L_structdist_mean_z + tree_discordance_score_z + (1|species)`

## Posterior Stability Summary

| model | raw_term | mean | hdi_3% | hdi_97% | posterior_prob_positive | replicate_n |
| --- | --- | --- | --- | --- | --- | --- |
| full_tip_model | overlap_opportunity_local | 0.295 | 0.074 | 0.531 | 0.989 | 1 |
| full_tip_model | arha_prev_local | -0.172 | -0.427 | 0.139 | 0.117 | 1 |
| full_tip_model | species_feat_hostdiv_faith_pd | -0.218 | -1.425 | 1.155 | 0.349 | 1 |
| full_tip_model | species_feat_mean_cramers_v_overall | -0.525 | -1.691 | 0.718 | 0.210 | 1 |
| full_tip_model | tree_discordance_score | 0.128 | -0.146 | 0.427 | 0.821 | 1 |
| one_tip_per_origin | overlap_opportunity_local | 0.227 | 0.177 | 0.275 | 1 | 8 |
| one_tip_per_origin | arha_prev_local | -0.179 | -0.210 | -0.151 | 0 | 8 |
| one_tip_per_origin | species_feat_hostdiv_faith_pd | -0.336 | -0.396 | -0.243 | 0 | 8 |
| one_tip_per_origin | species_feat_mean_cramers_v_overall | -0.596 | -0.650 | -0.544 | 0 | 8 |
| one_tip_per_origin | tree_discordance_score | 0.202 | 0.172 | 0.244 | 1 | 8 |
