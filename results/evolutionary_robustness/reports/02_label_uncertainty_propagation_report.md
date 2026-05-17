# Label Uncertainty Propagation

- Label sets evaluated: `5`

## Label-Set Agreement

| label_set | positive_n | agreement_with_current | positive_delta_vs_current |
| --- | --- | --- | --- |
| backfill_fast | 162 | 0.824 | -1 |
| backfill_full | 163 | 1 | 0 |
| current_main | 163 | 1 | 0 |
| distance_leq_1 | 113 | 0.903 | -50 |
| distance_leq_3 | 193 | 0.942 | 30 |

## Posterior Coefficient Ranges

| term | posterior_mean_min | posterior_mean_max | posterior_mean_median | posterior_prob_positive_min | posterior_prob_positive_max |
| --- | --- | --- | --- | --- | --- |
| overlap_opportunity_local | -0.043 | 0.299 | 0.286 | 0.390 | 0.995 |
| arha_prev_local | -0.317 | -0.083 | -0.184 | 0.020 | 0.325 |
| species_feat_hostdiv_faith_pd | -0.243 | -0.083 | -0.219 | 0.305 | 0.417 |
| species_feat_mean_cramers_v_overall | -0.619 | -0.447 | -0.552 | 0.195 | 0.270 |
| tree_discordance_score | 0.075 | 0.180 | 0.140 | 0.700 | 0.898 |
