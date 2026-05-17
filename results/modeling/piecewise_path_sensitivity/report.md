# Piecewise Path Sensitivity

## Data Structure
- Sequence rows: 516
- Events: 161
- Species: 7
- Events with mixed positive/negative sequence labels: 36

## Event-Level Path
Discordance model:
- `overlap_mean_z`: beta=-0.101, 95% CI=(-0.230, 0.027), p=0.1218
- `prev_mean_z`: beta=-0.109, 95% CI=(-0.218, 0.001), p=0.0514

Any-positive reassortment model:
- `discord_mean_z`: beta=-0.465, OR=0.628, 95% CI OR=(0.301, 1.311), p=0.2156
- `overlap_mean_z`: beta=0.151, OR=1.163, 95% CI OR=(0.724, 1.868), p=0.5327
- `prev_mean_z`: beta=-0.417, OR=0.659, 95% CI OR=(0.362, 1.199), p=0.1716

Weighted positive-fraction sensitivity:
- `discord_mean_z`: beta=0.032, OR=1.032, 95% CI OR=(0.740, 1.441), p=0.8523
- `overlap_mean_z`: beta=0.320, OR=1.377, 95% CI OR=(1.074, 1.766), p=0.0118
- `prev_mean_z`: beta=-0.375, OR=0.687, 95% CI OR=(0.364, 1.295), p=0.2459

## Sequence-Level Clustered Sensitivity
Discordance model clustered by event:
- `overlap_opportunity_local_z`: beta=-0.099, 95% CI=(-0.205, 0.007), p=0.0669
- `arha_prev_local_z`: beta=-0.078, 95% CI=(-0.150, -0.006), p=0.0347

Reassortment model clustered by event:
- `tree_discordance_score_z`: beta=0.091, OR=1.095, 95% CI OR=(0.785, 1.527), p=0.5937
- `overlap_opportunity_local_z`: beta=0.277, OR=1.319, 95% CI OR=(0.860, 2.022), p=0.2039
- `arha_prev_local_z`: beta=-0.158, OR=0.854, 95% CI OR=(0.530, 1.376), p=0.5155

## Indirect Effect
- Approximate event-level indirect effect `overlap -> discordance -> reassortment` on the log-odds scale: 0.047

## Interpretation
- Overlap is the most stable positive ecology term across the event-aware models.
- Discordance is positive at sequence level, but weak or unstable after event aggregation.
- Local prevalence tends negative in the event-aware outcome models.
- The path story is therefore modest: overlap has the clearest direct signal; mediation through discordance is weak.
