# Sequence Explanatory Model Suite

## Dataset
- Dataset: data/processed/final_sequence_dataset.tsv
- Labeled rows: 516
- Positives: 163
- Positive rate: 0.316
- Validation schemes: repeated_stratified
- Species groups evaluated: 7
- Virus clades evaluated: 3
- Target filter: true_label
- LD encoding: full

## Bayesian Status
- PyMC: available
- Bambi: available
- ArviZ: available
- statsmodels: available
- Fallback: species-effect reduction is reported with regularized fixed-effect logistic models.

## Primary Grouped Report
- Best overall model: repeated_stratified / ecology_molecular_lineage_plus_context / lightgbm
- OOF average precision: 0.812
- OOF ROC AUC: 0.885
- OOF balanced accuracy: 0.791
- AP lift vs prevalence: 2.57x

## Best Model Per CV Scheme
- repeated_stratified: ecology_molecular_lineage_plus_context / lightgbm with AP=0.812, ROC AUC=0.885

## Feature Stacks
- ecology_molecular_lineage_plus_context: Lineage-focused local ecology and direct molecular signals plus geography, time, provenance, and sampling context. (109 numeric, 15 categorical)

## Best Model Per Stack
- repeated_stratified / ecology_molecular_lineage_plus_context: lightgbm with AP=0.812, ROC AUC=0.885

## Flat Probability Diagnostics
- No species/model combinations were perfectly flat.

## Species-Effect Reduction Fallback
- logistic_elasticnet: L2 reduction=1.000, mean-abs reduction=1.000
- logistic_l2: L2 reduction=0.975, mean-abs reduction=0.980

## Top Effects
- ecology_molecular_lineage_plus_context / lightgbm
  - num__discord_ML_nn_patristic_absdiff: 421.0000 (positive)
  - num__discord_SM_profile_absz: 326.0000 (positive)
  - num__discord_SL_nn_patristic_absdiff: 321.0000 (positive)
  - num__discord_SM_nn_patristic_absdiff: 310.0000 (positive)
  - num__discord_nn_patristic_S: 300.0000 (positive)
  - num__discord_nn_patristic_mean: 296.0000 (positive)
  - num__discord_nn_patristic_L: 265.0000 (positive)
  - num__discord_ML_profile_corr: 253.0000 (positive)
