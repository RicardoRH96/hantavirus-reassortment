# Sequence Explanatory Model Suite

## Dataset
- Dataset: data/processed/final_sequence_dataset.tsv
- Labeled rows: 516
- Positives: 163
- Positive rate: 0.316
- Validation schemes: grouped_within_species_clade
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
- Best overall model: grouped_within_species_clade / ecology_molecular / xgboost
- OOF average precision: 0.504
- OOF ROC AUC: 0.702
- OOF balanced accuracy: 0.648
- AP lift vs prevalence: 1.59x

## Best Model Per CV Scheme
- grouped_within_species_clade: ecology_molecular / xgboost with AP=0.504, ROC AUC=0.702

## Feature Stacks
- ecology_molecular: Combined ecology, molecular, and control signals without species identity. (218 numeric, 69 categorical)

## Best Model Per Stack
- grouped_within_species_clade / ecology_molecular: xgboost with AP=0.504, ROC AUC=0.702

## Flat Probability Diagnostics
- No species/model combinations were perfectly flat.

## Species-Effect Reduction Fallback
- logistic_elasticnet: L2 reduction=1.000, mean-abs reduction=1.000
- logistic_l2: L2 reduction=0.975, mean-abs reduction=0.980

## Top Effects
- ecology_molecular / xgboost
  - num__species_feat_utr5_L_retention_rate: 0.0708 (positive)
  - num__species_feat_utr3_M_length_median: 0.0640 (positive)
  - num__species_feat_utr3_M_length_mean: 0.0423 (positive)
  - cat__ld_site_S_933_M_2701_TA: 0.0212 (positive)
  - num__species_feat_utr5_M_structdist_std: 0.0198 (positive)
  - cat__hosttrait_combine_biogeographical_realm_Palearctic: 0.0198 (positive)
  - num__hosttrait_combine_density_n_km2: 0.0183 (positive)
  - num__hosttrait_combine_weaning_age_d: 0.0179 (positive)
