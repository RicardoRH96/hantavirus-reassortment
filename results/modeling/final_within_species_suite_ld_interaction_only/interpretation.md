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
- LD encoding: interaction_only

## Bayesian Status
- PyMC: available
- Bambi: available
- ArviZ: available
- statsmodels: available
- Fallback: species-effect reduction is reported with regularized fixed-effect logistic models.

## Primary Grouped Report
- Best overall model: grouped_within_species_clade / ecology_molecular / xgboost
- OOF average precision: 0.486
- OOF ROC AUC: 0.685
- OOF balanced accuracy: 0.644
- AP lift vs prevalence: 1.54x

## Best Model Per CV Scheme
- grouped_within_species_clade: ecology_molecular / xgboost with AP=0.486, ROC AUC=0.685

## Feature Stacks
- ecology_molecular: Combined ecology, molecular, and control signals without species identity. (218 numeric, 40 categorical)

## Best Model Per Stack
- grouped_within_species_clade / ecology_molecular: xgboost with AP=0.486, ROC AUC=0.685

## Flat Probability Diagnostics
- grouped_within_species_clade / ecology_molecular / xgboost / orthohantavirus_tulaense::clade_3 / orthohantavirus_tulaense: flat predictions across 4 held-out rows

## Species-Effect Reduction Fallback
- logistic_elasticnet: L2 reduction=1.000, mean-abs reduction=1.000
- logistic_l2: L2 reduction=0.973, mean-abs reduction=0.979

## Top Effects
- ecology_molecular / xgboost
  - num__species_feat_utr5_L_retention_rate: 0.0942 (positive)
  - num__species_feat_utr3_M_length_mean: 0.0817 (positive)
  - num__species_feat_utr5_M_structdist_std: 0.0340 (positive)
  - num__hosttrait_combine_dispersal_km: 0.0246 (positive)
  - num__hosttrait_combine_altitude_breadth_m: 0.0166 (positive)
  - num__species_feat_n_site_pairs_SM: 0.0158 (positive)
  - cat__ld_site_S_531_L_6219_TA: 0.0143 (positive)
  - num__hosttrait_combine_det_diet_breadth_n: 0.0129 (positive)
