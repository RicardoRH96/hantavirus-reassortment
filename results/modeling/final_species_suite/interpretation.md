# Sequence Explanatory Model Suite

## Dataset
- Dataset: data/processed/final_sequence_dataset.tsv
- Labeled rows: 516
- Positives: 163
- Positive rate: 0.316
- Validation schemes: grouped_species
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
- Best overall model: grouped_species / species_only / random_forest_shallow
- OOF average precision: 0.424
- OOF ROC AUC: 0.668
- OOF balanced accuracy: 0.500
- AP lift vs prevalence: 1.34x

## Best Model Per CV Scheme
- grouped_species: species_only / random_forest_shallow with AP=0.424, ROC AUC=0.668

## Feature Stacks
- species_only: Taxon-only baseline using the held-out species label. (0 numeric, 1 categorical)

## Best Model Per Stack
- grouped_species / species_only: random_forest_shallow with AP=0.424, ROC AUC=0.668

## Flat Probability Diagnostics
- grouped_species / species_only / random_forest_shallow / orthohantavirus_andesense / orthohantavirus_andesense: flat predictions across 24 held-out rows
- grouped_species / species_only / random_forest_shallow / orthohantavirus_dobravaense / orthohantavirus_dobravaense: flat predictions across 18 held-out rows
- grouped_species / species_only / random_forest_shallow / orthohantavirus_hantanense / orthohantavirus_hantanense: flat predictions across 199 held-out rows
- grouped_species / species_only / random_forest_shallow / orthohantavirus_puumalaense / orthohantavirus_puumalaense: flat predictions across 69 held-out rows
- grouped_species / species_only / random_forest_shallow / orthohantavirus_seoulense / orthohantavirus_seoulense: flat predictions across 101 held-out rows
- grouped_species / species_only / random_forest_shallow / orthohantavirus_sinnombreense / orthohantavirus_sinnombreense: flat predictions across 34 held-out rows
- grouped_species / species_only / random_forest_shallow / orthohantavirus_tulaense / orthohantavirus_tulaense: flat predictions across 71 held-out rows

## Species-Effect Reduction Fallback
- logistic_elasticnet: L2 reduction=1.000, mean-abs reduction=1.000
- logistic_l2: L2 reduction=0.975, mean-abs reduction=0.980

## Top Effects
- species_only / random_forest_shallow
  - cat__species_orthohantavirus_hantanense: 0.4670 (positive)
  - cat__species_orthohantavirus_andesense: 0.2502 (positive)
  - cat__species_orthohantavirus_seoulense: 0.0933 (positive)
  - cat__species_orthohantavirus_sinnombreense: 0.0725 (positive)
  - cat__species_orthohantavirus_dobravaense: 0.0570 (positive)
  - cat__species_orthohantavirus_puumalaense: 0.0340 (positive)
  - cat__species_orthohantavirus_tulaense: 0.0260 (positive)
