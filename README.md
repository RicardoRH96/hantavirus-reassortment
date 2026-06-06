# hantavirus-reassortment

Code for The molecular and ecological determinants of effective reassortment in Hantaviruses.
Ricardo Rivero, David Simons, Lambodhar Damodaran, Irene Karegi, Sarah Gurev, Daniel J. Becker, Dan L. Warren, Nicola F. Müller, David Rasmussen, Stephanie N. Seifert.


## Layout

This repository contains the analysis code only. Input data, generated figures,
model outputs, and intermediate files are intentionally kept outside version
control.

```text
.
├── README.md
├── Scripts/
│   ├── buildFinalModelDataset.py
│   │   └── Builds the sequence-level table linking viral metadata, host taxonomy,
│   │       ecological predictors, and molecular features.
│   ├── TaxonomyReconciler/
│   │   └── Standardizes host and viral taxonomy, including hantavirus and
│   │       arenavirus classification utilities.
│   ├── ReassortmentInference/
│   │   └── Infers segment-tree discordance and reassortment histories from
│   │       S, M, and L segment phylogenies.
│   ├── SDM/
│   │   └── Builds and evaluates host species distribution models, habitat
│   │       overlap rasters, environmental predictors, and host occurrence checks.
│   ├── UTR/
│   │   └── Retrieves untranslated-region sequences and derives UTR-linked
│   │       molecular features for the viral segments.
│   ├── LinkageDissequilibrium/
│   │   └── Estimates intersegment linkage disequilibrium and extracts
│   │       segment-pair compatibility features.
│   ├── Epistasis/
│   │   └── Constructs and models intersegment epistatic features, including
│   │       comparisons with linkage disequilibrium signals.
│   └── EvolutionaryAnalysis/
│       └── Runs evolutionary robustness analyses for reassortment origins,
│           label uncertainty, lineage structure, phylogenetic nulls, sampling
│           sensitivity, establishment, and selection scans.
└── Notebooks/
    ├── Fig1.ipynb
    ├── Fig2.ipynb
    ├── Fig3.ipynb
    ├── Fig4.ipynb
    ├── Fig5.ipynb
    └── SDM_PDP_CircularPhylo.ipynb
```
