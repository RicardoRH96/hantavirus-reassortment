# Migration Manifest

Migration date: 2026-05-17

Source workspace: local manuscript analysis workspace

Target repository: `hantavirus-reassortment`

## Included

- Final modeling scripts and manuscript-facing model scripts, moved under `scripts/modeling/`.
- Manuscript and supplement figure scripts, moved under `scripts/figures/`.
- Evolutionary robustness scripts, moved under `scripts/evolutionary_robustness/`.
- Final sequence-level dataset, current and alternative label tables, compact tree inputs, and compact ARG outputs under `data/processed/`.
- Compact SDM validation tables, host-list inputs, and AEW43003 NP-UTR render inputs needed by retained notebooks.
- Compact final modeling, robustness, and figure outputs under `results/`.
- Retained notebooks with outputs stripped and paths rewritten to the new repository layout.

## Excluded

- Raw `Data/` bulk directories, large rasters, GBIF archives, downloaded GeoJSON collections, and raw model scratch space.
- Old exploratory scripts and notebooks not part of the final manuscript analysis set.
- Large posterior NetCDF files and model binary artifacts.
- Run directories and outputs carrying scratch names from intermediate analysis iterations.

## Naming Cleanup

Copied files were renamed or path-sanitized so scripts and manifests refer to the final repository layout:

- Final sequence dataset: `data/processed/final_sequence_dataset.tsv`.
- Repeated stratified suite: `results/modeling/final_repeated_stratified_suite/`.
- Within-species suite: `results/modeling/final_within_species_suite/`.
- Species suite: `results/modeling/final_species_suite/`.
- Hierarchical block comparison: `results/modeling/final_hierarchical_block_comparison/`.
- Evolutionary robustness outputs: `results/evolutionary_robustness/`.

## Verification

The migration was checked for:

- no files larger than 20 MB;
- no source-repo absolute paths;
- no old scratch dataset labels in scripts, notebooks, or result manifests;
- no figure title calls in retained scripts and notebooks;
- valid notebook JSON.
