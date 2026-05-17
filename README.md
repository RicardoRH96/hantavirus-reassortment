# hantavirus-reassortment

Minimal analysis repository for the final hantavirus reassortment manuscript analyses.

This repo contains the organized scripts, notebooks, compact processed inputs, and final-result tables/figures needed to reproduce and inspect the manuscript-facing analyses. Large raw downloads, raster stacks, model caches, posterior NetCDF files, and old exploratory iterations are intentionally excluded.

## Layout

- `data/processed/final_sequence_dataset.tsv`: final sequence-level analysis table.
- `data/processed/labels/`: current and alternative reassortment label tables.
- `data/processed/trees/` and `data/processed/arg/`: compact tree and ARG inputs used by robustness analyses.
- `data/processed/sdm/`: compact SDM validation and host-list inputs used by the retained notebooks.
- `scripts/modeling/`: final model, benchmark, and hierarchical analysis scripts.
- `scripts/figures/`: manuscript and supplement figure-rendering scripts.
- `scripts/evolutionary_robustness/`: robustness analyses for label definitions, lineage dependence, phylogenetic nulls, and sampling sensitivity.
- `notebooks/`: retained notebooks for figure development, SDM checks, host-list verification, and NP-UTR rendering.
- `results/`: compact final outputs copied from the source analysis.

## Environment

Create the conda environment when available:

```bash
conda env create -f config/environment.yml
conda activate hantavirus-reassortment
```

For a pip-only setup:

```bash
pip install -r config/requirements.txt
```

## Main Entry Points

Render manuscript-style result figures:

```bash
python scripts/figures/plot_manuscript_style_result_figures.py
```

Build the integrated explanation bundle:

```bash
python scripts/modeling/build_integrated_explanation_bundle.py
```

Run the robustness synthesis after prerequisite robustness tables exist:

```bash
python scripts/evolutionary_robustness/07_manuscript_robustness_summary.py
```

See `docs/migration_manifest.md` for the migration scope and exclusions.
