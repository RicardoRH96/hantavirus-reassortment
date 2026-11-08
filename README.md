# Molecular and ecological determinants of effective reassortment in hantaviruses

Analysis code accompanying:

> Ricardo Rivero, David Simons, Lambodhar Damodaran, Irene Karegi, Sarah Gurev, Daniel J. Becker, Dan L. Warren, Nicola F. Müller, David Rasmussen, Stephanie N. Seifert.

This repository contains the analysis scripts and figure notebooks used in the study. **Input sequences, occurrence records, environmental rasters, trained models, and intermediate outputs are not stored in version control.** Place those assets next to this repository (or set the path flags / environment variables described below) before running full analyses.

```text
.
├── README.md
├── Scripts/
│   ├── buildFinalModelDataset.py
│   ├── build_case_study_site_level_dataset.py
│   ├── benchmark_sequence_model_suite.py
│   ├── sequence_model_suite_explanatory_utils.py
│   ├── fit_hierarchical_block_comparison.py
│   ├── TaxonomyReconciler/
│   ├── ReassortmentInference/
│   ├── SDM/
│   ├── UTR/
│   ├── LinkageDissequilibrium/
│   ├── Epistasis/
│   └── EvolutionaryAnalysis/
└── Notebooks/
    ├── Fig1.ipynb … Fig5.ipynb
    └── SDM_PDP_CircularPhylo.ipynb
```

---

## 1. System requirements

### Operating systems

| Component | Supported / tested |
|-----------|--------------------|
| Operating system | **macOS** (tested on macOS 26.5 / Darwin 25.5, Apple Silicon) and **Linux** (SLURM HPC used for SDM batch jobs and HyPhy scans). Windows is not supported as a primary target (WSL2 may work for Python-only steps). |
| Python | **3.13.x** (notebooks executed with **Python 3.13.1**) |
| Shell | `bash` / `zsh` |

No non-standard hardware is required. Analyses run on a standard multi-core desktop or laptop. Bayesian MCMC, ensemble species distribution models (SDMs), and full Espalier ARG reconstruction benefit from ≥4 CPU cores and ≥16 GB RAM. Large geospatial rasters (WorldClim 30-arc-second covariates, host suitability stacks) need approximately 300GB of disk space and 24GB of RAM per model.

### Software dependencies

#### Core Python packages

Install via `pip` (or conda). Packages used across the codebase:

| Package | Role |
|---------|------|
| `pandas`, `numpy` | Tables and numerics |
| `matplotlib`, `seaborn` | Plotting |
| `scipy`, `scikit-learn`, `statsmodels` | Statistics / ML utilities |
| `biopython` | Sequence I/O and NCBI Entrez |
| `joblib`, `tqdm` | Parallelism / progress |
| `networkx` | Lineage graphs |
| `adjustText` | Notebook label placement (Fig. 3) |

#### Geospatial / SDM stack (`Scripts/SDM/`)

| Package | Role |
|---------|------|
| `geopandas`, `shapely`, `pyproj`, `affine` | Vector geometries and projections |
| `rasterio` | Raster I/O and windowed prediction |
| `xgboost`, `lightgbm` | Ensemble classifiers |
| `scikit-optimize` | Optional Bayesian hyperparameter search (`--tune`) |

#### Phylogenetics / reassortment inference

| Package or tool | Role |
|-----------------|------|
| `dendropy`, `ete3` | Tree I/O and taxonomy DB |
| [Espalier](https://espalier.readthedocs.io/en/latest/) | Ancestral recombination graph (ARG) reconstruction |
| `tskit` | ARG tree handling in evolutionary utilities |
| `baltic` | Tree-length diagnostics |
| **Nextstrain Augur** | `augur align` / `augur tree` / `augur refine` |
| **MAFFT** | Alignment (via Augur) |
| **IQ-TREE** | Maximum-likelihood trees (via Augur `--method iqtree`) |
| **TreeTime** | Time-tree refinement (via Augur refine) |
| **RAxML-NG** | Local tree inference inside Espalier (`raxml-ng`) |
| **HyPhy** | FEL selection scans (`hyphy fel`) |

#### Molecular / Bayesian modeling

| Package or tool | Role |
|-----------------|------|
| `ViennaRNA` (Python bindings) or `RNAfold` CLI | UTR secondary structure / MFE |
| `pymc`, `bambi`, `arviz` | Hierarchical Bayesian models and posterior summaries |

### Versions tested

| Software | Version tested |
|----------|----------------|
| Python | 3.13.1 (analysis notebooks); development machine also runs 3.13.7 |
| macOS | 26.5.2 (Apple Silicon) |
| Augur / IQ-TREE / MAFFT / TreeTime / RAxML-NG / HyPhy / Espalier | As installed in the analysis environment used for the manuscript; exact CLI binaries must be on `PATH` |

Python package versions were not locked in this repository at submission. For archival reproduction we recommend installing the packages listed above into a fresh Python 3.13 environment and recording `pip freeze` / `conda env export` alongside any data release.

### Required non-standard hardware

None. GPU acceleration is not required. Multi-core CPUs speed SDM training, UTR folding (`-t` / thread flags), Augur/IQ-TREE, and MCMC (`chains` / `cores`, default 2–4).

---

## 2. Installation guide

### Instructions

1. Clone the repository:

```bash
git clone https://github.com/RicardoRH96/hantavirus-reassortment.git
cd hantavirus-reassortment
```

2. Create and activate a Python 3.13 environment, then install Python dependencies. Minimal install for the taxonomy demo:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas ete3
```

Full analysis environment (approximate):

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels \
  biopython joblib tqdm networkx adjustText \
  geopandas shapely pyproj affine rasterio \
  xgboost lightgbm scikit-optimize \
  dendropy ete3 tskit baltic \
  pymc bambi arviz
```

Install Espalier following its documentation: https://espalier.readthedocs.io/en/latest/

Install ViennaRNA (Python bindings preferred; `RNAfold` CLI is an acceptable fallback for some UTR scripts).

3. Install external CLI tools and confirm they are on `PATH`:

```bash
augur --version
iqtree --version    # or iqtree2
mafft --version
raxml-ng -v
hyphy --version
```

Nextstrain Augur typically pulls in MAFFT, IQ-TREE, and TreeTime; follow current Nextstrain install docs for your OS.

4. (Optional) Place manuscript data next to the repo so default paths resolve. Scripts expect sibling directories such as:

```text
Data/                  # sequences, ArHa tables, LD features, TreeGrafting inputs
model_test/            # Espalier runs, isolate/sequence modeling tables
Covariates/            # WorldClim / environmental rasters
Inputs/                # GBIF / host GeoJSON occurrence files
specific_evolutionary_analyses/   # optional alternate output root for EGT analyses
```

Override defaults with CLI flags or environment variables when paths differ:

| Variable | Purpose |
|----------|---------|
| `HANTA_EVO_OUTPUT_DIR` | EvolutionaryAnalysis output root |
| `HANTA_EVO_DATASET` | Sequence-level modeling table |
| `HANTA_EVO_MAIN_LABELS` | Main reassortment labels |
| `HANTA_EVO_TREE_ROOT` | Segment / ARG tree root |
| `HANTA_EVO_TRUE_ROOT` | True-reassortment ARG trees |
| `NCBI_API_KEY` | NCBI Entrez rate limits (UTR scripts) |

### Typical install time

On a standard desktop or laptop with broadband internet:

| Step | Typical time |
|------|----------------|
| Clone repository | < 1 minute |
| Create venv + install core Python packages | **5-15 minutes** |
| Full geospatial + Bayesian stack | **15-40 minutes** |
| Augur / IQ-TREE / MAFFT / RAxML-NG / HyPhy / ViennaRNA / Espalier | **20-60 minutes** (OS- and method-dependent) |
| First-time `ete3` NCBI taxonomy database download | **5-20 minutes** (network) |

---

## 3. Demo
### Instructions

```bash
cd Scripts/TaxonomyReconciler
pip install pandas ete3   # if not already installed
python ClassifyHantas.py
```

Optional companion for arenaviruses:

```bash
python ClassifyArenas.py
```

### Expected output

- File: `hantaviridae_ncbi_taxonomy_lookup.tsv` in the working directory (`Scripts/TaxonomyReconciler/`).
- Console summary similar to:

```text
rows: <N>
unique lookup keys: <N>
ambiguous keys: <N>
```

The TSV contains NCBI taxids, scientific names, ranks, lineage fields, and normalized name keys used by `ApplyTaxonomy.py` / `TaxonomyReconciler.py`.

### Expected run time

On a desktop with internet access: **about 5-20 minutes** on the first run (NCBI taxonomy DB build + subtree export), and **under 2 minutes** on subsequent runs once the local `ete3` taxonomy database exists.

---

## 4. Instructions for use

How to run the software on your data. Commands assume the repository root unless noted.

### A. Standardize viral taxonomy

```bash
# Build lookup (see Demo)
python Scripts/TaxonomyReconciler/ClassifyHantas.py

# Apply to a master isolate/sequence table
python Scripts/TaxonomyReconciler/ApplyTaxonomy.py \
  --lookup Scripts/TaxonomyReconciler/hantaviridae_ncbi_taxonomy_lookup.tsv \
  --input /path/to/AllHanta.tsv \
  --output /path/to/AllHanta_taxonomy.tsv
```

General multi-family reconciler:

```bash
python Scripts/TaxonomyReconciler/TaxonomyReconciler.py \
  --input /path/to/table.tsv \
  --output /path/to/table_reconciled.tsv \
  --lookup Hantaviridae=Scripts/TaxonomyReconciler/hantaviridae_ncbi_taxonomy_lookup.tsv
```

### B. Infer segment trees and optional ARGs (Espalier)

Organize per-species inputs with `S_segment/`, `M_segment/`, and `L_segment/` FASTA + metadata folders, then:

```bash
python Scripts/ReassortmentInference/Espalier_pipeline.py \
  --species-root /path/to/species_inputs \
  --output-root /path/to/espalier_pipeline \
  --threads 4

# Optional ARG reconstruction (requires DendroPy, Espalier, ete3, raxml-ng)
python Scripts/ReassortmentInference/Espalier_pipeline.py \
  --species-root /path/to/species_inputs \
  --output-root /path/to/espalier_pipeline \
  --run-espalier \
  --raxml-bin raxml-ng
```

Prep-only (skip Augur):

```bash
python Scripts/ReassortmentInference/Espalier_pipeline.py \
  --species-root /path/to/species_inputs \
  --output-root /path/to/espalier_pipeline \
  --skip-augur
```

Outputs include harmonized FASTAs, Augur alignments/trees, common-tip trees, SPR distance summaries, and (if enabled) Espalier ARG products under `{output-root}/{species}/`.

### C. Host SDMs and habitat overlap

```bash
# Optional: refresh WorldClim 30s covariates
python Scripts/SDM/update_worldclim_rasters.py --covariates-dir /path/to/Covariates

# Merge curated occurrences into GBIF GeoJSONs (if using ArHa tables)
python Scripts/SDM/merge_arha_occurrences.py \
  --gbif-dir /path/to/Inputs \
  --arha-csv /path/to/host.csv \
  --output-dir /path/to/Inputs_merged

# Train / project one host species
python Scripts/SDM/Ensemble_SDM2.py \
  /path/to/species.geojson \
  /path/to/Covariates \
  /path/to/SDM_out/Species_name \
  "Genus species" \
  --thin

# Stack host suitability into overlap rasters
python Scripts/SDM/stack_habitat_overlap.py \
  --ensemble-dir /path/to/SDM_out \
  --out-dir /path/to/overlap
```

Related utilities: `clipRaster_IUCN.py`, `partialDepProcessing.py`, `evaluate_host_predictions.py`, `analyze_features.py`, `collect_spearman.py`, `gatherModelStats.py`.

### D. UTR features

```bash
# From local FASTA (ViennaRNA Python bindings required)
python Scripts/UTR/GetUTRs.py input.fasta out_utrs.fasta mfe_table.tsv -n 43 -t 4

# From NCBI accessions (requires --email; optional NCBI_API_KEY)
python Scripts/UTR/fetch_utrs_and_structures.py \
  --accession-file accessions.txt \
  --email you@institution.edu \
  --output-dir Data/UTR_downloads

python Scripts/UTR/build_utr5_from_entrez.py --manifest-root /path/to/manifests --email you@institution.edu
python Scripts/UTR/build_utr3_from_entrez.py --manifest-root /path/to/manifests --email you@institution.edu
```

### E. Linkage disequilibrium features

```bash
python Scripts/LinkageDissequilibrium/01_batch_ld_from_pipeline.py \
  --pipeline-root /path/to/espalier_pipeline \
  --threads 4

python Scripts/LinkageDissequilibrium/02_ld_feature_extraction.py \
  --pipeline-root /path/to/espalier_pipeline \
  --output-dir Data/ML/ld_features
```

### F. Sequence-level modeling table and mechanism model suites

```bash
python Scripts/buildFinalModelDataset.py \
  --espalier-root /path/to/espalier_pipeline \
  --species-input-root /path/to/species_inputs \
  --arha-sequence /path/to/sequence.csv \
  --arha-host /path/to/host.csv \
  --arha-pathogen /path/to/pathogen.csv \
  --sequence-ld-features Data/ML/ld_features/sequence_ld_features.csv \
  --output-dataset /path/to/sequence_level_final_dataset.tsv

# Supervised sequence-level model suite (grouped species, within-species lineage, and repeated stratified CV)
python Scripts/benchmark_sequence_model_suite.py \
  --dataset /path/to/sequence_level_final_dataset.tsv \
  --output-dir /path/to/model_suite_outputs

# Hierarchical Bayesian block comparison (ecology, molecular, UTR3, and interaction blocks)
python Scripts/fit_hierarchical_block_comparison.py \
  --dataset /path/to/sequence_level_final_dataset.tsv \
  --output-dir /path/to/hierarchical_block_comparison \
  --draws 3000 --chains 4
```

### G. Intersegment epistasis features and LD comparisons

```bash
python Scripts/Epistasis/build_intersegment_epistasis_features.py \
  --pipeline-root /path/to/espalier_pipeline \
  --base-dataset /path/to/sequence_level_final_dataset.tsv \
  --output-root /path/to/intersegment_epistasis

python Scripts/Epistasis/build_selected_epistatic_interaction_features.py \
  --epistasis-root /path/to/intersegment_epistasis

python Scripts/Epistasis/build_final_epistasis_modeling_dataset.py \
  --epistasis-root /path/to/intersegment_epistasis

python Scripts/Epistasis/fit_hierarchical_ld_epistasis_comparison.py \
  --dataset /path/to/epistasis_modeling_dataset.tsv \
  --output-dir /path/to/ld_epistasis_fits \
  --draws 3000 --chains 4

python Scripts/Epistasis/summarize_final_ld_epistasis_comparison.py \
  --root /path/to/ld_epistasis_fits
```

### H. Evolutionary robustness, establishment, and selection

Run from `Scripts/EvolutionaryAnalysis/` so local imports (`evo_utils`) resolve:

```bash
cd Scripts/EvolutionaryAnalysis
export HANTA_EVO_DATASET=/path/to/sequence_level_final_dataset.tsv
export HANTA_EVO_MAIN_LABELS=/path/to/sequence_true_labels.tsv
export HANTA_EVO_OUTPUT_DIR=/path/to/evo_outputs
export HANTA_EVO_TREE_ROOT=/path/to/tree_roots
export HANTA_EVO_TRUE_ROOT=/path/to/true_reassortment_args

python 01_origin_cluster_robustness.py
python 02_label_uncertainty_propagation.py
python 03_within_lineage_contrasts.py
python 04_phylogenetic_nulls.py
python 05_mechanism_ablation.py
python 06_sampling_sensitivity.py
python 07_manuscript_robustness_summary.py   # synthesizes 01-06

python 08_fixation_establishment_model.py   # heavy MCMC; multi-hour class
python 09_lineage_graph_supplement.py
python 10_combined_strategy_graph_layouts.py

python 11_direct516_reassortant_fraction_violin.py
python 12_direct516_reassortment_sampling_discovery.py
python 13_egt_ld_epistasis_comparison.py

python 13_hyphy_fel_selection_scan.py --tree-root "$HANTA_EVO_TREE_ROOT" --output-dir /path/to/fel --run-hyphy
python 14_baltic_tree_length_diagnostics.py --tree-root "$HANTA_EVO_TREE_ROOT" --output-dir /path/to/baltic
python 15_andes_common_tip_fel_selection_overview.py --scan-dir /path/to/fel --output-dir /path/to/fel_overview
```

### I. Manuscript figures

With precomputed tables/model outputs available at the paths expected by each notebook:

```bash
jupyter notebook Notebooks/Fig1.ipynb
jupyter notebook Notebooks/Fig2.ipynb
jupyter notebook Notebooks/Fig3.ipynb
jupyter notebook Notebooks/Fig4.ipynb
jupyter notebook Notebooks/Fig5.ipynb
jupyter notebook Notebooks/SDM_PDP_CircularPhylo.ipynb
```

`Fig5.ipynb` redraws the establishment / strategy landscape from saved EvolutionaryAnalysis outputs and does not refit MCMC by default.

---

## 5. Reproduction instructions (quantitative manuscript results)

End-to-end reproduction requires the released analysis data (sequences, labels, SDM inputs/outputs or covariates, and modeling tables). Suggested order:

```text
TaxonomyReconciler
    -> Espalier_pipeline (Augur trees +/- Espalier ARGs)
    -> LinkageDissequilibrium (01 -> 02)
    -> UTR feature scripts
    -> SDM (occurrences -> Ensemble_SDM2 -> stack_habitat_overlap)
    -> buildFinalModelDataset
    -> Epistasis feature build -> hierarchical LD/epistasis fits
    -> EvolutionaryAnalysis 01-07 (robustness)
    -> EvolutionaryAnalysis 08-10 (establishment / Figure 5)
    -> EvolutionaryAnalysis 11-15 (descriptive, sampling, FEL, diagnostics)
    -> Notebooks Fig1-Fig5
```

| Manuscript product | Primary code |
|--------------------|--------------|
| Figure 1 | `Scripts/ReassortmentInference/build_reassortment_validation_outputs.py` / `Notebooks/Fig1.ipynb` |
| Figure 2 | `Notebooks/Fig2.ipynb` |
| Figure 3 (model performance) | `Scripts/benchmark_sequence_model_suite.py` / `Notebooks/Fig3.ipynb` |
| Figure 4 | `Scripts/fit_hierarchical_block_comparison.py` / `Notebooks/Fig4.ipynb` |
| Figure 5 (EGT strategy landscape) | `08_fixation_establishment_model.py` -> `09_lineage_graph_supplement.py` -> `10_combined_strategy_graph_layouts.py` / `Notebooks/Fig5.ipynb` |
| SDM / habitat overlap panels | `Scripts/SDM/Ensemble_SDM2.py`, `stack_habitat_overlap.py`, `partialDepProcessing.py`, `Notebooks/SDM_PDP_CircularPhylo.ipynb` |
| Robustness supplements | `Scripts/EvolutionaryAnalysis/01`-`07_*.py` |
| LD vs epistasis | `Scripts/Epistasis/*`, `13_egt_ld_epistasis_comparison.py` |
| FEL selection | `13_hyphy_fel_selection_scan.py`, `15_andes_common_tip_fel_selection_overview.py` |

### Expected run times (normal desktop / workstation)

| Analysis | Typical wall time |
|----------|-------------------|
| Taxonomy demo | minutes |
| Single-species Augur prep (no Espalier) | tens of minutes |
| Single-species Espalier ARG (`--run-espalier`) | tens of minutes to several hours (isolate count dependent) |
| One host SDM (`Ensemble_SDM2.py`, no `--tune`) | ~30 minutes to a few hours |
| SDM with `--tune` (BayesSearchCV) | several hours |
| WorldClim 30s download | tens of minutes to hours (network / disk) |
| EvolutionaryAnalysis 01-06 | tens of minutes to a few hours each (MCMC settings) |
| `08_fixation_establishment_model.py` | multi-hour (default multi-chain MCMC) |
| Full manuscript stack | days on a desktop; shorter on a multi-core workstation / cluster |

Seeds used in SDM code include `RNG_SEED = 72724`. Bayesian scripts expose `draws` / `chains` / `cores` (and related `HANTA_EGT_*` environment variables for establishment models). Exact posterior draws can differ slightly across platforms; manuscript figures were produced from the archived model outputs associated with the paper's data release.

### Data availability

Sequence metadata, modeling tables, rasters, and posterior objects required for full reproduction should be obtained from the paper's data availability statement / archival deposit (DOI to be inserted upon publication). This git repository intentionally excludes large binaries (`*.tif`, `*.joblib`, `*.nc`, etc.).
