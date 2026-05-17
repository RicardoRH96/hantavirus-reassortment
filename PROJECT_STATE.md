# Project State

Last updated: 2026-05-17

## Current Scope

This repository is the cleaned final-analysis workspace for the hantavirus reassortment manuscript. It contains only final-facing scripts, retained notebooks, compact processed data, and compact final outputs.

## Migration Status

- Initial migration from the local manuscript analysis workspace is complete.
- Large raw data directories and old exploratory iterations were not copied.
- Paths in scripts, notebooks, and result manifests have been rewritten to the new layout.
- Figure scripts have title calls removed and use Helvetica Neue fallbacks where configured.

## Next Checks

- Rerun the final figure scripts after any manuscript figure edits.
- Refit Bayesian models only when changing model specifications; posterior NetCDF files are intentionally ignored.
- Keep future raw downloads outside this repo or under ignored paths.
