# Biology-only repeated-stratified explanatory companion

This directory materializes the paper-facing control-free repeated-stratified LightGBM companion for interpretation.

Current verdict:
- the promoted explanatory bundle is `ecology_molecular_lineage_core`
- it keeps local ecology, direct discordance, direct terminal `utrterm_*` features, and the biology-only LD interaction encoding
- the later UTR-augmented audit did not improve AP/AUC enough to replace this lineage-core bundle, so broader species-level `species_feat_utr5_*` / `species_feat_utr3_*` summaries remain sensitivity-only here
