# Species Lineage Graph Supplement

This figure visualizes the lineage graph used to calculate graph-neighborhood opportunity in the fixation-inspired establishment analysis. The raw graph is almost complete within species, so the plotted network emphasizes a reproducible backbone: the maximum-spanning tree, each lineage's two strongest links, and species-specific top-decile edge weights. The full raw graph remains visible as a faint scaffold.

Node marker area is min-max scaled from the observed lineage sample-size range with a very small singleton area, so one-tip lineages remain visually distinct from larger sampled clades.

- Input nodes: `results/evolutionary_robustness/reports/08_species_lineage_graph_nodes.tsv`
- Input edges: `results/evolutionary_robustness/reports/08_species_lineage_graph_edges.tsv`
- Backbone edges: `results/evolutionary_robustness/reports/09_species_lineage_graph_backbone_edges.tsv`
- Figure PNG: `results/evolutionary_robustness/figures/09_species_lineage_graph_supplement.png`
- Figure SVG: `results/evolutionary_robustness/figures/09_species_lineage_graph_supplement.svg`

| Species | Lineages | Sampled tips | Origin lineages | Raw edges | Backbone edges | Backbone fraction | Mean neighbor overlap | Median edge weight | Median backbone weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Andes | 4 | 24 | 0 | 6 | 5 | 0.833 | 26.572 | 0.720 | 0.748 |
| Dobrava | 4 | 18 | 2 | 6 | 5 | 0.833 | 22.618 | 0.205 | 0.229 |
| Hantaan | 51 | 199 | 12 | 1275 | 152 | 0.119 | 30.418 | 0.648 | 0.940 |
| Puumala | 19 | 69 | 10 | 171 | 35 | 0.205 | 27.770 | 0.501 | 0.902 |
| Seoul | 39 | 101 | 12 | 741 | 103 | 0.139 | 17.903 | 0.754 | 0.942 |
| Sin Nombre | 8 | 34 | 6 | 28 | 11 | 0.393 | 13.406 | 0.478 | 0.533 |
| Tula | 18 | 71 | 10 | 153 | 31 | 0.203 | 48.280 | 0.915 | 0.974 |
