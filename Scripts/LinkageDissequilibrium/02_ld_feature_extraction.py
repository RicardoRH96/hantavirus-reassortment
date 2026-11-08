#!/usr/bin/env python3
"""
Extract species-level LD summaries, cross-species conserved LD site pairs, and
per-sequence nucleotide/haplotype features for reassortment modeling.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Sequence

import numpy as np
import pandas as pd
from Bio import AlignIO

LOGGER = logging.getLogger("ld_features")

R2_THRESHOLD = 0.1
CRAMERS_V_THRESHOLD = 0.1
NMI_THRESHOLD = 0.05
P_VALUE_THRESHOLD = 0.05


@dataclass
class LDSitePair:
    segment_a: str
    segment_b: str
    site_a: int
    site_b: int
    n_strains: int
    d_prime: float
    chi_sq: float
    cramers_v: float
    nmi: float
    maf1: float
    maf2: float


@dataclass
class SpeciesLDSummary:
    species: str
    n_isolates: int

    mean_cramers_v_SM: float = 0.0
    mean_cramers_v_SL: float = 0.0
    mean_cramers_v_ML: float = 0.0

    median_cramers_v_SM: float = 0.0
    median_cramers_v_SL: float = 0.0
    median_cramers_v_ML: float = 0.0

    mean_nmi_SM: float = 0.0
    mean_nmi_SL: float = 0.0
    mean_nmi_ML: float = 0.0

    prop_significant_SM: float = 0.0
    prop_significant_SL: float = 0.0
    prop_significant_ML: float = 0.0

    n_site_pairs_SM: int = 0
    n_site_pairs_SL: int = 0
    n_site_pairs_ML: int = 0

    n_significant_SM: int = 0
    n_significant_SL: int = 0
    n_significant_ML: int = 0

    mean_cramers_v_overall: float = 0.0
    median_cramers_v_overall: float = 0.0
    mean_nmi_overall: float = 0.0
    prop_significant_overall: float = 0.0

    cramers_v_variance: float = 0.0
    cramers_v_skewness: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "species": self.species,
            "n_isolates": self.n_isolates,
            "mean_cramers_v_SM": self.mean_cramers_v_SM,
            "mean_cramers_v_SL": self.mean_cramers_v_SL,
            "mean_cramers_v_ML": self.mean_cramers_v_ML,
            "median_cramers_v_SM": self.median_cramers_v_SM,
            "median_cramers_v_SL": self.median_cramers_v_SL,
            "median_cramers_v_ML": self.median_cramers_v_ML,
            "mean_nmi_SM": self.mean_nmi_SM,
            "mean_nmi_SL": self.mean_nmi_SL,
            "mean_nmi_ML": self.mean_nmi_ML,
            "prop_significant_SM": self.prop_significant_SM,
            "prop_significant_SL": self.prop_significant_SL,
            "prop_significant_ML": self.prop_significant_ML,
            "n_site_pairs_SM": self.n_site_pairs_SM,
            "n_site_pairs_SL": self.n_site_pairs_SL,
            "n_site_pairs_ML": self.n_site_pairs_ML,
            "n_significant_SM": self.n_significant_SM,
            "n_significant_SL": self.n_significant_SL,
            "n_significant_ML": self.n_significant_ML,
            "mean_cramers_v_overall": self.mean_cramers_v_overall,
            "median_cramers_v_overall": self.median_cramers_v_overall,
            "mean_nmi_overall": self.mean_nmi_overall,
            "prop_significant_overall": self.prop_significant_overall,
            "cramers_v_variance": self.cramers_v_variance,
            "cramers_v_skewness": self.cramers_v_skewness,
        }


@dataclass
class ConservedLDSite:
    segment_a: str
    segment_b: str
    site_a: int
    site_b: int
    species_with_ld: List[str] = field(default_factory=list)
    mean_cramers_v: float = 0.0
    mean_nmi: float = 0.0

    @property
    def n_species(self) -> int:
        return len(self.species_with_ld)

    @property
    def site_key(self) -> Tuple[str, str, int, int]:
        return (self.segment_a, self.segment_b, self.site_a, self.site_b)


def load_ld_results(ld_csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(ld_csv_path)


def load_all_species_ld(
    pipeline_root: Path,
    species_list: Optional[List[str]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    results = {}

    for species_dir in sorted(pipeline_root.iterdir()):
        if not species_dir.is_dir() or species_dir.name == "inputs":
            continue
        if species_list and species_dir.name not in species_list:
            continue

        ld_dir = species_dir / "ld" / "conservative"
        if not ld_dir.exists():
            continue

        species_results = {}
        for segment_pair in ["S_M", "S_L", "M_L"]:
            ld_file = ld_dir / f"{segment_pair}_LD.csv"
            if ld_file.exists():
                df = load_ld_results(ld_file)
                if len(df) > 0:
                    species_results[segment_pair] = df

        if species_results:
            results[species_dir.name] = species_results
            LOGGER.info("Loaded LD results for %s: %s",
                       species_dir.name, list(species_results.keys()))

    return results


def compute_species_summary(
    species: str,
    ld_data: Dict[str, pd.DataFrame],
    cramers_v_threshold: float = CRAMERS_V_THRESHOLD,
) -> SpeciesLDSummary:
    n_isolates = 0
    for df in ld_data.values():
        if len(df) > 0 and "n_strains" in df.columns:
            n_isolates = int(df["n_strains"].iloc[0])
            break

    summary = SpeciesLDSummary(species=species, n_isolates=n_isolates)

    all_cramers_v = []
    all_nmi = []

    segment_pair_map = {
        "S_M": ("SM", summary),
        "S_L": ("SL", summary),
        "M_L": ("ML", summary),
    }

    for seg_pair, df in ld_data.items():
        if seg_pair not in segment_pair_map:
            continue

        suffix = segment_pair_map[seg_pair][0]

        cramers_v_col = df["CramerV"].dropna()
        nmi_col = df["NMI"].dropna() if "NMI" in df.columns else pd.Series([])

        if len(cramers_v_col) == 0:
            continue

        mean_cv = cramers_v_col.mean()
        median_cv = cramers_v_col.median()
        n_pairs = len(cramers_v_col)
        n_significant = (cramers_v_col >= cramers_v_threshold).sum()
        prop_significant = n_significant / n_pairs if n_pairs > 0 else 0

        mean_nmi = nmi_col.mean() if len(nmi_col) > 0 else 0.0

        setattr(summary, f"mean_cramers_v_{suffix}", mean_cv)
        setattr(summary, f"median_cramers_v_{suffix}", median_cv)
        setattr(summary, f"mean_nmi_{suffix}", mean_nmi)
        setattr(summary, f"prop_significant_{suffix}", prop_significant)
        setattr(summary, f"n_site_pairs_{suffix}", n_pairs)
        setattr(summary, f"n_significant_{suffix}", int(n_significant))

        all_cramers_v.extend(cramers_v_col.tolist())
        all_nmi.extend(nmi_col.tolist())

    if all_cramers_v:
        all_cv = np.array(all_cramers_v)
        summary.mean_cramers_v_overall = float(np.mean(all_cv))
        summary.median_cramers_v_overall = float(np.median(all_cv))
        summary.cramers_v_variance = float(np.var(all_cv))
        if len(all_cv) > 2:
            summary.cramers_v_skewness = float(
                pd.Series(all_cv).skew()
            )
        n_total = len(all_cv)
        n_sig_total = (all_cv >= cramers_v_threshold).sum()
        summary.prop_significant_overall = n_sig_total / n_total

    if all_nmi:
        summary.mean_nmi_overall = float(np.mean(all_nmi))

    return summary


def extract_species_level_features(
    pipeline_root: Path,
    species_list: Optional[List[str]] = None,
    cramers_v_threshold: float = CRAMERS_V_THRESHOLD,
) -> pd.DataFrame:
    ld_data = load_all_species_ld(pipeline_root, species_list)

    summaries = []
    for species, data in ld_data.items():
        summary = compute_species_summary(species, data, cramers_v_threshold)
        summaries.append(summary.to_dict())

    return pd.DataFrame(summaries)


def find_conserved_ld_sites(
    pipeline_root: Path,
    min_species: int = 2,
    cramers_v_threshold: float = CRAMERS_V_THRESHOLD,
    species_list: Optional[List[str]] = None,
) -> List[ConservedLDSite]:
    ld_data = load_all_species_ld(pipeline_root, species_list)

    site_species_map: Dict[Tuple, Dict[str, float]] = defaultdict(dict)
    site_nmi_map: Dict[Tuple, Dict[str, float]] = defaultdict(dict)

    for species, seg_data in ld_data.items():
        for seg_pair, df in seg_data.items():
            seg_a, seg_b = seg_pair.split("_")

            significant = df[df["CramerV"] >= cramers_v_threshold]

            for _, row in significant.iterrows():
                site_key = (seg_a, seg_b, int(row["site1"]), int(row["site2"]))
                site_species_map[site_key][species] = row["CramerV"]
                if "NMI" in row and pd.notna(row["NMI"]):
                    site_nmi_map[site_key][species] = row["NMI"]

    conserved_sites = []
    for site_key, species_values in site_species_map.items():
        if len(species_values) >= min_species:
            site = ConservedLDSite(
                segment_a=site_key[0],
                segment_b=site_key[1],
                site_a=site_key[2],
                site_b=site_key[3],
                species_with_ld=list(species_values.keys()),
                mean_cramers_v=np.mean(list(species_values.values())),
                mean_nmi=np.mean(list(site_nmi_map.get(site_key, {0: 0}).values())),
            )
            conserved_sites.append(site)

    conserved_sites.sort(key=lambda x: (-x.n_species, -x.mean_cramers_v))

    LOGGER.info("Found %d conserved LD site pairs (in >= %d species)",
               len(conserved_sites), min_species)

    return conserved_sites


def conserved_sites_to_dataframe(sites: List[ConservedLDSite]) -> pd.DataFrame:
    records = []
    for site in sites:
        records.append({
            "segment_a": site.segment_a,
            "segment_b": site.segment_b,
            "site_a": site.site_a,
            "site_b": site.site_b,
            "n_species": site.n_species,
            "species": ",".join(sorted(site.species_with_ld)),
            "mean_cramers_v": site.mean_cramers_v,
            "mean_nmi": site.mean_nmi,
        })
    return pd.DataFrame(records)


def load_alignment(fasta_path: Path) -> Dict[str, str]:
    aln = AlignIO.read(fasta_path, "fasta")
    return {rec.id: str(rec.seq) for rec in aln}


def extract_sequence_features_at_sites(
    alignment_dir: Path,
    conserved_sites: List[ConservedLDSite],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract nucleotide and two-locus haplotype states at conserved LD positions.
    Column names follow `<seg>_<pos>` and `<segA>_<posA>_<segB>_<posB>`.
    """
    if top_n:
        sites_to_use = conserved_sites[:top_n]
    else:
        sites_to_use = conserved_sites

    if not sites_to_use:
        return pd.DataFrame()

    alignments: Dict[str, Dict[str, str]] = {}
    for seg in ["S", "M", "L"]:
        fasta_path = alignment_dir / f"{seg}.fasta"
        if fasta_path.exists():
            alignments[seg] = load_alignment(fasta_path)

    if not alignments:
        LOGGER.warning("No alignments found in %s", alignment_dir)
        return pd.DataFrame()

    common_ids = set(alignments[list(alignments.keys())[0]].keys())
    for seg_aln in alignments.values():
        common_ids &= set(seg_aln.keys())

    if not common_ids:
        LOGGER.warning("No common sequence IDs across segments")
        return pd.DataFrame()

    common_ids = sorted(common_ids)
    feature_data = {seq_id: {} for seq_id in common_ids}

    for site in sites_to_use:
        seg_a, seg_b = site.segment_a, site.segment_b
        pos_a, pos_b = site.site_a, site.site_b

        if seg_a not in alignments or seg_b not in alignments:
            continue

        aln_a = alignments[seg_a]
        aln_b = alignments[seg_b]

        col_name_a = f"{seg_a}_{pos_a}"
        col_name_b = f"{seg_b}_{pos_b}"
        col_name_pair = f"{seg_a}_{pos_a}_{seg_b}_{pos_b}"

        for seq_id in common_ids:
            seq_a = aln_a.get(seq_id, "")
            seq_b = aln_b.get(seq_id, "")

            # Ambiguous N states are treated as missing rather than a distinct allele category.
            nt_a = seq_a[pos_a] if pos_a < len(seq_a) else "N"
            nt_b = seq_b[pos_b] if pos_b < len(seq_b) else "N"
            nt_a = pd.NA if nt_a == "N" else nt_a
            nt_b = pd.NA if nt_b == "N" else nt_b

            feature_data[seq_id][col_name_a] = nt_a
            feature_data[seq_id][col_name_b] = nt_b
            feature_data[seq_id][col_name_pair] = (
                pd.NA if pd.isna(nt_a) or pd.isna(nt_b) else f"{nt_a}{nt_b}"
            )

    df = pd.DataFrame.from_dict(feature_data, orient="index")
    df.index.name = "sequence_id"

    return df


def extract_all_sequence_features(
    pipeline_root: Path,
    conserved_sites: List[ConservedLDSite],
    top_n: Optional[int] = 50,
) -> pd.DataFrame:
    all_features = []

    for species_dir in sorted(pipeline_root.iterdir()):
        if not species_dir.is_dir() or species_dir.name == "inputs":
            continue

        aln_dir = species_dir / "ld" / "conservative" / "inputs"
        if not aln_dir.exists():
            continue

        species_features = extract_sequence_features_at_sites(
            aln_dir, conserved_sites, top_n
        )

        if len(species_features) > 0:
            species_features = species_features.reset_index()
            species_features.insert(0, "species", species_dir.name)
            all_features.append(species_features)
            LOGGER.info("Extracted features for %d sequences from %s",
                       len(species_features), species_dir.name)

    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()


def compute_ld_site_variability(
    pipeline_root: Path,
    species_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    ld_data = load_all_species_ld(pipeline_root, species_list)
    site_values: Dict[Tuple, List[Tuple[str, float]]] = defaultdict(list)

    for species, seg_data in ld_data.items():
        for seg_pair, df in seg_data.items():
            seg_a, seg_b = seg_pair.split("_")

            for _, row in df.iterrows():
                if pd.isna(row["CramerV"]):
                    continue
                site_key = (seg_a, seg_b, int(row["site1"]), int(row["site2"]))
                site_values[site_key].append((species, row["CramerV"]))

    records = []
    for site_key, values in site_values.items():
        if len(values) < 2:
            continue

        species_names = [v[0] for v in values]
        cramers_v_values = [v[1] for v in values]

        records.append({
            "segment_a": site_key[0],
            "segment_b": site_key[1],
            "site_a": site_key[2],
            "site_b": site_key[3],
            "n_species": len(values),
            "mean_cramers_v": np.mean(cramers_v_values),
            "std_cramers_v": np.std(cramers_v_values),
            "cv_cramers_v": np.std(cramers_v_values) / np.mean(cramers_v_values) if np.mean(cramers_v_values) > 0 else 0,
            "min_cramers_v": np.min(cramers_v_values),
            "max_cramers_v": np.max(cramers_v_values),
            "range_cramers_v": np.max(cramers_v_values) - np.min(cramers_v_values),
            "species": ",".join(sorted(species_names)),
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("n_species", ascending=False)
    return df


def encode_nucleotide_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    if columns is None:
        columns = [c for c in df.columns
                   if df[c].dtype == object and
                   df[c].str.len().max() == 1 and
                   c not in ["species", "sequence_id"]]

    encoded_dfs = [df[["species", "sequence_id"]]] if "species" in df.columns else []

    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col)
        encoded_dfs.append(dummies)

    return pd.concat(encoded_dfs, axis=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract LD features for reassortment ML models"
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path("Data/TreeGrafting/espalier_pipeline"),
        help="Root directory with species LD results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/ML/ld_features"),
        help="Output directory for feature files",
    )
    parser.add_argument(
        "--min-species-conserved",
        type=int,
        default=2,
        help="Minimum species for a site to be considered conserved (default: 2)",
    )
    parser.add_argument(
        "--cramers-v-threshold",
        type=float,
        default=0.1,
        help="Cramer's V threshold for significant LD (default: 0.1)",
    )
    parser.add_argument(
        "--top-n-sites",
        type=int,
        default=50,
        help="Number of top conserved sites for sequence features (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    args.pipeline_root = args.pipeline_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Extracting LD features from %s", args.pipeline_root)

    LOGGER.info("Computing species-level LD summaries...")
    species_features = extract_species_level_features(
        args.pipeline_root,
        cramers_v_threshold=args.cramers_v_threshold,
    )
    species_out = args.output_dir / "species_ld_features.csv"
    species_features.to_csv(species_out, index=False)
    LOGGER.info("Wrote species features to %s", species_out)
    print(f"\nSpecies-level features ({len(species_features)} species):")
    print(species_features.to_string())

    LOGGER.info("Finding conserved LD sites...")
    conserved_sites = find_conserved_ld_sites(
        args.pipeline_root,
        min_species=args.min_species_conserved,
        cramers_v_threshold=args.cramers_v_threshold,
    )
    conserved_df = conserved_sites_to_dataframe(conserved_sites)
    conserved_out = args.output_dir / "conserved_ld_sites.csv"
    conserved_df.to_csv(conserved_out, index=False)
    LOGGER.info("Wrote %d conserved sites to %s", len(conserved_df), conserved_out)

    if len(conserved_sites) > 0:
        print(f"\nTop conserved LD sites (n={len(conserved_sites)}):")
        print(conserved_df.head(20).to_string())

    LOGGER.info("Computing site-level LD variability...")
    variability_df = compute_ld_site_variability(args.pipeline_root)
    variability_out = args.output_dir / "ld_site_variability.csv"
    variability_df.to_csv(variability_out, index=False)
    LOGGER.info("Wrote site variability to %s", variability_out)

    if conserved_sites:
        LOGGER.info("Extracting sequence-level features at top %d conserved sites...",
                   args.top_n_sites)
        seq_features = extract_all_sequence_features(
            args.pipeline_root,
            conserved_sites,
            top_n=args.top_n_sites,
        )

        if len(seq_features) > 0:
            seq_out = args.output_dir / "sequence_ld_features.csv"
            seq_features.to_csv(seq_out, index=False)
            LOGGER.info("Wrote sequence features for %d sequences to %s",
                       len(seq_features), seq_out)

            nt_cols = [c for c in seq_features.columns
                      if c not in ["species", "sequence_id"] and
                      "_" in c and c.count("_") <= 1]
            if nt_cols:
                encoded = encode_nucleotide_features(seq_features, nt_cols)
                encoded_out = args.output_dir / "sequence_ld_features_encoded.csv"
                encoded.to_csv(encoded_out, index=False)
                LOGGER.info("Wrote one-hot encoded features to %s", encoded_out)

    LOGGER.info("Feature extraction complete!")
    print(f"\nOutput files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
