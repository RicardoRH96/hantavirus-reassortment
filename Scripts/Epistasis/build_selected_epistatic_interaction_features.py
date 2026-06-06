#!/usr/bin/env python3
"""Encode significant intersegment epistatic site pairs as sequence features.

This script consumes the full intersegment epistasis scan outputs and keeps a
small, FDR-gated set of directional site-pair interactions. It then turns each
selected interaction into per-sequence categorical nucleotide and amino-acid
pair-state features while retaining the species-level epistasis summaries
already present in the input dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO


LOGGER = logging.getLogger("build_selected_epistatic_interaction_features")
REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENTS = ("S", "M", "L")
VALID_BASES = {"A", "C", "G", "T"}
STOP_CODONS = {"TAA", "TAG", "TGA"}
GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


@dataclass(frozen=True)
class OrfSpan:
    start0: int
    end0: int
    source: str


@dataclass(frozen=True)
class InteractionFeature:
    interaction_id: str
    species: str
    species_token: str
    lead_segment: str
    trail_segment: str
    mode: str
    lead_site: int
    trail_site: int
    observed_score: float
    empirical_p_ge: float
    n_lead_events: int
    n_trail_events: int
    n_projected_lead_events: int
    analysis_fdr: float
    observed_positives: int
    expected_false_positives: float
    robust_positive_fraction_p: float
    robust_observed_positives: int
    robust_total_pairs: int
    n_permutations: int
    lead_aa_position: Optional[int]
    trail_aa_position: Optional[int]
    lead_orf_source: str
    trail_orf_source: str
    nt_feature: str
    aa_feature: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    epistasis_root = REPO_ROOT / "model_test" / "runs" / "intersegment_epistasis_fullscan_direct553"
    parser = argparse.ArgumentParser(
        description="Select FDR-gated intersegment epistatic interactions and encode per-sequence pair states."
    )
    parser.add_argument("--epistasis-root", type=Path, default=epistasis_root)
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "pathoplexus_andes_expansion" / "mixed_pipeline",
    )
    parser.add_argument(
        "--true-reassortment-root",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "pathoplexus_andes_expansion" / "true_reassortment_mixed",
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=epistasis_root / "datasets" / "sequence_level_final_dataset_direct516_plus_epistasis.tsv",
    )
    parser.add_argument("--output-root", type=Path, default=epistasis_root)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--max-analysis-fdr",
        type=float,
        default=0.20,
        help="Maximum analysis-level EFP/OP FDR for a directional species/segment/mode scan.",
    )
    parser.add_argument(
        "--max-robust-positive-fraction-p",
        type=float,
        default=0.10,
        help="Maximum paper-style hitchhiking-robust positive-fraction P-value for the analysis.",
    )
    parser.add_argument("--max-pair-p", type=float, default=0.01)
    parser.add_argument(
        "--mode",
        action="append",
        choices=["time", "div"],
        default=None,
        help="Mode(s) to consider. Defaults to both time and div.",
    )
    parser.add_argument(
        "--allow-single-leading-event",
        action="store_true",
        help="Do not require the paper-style two-or-more non-virtual leading-mutation pair filter.",
    )
    parser.add_argument("--top-pairs-per-analysis", type=int, default=25)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


def safe_token(value: str) -> str:
    token = re.sub(r"^orthohantavirus_", "", str(value))
    token = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").lower()
    return token or "species"


def translate_codon(codon: str) -> str:
    codon = codon.upper()
    if len(codon) != 3 or any(base not in VALID_BASES for base in codon):
        return "X"
    return GENETIC_CODE.get(codon, "X")


def infer_longest_orf(seq: str) -> OrfSpan:
    seq = str(seq or "").upper().replace("-", "N")
    best: Optional[Tuple[int, int, str]] = None
    for frame in range(3):
        for start in range(frame, max(frame, len(seq) - 2), 3):
            if seq[start : start + 3] != "ATG":
                continue
            stop = None
            for pos in range(start + 3, len(seq) - 2, 3):
                if seq[pos : pos + 3] in STOP_CODONS:
                    stop = pos
                    break
            end = stop if stop is not None else start + 3 * ((len(seq) - start) // 3)
            if end <= start:
                continue
            if best is None or (end - start) > (best[1] - best[0]):
                source = "longest_atg_to_stop" if stop is not None else "longest_atg_to_end"
                best = (start, end, source)
    if best and (best[1] - best[0]) >= 300:
        return OrfSpan(*best)

    fallback = (0, 3 * (len(seq) // 3), "fallback_frame0")
    for frame in range(3):
        run_start = frame
        pos = frame
        while pos <= len(seq) - 3:
            codon = seq[pos : pos + 3]
            if codon in STOP_CODONS:
                if pos - run_start > fallback[1] - fallback[0]:
                    fallback = (run_start, pos, "fallback_longest_stop_free_frame")
                run_start = pos + 3
            pos += 3
        end = frame + 3 * ((len(seq) - frame) // 3)
        if end - run_start > fallback[1] - fallback[0]:
            fallback = (run_start, end, "fallback_longest_stop_free_frame")
    return OrfSpan(*fallback)


def load_ancestral_root_sequence(true_root: Path, species: str, segment: str) -> Optional[str]:
    path = true_root / species / segment / f"{segment}_ancestral_node_data.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    nodes = data.get("nodes", {})
    if not isinstance(nodes, MappingABC):
        return None
    for payload in nodes.values():
        if isinstance(payload, MappingABC) and payload.get("sequence"):
            return str(payload["sequence"])
    return None


def alignment_candidates(pipeline_root: Path, species: str, segment: str) -> List[Path]:
    return [
        pipeline_root / species / "augur" / segment / f"{segment}_aligned.fasta",
        pipeline_root / species / "espalier" / "alignments" / f"{segment}_aligned_common.fasta",
        pipeline_root / species / "espalier" / "alignments" / f"{segment}_aligned.fasta",
    ]


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def load_alignment(pipeline_root: Path, species: str, segment: str) -> Dict[str, str]:
    path = first_existing(alignment_candidates(pipeline_root, species, segment))
    if path is None:
        raise FileNotFoundError(f"No aligned FASTA found for {species} {segment}")
    return {record.id: str(record.seq).upper() for record in SeqIO.parse(str(path), "fasta")}


def infer_orf(
    true_root: Path,
    pipeline_root: Path,
    species: str,
    segment: str,
    alignment: Mapping[str, str],
) -> OrfSpan:
    seq = load_ancestral_root_sequence(true_root, species, segment)
    if seq is None and alignment:
        seq = next(iter(alignment.values()))
    if seq is None:
        return OrfSpan(0, 0, "missing_sequence")
    return infer_longest_orf(seq)


def aa_position_for_site(site: int, orf: OrfSpan) -> Optional[int]:
    index0 = int(site) - 1
    if index0 < orf.start0 or index0 >= orf.end0:
        return None
    return ((index0 - orf.start0) // 3) + 1


def base_at(seq: Optional[str], site: int) -> str:
    if not seq:
        return "missing"
    index0 = int(site) - 1
    if index0 < 0 or index0 >= len(seq):
        return "missing"
    base = seq[index0].upper()
    if base in VALID_BASES:
        return base
    if base == "-":
        return "gap"
    if base == "N":
        return "N"
    return "ambiguous"


def aa_at(seq: Optional[str], site: int, orf: OrfSpan) -> str:
    if not seq:
        return "missing"
    index0 = int(site) - 1
    if index0 < orf.start0 or index0 >= orf.end0:
        return "not_coding"
    codon_start = orf.start0 + 3 * ((index0 - orf.start0) // 3)
    codon = seq[codon_start : codon_start + 3].upper().replace("-", "N")
    aa = translate_codon(codon)
    return aa if aa != "X" else "ambiguous"


def filter_analysis_summary(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    modes = args.mode or ["time", "div"]
    alpha_mask = np.isclose(pd.to_numeric(summary["alpha"], errors="coerce"), args.alpha)
    analysis = summary.loc[alpha_mask & summary["mode"].isin(modes)].copy()
    numeric_cols = [
        "fdr",
        "observed_positive_count_p",
        "robust_positive_fraction_p",
        "observed_positives",
        "expected_false_positives",
        "robust_observed_positives",
        "robust_total_pairs",
    ]
    for col in numeric_cols:
        if col in analysis.columns:
            analysis[col] = pd.to_numeric(analysis[col], errors="coerce")
    analysis["passes_analysis_fdr"] = analysis["fdr"].le(args.max_analysis_fdr)
    analysis["passes_robust_fraction"] = analysis["robust_positive_fraction_p"].le(
        args.max_robust_positive_fraction_p
    )
    analysis["selected_analysis"] = analysis["passes_analysis_fdr"] & analysis["passes_robust_fraction"]
    return analysis


def select_pairs(epistasis_root: Path, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = epistasis_root / "reports" / "intersegment_epistasis_fdr_summary.tsv"
    pairs_path = epistasis_root / "reports" / "intersegment_epistasis_fdr_significant_pairs.tsv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing FDR summary: {summary_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing FDR significant pair table: {pairs_path}")

    summary = pd.read_csv(summary_path, sep="\t")
    analysis = filter_analysis_summary(summary, args)
    selected_analysis = analysis.loc[analysis["selected_analysis"]].copy()

    pairs = pd.read_csv(pairs_path, sep="\t")
    merge_cols = ["species", "lead_segment", "trail_segment", "segment_pair", "mode"]
    analysis_cols = merge_cols + [
        "alpha",
        "observed_positives",
        "expected_false_positives",
        "fdr",
        "observed_positive_count_p",
        "robust_min_leading_events",
        "robust_total_pairs",
        "robust_observed_positives",
        "robust_observed_positive_fraction",
        "robust_expected_fake_positive_fraction",
        "robust_positive_fraction_p",
        "selected_analysis",
    ]
    merged = pairs.merge(selected_analysis[analysis_cols], on=merge_cols, how="inner")
    merged["empirical_p_ge"] = pd.to_numeric(merged["empirical_p_ge"], errors="coerce")
    merged["observed_score"] = pd.to_numeric(merged["observed_score"], errors="coerce")
    selected = merged.loc[merged["empirical_p_ge"].le(args.max_pair_p)].copy()
    if not args.allow_single_leading_event:
        selected = selected.loc[pd.to_numeric(selected["passes_hitchhiking_min_leads"], errors="coerce").eq(1)]

    selected_chunks: List[pd.DataFrame] = []
    group_cols = ["species", "lead_segment", "trail_segment", "mode"]
    for _, group in selected.groupby(group_cols, sort=True):
        ranked = group.sort_values(
            ["empirical_p_ge", "observed_score", "lead_site", "trail_site"],
            ascending=[True, False, True, True],
        )
        selected_chunks.append(ranked.head(args.top_pairs_per_analysis))
    selected = pd.concat(selected_chunks, ignore_index=True) if selected_chunks else pd.DataFrame()
    if selected.empty:
        return analysis, selected
    selected = selected.sort_values(
        ["fdr", "robust_positive_fraction_p", "species", "lead_segment", "trail_segment", "mode", "empirical_p_ge", "observed_score"],
        ascending=[True, True, True, True, True, True, True, False],
    ).reset_index(drop=True)
    selected.insert(0, "interaction_rank", np.arange(1, selected.shape[0] + 1))
    return analysis, selected


def build_interaction_features(
    selected: pd.DataFrame,
    *,
    true_root: Path,
    pipeline_root: Path,
) -> List[InteractionFeature]:
    alignments: Dict[Tuple[str, str], Dict[str, str]] = {}
    orfs: Dict[Tuple[str, str], OrfSpan] = {}
    features: List[InteractionFeature] = []

    for row in selected.itertuples(index=False):
        species = str(row.species)
        lead_segment = str(row.lead_segment)
        trail_segment = str(row.trail_segment)
        for segment in [lead_segment, trail_segment]:
            key = (species, segment)
            if key not in alignments:
                alignments[key] = load_alignment(pipeline_root, species, segment)
                orfs[key] = infer_orf(true_root, pipeline_root, species, segment, alignments[key])

        rank = int(row.interaction_rank)
        token = safe_token(species)
        base_name = (
            f"epi_pair_state_{rank:03d}_{token}_{row.mode}_"
            f"{lead_segment}{int(row.lead_site)}_{trail_segment}{int(row.trail_site)}"
        )
        lead_orf = orfs[(species, lead_segment)]
        trail_orf = orfs[(species, trail_segment)]
        features.append(
            InteractionFeature(
                interaction_id=base_name.replace("epi_pair_state_", "epi_interaction_", 1),
                species=species,
                species_token=token,
                lead_segment=lead_segment,
                trail_segment=trail_segment,
                mode=str(row.mode),
                lead_site=int(row.lead_site),
                trail_site=int(row.trail_site),
                observed_score=float(row.observed_score),
                empirical_p_ge=float(row.empirical_p_ge),
                n_lead_events=int(row.n_lead_events),
                n_trail_events=int(row.n_trail_events),
                n_projected_lead_events=int(row.n_projected_lead_events),
                analysis_fdr=float(row.fdr),
                observed_positives=int(row.observed_positives),
                expected_false_positives=float(row.expected_false_positives),
                robust_positive_fraction_p=float(row.robust_positive_fraction_p),
                robust_observed_positives=int(row.robust_observed_positives),
                robust_total_pairs=int(row.robust_total_pairs),
                n_permutations=int(row.n_permutations),
                lead_aa_position=aa_position_for_site(int(row.lead_site), lead_orf),
                trail_aa_position=aa_position_for_site(int(row.trail_site), trail_orf),
                lead_orf_source=lead_orf.source,
                trail_orf_source=trail_orf.source,
                nt_feature=f"{base_name}_nt",
                aa_feature=f"{base_name}_aa",
            )
        )
    return features


def encode_features(
    df: pd.DataFrame,
    interactions: Sequence[InteractionFeature],
    *,
    true_root: Path,
    pipeline_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    augmented = df.reset_index(drop=True).copy()
    alignments: Dict[Tuple[str, str], Dict[str, str]] = {}
    orfs: Dict[Tuple[str, str], OrfSpan] = {}
    dictionary_rows: List[Dict[str, object]] = []

    if "species" not in augmented.columns or "tip_name" not in augmented.columns:
        raise ValueError("Input dataset must contain species and tip_name columns.")

    species_series = augmented["species"].astype(str)
    tip_series = augmented["tip_name"].astype(str)

    for interaction in interactions:
        for segment in [interaction.lead_segment, interaction.trail_segment]:
            key = (interaction.species, segment)
            if key not in alignments:
                alignments[key] = load_alignment(pipeline_root, interaction.species, segment)
                orfs[key] = infer_orf(true_root, pipeline_root, interaction.species, segment, alignments[key])

        lead_alignment = alignments[(interaction.species, interaction.lead_segment)]
        trail_alignment = alignments[(interaction.species, interaction.trail_segment)]
        lead_orf = orfs[(interaction.species, interaction.lead_segment)]
        trail_orf = orfs[(interaction.species, interaction.trail_segment)]
        species_mask = species_series.eq(interaction.species)

        nt_values = np.full(augmented.shape[0], "not_applicable", dtype=object)
        aa_values = np.full(augmented.shape[0], "not_applicable", dtype=object)
        for idx, tip in tip_series.loc[species_mask].items():
            lead_seq = lead_alignment.get(tip)
            trail_seq = trail_alignment.get(tip)
            lead_base = base_at(lead_seq, interaction.lead_site)
            trail_base = base_at(trail_seq, interaction.trail_site)
            nt_values[idx] = f"{lead_base}|{trail_base}"
            lead_aa = aa_at(lead_seq, interaction.lead_site, lead_orf)
            trail_aa = aa_at(trail_seq, interaction.trail_site, trail_orf)
            aa_values[idx] = f"{lead_aa}|{trail_aa}"

        augmented[interaction.nt_feature] = nt_values
        augmented[interaction.aa_feature] = aa_values

        in_species = augmented.loc[species_mask]
        nt_counts = in_species[interaction.nt_feature].value_counts(dropna=False)
        aa_counts = in_species[interaction.aa_feature].value_counts(dropna=False)
        dictionary_rows.append(
            {
                "interaction_id": interaction.interaction_id,
                "species": interaction.species,
                "mode": interaction.mode,
                "direction": f"{interaction.lead_segment}->{interaction.trail_segment}",
                "lead_segment": interaction.lead_segment,
                "lead_alignment_site": interaction.lead_site,
                "lead_aa_position": interaction.lead_aa_position,
                "trail_segment": interaction.trail_segment,
                "trail_alignment_site": interaction.trail_site,
                "trail_aa_position": interaction.trail_aa_position,
                "observed_score": interaction.observed_score,
                "empirical_p_ge": interaction.empirical_p_ge,
                "analysis_fdr": interaction.analysis_fdr,
                "analysis_observed_positives": interaction.observed_positives,
                "analysis_expected_false_positives": interaction.expected_false_positives,
                "robust_positive_fraction_p": interaction.robust_positive_fraction_p,
                "robust_observed_positives": interaction.robust_observed_positives,
                "robust_total_pairs": interaction.robust_total_pairs,
                "n_lead_events": interaction.n_lead_events,
                "n_trail_events": interaction.n_trail_events,
                "n_projected_lead_events": interaction.n_projected_lead_events,
                "n_permutations": interaction.n_permutations,
                "nt_feature": interaction.nt_feature,
                "aa_feature": interaction.aa_feature,
                "n_species_rows": int(in_species.shape[0]),
                "n_nt_states": int(nt_counts.shape[0]),
                "top_nt_states": ";".join(f"{state}:{count}" for state, count in nt_counts.head(5).items()),
                "n_aa_states": int(aa_counts.shape[0]),
                "top_aa_states": ";".join(f"{state}:{count}" for state, count in aa_counts.head(5).items()),
                "lead_orf_source": interaction.lead_orf_source,
                "trail_orf_source": interaction.trail_orf_source,
            }
        )

    return augmented, pd.DataFrame(dictionary_rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    reports_dir = args.output_root / "reports"
    datasets_dir = args.output_root / "datasets"
    reports_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    analysis, selected = select_pairs(args.epistasis_root, args)
    analysis_path = reports_dir / "intersegment_epistasis_selected_interaction_analysis_summary.tsv"
    analysis.to_csv(analysis_path, sep="\t", index=False)
    LOGGER.info("Wrote analysis-gate summary: %s", analysis_path)

    if selected.empty:
        message = "No interactions passed the requested FDR/pair/hitchhiking gates."
        if args.strict:
            raise RuntimeError(message)
        LOGGER.warning(message)
        selected_path = reports_dir / "intersegment_epistasis_selected_interactions.tsv"
        pd.DataFrame().to_csv(selected_path, sep="\t", index=False)
        return 0

    interactions = build_interaction_features(
        selected,
        true_root=args.true_reassortment_root,
        pipeline_root=args.pipeline_root,
    )
    dataset = pd.read_csv(args.base_dataset, sep="\t", low_memory=False)
    augmented, dictionary = encode_features(
        dataset,
        interactions,
        true_root=args.true_reassortment_root,
        pipeline_root=args.pipeline_root,
    )

    selected_path = reports_dir / "intersegment_epistasis_selected_interactions.tsv"
    dictionary.to_csv(selected_path, sep="\t", index=False)

    dataset_path = datasets_dir / "sequence_level_final_dataset_direct553_plus_selected_epistasis_interactions.tsv"
    augmented.to_csv(dataset_path, sep="\t", index=False)

    summary = {
        "base_dataset": str(args.base_dataset),
        "augmented_dataset": str(dataset_path),
        "selected_interactions": int(dictionary.shape[0]),
        "new_epistasis_pair_state_columns": int(2 * dictionary.shape[0]),
        "selection_alpha": float(args.alpha),
        "max_analysis_fdr": float(args.max_analysis_fdr),
        "max_robust_positive_fraction_p": float(args.max_robust_positive_fraction_p),
        "max_pair_p": float(args.max_pair_p),
        "modes": args.mode or ["time", "div"],
        "top_pairs_per_analysis": int(args.top_pairs_per_analysis),
        "allow_single_leading_event": bool(args.allow_single_leading_event),
    }
    summary_path = reports_dir / "intersegment_epistasis_selected_interaction_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    LOGGER.info("Wrote selected interactions: %s", selected_path)
    LOGGER.info("Wrote augmented dataset: %s", dataset_path)
    LOGGER.info("Selected %d interactions and added %d pair-state columns.", dictionary.shape[0], 2 * dictionary.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
