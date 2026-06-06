#!/usr/bin/env python3
"""Prepare and run HyPhy FEL scans for species-by-segment hantavirus alignments."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from Bio import Phylo, SeqIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TREE_ROOT = (
    REPO_ROOT
    / "model_test"
    / "runs"
    / "direct516_plus_pathoplexus_andes_egt"
    / "tree_roots_common61"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "specific_evolutionary_analyses"
    / "outputs_direct516_pathoplexus_andes"
    / "hyphy_fel_selection_scan"
)

SPECIES = [
    "orthohantavirus_andesense",
    "orthohantavirus_dobravaense",
    "orthohantavirus_hantanense",
    "orthohantavirus_puumalaense",
    "orthohantavirus_seoulense",
    "orthohantavirus_sinnombreense",
    "orthohantavirus_tulaense",
]
SEGMENTS = ["S", "M", "L"]
STOP_CODONS = {"TAA", "TAG", "TGA"}
VALID_BASES = set("ACGT")


@dataclass
class PrepStats:
    species: str
    segment: str
    n_sequences: int
    input_sites: int
    frame: int
    codon_sites_before_filter: int
    codon_sites_after_filter: int
    dropped_stop_columns: int
    dropped_gap_columns: int
    tree_tips: int
    tree_alignment_tip_match: bool
    alignment_path: Path
    tree_path: Path
    fel_json: Path
    status: str
    note: str = ""
    common_tip_count: int = 0
    segment_tree_tips_before_common_filter: int = 0
    removed_noncommon_tips: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-root", type=Path, default=DEFAULT_TREE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--species", nargs="*", default=SPECIES)
    parser.add_argument("--segments", nargs="*", default=SEGMENTS)
    parser.add_argument("--run-hyphy", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pvalue", type=float, default=0.1)
    parser.add_argument("--max-gap-fraction", type=float, default=0.5)
    parser.add_argument("--hyphy-bin", default="hyphy")
    return parser.parse_args()


def read_alignment(path: Path) -> Dict[str, str]:
    return {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(str(path), "fasta")}


def codon_chunks(seq: str, frame: int) -> List[str]:
    trimmed = seq[frame:]
    usable = len(trimmed) - (len(trimmed) % 3)
    return [trimmed[i : i + 3] for i in range(0, usable, 3)]


def choose_reading_frame(alignment: Dict[str, str]) -> int:
    scores: List[Tuple[int, int, int]] = []
    for frame in range(3):
        stop_count = 0
        informative_codons = 0
        for seq in alignment.values():
            for codon in codon_chunks(seq.replace("-", ""), frame):
                if set(codon) <= VALID_BASES:
                    informative_codons += 1
                    if codon in STOP_CODONS:
                        stop_count += 1
        scores.append((stop_count, -informative_codons, frame))
    return sorted(scores)[0][2]


def filter_codon_alignment(
    alignment: Dict[str, str], frame: int, max_gap_fraction: float
) -> Tuple[Dict[str, str], int, int, int]:
    chunks = {name: codon_chunks(seq, frame) for name, seq in alignment.items()}
    if not chunks:
        raise ValueError("empty alignment")
    n_sites = min(len(v) for v in chunks.values())
    names = list(chunks)
    kept: List[int] = []
    stop_dropped = 0
    gap_dropped = 0
    for idx in range(n_sites):
        column = [chunks[name][idx] for name in names]
        concrete = [codon for codon in column if set(codon) <= VALID_BASES]
        if any(codon in STOP_CODONS for codon in concrete):
            stop_dropped += 1
            continue
        gap_fraction = sum("-" in codon for codon in column) / len(column)
        if gap_fraction > max_gap_fraction:
            gap_dropped += 1
            continue
        kept.append(idx)
    filtered = {name: "".join(chunks[name][idx] for idx in kept) for name in names}
    return filtered, n_sites, stop_dropped, gap_dropped


def write_fasta(seqs: Dict[str, str], path: Path) -> None:
    with path.open("w") as handle:
        for name, seq in seqs.items():
            handle.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def tree_tip_names(path: Path) -> List[str]:
    tree = Phylo.read(str(path), "newick")
    return [tip.name for tip in tree.get_terminals()]


def species_common_tip_sets(
    tree_root: Path, species_list: Sequence[str], segment_list: Sequence[str]
) -> Dict[str, set[str]]:
    common_by_species: Dict[str, set[str]] = {}
    for species in species_list:
        segment_sets: List[set[str]] = []
        for segment in segment_list:
            source_tree = tree_root / species / "augur" / segment / f"{segment}_tree_refined.newick"
            if source_tree.exists():
                segment_sets.append(set(tree_tip_names(source_tree)))
        common_by_species[species] = set.intersection(*segment_sets) if segment_sets else set()
    return common_by_species


def write_pruned_tree(source_tree: Path, keep_tips: set[str], output_tree: Path) -> None:
    tree = Phylo.read(str(source_tree), "newick")
    for tip in list(tree.get_terminals()):
        if tip.name not in keep_tips:
            tree.prune(tip)
    Phylo.write(tree, str(output_tree), "newick")


def run_hyphy(
    hyphy_bin: str,
    alignment_path: Path,
    tree_path: Path,
    output_path: Path,
    pvalue: float,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        hyphy_bin,
        "fel",
        "--alignment",
        str(alignment_path),
        "--tree",
        str(tree_path),
        "--branches",
        "All",
        "--srv",
        "Yes",
        "--multiple-hits",
        "None",
        "--pvalue",
        str(pvalue),
        "--ci",
        "No",
        "--output",
        str(output_path),
    ]
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def summarize_fel_json(path: Path, pvalue: float) -> Dict[str, float]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    table = data.get("MLE", {}).get("content", {}).get("0", [])
    headers = data.get("MLE", {}).get("headers", [])
    header_names = []
    for header in headers:
        if isinstance(header, dict):
            header_names.append(str(header.get("0", "")))
        elif isinstance(header, (list, tuple)) and header:
            header_names.append(str(header[0]))
        else:
            header_names.append(str(header))
    try:
        alpha_idx = header_names.index("alpha")
        beta_idx = header_names.index("beta")
        p_idx = header_names.index("p-value")
    except ValueError:
        return {}
    tested = 0
    negative = 0
    positive = 0
    median_alpha = []
    median_beta = []
    for row in table:
        try:
            alpha = float(row[alpha_idx])
            beta = float(row[beta_idx])
            p = float(row[p_idx])
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(alpha) and math.isfinite(beta) and math.isfinite(p)):
            continue
        tested += 1
        median_alpha.append(alpha)
        median_beta.append(beta)
        if p <= pvalue and beta < alpha:
            negative += 1
        elif p <= pvalue and beta > alpha:
            positive += 1
    return {
        "fel_tested_sites": tested,
        "fel_negative_sites": negative,
        "fel_positive_sites": positive,
        "fel_negative_fraction": negative / tested if tested else math.nan,
        "fel_positive_fraction": positive / tested if tested else math.nan,
        "fel_median_alpha": float(pd.Series(median_alpha).median()) if median_alpha else math.nan,
        "fel_median_beta": float(pd.Series(median_beta).median()) if median_beta else math.nan,
    }


def main() -> int:
    args = parse_args()
    input_dir = args.output_dir / "inputs"
    report_dir = args.output_dir / "reports"
    log_dir = args.output_dir / "logs"
    for path in [input_dir, report_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    common_tips_by_species = species_common_tip_sets(args.tree_root, args.species, args.segments)
    prep_rows: List[PrepStats] = []
    summary_rows: List[Dict[str, object]] = []
    for species in args.species:
        for segment in args.segments:
            source_alignment = args.tree_root / species / "augur" / segment / f"{segment}_aligned.fasta"
            source_tree = args.tree_root / species / "augur" / segment / f"{segment}_tree_refined.newick"
            out_prefix = f"{species}_{segment}"
            prepared_alignment = input_dir / f"{out_prefix}_codon.fasta"
            prepared_tree = input_dir / f"{out_prefix}_common_tree.newick"
            fel_json = report_dir / f"{out_prefix}_FEL.json"
            log_path = log_dir / f"{out_prefix}_hyphy.log"
            try:
                if not source_alignment.exists() or not source_tree.exists():
                    raise FileNotFoundError("missing alignment or tree")
                tips = tree_tip_names(source_tree)
                common_tips = common_tips_by_species.get(species, set())
                if not common_tips:
                    raise ValueError("no common tips across selected segments")
                aln = read_alignment(source_alignment)
                alignment_only = set(aln) - set(tips)
                tree_only = set(tips) - set(aln)
                note = ""
                if tree_only:
                    note = f"tree/alignment mismatch: {len(alignment_only)} alignment-only, {len(tree_only)} tree-only"
                elif alignment_only:
                    aln = {name: aln[name] for name in tips if name in aln}
                    note = f"pruned {len(alignment_only)} alignment-only tips to match tree"
                noncommon_tips = set(tips) - common_tips
                if noncommon_tips:
                    aln = {name: seq for name, seq in aln.items() if name in common_tips}
                    common_note = f"removed {len(noncommon_tips)} noncommon segment tips"
                    note = f"{note}; {common_note}" if note else common_note
                write_pruned_tree(source_tree, common_tips, prepared_tree)
                frame = choose_reading_frame(aln)
                codon_aln, n_sites, stop_drop, gap_drop = filter_codon_alignment(
                    aln, frame, args.max_gap_fraction
                )
                prepared_tips = set(tree_tip_names(prepared_tree))
                tree_match = set(codon_aln) == prepared_tips
                write_fasta(codon_aln, prepared_alignment)
                status = "prepared"
                if not tree_match and not note:
                    note = f"tree/alignment mismatch: {len(set(codon_aln) - prepared_tips)} alignment-only, {len(prepared_tips - set(codon_aln))} tree-only"
                if args.run_hyphy:
                    if fel_json.exists() and fel_json.stat().st_size > 0 and not args.force:
                        status = "hyphy_cached"
                    elif not tree_match:
                        status = "skipped"
                    else:
                        proc = run_hyphy(
                            args.hyphy_bin,
                            prepared_alignment,
                            prepared_tree,
                            fel_json,
                            args.pvalue,
                        )
                        log_path.write_text(proc.stdout)
                        status = "hyphy_ok" if proc.returncode == 0 and fel_json.exists() else "hyphy_failed"
                        if proc.returncode != 0:
                            note = f"HyPhy exit {proc.returncode}"
                prep = PrepStats(
                    species=species,
                    segment=segment,
                    n_sequences=len(codon_aln),
                    input_sites=len(next(iter(aln.values()))) if aln else 0,
                    frame=frame,
                    codon_sites_before_filter=n_sites,
                    codon_sites_after_filter=len(next(iter(codon_aln.values()))) // 3 if codon_aln else 0,
                    dropped_stop_columns=stop_drop,
                    dropped_gap_columns=gap_drop,
                    tree_tips=len(prepared_tips),
                    tree_alignment_tip_match=tree_match,
                    alignment_path=prepared_alignment,
                    tree_path=prepared_tree,
                    fel_json=fel_json,
                    status=status,
                    note=note,
                    common_tip_count=len(common_tips),
                    segment_tree_tips_before_common_filter=len(tips),
                    removed_noncommon_tips=len(noncommon_tips),
                )
            except Exception as exc:  # noqa: BLE001
                prep = PrepStats(
                    species=species,
                    segment=segment,
                    n_sequences=0,
                    input_sites=0,
                    frame=-1,
                    codon_sites_before_filter=0,
                    codon_sites_after_filter=0,
                    dropped_stop_columns=0,
                    dropped_gap_columns=0,
                    tree_tips=0,
                    tree_alignment_tip_match=False,
                    alignment_path=prepared_alignment,
                    tree_path=source_tree,
                    fel_json=fel_json,
                    status="error",
                    note=str(exc),
                )
            prep_rows.append(prep)
            row: Dict[str, object] = {
                **prep.__dict__,
                **summarize_fel_json(fel_json, args.pvalue),
            }
            summary_rows.append(row)
            print(f"{prep.species}\t{prep.segment}\t{prep.status}\t{prep.codon_sites_after_filter} codons\t{prep.note}")

    pd.DataFrame([p.__dict__ for p in prep_rows]).to_csv(report_dir / "fel_input_preparation_summary.tsv", sep="\t", index=False)
    pd.DataFrame(summary_rows).to_csv(report_dir / "fel_selection_summary.tsv", sep="\t", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
