#!/usr/bin/env python3
"""Build intersegment epistasis features.

This implements a pragmatic Kryazhimskiy/Neverov-style screen using existing
segment trees and Augur ancestral reconstructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from Bio import Phylo


LOGGER = logging.getLogger("build_intersegment_epistasis_features")
REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENTS = ("S", "M", "L")
SEGMENT_PAIRS = (("S", "M"), ("S", "L"), ("M", "L"))
FDR_DEFAULT_ALPHAS = (0.05, 0.01)
MUT_RE = re.compile(r"^([ACGTN-])(\d+)([ACGTN-])$")
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

    @property
    def n_codons(self) -> int:
        return max(0, (self.end0 - self.start0) // 3)


@dataclass(frozen=True)
class MutationEvent:
    species: str
    segment: str
    node: str
    parent: str
    site: int
    ancestral_base: str
    derived_base: str
    terminal_branch: bool
    is_nonsynonymous: bool
    aa_from: str
    aa_to: str
    codon_index: Optional[int]
    descendant_tips: frozenset[str]
    numdate: float
    div: float
    mutcount_cum: float

    @property
    def primary_included(self) -> bool:
        return (not self.terminal_branch) and self.is_nonsynonymous


@dataclass
class SegmentData:
    species: str
    segment: str
    tree_path: Path
    node_data_path: Optional[Path]
    ancestral_path: Path
    tree: object
    clade_by_name: Dict[str, object]
    parent_by_name: Dict[str, str]
    terminal_by_name: Dict[str, bool]
    desc_tips_by_name: Dict[str, frozenset[str]]
    tip_names: frozenset[str]
    numdate_by_name: Dict[str, float]
    div_by_name: Dict[str, float]
    mutcount_cum_by_name: Dict[str, float]
    branch_mut_count_by_name: Dict[str, int]
    events_by_site: Dict[int, List[MutationEvent]]
    orf_span: OrfSpan
    div_source: str
    qc: Dict[str, object]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build intersegment epistasis features from ASR node mutations and segment trees."
    )
    parser.add_argument(
        "--true-reassortment-root",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "pathoplexus_andes_expansion" / "true_reassortment_mixed",
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "pathoplexus_andes_expansion" / "mixed_pipeline",
    )
    parser.add_argument(
        "--ld-root",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "ld_expanded_all_backfills_pipeline",
    )
    parser.add_argument(
        "--species-file",
        type=Path,
        default=None,
        help="Optional newline-delimited species list. If omitted, species are inferred from the input roots.",
    )
    parser.add_argument(
        "--species",
        action="append",
        default=[],
        help="Restrict to one species slug. Can be passed multiple times.",
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=REPO_ROOT
        / "model_test"
        / "runs"
        / "model_site"
        / "datasets"
        / "sequence_level_final_dataset_combined_relaxed_andesense_all_backfills_full_geooverlap_manual_plus_tula_coords_main.tsv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "model_test" / "runs" / "intersegment_epistasis_fullscan_direct553",
    )
    parser.add_argument(
        "--candidate-source",
        choices=["all_nonsyn", "ld_top"],
        default="all_nonsyn",
        help="all_nonsyn scans all intersegment sites with internal nonsynonymous events; ld_top uses LD-ranked site pairs.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help="Per-pair null permutations. Use 0 for full scans unless null z/p features are specifically needed.",
    )
    parser.add_argument("--top-ld-pairs-per-segment-pair", type=int, default=250)
    parser.add_argument("--min-events-per-site", type=int, default=1)
    parser.add_argument("--tau-time", type=float, default=5.0, help="Exponential scale in years for time scores.")
    parser.add_argument("--tau-div", type=float, default=0.01, help="Exponential scale for genetic-divergence scores.")
    parser.add_argument("--tau-mutcount", type=float, default=5.0, help="Exponential scale for mutation-count audit scores.")
    parser.add_argument(
        "--fdr-permutations",
        type=int,
        default=0,
        help="Paper-style simultaneous-null permutations for FDR. Set to 10000 for the full paper setting.",
    )
    parser.add_argument(
        "--fdr-fake-datasets",
        type=int,
        default=400,
        help="Number of null permutations to use as fake datasets for FDR estimation.",
    )
    parser.add_argument(
        "--fdr-alpha",
        action="append",
        type=float,
        default=None,
        help="Nominal P-value threshold for FDR summaries. Can be passed multiple times.",
    )
    parser.add_argument(
        "--fdr-mode",
        action="append",
        choices=["time", "div", "mutcount_audit"],
        default=None,
        help="Distance mode(s) for FDR testing. Defaults to time and div.",
    )
    parser.add_argument(
        "--hitchhiking-min-leading-events",
        type=int,
        default=2,
        help="Minimum internal nonsynonymous leading events for the hitchhiking-robust fraction-positive test.",
    )
    parser.add_argument(
        "--fdr-report-max-pairs-per-analysis",
        type=int,
        default=5000,
        help="Maximum significant site-pair rows to retain per species/ordered segment/mode FDR analysis.",
    )
    parser.add_argument(
        "--fdr-checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for per-species/order FDR checkpoints. Defaults to reports/fdr_checkpoints.",
    )
    parser.add_argument(
        "--no-resume-fdr",
        action="store_true",
        help="Recompute FDR checkpoints instead of loading compatible existing checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument(
        "--coding-spans",
        type=Path,
        default=None,
        help="Optional TSV with species, segment, start_1based, end_1based_inclusive coding spans.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


def discover_species(true_root: Path, pipeline_root: Path) -> List[str]:
    species: Set[str] = set()
    for root in [true_root, pipeline_root]:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if any((child / segment).exists() for segment in SEGMENTS) or (child / "augur").exists():
                species.add(child.name)
    return sorted(species)


def read_species(
    path: Optional[Path],
    requested: Sequence[str],
    true_root: Path,
    pipeline_root: Path,
) -> List[str]:
    if requested:
        return list(dict.fromkeys(requested))
    if path is None:
        species = discover_species(true_root, pipeline_root)
        if not species:
            raise FileNotFoundError(
                f"No species could be inferred from {true_root} or {pipeline_root}; pass --species-file or --species."
            )
        return species
    if not path.exists():
        raise FileNotFoundError(f"Species file not found: {path}")
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def parse_mutation(mut: object) -> Optional[Tuple[str, int, str]]:
    match = MUT_RE.match(str(mut or "").strip().upper())
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


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
                best = (start, end, "longest_atg_to_stop" if stop is not None else "longest_atg_to_end")
    if best and (best[1] - best[0]) >= 300:
        return OrfSpan(best[0], best[1], best[2])

    # Fallback: longest stop-free reading frame, preserving alignment coordinates.
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


def load_coding_spans(path: Optional[Path]) -> Dict[Tuple[str, str], OrfSpan]:
    if path is None:
        return {}
    df = pd.read_csv(path, sep="\t")
    required = {"species", "segment", "start_1based", "end_1based_inclusive"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coding span file is missing columns: {sorted(missing)}")
    out: Dict[Tuple[str, str], OrfSpan] = {}
    for row in df.itertuples(index=False):
        start0 = int(getattr(row, "start_1based")) - 1
        end0 = int(getattr(row, "end_1based_inclusive"))
        out[(str(getattr(row, "species")), str(getattr(row, "segment")))] = OrfSpan(
            start0=start0,
            end0=end0,
            source="coding_spans_override",
        )
    return out


def load_tree(path: Path) -> object:
    schema = "nexus" if path.suffix.lower() in {".nex", ".nexus"} else "newick"
    return Phylo.read(str(path), schema)


def build_parent_map(tree: object) -> Dict[object, object]:
    parent: Dict[object, object] = {}
    for clade in tree.find_clades(order="level"):
        for child in getattr(clade, "clades", []):
            parent[child] = clade
    return parent


def compute_descendant_tips(tree: object) -> Dict[object, frozenset[str]]:
    desc: Dict[object, frozenset[str]] = {}
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            desc[clade] = frozenset([clade.name] if clade.name else [])
        else:
            tips: Set[str] = set()
            for child in getattr(clade, "clades", []):
                tips.update(desc.get(child, frozenset()))
            desc[clade] = frozenset(tips)
    return desc


def load_node_data(path: Optional[Path]) -> Dict[str, Mapping[str, object]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    nodes = data.get("nodes", {})
    return nodes if isinstance(nodes, dict) else {}


def load_ancestral_data(path: Path) -> Dict[str, Mapping[str, object]]:
    data = json.loads(path.read_text())
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError(f"Ancestral node data has no node map: {path}")
    return nodes


def tree_path_candidates(pipeline_root: Path, true_root: Path, species: str, segment: str) -> List[Path]:
    return [
        pipeline_root / species / "augur" / segment / f"{segment}_tree_refined.newick",
        pipeline_root / species / "augur" / segment / f"{segment}_tree_refined.nwk",
        pipeline_root / species / "augur" / segment / f"{segment}_tree_raw.nwk",
        true_root / species / segment / f"{segment}_tree_common_binary.newick",
        true_root / species / segment / f"{segment}_tree_common_binary.nwk",
        true_root / species / segment / f"{segment}_tree_common_binary_annotated.nexus",
    ]


def load_segment_data(
    *,
    species: str,
    segment: str,
    true_root: Path,
    pipeline_root: Path,
    candidate_sites: Optional[Set[int]],
    coding_spans: Mapping[Tuple[str, str], OrfSpan],
) -> SegmentData:
    tree_path = first_existing(tree_path_candidates(pipeline_root, true_root, species, segment))
    if tree_path is None:
        raise FileNotFoundError(f"No tree found for {species} {segment}")
    node_data_path = first_existing(
        [
            pipeline_root / species / "augur" / segment / f"{segment}_node_data.json",
            pipeline_root / species / "augur" / segment / f"{segment}_branch_lengths.json",
        ]
    )
    ancestral_path = true_root / species / segment / f"{segment}_ancestral_node_data.json"
    if not ancestral_path.exists():
        raise FileNotFoundError(f"No ancestral node data found for {species} {segment}: {ancestral_path}")

    tree = load_tree(tree_path)
    node_data = load_node_data(node_data_path)
    ancestral = load_ancestral_data(ancestral_path)
    parent_by_clade = build_parent_map(tree)
    desc_by_clade = compute_descendant_tips(tree)

    clade_by_name: Dict[str, object] = {}
    parent_by_name: Dict[str, str] = {}
    terminal_by_name: Dict[str, bool] = {}
    desc_tips_by_name: Dict[str, frozenset[str]] = {}
    unnamed_internal = 0
    duplicate_names = 0
    for clade in tree.find_clades(order="preorder"):
        name = getattr(clade, "name", None)
        if not name:
            if not clade.is_terminal():
                unnamed_internal += 1
            continue
        if name in clade_by_name:
            duplicate_names += 1
            continue
        clade_by_name[name] = clade
        terminal_by_name[name] = bool(clade.is_terminal())
        desc_tips_by_name[name] = desc_by_clade.get(clade, frozenset())
        parent = parent_by_clade.get(clade)
        parent_name = getattr(parent, "name", None) if parent is not None else None
        if parent_name:
            parent_by_name[name] = parent_name

    root = tree.root
    root_name = getattr(root, "name", None)

    branch_mut_count_by_name: Dict[str, int] = {}
    for name in clade_by_name:
        muts = ancestral.get(name, {}).get("muts", [])
        branch_mut_count_by_name[name] = len(muts) if isinstance(muts, list) else 0

    has_cumulative_div = any("div" in payload for payload in node_data.values() if isinstance(payload, Mapping))
    div_source = "node_div" if has_cumulative_div else "cumulative_mutation_length"
    numdate_by_name: Dict[str, float] = {}
    div_by_name: Dict[str, float] = {}
    mutcount_cum_by_name: Dict[str, float] = {}
    for clade in tree.find_clades(order="preorder"):
        name = getattr(clade, "name", None)
        if not name or name not in clade_by_name:
            continue
        payload = node_data.get(name, {})
        parent = parent_by_clade.get(clade)
        parent_name = getattr(parent, "name", None) if parent is not None else None
        numdate_by_name[name] = pd.to_numeric(payload.get("numdate"), errors="coerce") if payload else np.nan
        if has_cumulative_div and payload and pd.notna(pd.to_numeric(payload.get("div"), errors="coerce")):
            div_by_name[name] = float(pd.to_numeric(payload.get("div"), errors="coerce"))
        elif parent_name and parent_name in div_by_name:
            branch_div = pd.to_numeric(payload.get("mutation_length"), errors="coerce") if payload else np.nan
            div_by_name[name] = float(div_by_name[parent_name] + (0.0 if pd.isna(branch_div) else branch_div))
        else:
            div_by_name[name] = 0.0
        if parent_name and parent_name in mutcount_cum_by_name:
            mutcount_cum_by_name[name] = mutcount_cum_by_name[parent_name] + branch_mut_count_by_name.get(name, 0)
        else:
            mutcount_cum_by_name[name] = 0.0

    root_seq = None
    if root_name and root_name in ancestral:
        root_seq = ancestral[root_name].get("sequence")
    if root_seq is None and ancestral:
        root_seq = next((payload.get("sequence") for payload in ancestral.values() if payload.get("sequence")), "")
    override = coding_spans.get((species, segment))
    orf_span = override if override is not None else infer_longest_orf(str(root_seq or ""))

    site_filter_applied = candidate_sites is not None
    site_filter = set(candidate_sites or set())
    events_by_site: Dict[int, List[MutationEvent]] = {site: [] for site in site_filter}
    parsed_muts = 0
    candidate_muts = 0
    nonsyn_muts = 0
    terminal_muts = 0
    for name, clade in clade_by_name.items():
        if name == root_name:
            continue
        payload = ancestral.get(name)
        if not payload:
            continue
        muts = payload.get("muts", [])
        child_seq = str(payload.get("sequence", "")).upper()
        if not isinstance(muts, list) or not child_seq:
            continue
        parsed = [parse_mutation(mut) for mut in muts]
        parsed = [mut for mut in parsed if mut is not None]
        parsed_muts += len(parsed)
        pre_seq = list(child_seq)
        for anc_base, pos, _der_base in parsed:
            if 1 <= pos <= len(pre_seq):
                pre_seq[pos - 1] = anc_base
        pre_seq_s = "".join(pre_seq)
        parent_name = parent_by_name.get(name, "")
        for anc_base, pos, der_base in parsed:
            if site_filter_applied and pos not in site_filter:
                continue
            candidate_muts += 1
            codon_index: Optional[int] = None
            aa_from = ""
            aa_to = ""
            is_nonsyn = False
            if (
                orf_span.start0 <= pos - 1 < orf_span.end0
                and anc_base in VALID_BASES
                and der_base in VALID_BASES
            ):
                codon_start = orf_span.start0 + 3 * ((pos - 1 - orf_span.start0) // 3)
                codon_pre = pre_seq_s[codon_start : codon_start + 3]
                codon_post = child_seq[codon_start : codon_start + 3]
                aa_from = translate_codon(codon_pre)
                aa_to = translate_codon(codon_post)
                codon_index = ((codon_start - orf_span.start0) // 3) + 1
                is_nonsyn = aa_from != "X" and aa_to != "X" and aa_from != aa_to
            terminal = bool(terminal_by_name.get(name, False))
            terminal_muts += int(terminal)
            nonsyn_muts += int(is_nonsyn)
            events_by_site.setdefault(pos, []).append(
                MutationEvent(
                    species=species,
                    segment=segment,
                    node=name,
                    parent=parent_name,
                    site=pos,
                    ancestral_base=anc_base,
                    derived_base=der_base,
                    terminal_branch=terminal,
                    is_nonsynonymous=is_nonsyn,
                    aa_from=aa_from,
                    aa_to=aa_to,
                    codon_index=codon_index,
                    descendant_tips=desc_tips_by_name.get(name, frozenset()),
                    numdate=float(numdate_by_name.get(name, np.nan)),
                    div=float(div_by_name.get(name, np.nan)),
                    mutcount_cum=float(mutcount_cum_by_name.get(name, np.nan)),
                )
            )

    qc = {
        "species": species,
        "segment": segment,
        "tree_path": str(tree_path),
        "node_data_path": "" if node_data_path is None else str(node_data_path),
        "ancestral_path": str(ancestral_path),
        "n_tree_nodes_named": int(len(clade_by_name)),
        "n_tree_tips": int(len(tree.get_terminals())),
        "n_ancestral_nodes": int(len(ancestral)),
        "n_node_data_nodes": int(len(node_data)),
        "n_named_tree_nodes_with_ancestral": int(len(set(clade_by_name) & set(ancestral))),
        "n_named_tree_nodes_with_node_data": int(len(set(clade_by_name) & set(node_data))),
        "n_unnamed_internal_nodes": int(unnamed_internal),
        "n_duplicate_names": int(duplicate_names),
        "orf_start_1based": int(orf_span.start0 + 1),
        "orf_end_1based_inclusive": int(orf_span.end0),
        "orf_n_codons": int(orf_span.n_codons),
        "orf_source": orf_span.source,
        "div_source": div_source,
        "site_filter_mode": "provided_sites" if site_filter_applied else "all_sites",
        "n_candidate_sites": int(len(site_filter) if site_filter_applied else len(events_by_site)),
        "n_branch_muts_parsed": int(parsed_muts),
        "n_candidate_site_muts": int(candidate_muts),
        "n_candidate_site_nonsyn_muts": int(nonsyn_muts),
        "n_candidate_site_terminal_muts": int(terminal_muts),
    }
    return SegmentData(
        species=species,
        segment=segment,
        tree_path=tree_path,
        node_data_path=node_data_path,
        ancestral_path=ancestral_path,
        tree=tree,
        clade_by_name=clade_by_name,
        parent_by_name=parent_by_name,
        terminal_by_name=terminal_by_name,
        desc_tips_by_name=desc_tips_by_name,
        tip_names=frozenset(tip.name for tip in tree.get_terminals() if tip.name),
        numdate_by_name=numdate_by_name,
        div_by_name=div_by_name,
        mutcount_cum_by_name=mutcount_cum_by_name,
        branch_mut_count_by_name=branch_mut_count_by_name,
        events_by_site=events_by_site,
        orf_span=orf_span,
        div_source=div_source,
        qc=qc,
    )


def load_candidate_pairs(ld_root: Path, species: str, top_n: int) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for seg_a, seg_b in SEGMENT_PAIRS:
        path = ld_root / species / "ld" / "conservative" / f"{seg_a}_{seg_b}_LD.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["CramerV"] = pd.to_numeric(df.get("CramerV"), errors="coerce")
        df = df[np.isfinite(df["CramerV"])].copy()
        if df.empty:
            continue
        df = df.sort_values("CramerV", ascending=False).head(top_n).copy()
        df["segmentA"] = seg_a
        df["segmentB"] = seg_b
        df["site1"] = pd.to_numeric(df["site1"], errors="coerce").astype("Int64")
        df["site2"] = pd.to_numeric(df["site2"], errors="coerce").astype("Int64")
        rows.append(df[["segmentA", "segmentB", "site1", "site2", "CramerV", "NMI", "n_strains"]])
    if not rows:
        return pd.DataFrame(columns=["species", "segmentA", "segmentB", "site1", "site2", "CramerV", "NMI", "n_strains"])
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "species", species)
    out["candidate_source"] = "ld_top"
    return out


def primary_sites(data: SegmentData) -> List[int]:
    return sorted(
        site
        for site, events in data.events_by_site.items()
        if any(event.primary_included for event in events)
    )


def build_all_nonsyn_candidate_pairs(species: str, segments: Mapping[str, SegmentData]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    sites_by_segment = {segment: primary_sites(data) for segment, data in segments.items()}
    for seg_a, seg_b in SEGMENT_PAIRS:
        sites_a = sites_by_segment.get(seg_a, [])
        sites_b = sites_by_segment.get(seg_b, [])
        for site_a in sites_a:
            for site_b in sites_b:
                rows.append(
                    {
                        "species": species,
                        "segmentA": seg_a,
                        "segmentB": seg_b,
                        "site1": int(site_a),
                        "site2": int(site_b),
                        "CramerV": np.nan,
                        "NMI": np.nan,
                        "n_strains": 0,
                        "candidate_source": "all_nonsyn",
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["species", "segmentA", "segmentB", "site1", "site2", "CramerV", "NMI", "n_strains", "candidate_source"],
    )


def required_sites_by_segment(candidate_df: pd.DataFrame) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {seg: set() for seg in SEGMENTS}
    for row in candidate_df.itertuples(index=False):
        if pd.notna(row.site1):
            out[str(row.segmentA)].add(int(row.site1))
        if pd.notna(row.site2):
            out[str(row.segmentB)].add(int(row.site2))
    return out


def segment_pair_label(seg_a: str, seg_b: str) -> str:
    ordered = [seg for seg in SEGMENTS if seg in {seg_a, seg_b}]
    return "".join(ordered)


def metric_value(data: SegmentData, node: str, mode: str) -> float:
    if mode == "time":
        return float(data.numdate_by_name.get(node, np.nan))
    if mode == "div":
        return float(data.div_by_name.get(node, np.nan))
    if mode == "mutcount_audit":
        return float(data.mutcount_cum_by_name.get(node, np.nan))
    raise ValueError(f"Unsupported mode: {mode}")


def tau_for_mode(args: argparse.Namespace, mode: str) -> float:
    if mode == "time":
        return float(args.tau_time)
    if mode == "div":
        return float(args.tau_div)
    if mode == "mutcount_audit":
        return float(args.tau_mutcount)
    raise ValueError(f"Unsupported mode: {mode}")


def project_to_segment(
    lead_event: MutationEvent,
    target: SegmentData,
    cache: Dict[frozenset[str], Optional[str]],
) -> Tuple[Optional[str], int]:
    shared = frozenset(lead_event.descendant_tips & target.tip_names)
    if not shared:
        return None, 0
    if shared in cache:
        return cache[shared], len(shared)
    try:
        clade = target.tree.common_ancestor(list(shared))
    except Exception:
        cache[shared] = None
        return None, len(shared)
    name = getattr(clade, "name", None)
    cache[shared] = name if name in target.desc_tips_by_name else None
    return cache[shared], len(shared)


def compute_score_for_nodes(
    *,
    projection_nodes: Sequence[str],
    trail_nodes: Sequence[str],
    target: SegmentData,
    mode: str,
    tau: float,
) -> Tuple[float, int, int, int]:
    score = 0.0
    n_links = 0
    n_same = 0
    n_negative = 0
    if tau <= 0:
        raise ValueError("Tau must be positive.")
    for proj in projection_nodes:
        proj_desc = target.desc_tips_by_name.get(proj)
        proj_value = metric_value(target, proj, mode)
        if proj_desc is None or not np.isfinite(proj_value):
            continue
        for trail in trail_nodes:
            trail_desc = target.desc_tips_by_name.get(trail)
            trail_value = metric_value(target, trail, mode)
            if trail_desc is None or not np.isfinite(trail_value):
                continue
            if not trail_desc.issubset(proj_desc):
                continue
            distance = trail_value - proj_value
            if distance < 0:
                n_negative += 1
                continue
            n_links += 1
            n_same += int(trail == proj)
            score += math.exp(-float(distance) / tau)
    return float(score), int(n_links), int(n_same), int(n_negative)


def score_ordered_pair(
    *,
    species: str,
    segment_a: str,
    site_a: int,
    segment_b: str,
    site_b: int,
    cramer_v: float,
    nmi: float,
    n_strains: int,
    segments: Mapping[str, SegmentData],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    lead = segments[segment_a]
    trail = segments[segment_b]
    lead_events = [event for event in lead.events_by_site.get(site_a, []) if event.primary_included]
    trail_events = [event for event in trail.events_by_site.get(site_b, []) if event.primary_included]
    segment_pair = segment_pair_label(segment_a, segment_b)

    base = {
        "species": species,
        "segment_pair": segment_pair,
        "lead_segment": segment_a,
        "trail_segment": segment_b,
        "lead_site": int(site_a),
        "trail_site": int(site_b),
        "cramers_v": float(cramer_v) if pd.notna(cramer_v) else np.nan,
        "nmi": float(nmi) if pd.notna(nmi) else np.nan,
        "n_strains": int(n_strains) if pd.notna(n_strains) else 0,
        "n_lead_events": int(len(lead_events)),
        "n_trail_events": int(len(trail_events)),
    }
    modes = ["time", "div", "mutcount_audit"]
    if len(lead_events) < args.min_events_per_site or len(trail_events) < args.min_events_per_site:
        return [
            {
                **base,
                "mode": mode,
                "status": "insufficient_primary_events",
                "n_projected_leads": 0,
                "n_projection_shared_tips_total": 0,
                "observed_score": np.nan,
                "null_mean": np.nan,
                "null_sd": np.nan,
                "z_score": np.nan,
                "empirical_p_ge": np.nan,
                "n_scored_links": 0,
                "n_same_projected_branch_links": 0,
                "n_negative_distance_links": 0,
                "n_permutations": int(args.n_permutations),
            }
            for mode in modes
        ]

    projection_cache: Dict[frozenset[str], Optional[str]] = {}
    projection_nodes: List[str] = []
    n_shared_total = 0
    for event in lead_events:
        projection, n_shared = project_to_segment(event, trail, projection_cache)
        n_shared_total += n_shared
        if projection is not None:
            projection_nodes.append(projection)
    trail_nodes = [event.node for event in trail_events]
    eligible_nodes = [
        name
        for name, terminal in trail.terminal_by_name.items()
        if not terminal and name in trail.desc_tips_by_name and name in trail.parent_by_name
    ]
    if not projection_nodes or not eligible_nodes:
        return [
            {
                **base,
                "mode": mode,
                "status": "no_projected_leads_or_null_pool",
                "n_projected_leads": int(len(projection_nodes)),
                "n_projection_shared_tips_total": int(n_shared_total),
                "observed_score": np.nan,
                "null_mean": np.nan,
                "null_sd": np.nan,
                "z_score": np.nan,
                "empirical_p_ge": np.nan,
                "n_scored_links": 0,
                "n_same_projected_branch_links": 0,
                "n_negative_distance_links": 0,
                "n_permutations": int(args.n_permutations),
            }
            for mode in modes
        ]

    rows: List[Dict[str, object]] = []
    for mode in modes:
        tau = tau_for_mode(args, mode)
        observed, n_links, n_same, n_negative = compute_score_for_nodes(
            projection_nodes=projection_nodes,
            trail_nodes=trail_nodes,
            target=trail,
            mode=mode,
            tau=tau,
        )
        null_scores: List[float] = []
        replace = len(trail_nodes) > len(eligible_nodes)
        for _idx in range(int(args.n_permutations)):
            sampled = rng.choice(eligible_nodes, size=len(trail_nodes), replace=replace)
            null_score, _null_links, _null_same, _null_negative = compute_score_for_nodes(
                projection_nodes=projection_nodes,
                trail_nodes=[str(node) for node in sampled],
                target=trail,
                mode=mode,
                tau=tau,
            )
            null_scores.append(null_score)
        null = np.asarray(null_scores, dtype=float)
        null_mean = float(np.nanmean(null)) if null.size else np.nan
        null_sd = float(np.nanstd(null, ddof=1)) if null.size > 1 else np.nan
        z_score = np.nan if not np.isfinite(null_sd) or null_sd == 0 else float((observed - null_mean) / null_sd)
        p_ge = np.nan
        if null.size:
            p_ge = float((np.sum(null >= observed) + 1.0) / (null.size + 1.0))
        rows.append(
            {
                **base,
                "mode": mode,
                "status": "ok" if n_links > 0 else "no_downstream_links",
                "n_projected_leads": int(len(projection_nodes)),
                "n_projection_shared_tips_total": int(n_shared_total),
                "observed_score": float(observed),
                "null_mean": null_mean,
                "null_sd": null_sd,
                "z_score": z_score,
                "empirical_p_ge": p_ge,
                "n_scored_links": int(n_links),
                "n_same_projected_branch_links": int(n_same),
                "n_negative_distance_links": int(n_negative),
                "n_permutations": int(args.n_permutations),
            }
        )
    return rows


def fdr_modes(args: argparse.Namespace) -> List[str]:
    return list(dict.fromkeys(args.fdr_mode or ["time", "div"]))


def fdr_alphas(args: argparse.Namespace) -> List[float]:
    values = args.fdr_alpha if args.fdr_alpha else list(FDR_DEFAULT_ALPHAS)
    return sorted({float(value) for value in values if 0.0 < float(value) < 1.0}, reverse=True)


def eligible_internal_nodes(data: SegmentData) -> List[str]:
    return [
        name
        for name, terminal in data.terminal_by_name.items()
        if not terminal and name in data.desc_tips_by_name and name in data.parent_by_name
    ]


def build_lead_weight_matrix(
    *,
    lead_sites: Sequence[int],
    lead_events_by_site: Mapping[int, Sequence[MutationEvent]],
    target: SegmentData,
    eligible_nodes: Sequence[str],
    mode: str,
    tau: float,
) -> Tuple[np.ndarray, List[int], List[int]]:
    node_desc = [target.desc_tips_by_name.get(node, frozenset()) for node in eligible_nodes]
    node_values = np.asarray([metric_value(target, node, mode) for node in eligible_nodes], dtype=float)
    weights = np.zeros((len(lead_sites), len(eligible_nodes)), dtype=float)
    lead_event_counts: List[int] = []
    projection_counts: List[int] = []
    projection_cache: Dict[frozenset[str], Optional[str]] = {}
    node_index = {node: idx for idx, node in enumerate(eligible_nodes)}
    del node_index
    for lead_i, site in enumerate(lead_sites):
        events = list(lead_events_by_site.get(int(site), []))
        lead_event_counts.append(len(events))
        projections: List[str] = []
        for event in events:
            projected, _shared = project_to_segment(event, target, projection_cache)
            if projected is not None:
                projections.append(projected)
        projection_counts.append(len(projections))
        for projected in projections:
            projected_desc = target.desc_tips_by_name.get(projected)
            projected_value = metric_value(target, projected, mode)
            if projected_desc is None or not np.isfinite(projected_value):
                continue
            for node_i, (desc, node_value) in enumerate(zip(node_desc, node_values)):
                if not np.isfinite(node_value) or not desc.issubset(projected_desc):
                    continue
                distance = node_value - projected_value
                if distance < 0:
                    continue
                weights[lead_i, node_i] += float(math.exp(-float(distance) / tau))
    return weights, lead_event_counts, projection_counts


def build_trail_count_matrix(
    *,
    trail_sites: Sequence[int],
    trail_events_by_site: Mapping[int, Sequence[MutationEvent]],
    eligible_node_index: Mapping[str, int],
    n_nodes: int,
) -> Tuple[np.ndarray, List[int], List[int]]:
    counts = np.zeros((len(trail_sites), n_nodes), dtype=float)
    site_event_counts: List[int] = []
    branch_pool: List[int] = []
    for site_i, site in enumerate(trail_sites):
        n_site_events = 0
        for event in trail_events_by_site.get(int(site), []):
            node_i = eligible_node_index.get(event.node)
            if node_i is None:
                continue
            counts[site_i, node_i] += 1.0
            branch_pool.append(node_i)
            n_site_events += 1
        site_event_counts.append(n_site_events)
    return counts, site_event_counts, branch_pool


def permuted_count_matrix(
    *,
    branch_pool: Sequence[int],
    site_event_counts: Sequence[int],
    n_nodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = np.asarray(branch_pool, dtype=int).copy()
    rng.shuffle(shuffled)
    counts = np.zeros((len(site_event_counts), n_nodes), dtype=float)
    cursor = 0
    for site_i, n_events in enumerate(site_event_counts):
        if n_events <= 0:
            continue
        assigned = shuffled[cursor : cursor + n_events]
        cursor += n_events
        for node_i in assigned:
            counts[site_i, int(node_i)] += 1.0
    return counts


def score_matrix_from_counts(weights: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if weights.size == 0 or counts.size == 0:
        return np.zeros((weights.shape[0], counts.shape[0]), dtype=np.float32)
    scores = np.einsum("ij,kj->ik", weights, counts, optimize=True)
    scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
    return np.asarray(scores, dtype=np.float32)


def compute_ordered_fdr(
    *,
    species: str,
    lead_segment: str,
    trail_segment: str,
    lead: SegmentData,
    trail: SegmentData,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    n_perm = int(args.fdr_permutations)
    if n_perm <= 0:
        return [], []
    alphas = fdr_alphas(args)
    if not alphas:
        return [], []

    lead_events_by_site = {
        site: [event for event in events if event.primary_included]
        for site, events in lead.events_by_site.items()
    }
    trail_events_by_site = {
        site: [event for event in events if event.primary_included]
        for site, events in trail.events_by_site.items()
    }
    lead_sites = sorted(site for site, events in lead_events_by_site.items() if events)
    trail_sites = sorted(site for site, events in trail_events_by_site.items() if events)
    eligible_nodes = eligible_internal_nodes(trail)
    if not lead_sites or not trail_sites or not eligible_nodes:
        return [], []

    eligible_node_index = {node: idx for idx, node in enumerate(eligible_nodes)}
    observed_counts, trail_event_counts, branch_pool = build_trail_count_matrix(
        trail_sites=trail_sites,
        trail_events_by_site=trail_events_by_site,
        eligible_node_index=eligible_node_index,
        n_nodes=len(eligible_nodes),
    )
    if not branch_pool:
        return [], []

    summary_rows: List[Dict[str, object]] = []
    significant_rows: List[Dict[str, object]] = []
    n_pairs = len(lead_sites) * len(trail_sites)
    n_fake = min(int(args.fdr_fake_datasets), n_perm)
    fake_indices = np.sort(rng.choice(np.arange(n_perm), size=n_fake, replace=False)) if n_fake > 0 else np.asarray([], dtype=int)
    max_alpha = max(alphas)

    for mode in fdr_modes(args):
        tau = tau_for_mode(args, mode)
        weights, lead_event_counts, projection_counts = build_lead_weight_matrix(
            lead_sites=lead_sites,
            lead_events_by_site=lead_events_by_site,
            target=trail,
            eligible_nodes=eligible_nodes,
            mode=mode,
            tau=tau,
        )
        if not np.isfinite(weights).any() or float(np.nansum(weights)) == 0.0:
            continue
        observed_scores = score_matrix_from_counts(weights, observed_counts).reshape(-1)
        null_scores = np.empty((n_perm, n_pairs), dtype=np.float32)
        for perm_i in range(n_perm):
            perm_counts = permuted_count_matrix(
                branch_pool=branch_pool,
                site_event_counts=trail_event_counts,
                n_nodes=len(eligible_nodes),
                rng=rng,
            )
            null_scores[perm_i, :] = score_matrix_from_counts(weights, perm_counts).reshape(-1)

        p_obs = ((null_scores >= observed_scores[None, :]).sum(axis=0).astype(float) + 1.0) / (n_perm + 1.0)
        lead_event_counts_arr = np.repeat(np.asarray(lead_event_counts, dtype=int), len(trail_sites))
        trail_event_counts_arr = np.tile(np.asarray(trail_event_counts, dtype=int), len(lead_sites))
        projection_counts_arr = np.repeat(np.asarray(projection_counts, dtype=int), len(trail_sites))
        robust_mask = lead_event_counts_arr >= int(args.hitchhiking_min_leading_events)
        robust_total = int(robust_mask.sum())

        fake_positive_by_alpha: Dict[float, List[int]] = {alpha: [] for alpha in alphas}
        fake_fraction_by_alpha: Dict[float, List[float]] = {alpha: [] for alpha in alphas}
        for fake_idx in fake_indices:
            fake_scores = null_scores[int(fake_idx), :]
            p_fake = ((null_scores >= fake_scores[None, :]).sum(axis=0).astype(float) + 1.0) / (n_perm + 1.0)
            for alpha in alphas:
                fake_positive = p_fake <= alpha
                fake_positive_by_alpha[alpha].append(int(fake_positive.sum()))
                if robust_total > 0:
                    fake_fraction_by_alpha[alpha].append(float((fake_positive & robust_mask).sum() / robust_total))

        for alpha in alphas:
            observed_positive = p_obs <= alpha
            op = int(observed_positive.sum())
            fake_counts = np.asarray(fake_positive_by_alpha[alpha], dtype=float)
            efp = float(np.nanmean(fake_counts)) if fake_counts.size else np.nan
            op_p = float((np.sum(fake_counts >= op) + 1.0) / (fake_counts.size + 1.0)) if fake_counts.size else np.nan
            robust_op = int((observed_positive & robust_mask).sum()) if robust_total > 0 else 0
            robust_obs_fraction = float(robust_op / robust_total) if robust_total > 0 else np.nan
            fake_fractions = np.asarray(fake_fraction_by_alpha[alpha], dtype=float)
            robust_expected_fraction = float(np.nanmean(fake_fractions)) if fake_fractions.size else np.nan
            robust_fraction_p = (
                float((np.sum(fake_fractions >= robust_obs_fraction) + 1.0) / (fake_fractions.size + 1.0))
                if fake_fractions.size and np.isfinite(robust_obs_fraction)
                else np.nan
            )
            summary_rows.append(
                {
                    "species": species,
                    "lead_segment": lead_segment,
                    "trail_segment": trail_segment,
                    "segment_pair": segment_pair_label(lead_segment, trail_segment),
                    "mode": mode,
                    "alpha": float(alpha),
                    "n_lead_sites": int(len(lead_sites)),
                    "n_trail_sites": int(len(trail_sites)),
                    "n_site_pairs": int(n_pairs),
                    "n_permutations": int(n_perm),
                    "n_fake_datasets": int(n_fake),
                    "observed_positives": int(op),
                    "expected_false_positives": efp,
                    "fdr": float(efp / op) if op > 0 and np.isfinite(efp) else np.nan,
                    "observed_positive_count_p": op_p,
                    "robust_min_leading_events": int(args.hitchhiking_min_leading_events),
                    "robust_total_pairs": int(robust_total),
                    "robust_observed_positives": int(robust_op),
                    "robust_observed_positive_fraction": robust_obs_fraction,
                    "robust_expected_fake_positive_fraction": robust_expected_fraction,
                    "robust_positive_fraction_p": robust_fraction_p,
                }
            )

        sig_idx = np.flatnonzero(p_obs <= max_alpha)
        if sig_idx.size:
            order = sorted(
                sig_idx.tolist(),
                key=lambda idx: (float(p_obs[idx]), -float(observed_scores[idx])),
            )
            if int(args.fdr_report_max_pairs_per_analysis) > 0:
                order = order[: int(args.fdr_report_max_pairs_per_analysis)]
            for flat_idx in order:
                lead_i = int(flat_idx // len(trail_sites))
                trail_i = int(flat_idx % len(trail_sites))
                significant_rows.append(
                    {
                        "species": species,
                        "lead_segment": lead_segment,
                        "trail_segment": trail_segment,
                        "segment_pair": segment_pair_label(lead_segment, trail_segment),
                        "mode": mode,
                        "lead_site": int(lead_sites[lead_i]),
                        "trail_site": int(trail_sites[trail_i]),
                        "observed_score": float(observed_scores[flat_idx]),
                        "empirical_p_ge": float(p_obs[flat_idx]),
                        "n_lead_events": int(lead_event_counts_arr[flat_idx]),
                        "n_trail_events": int(trail_event_counts_arr[flat_idx]),
                        "n_projected_lead_events": int(projection_counts_arr[flat_idx]),
                        "passes_hitchhiking_min_leads": int(robust_mask[flat_idx]),
                        "n_permutations": int(n_perm),
                    }
                )
    return summary_rows, significant_rows


def event_rows(segments: Mapping[str, SegmentData]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for data in segments.values():
        for events in data.events_by_site.values():
            for event in events:
                rows.append(
                    {
                        "species": event.species,
                        "segment": event.segment,
                        "node": event.node,
                        "parent": event.parent,
                        "site": event.site,
                        "ancestral_base": event.ancestral_base,
                        "derived_base": event.derived_base,
                        "terminal_branch": int(event.terminal_branch),
                        "is_nonsynonymous": int(event.is_nonsynonymous),
                        "primary_included": int(event.primary_included),
                        "aa_from": event.aa_from,
                        "aa_to": event.aa_to,
                        "codon_index": event.codon_index if event.codon_index is not None else np.nan,
                        "n_descendant_tips": int(len(event.descendant_tips)),
                        "numdate": event.numdate,
                        "div": event.div,
                        "mutcount_cum": event.mutcount_cum,
                    }
                )
    return rows


def build_species_features(pair_scores: pd.DataFrame, species: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    modes = ["time", "div", "mutcount_audit"]
    segment_labels = ["all", "SM", "SL", "ML"]
    for sp in species:
        row: Dict[str, object] = {"species": sp}
        if pair_scores.empty or "species" not in pair_scores.columns or "status" not in pair_scores.columns:
            sp_df = pd.DataFrame()
        else:
            sp_df = pair_scores[(pair_scores["species"] == sp) & (pair_scores["status"] == "ok")].copy()
        for mode in modes:
            mode_df = sp_df[sp_df["mode"] == mode].copy()
            for seg_label in segment_labels:
                sub = mode_df if seg_label == "all" else mode_df[mode_df["segment_pair"] == seg_label]
                prefix = f"species_feat_epi_{mode}"
                suffix = "" if seg_label == "all" else f"_{seg_label}"
                row[f"{prefix}_n_pairs{suffix}"] = int(len(sub))
                if sub.empty:
                    row[f"{prefix}_mean_score{suffix}"] = np.nan
                    row[f"{prefix}_max_score{suffix}"] = np.nan
                    row[f"{prefix}_mean_z{suffix}"] = np.nan
                    row[f"{prefix}_max_z{suffix}"] = np.nan
                    row[f"{prefix}_prop_p_le_0_05{suffix}"] = np.nan
                    row[f"{prefix}_mean_neglog10_p{suffix}"] = np.nan
                    continue
                pvals = pd.to_numeric(sub["empirical_p_ge"], errors="coerce")
                row[f"{prefix}_mean_score{suffix}"] = float(pd.to_numeric(sub["observed_score"], errors="coerce").mean())
                row[f"{prefix}_max_score{suffix}"] = float(pd.to_numeric(sub["observed_score"], errors="coerce").max())
                row[f"{prefix}_mean_z{suffix}"] = float(pd.to_numeric(sub["z_score"], errors="coerce").mean())
                row[f"{prefix}_max_z{suffix}"] = float(pd.to_numeric(sub["z_score"], errors="coerce").max())
                valid_p = pvals.dropna()
                if valid_p.empty:
                    row[f"{prefix}_prop_p_le_0_05{suffix}"] = np.nan
                    row[f"{prefix}_mean_neglog10_p{suffix}"] = np.nan
                else:
                    clipped_p = valid_p.clip(lower=1e-12)
                    row[f"{prefix}_prop_p_le_0_05{suffix}"] = float((valid_p <= 0.05).mean())
                    row[f"{prefix}_mean_neglog10_p{suffix}"] = float((-np.log10(clipped_p)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def augment_dataset(base_dataset: Path, species_features: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if not base_dataset.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_dataset}")
    base = pd.read_csv(base_dataset, sep="\t", low_memory=False)
    drop_cols = [col for col in base.columns if col.startswith("species_feat_epi_")]
    if drop_cols:
        base = base.drop(columns=drop_cols)
    features = species_features.copy()
    merged = base.merge(features, on="species", how="left", sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, sep="\t", index=False)
    return merged


def write_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def append_table(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, mode="a", header=not path.exists())


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    child_seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    return np.random.default_rng(child_seed)


def fdr_checkpoint_paths(checkpoint_dir: Path, species: str, lead_segment: str, trail_segment: str) -> Tuple[Path, Path]:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{species}_{lead_segment}_to_{trail_segment}")
    return checkpoint_dir / f"{token}_summary.tsv", checkpoint_dir / f"{token}_significant_pairs.tsv"


def checkpoint_matches(summary: pd.DataFrame, args: argparse.Namespace) -> bool:
    if summary.empty:
        return False
    if "n_permutations" not in summary.columns:
        return False
    n_permutations = pd.to_numeric(summary["n_permutations"], errors="coerce").dropna()
    if n_permutations.empty or int(n_permutations.iloc[0]) != int(args.fdr_permutations):
        return False
    expected_fake = min(int(args.fdr_fake_datasets), int(args.fdr_permutations))
    if "n_fake_datasets" not in summary.columns:
        return False
    n_fake = pd.to_numeric(summary["n_fake_datasets"], errors="coerce").dropna()
    if n_fake.empty or int(n_fake.iloc[0]) != expected_fake:
        return False
    observed_modes = set(summary.get("mode", pd.Series(dtype=str)).astype(str))
    if observed_modes != set(fdr_modes(args)):
        return False
    observed_alphas = {
        round(float(value), 12)
        for value in pd.to_numeric(summary.get("alpha", pd.Series(dtype=float)), errors="coerce").dropna()
    }
    expected_alphas = {round(float(value), 12) for value in fdr_alphas(args)}
    return observed_alphas == expected_alphas


def load_fdr_checkpoint(
    checkpoint_dir: Path,
    species: str,
    lead_segment: str,
    trail_segment: str,
    args: argparse.Namespace,
) -> Optional[Tuple[List[Dict[str, object]], List[Dict[str, object]]]]:
    summary_path, significant_path = fdr_checkpoint_paths(checkpoint_dir, species, lead_segment, trail_segment)
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path, sep="\t")
    if not checkpoint_matches(summary, args):
        return None
    if significant_path.exists():
        try:
            significant = pd.read_csv(significant_path, sep="\t")
        except pd.errors.EmptyDataError:
            significant = pd.DataFrame()
    else:
        significant = pd.DataFrame()
    return summary.to_dict("records"), significant.to_dict("records")


def write_fdr_checkpoint(
    checkpoint_dir: Path,
    species: str,
    lead_segment: str,
    trail_segment: str,
    summary_rows: Sequence[Mapping[str, object]],
    significant_rows: Sequence[Mapping[str, object]],
) -> None:
    if not summary_rows:
        return
    summary_path, significant_path = fdr_checkpoint_paths(checkpoint_dir, species, lead_segment, trail_segment)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    pd.DataFrame(significant_rows).to_csv(significant_path, sep="\t", index=False)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)
    species = read_species(args.species_file, args.species, args.true_reassortment_root, args.pipeline_root)
    coding_spans = load_coding_spans(args.coding_spans)
    report_dir = args.output_root / "reports"
    dataset_dir = args.output_root / "datasets"
    fdr_checkpoint_dir = args.fdr_checkpoint_dir or (report_dir / "fdr_checkpoints")
    rng = np.random.default_rng(args.seed)
    null_cols = [
        "species",
        "segment_pair",
        "lead_segment",
        "trail_segment",
        "lead_site",
        "trail_site",
        "mode",
        "observed_score",
        "null_mean",
        "null_sd",
        "z_score",
        "empirical_p_ge",
        "n_permutations",
        "status",
    ]
    pair_score_path = report_dir / "intersegment_epistasis_pair_scores.tsv"
    null_summary_path = report_dir / "intersegment_epistasis_null_summary.tsv"
    for path in [pair_score_path, null_summary_path]:
        if path.exists():
            path.unlink()

    all_candidates: List[pd.DataFrame] = []
    all_events: List[Dict[str, object]] = []
    all_qc: List[Dict[str, object]] = []
    all_species_features: List[pd.DataFrame] = []
    all_fdr_summary: List[Dict[str, object]] = []
    all_fdr_significant: List[Dict[str, object]] = []
    n_pair_score_rows = 0

    for sp in species:
        LOGGER.info("Processing %s", sp)
        if args.candidate_source == "ld_top":
            candidates = load_candidate_pairs(args.ld_root, sp, args.top_ld_pairs_per_segment_pair)
            sites_by_segment: Dict[str, Optional[Set[int]]] = required_sites_by_segment(candidates)
        else:
            candidates = pd.DataFrame()
            sites_by_segment = {seg: None for seg in SEGMENTS}
        if args.candidate_source == "ld_top" and candidates.empty:
            all_qc.append({"species": sp, "segment": "all", "status": "no_ld_candidates"})
            all_species_features.append(build_species_features(pd.DataFrame(), [sp]))
            if args.strict:
                raise RuntimeError(f"No LD candidates for {sp}")
            continue
        try:
            segments = {
                seg: load_segment_data(
                    species=sp,
                    segment=seg,
                    true_root=args.true_reassortment_root,
                    pipeline_root=args.pipeline_root,
                    candidate_sites=sites_by_segment.get(seg, set()),
                    coding_spans=coding_spans,
                )
                for seg in SEGMENTS
            }
        except Exception as exc:
            LOGGER.warning("Skipping %s after segment-load failure: %s", sp, exc)
            all_qc.append({"species": sp, "segment": "all", "status": f"segment_load_failed:{type(exc).__name__}", "error": str(exc)})
            all_species_features.append(build_species_features(pd.DataFrame(), [sp]))
            if args.strict:
                raise
            continue
        if args.candidate_source == "all_nonsyn":
            candidates = build_all_nonsyn_candidate_pairs(sp, segments)
        all_candidates.append(candidates)
        if candidates.empty:
            all_qc.append({"species": sp, "segment": "all", "status": f"no_{args.candidate_source}_candidates"})
            all_species_features.append(build_species_features(pd.DataFrame(), [sp]))
            if args.strict:
                raise RuntimeError(f"No {args.candidate_source} candidates for {sp}")
            continue
        for seg_data in segments.values():
            qc = dict(seg_data.qc)
            qc["status"] = "ok"
            qc["candidate_source"] = args.candidate_source
            all_qc.append(qc)
        all_events.extend(event_rows(segments))

        species_scores: List[Dict[str, object]] = []
        for row in candidates.itertuples(index=False):
            seg_a = str(row.segmentA)
            seg_b = str(row.segmentB)
            site_a = int(row.site1)
            site_b = int(row.site2)
            for lead_seg, lead_site, trail_seg, trail_site in [
                (seg_a, site_a, seg_b, site_b),
                (seg_b, site_b, seg_a, site_a),
            ]:
                species_scores.extend(
                    score_ordered_pair(
                        species=sp,
                        segment_a=lead_seg,
                        site_a=lead_site,
                        segment_b=trail_seg,
                        site_b=trail_site,
                        cramer_v=row.CramerV,
                        nmi=getattr(row, "NMI", np.nan),
                        n_strains=getattr(row, "n_strains", 0),
                        segments=segments,
                        args=args,
                        rng=rng,
                    )
                )
        species_score_df = pd.DataFrame(species_scores)
        n_pair_score_rows += int(len(species_score_df))
        append_table(pair_score_path, species_score_df)
        available_null_cols = [col for col in null_cols if col in species_score_df.columns]
        append_table(null_summary_path, species_score_df[available_null_cols] if available_null_cols else pd.DataFrame())
        all_species_features.append(build_species_features(species_score_df, [sp]))

        if int(args.fdr_permutations) > 0:
            LOGGER.info("Running simultaneous-null FDR for %s with %s permutations", sp, args.fdr_permutations)
            for lead_seg, trail_seg in [(a, b) for a, b in SEGMENT_PAIRS] + [(b, a) for a, b in SEGMENT_PAIRS]:
                if not args.no_resume_fdr:
                    cached = load_fdr_checkpoint(fdr_checkpoint_dir, sp, lead_seg, trail_seg, args)
                    if cached is not None:
                        summary_rows, significant_rows = cached
                        LOGGER.info(
                            "Loaded FDR checkpoint for %s %s->%s (%s summary rows, %s significant rows)",
                            sp,
                            lead_seg,
                            trail_seg,
                            len(summary_rows),
                            len(significant_rows),
                        )
                        all_fdr_summary.extend(summary_rows)
                        all_fdr_significant.extend(significant_rows)
                        continue
                summary_rows, significant_rows = compute_ordered_fdr(
                    species=sp,
                    lead_segment=lead_seg,
                    trail_segment=trail_seg,
                    lead=segments[lead_seg],
                    trail=segments[trail_seg],
                    args=args,
                    rng=stable_rng(args.seed, "fdr", sp, lead_seg, trail_seg),
                )
                write_fdr_checkpoint(fdr_checkpoint_dir, sp, lead_seg, trail_seg, summary_rows, significant_rows)
                all_fdr_summary.extend(summary_rows)
                all_fdr_significant.extend(significant_rows)

    candidate_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    event_df = pd.DataFrame(all_events)
    qc_df = pd.DataFrame(all_qc)
    species_features = (
        pd.concat(all_species_features, ignore_index=True)
        if all_species_features
        else build_species_features(pd.DataFrame(), species)
    )

    write_table(report_dir / "intersegment_epistasis_candidate_pairs.tsv", candidate_df)
    write_table(report_dir / "intersegment_epistasis_event_table.tsv", event_df)
    if not pair_score_path.exists():
        write_table(pair_score_path, pd.DataFrame())
    if not null_summary_path.exists():
        write_table(null_summary_path, pd.DataFrame())
    fdr_summary_columns = [
        "species",
        "lead_segment",
        "trail_segment",
        "segment_pair",
        "mode",
        "alpha",
        "n_lead_sites",
        "n_trail_sites",
        "n_site_pairs",
        "n_permutations",
        "n_fake_datasets",
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
    ]
    fdr_significant_columns = [
        "species",
        "lead_segment",
        "trail_segment",
        "segment_pair",
        "mode",
        "lead_site",
        "trail_site",
        "observed_score",
        "empirical_p_ge",
        "n_lead_events",
        "n_trail_events",
        "n_projected_lead_events",
        "passes_hitchhiking_min_leads",
        "n_permutations",
    ]
    write_table(report_dir / "intersegment_epistasis_fdr_summary.tsv", pd.DataFrame(all_fdr_summary, columns=fdr_summary_columns))
    write_table(
        report_dir / "intersegment_epistasis_fdr_significant_pairs.tsv",
        pd.DataFrame(all_fdr_significant, columns=fdr_significant_columns),
    )
    write_table(report_dir / "intersegment_epistasis_qc.tsv", qc_df)
    write_table(dataset_dir / "species_intersegment_epistasis_features.tsv", species_features)
    augmented = augment_dataset(
        args.base_dataset,
        species_features,
        dataset_dir / "sequence_level_final_dataset_direct516_plus_epistasis.tsv",
    )

    summary = pd.DataFrame(
        [
            {
                "n_species_requested": int(len(species)),
                "candidate_source": str(args.candidate_source),
                "n_candidate_pairs": int(len(candidate_df)),
                "n_event_rows": int(len(event_df)),
                "n_pair_score_rows": int(n_pair_score_rows),
                "n_species_feature_columns": int(max(0, species_features.shape[1] - 1)),
                "n_fdr_permutations": int(args.fdr_permutations),
                "n_fdr_summary_rows": int(len(all_fdr_summary)),
                "n_fdr_significant_pair_rows": int(len(all_fdr_significant)),
                "base_dataset": str(args.base_dataset),
                "augmented_dataset": str(dataset_dir / "sequence_level_final_dataset_direct516_plus_epistasis.tsv"),
                "base_dataset_rows": int(pd.read_csv(args.base_dataset, sep="\t", usecols=["species"]).shape[0])
                if args.base_dataset.exists()
                else 0,
                "augmented_dataset_rows": int(len(augmented)),
            }
        ]
    )
    write_table(report_dir / "intersegment_epistasis_run_summary.tsv", summary)
    LOGGER.info("Wrote outputs under %s", args.output_root)


if __name__ == "__main__":
    main()
