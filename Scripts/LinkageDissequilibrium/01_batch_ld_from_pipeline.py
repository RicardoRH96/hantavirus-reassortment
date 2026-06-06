#!/usr/bin/env python3
"""
Batch linkage disequilibrium runner using matched isolates from the
puumala_espalier_pipeline outputs.

By default, this script takes the conservative route: it loads the
per-species alignments that have already been pruned to the common tip
set (espalier/alignments/*_aligned_common.fasta) and restricts LD
calculations to taxa present in every segment. If those common
alignments are missing, it falls back to the Augur alignments built
from matched isolates. An optional "allow-missing" strategy lets you
compute LD without enforcing identical taxon sets.

When alignments don't exist, this script can generate them from raw
species input directories (as produced by split_hanta_master.py) by:
1. Filtering sequences by length and metadata completeness
2. Matching isolates across segments by canonical isolate name
3. Running augur align on matched sequences

LD is calculated ONLY between segments (S-M, S-L, M-L), not within
segments, as within-segment LD is not relevant for reassortment studies.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from Bio import AlignIO

from extened_ld_calculator import compute_ld_segment_pair


LOGGER = logging.getLogger("batch_ld")


# =============================================================================
# Constants for segment inference and sequence filtering
# =============================================================================

SEGMENT_KEYWORDS = {
    "S": ["nucleocapsid", "n protein", "np protein", "small"],
    "M": ["glycoprotein", "gn", "gc", "gpc", "medium"],
    "L": ["rna-dependent", "polymerase", "rdp", "l segment"],
}

# Expected segment length ranges (min, max) for hantaviruses
# Uses lenient ranges to accommodate partial sequences while excluding chimeras
SEGMENT_LENGTH_RANGES = {
    "S": (800, 2500),    # Small segment: complete ~1.7-2.1 kb, allow partial down to 800bp
    "M": (2000, 4500),   # Medium segment: complete ~3.5-3.8 kb, allow partial down to 2000bp
    "L": (4000, 7000),   # Large segment: complete ~6.4-6.6 kb, allow partial down to 4000bp
}

ACC_PATTERN = re.compile(r"([A-Za-z0-9_]+(?:\.\d+)?)")
DATE_PATTERN = re.compile(r"^(?P<year>\d{4})(?:[-/](?P<month>\d{1,2})(?:[-/](?P<day>\d{1,2}))?)?$")
TITLE_ISOLATE_PATTERN = re.compile(r"(?:strain|isolate)\s+([^,;]+)", re.IGNORECASE)


# =============================================================================
# Data classes for sequence/isolate handling
# =============================================================================

@dataclass(frozen=True)
class NormalizedDate:
    iso: str
    precision: str  # year, month, day


@dataclass
class SequenceEntry:
    accession: str
    canonical_isolate: str
    isolate_display: str
    normalized_date: NormalizedDate
    sequence: str
    length: int
    raw_header: str
    metadata: Dict[str, str] = field(repr=False)


@dataclass
class SegmentResult:
    segment: str
    fasta_path: Path
    metadata_path: Path
    reference_path: Optional[Path]
    sequences: Dict[str, SequenceEntry]
    dropped: List[Dict[str, str]]
    max_length: int
    min_length_threshold: float


@dataclass
class TipAssignment:
    canonical_isolate: str
    display_name: str
    slug: str
    normalized_date: NormalizedDate
    tip_name: str
    present_segments: List[str]


# =============================================================================
# Helper functions for sequence processing (adapted from puumala_espalier_pipeline)
# =============================================================================

def clean_accession(value: str) -> str:
    """Extract clean accession number from a string."""
    if not value:
        return ""
    match = ACC_PATTERN.search(value)
    if not match:
        return value.strip()
    acc = match.group(1)
    return acc.split(".")[0]


def normalize_date(value: Optional[str]) -> Optional[NormalizedDate]:
    """Parse date string into normalized ISO format with precision."""
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    raw = raw.replace("/", "-")
    match = DATE_PATTERN.match(raw)
    if not match:
        return None
    year = int(match.group("year"))
    month_str = match.group("month")
    day_str = match.group("day")
    month = int(month_str) if month_str else 1
    day = int(day_str) if day_str else 1
    precision = "day" if day_str else ("month" if month_str else "year")
    try:
        normalized = datetime(year=year, month=month, day=day).strftime("%Y-%m-%d")
    except ValueError:
        return None
    return NormalizedDate(iso=normalized, precision=precision)


def derive_isolate_name(row: Dict[str, str], header: str) -> Optional[str]:
    """Extract isolate name from metadata row or FASTA header."""
    candidates: List[str] = []
    for key in ("Isolate", "isolate", "Strain", "strain"):
        val = row.get(key)
        if val:
            candidates.append(val.strip())
    title = row.get("GenBank_Title") or row.get("Title") or row.get("Definition")
    if title:
        match = TITLE_ISOLATE_PATTERN.search(title)
        if match:
            candidates.append(match.group(1).strip())
    acc = clean_accession(row.get("Accession", ""))
    if acc:
        candidates.append(acc)
    for cand in candidates:
        if cand:
            return cand
    return None


def canonicalize_isolate(value: str) -> str:
    """Normalize isolate name to a canonical lowercase form for matching."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    trimmed = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return trimmed or "isolate"


def slugify_isolate(value: str, fallback: str) -> str:
    """Create a URL-safe slug from isolate name."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    if not cleaned:
        cleaned = fallback
    return cleaned


def slugify_species(name: str) -> str:
    """Create a URL-safe slug from species name."""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_").lower()
    return cleaned or "species"


def read_tsv(path: Path) -> List[Dict[str, str]]:
    """Read a TSV file into a list of dictionaries."""
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader]


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    """Read a FASTA file into a list of (header, sequence) tuples."""
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    seq_chunks: List[str] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if header:
                    records.append((header, "".join(seq_chunks).upper()))
                header = stripped[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(stripped.replace(" ", "").replace("\t", "").upper())
        if header:
            records.append((header, "".join(seq_chunks).upper()))
    return records


def select_longest(existing: Optional[SequenceEntry], incoming: SequenceEntry) -> SequenceEntry:
    """Return the longer of two sequence entries."""
    if existing is None or incoming.length > existing.length:
        return incoming
    return existing


def choose_display(displays: Iterable[str], fallback: str) -> str:
    """Choose the best display name from candidates."""
    valid = [d for d in displays if d]
    if not valid:
        return fallback
    valid.sort(key=lambda item: (-len(item), item))
    return valid[0]


def choose_date(dates: Iterable[NormalizedDate]) -> NormalizedDate:
    """Choose the most precise date from candidates."""
    order = {"day": 2, "month": 1, "year": 0}
    best = None
    best_score = -1
    for date in dates:
        score = order.get(date.precision, 0)
        if score > best_score:
            best = date
            best_score = score
        elif score == best_score and best and date.iso < best.iso:
            best = date
    if best is None:
        raise ValueError("No dates provided for isolate assignment")
    return best


# =============================================================================
# Augur alignment runner
# =============================================================================

class AugurRunner:
    """Run Augur alignment commands."""

    def __init__(self, binary: str = "augur", threads: int = 4) -> None:
        self.binary = binary
        self.threads = threads

    def run(self, args: Sequence[str], cwd: Optional[Path] = None) -> None:
        cmd = [self.binary, *args]
        LOGGER.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

    def align(self, fasta: Path, output: Path, reference: Optional[Path] = None) -> None:
        args = [
            "align",
            "--sequences",
            str(fasta),
            "--output",
            str(output),
            "--nthreads",
            str(self.threads),
        ]
        if reference and reference.exists():
            args.extend(["--reference-sequence", str(reference), "--remove-reference", "--fill-gaps"])
        self.run(args)


# =============================================================================
# Segment discovery and processing functions
# =============================================================================

def discover_segment_files(data_root: Path, segment: str) -> Tuple[Path, Path, Optional[Path]]:
    """
    Discover FASTA and metadata files for a segment in a species input directory.

    Expected structure:
        data_root/<segment>_segment/sequences.fasta
        data_root/<segment>_segment/metadata.tsv
    """
    segment_dir = data_root / f"{segment}_segment"
    if not segment_dir.exists():
        raise FileNotFoundError(f"Missing directory for segment {segment}: {segment_dir}")

    metadata_candidates = sorted(segment_dir.glob("*.tsv"))
    if not metadata_candidates:
        raise FileNotFoundError(f"No metadata TSV found for segment {segment} under {segment_dir}")
    metadata_path = metadata_candidates[0]

    fasta_candidates = [
        path for path in sorted(segment_dir.glob("*.fasta"))
        if not any(
            flag in path.name.lower()
            for flag in ["aligned", "ancestral", "excluded", "common", "binary", "matched"]
        )
    ]
    if not fasta_candidates:
        raise FileNotFoundError(f"No raw FASTA found for segment {segment} under {segment_dir}")
    fasta_path = fasta_candidates[0]

    ref_path = None
    ref_candidate = data_root / "refs" / f"{segment}_ref.fasta"
    if ref_candidate.exists():
        ref_path = ref_candidate

    return fasta_path, metadata_path, ref_path


def process_segment(
    data_root: Path,
    segment: str,
    length_threshold: float = 0.7,
) -> SegmentResult:
    """
    Process a segment's raw sequences: filter by length, metadata, and date.
    Returns a SegmentResult containing kept sequences keyed by canonical isolate.
    """
    fasta_path, metadata_path, ref_path = discover_segment_files(data_root, segment)
    LOGGER.debug("Processing segment %s (FASTA=%s, metadata=%s)", segment, fasta_path.name, metadata_path.name)

    metadata_rows = read_tsv(metadata_path)
    metadata_by_acc = {}
    for row in metadata_rows:
        acc = clean_accession(row.get("Accession", ""))
        if acc:
            metadata_by_acc[acc] = row

    fasta_records = read_fasta(fasta_path)
    if not fasta_records:
        raise ValueError(f"No sequences found in {fasta_path}")

    # Use predefined length ranges for segment filtering
    # This avoids issues with mislabeled sequences inflating max_length
    length_range = SEGMENT_LENGTH_RANGES.get(segment.upper())
    if length_range:
        min_length, max_length = length_range
    else:
        # Fallback: use data-derived max with threshold
        max_length = max(len(seq) for _, seq in fasta_records)
        min_length = max_length * length_threshold

    kept: Dict[str, SequenceEntry] = {}
    dropped: List[Dict[str, str]] = []

    for header, sequence in fasta_records:
        seq_len = len(sequence)
        if seq_len < min_length or seq_len > max_length:
            dropped.append({"header": header, "reason": "length_out_of_range", "length": str(seq_len)})
            continue
        accession = clean_accession(header)
        if not accession:
            dropped.append({"header": header, "reason": "missing_accession", "length": str(seq_len)})
            continue
        metadata_row = metadata_by_acc.get(accession)
        if not metadata_row:
            dropped.append({"header": header, "accession": accession, "reason": "metadata_missing", "length": str(seq_len)})
            continue
        normalized_date = normalize_date(metadata_row.get("Collection_Date"))
        if not normalized_date:
            dropped.append({"header": header, "accession": accession, "reason": "date_missing", "length": str(seq_len)})
            continue
        isolate_display = derive_isolate_name(metadata_row, header)
        if not isolate_display:
            dropped.append({"header": header, "accession": accession, "reason": "isolate_missing", "length": str(seq_len)})
            continue
        canonical = canonicalize_isolate(isolate_display)
        entry = SequenceEntry(
            accession=accession,
            canonical_isolate=canonical,
            isolate_display=isolate_display,
            normalized_date=normalized_date,
            sequence=sequence,
            length=seq_len,
            raw_header=header,
            metadata=metadata_row,
        )
        existing = kept.get(canonical)
        kept[canonical] = select_longest(existing, entry)

    return SegmentResult(
        segment=segment,
        fasta_path=fasta_path,
        metadata_path=metadata_path,
        reference_path=ref_path,
        sequences=kept,
        dropped=dropped,
        max_length=max_length,
        min_length_threshold=min_length,
    )


def build_tip_assignments(segment_results: Sequence[SegmentResult]) -> Dict[str, TipAssignment]:
    """
    Build tip name assignments by matching isolates across segments.
    Returns a dict mapping canonical isolate -> TipAssignment.
    """
    isolates: Dict[str, Dict[str, object]] = {}
    for result in segment_results:
        for canonical, entry in result.sequences.items():
            bucket = isolates.setdefault(
                canonical,
                {
                    "displays": set(),
                    "dates": [],
                    "segments": [],
                },
            )
            bucket["displays"].add(entry.isolate_display)
            bucket["dates"].append(entry.normalized_date)
            bucket["segments"].append(result.segment)

    assignments: Dict[str, TipAssignment] = {}
    for canonical, bucket in isolates.items():
        display = choose_display(bucket["displays"], fallback=canonical.upper())
        date = choose_date(bucket["dates"])
        slug = slugify_isolate(display, fallback=canonical.upper())
        tip_name = f"{slug}|{date.iso}"
        assignments[canonical] = TipAssignment(
            canonical_isolate=canonical,
            display_name=display,
            slug=slug,
            normalized_date=date,
            tip_name=tip_name,
            present_segments=sorted(set(bucket["segments"])),
        )
    return assignments


def partition_isolates(
    assignments: Dict[str, TipAssignment],
    required_segments: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """
    Partition isolates into matched (have all required segments) and partial.
    Returns (matched_canonical_isolates, partial_canonical_isolates).
    """
    matched: List[str] = []
    partial: List[str] = []
    required = set(required_segments)
    for canonical, assignment in assignments.items():
        present = set(assignment.present_segments)
        if required.issubset(present):
            matched.append(canonical)
        elif present:
            partial.append(canonical)
    return matched, partial


def load_species_label(dataset_root: Path) -> Tuple[str, str]:
    """Load species label and slug from species_info.json or directory name."""
    info_file = dataset_root / "species_info.json"
    if info_file.exists():
        try:
            info = json.loads(info_file.read_text())
            species_label = info.get("species_label") or info.get("species") or dataset_root.name
            slug = info.get("slug") or slugify_species(species_label)
            return species_label, slug
        except Exception as exc:
            LOGGER.warning("Failed to parse %s (%s); using directory name.", info_file, exc)
    species_label = dataset_root.name
    slug = slugify_species(species_label)
    return species_label, slug


def generate_alignments_for_species(
    species_input_dir: Path,
    species_output_dir: Path,
    segments: Sequence[str],
    min_taxa: int,
    threads: int = 4,
) -> Dict[str, Path]:
    """
    Generate alignments for a species from raw input data.

    1. Process each segment to filter sequences
    2. Match isolates across segments
    3. Write matched FASTAs
    4. Run augur align

    Returns a dict mapping segment -> aligned FASTA path.
    """
    LOGGER.info("Generating alignments for %s", species_input_dir.name)

    # Process each segment
    segment_results: List[SegmentResult] = []
    for segment in segments:
        try:
            result = process_segment(species_input_dir, segment)
        except FileNotFoundError:
            LOGGER.warning("Segment %s missing for %s; skipping.", segment, species_input_dir.name)
            continue
        if not result.sequences:
            LOGGER.warning("No sequences passed filters for %s segment %s.", species_input_dir.name, segment)
            continue
        segment_results.append(result)

    if len(segment_results) < 2:
        LOGGER.warning("Need at least 2 segments for LD; %s has %d.", species_input_dir.name, len(segment_results))
        return {}

    # Build tip assignments and partition isolates
    assignments = build_tip_assignments(segment_results)
    available = [r.segment for r in segment_results]
    matched_isolates, _ = partition_isolates(assignments, available)

    if len(matched_isolates) < min_taxa:
        LOGGER.warning(
            "Only %d matched isolates for %s (need >= %d); skipping alignment generation.",
            len(matched_isolates),
            species_input_dir.name,
            min_taxa,
        )
        return {}

    LOGGER.info(
        "%s: %d isolates matched across segments %s",
        species_input_dir.name,
        len(matched_isolates),
        ",".join(available),
    )

    # Create output directories
    augur_dir = species_output_dir / "augur"
    augur_dir.mkdir(parents=True, exist_ok=True)

    runner = AugurRunner(threads=threads)
    alignments: Dict[str, Path] = {}

    for result in segment_results:
        seg_augur_dir = augur_dir / result.segment
        seg_augur_dir.mkdir(parents=True, exist_ok=True)

        # Write matched FASTA with harmonized tip names
        matched_fasta = seg_augur_dir / f"{result.segment}_matched.fasta"
        records: List[Tuple[str, str]] = []
        for canonical in sorted(matched_isolates):
            entry = result.sequences.get(canonical)
            if entry:
                assignment = assignments[canonical]
                records.append((assignment.tip_name, entry.sequence))

        write_fasta(matched_fasta, records)

        # Run alignment
        aligned_path = seg_augur_dir / f"{result.segment}_aligned.fasta"
        try:
            runner.align(matched_fasta, aligned_path, result.reference_path)
            alignments[result.segment] = aligned_path
        except subprocess.CalledProcessError as e:
            LOGGER.error("Augur align failed for %s segment %s: %s", species_input_dir.name, result.segment, e)
            continue

    return alignments


# =============================================================================
# Argument parsing and main functions
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LD for all species using matched isolates.")
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path("Data") / "TreeGrafting" / "espalier_pipeline",
        help="Root directory produced by puumala_espalier_pipeline (default: Data/TreeGrafting/espalier_pipeline).",
    )
    parser.add_argument(
        "--species-root",
        type=Path,
        help="Directory containing per-species input folders (each with <segment>_segment subdirs). "
             "When provided, alignments will be generated from raw inputs if they don't exist.",
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        default=["S", "M", "L"],
        help="Segment codes to include (default: S M L).",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        help="Optional subset of species slugs to process (matches directory names).",
    )
    parser.add_argument(
        "--mode",
        choices=["nt", "aa", "3di"],
        default="nt",
        help="Alignment mode passed to the LD calculator (default: nt).",
    )
    parser.add_argument(
        "--strategy",
        choices=["conservative", "allow-missing"],
        default="conservative",
        help="Conservative restricts to taxa shared by all segments; allow-missing pairs sequences by order.",
    )
    parser.add_argument(
        "--min-taxa",
        type=int,
        default=10,
        help="Minimum taxa required to run LD for a species (default: 10).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for alignment (default: 4).",
    )
    parser.add_argument(
        "--use-common-alignments",
        action="store_true",
        default=True,
        help="Prefer alignments filtered to common taxa (espalier/alignments) when present.",
    )
    parser.add_argument(
        "--no-use-common-alignments",
        dest="use_common_alignments",
        action="store_false",
        help="Skip common alignments and always fall back to Augur outputs.",
    )
    parser.add_argument(
        "--generate-alignments",
        action="store_true",
        help="Generate alignments from raw inputs when they don't exist (requires --species-root).",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of alignments even if they already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)
    args.pipeline_root = args.pipeline_root.resolve()
    if args.species_root:
        args.species_root = args.species_root.resolve()
    return args


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def write_fasta(path: Path, records: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for header, seq in records:
            handle.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def load_alignment(path: Path) -> List[Tuple[str, str]]:
    aln = AlignIO.read(path, "fasta")
    return [(rec.id, str(rec.seq)) for rec in aln]


def list_species_dirs(pipeline_root: Path, allowed: Optional[Iterable[str]]) -> List[Path]:
    requested = {a.lower() for a in allowed} if allowed else None
    dirs: List[Path] = []
    for child in sorted(pipeline_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "inputs":
            continue
        if requested and child.name.lower() not in requested:
            continue
        dirs.append(child)
    return dirs


def collect_alignments(species_dir: Path, segments: Sequence[str], use_common: bool) -> Dict[str, Path]:
    alignments: Dict[str, Path] = {}
    if use_common:
        common_dir = species_dir / "espalier" / "alignments"
        if common_dir.exists():
            for path in common_dir.glob("*_aligned_common.fasta"):
                seg = path.name.split("_", 1)[0]
                if seg in segments:
                    alignments[seg] = path
    if not alignments:
        augur_dir = species_dir / "augur"
        if augur_dir.exists():
            for seg in segments:
                candidate = augur_dir / seg / f"{seg}_aligned.fasta"
                if candidate.exists():
                    alignments[seg] = candidate
    return alignments


def prepare_alignments_for_ld(
    species_dir: Path,
    alignments: Dict[str, Path],
    strategy: str,
    min_taxa: int,
) -> Tuple[Dict[str, Path], int]:
    """
    Normalize alignments into a dedicated ld/inputs folder. When strategy is
    conservative, we prune to taxa present in every segment; otherwise we
    copy alignments as-is and allow the LD calculator to pair by order.
    """
    if len(alignments) < 2:
        return {}, 0
    input_dir = species_dir / "ld" / strategy / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    if strategy == "conservative":
        taxon_sets: Dict[str, set[str]] = {}
        seq_maps: Dict[str, Dict[str, str]] = {}
        for seg, path in alignments.items():
            records = load_alignment(path)
            seq_maps[seg] = {h: s for h, s in records}
            taxon_sets[seg] = set(seq_maps[seg].keys())
        common_taxa = set.intersection(*taxon_sets.values())
        if len(common_taxa) < min_taxa:
            LOGGER.warning(
                "Skipping %s: only %d common taxa across %d segments (need >= %d).",
                species_dir.name,
                len(common_taxa),
                len(alignments),
                min_taxa,
            )
            return {}, len(common_taxa)
        for seg, seq_map in seq_maps.items():
            dest = input_dir / f"{seg}.fasta"
            filtered = [(tax, seq_map[tax]) for tax in sorted(common_taxa)]
            write_fasta(dest, filtered)
            alignments[seg] = dest
        return alignments, len(common_taxa)

    # allow-missing: copy alignments without pruning
    min_records = None
    for seg, path in alignments.items():
        dest = input_dir / f"{seg}.fasta"
        write_fasta(dest, load_alignment(path))
        alignments[seg] = dest
        count = len(load_alignment(dest))
        min_records = count if min_records is None else min(min_records, count)
    if min_records is None or min_records < min_taxa:
        LOGGER.warning(
            "Skipping %s: insufficient taxa (%s) for allow-missing strategy (need >= %d).",
            species_dir.name,
            min_records or 0,
            min_taxa,
        )
        return {}, min_records or 0
    return alignments, min_records or 0


def compute_ld_for_species(
    species_dir: Path,
    alignments: Dict[str, Path],
    segments: Sequence[str],
    mode: str,
    enforce_taxa: bool,
) -> Optional[Path]:
    if len(alignments) < 2:
        return None
    ld_dir = species_dir / "ld"
    output_dir = ld_dir / ("conservative" if enforce_taxa else "allow-missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "inputs"
    # alignments already materialized into input_dir; ensure we point there
    input_dir.mkdir(parents=True, exist_ok=True)
    all_dfs: List[pd.DataFrame] = []
    segs = [s for s in segments if s in alignments]
    # Only compute LD between segments (not within segments) for reassortment studies
    for idx, segA in enumerate(segs):
        for segB in segs[idx + 1:]:
            df = compute_ld_segment_pair(
                segA,
                segB,
                mode,
                input_dir,
                "{segment}.fasta",
                "{segment}.fasta",
                "{segment}.fasta",
                excluded_taxa=None,
                enforce_taxon_ids=enforce_taxa,
            )
            if not df.empty:
                out = output_dir / f"{segA}_{segB}_LD.csv"
                df.to_csv(out, index=False)
                LOGGER.info("Wrote %s", out)
            all_dfs.append(df)
    if not all_dfs:
        return None
    big = pd.concat(all_dfs, ignore_index=True)
    combined = output_dir / "ALL_SEGMENT_PAIRS_LD.csv"
    big.to_csv(combined, index=False)
    LOGGER.info("Combined LD table -> %s", combined)
    return combined


def list_species_input_dirs(species_root: Path, allowed: Optional[Iterable[str]]) -> List[Tuple[Path, str, str]]:
    """
    List species input directories from the species_root.
    Returns list of (input_dir, species_label, slug) tuples.
    """
    requested = {a.lower() for a in allowed} if allowed else None
    dirs: List[Tuple[Path, str, str]] = []
    for child in sorted(species_root.iterdir()):
        if not child.is_dir():
            continue
        species_label, slug = load_species_label(child)
        if requested and child.name.lower() not in requested and slug.lower() not in requested:
            continue
        dirs.append((child, species_label, slug))
    return dirs


def run(args: argparse.Namespace) -> None:
    # Mode 1: Generate alignments from species-root if requested
    if args.generate_alignments and args.species_root:
        if not args.species_root.exists():
            raise SystemExit(f"Species root does not exist: {args.species_root}")

        species_inputs = list_species_input_dirs(args.species_root, args.species)
        if not species_inputs:
            raise SystemExit(f"No species directories found under {args.species_root}")

        LOGGER.info("Generating alignments for %d species from %s", len(species_inputs), args.species_root)

        for species_input_dir, species_label, slug in species_inputs:
            LOGGER.info("=== %s ===", species_label)
            species_output_dir = args.pipeline_root / slug
            species_output_dir.mkdir(parents=True, exist_ok=True)

            # Check if alignments already exist (skip check if --force-regenerate)
            if not args.force_regenerate:
                existing_alignments = collect_alignments(species_output_dir, args.segments, args.use_common_alignments)
                if len(existing_alignments) >= 2:
                    LOGGER.info("Alignments already exist for %s; skipping generation.", slug)
                    alignments = existing_alignments
                else:
                    alignments = None
            else:
                LOGGER.info("Force regenerating alignments for %s", slug)
                alignments = None

            if alignments is None:
                # Generate alignments from raw inputs
                alignments = generate_alignments_for_species(
                    species_input_dir=species_input_dir,
                    species_output_dir=species_output_dir,
                    segments=args.segments,
                    min_taxa=args.min_taxa,
                    threads=args.threads,
                )
                if not alignments:
                    LOGGER.warning("Failed to generate alignments for %s; skipping.", slug)
                    continue

            # Prepare alignments for LD and compute
            ready_alignments, n_taxa = prepare_alignments_for_ld(
                species_dir=species_output_dir,
                alignments=alignments,
                strategy=args.strategy,
                min_taxa=args.min_taxa,
            )
            if not ready_alignments:
                continue

            enforce = args.strategy == "conservative"
            compute_ld_for_species(
                species_dir=species_output_dir,
                alignments=ready_alignments,
                segments=args.segments,
                mode=args.mode,
                enforce_taxa=enforce,
            )
            LOGGER.info(
                "Finished %s with strategy=%s using %d taxa.",
                slug,
                args.strategy,
                n_taxa,
            )
        return

    # Mode 2: Use existing alignments from pipeline-root (original behavior)
    species_dirs = list_species_dirs(args.pipeline_root, args.species)
    if not species_dirs:
        raise SystemExit(f"No species directories found under {args.pipeline_root}")

    for species_dir in species_dirs:
        LOGGER.info("=== %s ===", species_dir.name)
        alignments = collect_alignments(species_dir, args.segments, args.use_common_alignments)
        if len(alignments) < 2:
            LOGGER.warning("No usable alignments for %s; skipping.", species_dir.name)
            continue
        ready_alignments, n_taxa = prepare_alignments_for_ld(
            species_dir=species_dir,
            alignments=alignments,
            strategy=args.strategy,
            min_taxa=args.min_taxa,
        )
        if not ready_alignments:
            continue
        enforce = args.strategy == "conservative"
        # Alignments were rewritten into ld/<strategy>/inputs during preparation.
        compute_ld_for_species(
            species_dir=species_dir,
            alignments=ready_alignments,
            segments=args.segments,
            mode=args.mode,
            enforce_taxa=enforce,
        )
        LOGGER.info(
            "Finished %s with strategy=%s using %d taxa.",
            species_dir.name,
            args.strategy,
            n_taxa,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)
    run(args)


if __name__ == "__main__":
    main()
