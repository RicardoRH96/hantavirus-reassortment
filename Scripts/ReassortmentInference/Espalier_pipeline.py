#!/usr/bin/env python3
"""
Hantavirus segment harmonization pipeline for Espalier + Augur.

This script can operate on a single dataset containing S, M, and L segment
FASTA/metadata files or iterate across multiple species that have already
been split into per-segment folders. It filters sequences by completeness
and date metadata, harmonizes isolate names across segments, runs Augur's
align/tree/refine workflow, enforces a common tip set, resolves polytomies,
computes pairwise SPR distances, and optionally launches Espalier to build
ancestral recombination graphs (ARGs). It also records Espalier
documentation links for downstream reference.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:
    import dendropy  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dendropy = None
else:
    DENDROPY_DEFAULTS = {
        "rooting": "default-rooted",
    }

MAF = None
ESPALIER_PATH: Optional[Path] = None


def require_module(module, task: str) -> None:
    if module is None:
        raise RuntimeError(f"{task} requires optional dependency that is not available in this environment.")


def dendropy_tree_get(*, path: Optional[str] = None, schema: str = "newick", preserve_underscores: bool = True, **kwargs):
    require_module(dendropy, "Tree loading")
    params = dict(DENDROPY_DEFAULTS)
    params.update(kwargs)
    if path is not None:
        params["path"] = path
    params["schema"] = schema
    params["preserve_underscores"] = preserve_underscores
    tree = dendropy.Tree.get(**params)
    if preserve_underscores:
        tree.preserve_underscores = True
    return tree

ESPALIER_DOC_URL = "https://espalier.readthedocs.io/en/latest/"
ESPALIER_DOC_BLURB = (
    "Espalier is a Python package for reconciling discordant phylogenetic "
    "trees with maximum agreement forests so that approximate ancestral "
    "recombination graphs (ARGs) can be inferred. Documentation: "
    f"{ESPALIER_DOC_URL}"
)

LOGGER = logging.getLogger("puumala_pipeline")


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
    metadata_stats: Dict[str, Dict[str, float]]
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


@dataclass
class SegmentAugurOutput:
    segment: str
    matched_fasta: Path
    aligned_fasta: Path
    raw_tree_newick: Path
    refined_tree_nexus: Path
    refined_tree_newick: Path
    metadata_tsv: Path


@dataclass
class SpeciesRunResult:
    species_label: str
    slug: str
    dataset_root: Path
    output_dir: Path
    matched_isolates: int
    partial_isolates: int
    n_common_taxa: int
    spr_table: Optional[Path]
    arg_tree_dir: Optional[Path]


def slugify_species(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_").lower()
    return cleaned or "species"


def load_species_label(dataset_root: Path) -> Tuple[str, str]:
    info_file = dataset_root / "species_info.json"
    if info_file.exists():
        try:
            info = json.loads(info_file.read_text())
            species_label = info.get("species_label") or info.get("species") or dataset_root.name
            slug = info.get("slug") or slugify_species(species_label)
            return species_label, slug
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Failed to parse %s (%s); using directory name.", info_file, exc)
    species_label = dataset_root.name
    slug = slugify_species(species_label)
    return species_label, slug


@contextlib.contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def enumerate_dataset_roots(args: argparse.Namespace) -> List[Tuple[str, Path, str]]:
    """
    Returns a list of tuples (species_label, dataset_root, slug).
    """
    dataset_entries: List[Tuple[str, Path, str]] = []
    if args.species_root:
        if not args.species_root.exists():
            raise FileNotFoundError(f"Species root {args.species_root} does not exist.")
        requested = set(x.lower() for x in (args.species or []))
        for child in sorted(args.species_root.iterdir()):
            if not child.is_dir():
                continue
            label, slug = load_species_label(child)
            if requested and (child.name.lower() not in requested and label.lower() not in requested and slug.lower() not in requested):
                continue
            dataset_entries.append((label, child.resolve(), slug))
        if not dataset_entries:
            raise ValueError("No species directories matched the requested filters.")
    else:
        label, slug = load_species_label(args.data_root)
        dataset_entries.append((label, args.data_root.resolve(), slug))
    return dataset_entries


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare hantavirus segments for Augur + Espalier analysis."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data") / "TreeGrafting",
        help="Directory containing <segment>_segment folders when running a single dataset (default: Data/TreeGrafting).",
    )
    parser.add_argument(
        "--species-root",
        type=Path,
        help="Optional directory containing per-species datasets (each with <segment>_segment folders). When provided the pipeline iterates across all subdirectories or the subset named via --species.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        help="Subset of species to run when --species-root is provided. Matches either the directory name or the species label stored in species_info.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data") / "TreeGrafting" / "espalier_pipeline",
        help="Where to write harmonized FASTAs, metadata, and reports.",
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        default=["S", "M", "L"],
        help="Segment codes to process (default: S M L).",
    )
    parser.add_argument(
        "--length-threshold",
        type=float,
        default=0.7,
        help="Retain sequences that are >= this fraction of the max length per segment.",
    )
    parser.add_argument(
        "--min-complete-isolates",
        type=int,
        default=10,
        help="Minimum isolates with all requested segments required to run Augur/Espalier steps (default: 10).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to hand to Augur (and IQ-TREE).",
    )
    parser.add_argument(
        "--skip-augur",
        action="store_true",
        help="Prepare datasets but skip Augur align/tree/refine calls.",
    )
    parser.add_argument(
        "--common-tree-source",
        choices=["refined", "raw"],
        default="refined",
        help=(
            "Tree source used for common-tip Espalier inputs. The default refined "
            "source preserves the current Augur/TreeTime behavior; raw keeps the "
            "full matched-tip IQ-TREE set for rescue/sensitivity runs."
        ),
    )
    parser.add_argument(
        "--run-espalier",
        action="store_true",
        help="Run Espalier ARG reconstruction (requires DendroPy, Espalier, ete3, and raxml-ng).",
    )
    parser.add_argument(
        "--raxml-bin",
        default="raxml-ng",
        help="Path to raxml-ng binary (used when --run-espalier is enabled).",
    )
    parser.add_argument(
        "--recombination-rate",
        type=float,
        default=0.0002,
        help="Recombination rate per site for Espalier (default: 0.0002).",
    )
    parser.add_argument(
        "--lower-bound-ratio",
        type=float,
        default=0.1,
        help="Lower bound ratio for Espalier reconciler (default: 0.1).",
    )
    parser.add_argument(
        "--prior-gamma",
        type=float,
        help="Decay rate for weighting site likelihoods outside each segment (defaults to recombination rate).",
    )
    parser.add_argument(
        "--espalier-path",
        type=Path,
        help="Path to a local Espalier fork to prefer (useful for development branches).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)
    if args.data_root:
        args.data_root = args.data_root.resolve()
    if args.species_root:
        args.species_root = args.species_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.espalier_path:
        args.espalier_path = args.espalier_path.resolve()
    return args


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def configure_espalier_imports(espalier_path: Optional[Path]) -> None:
    """
    Optionally prepend a local Espalier checkout (e.g., a development fork)
    to sys.path and import the MAF helper used for SPR distance estimation.
    """
    global MAF, ESPALIER_PATH
    if espalier_path:
        resolved = espalier_path.resolve()
        ESPALIER_PATH = resolved
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
        LOGGER.info("Preferring Espalier from %s", resolved)
    try:
        from Espalier import MAF as maf_mod  # type: ignore
    except ImportError:
        LOGGER.debug("Espalier not available; skipping optional imports.", exc_info=True)
        MAF = None
    else:
        MAF = maf_mod


def ensure_espalier_path() -> None:
    if ESPALIER_PATH and str(ESPALIER_PATH) not in sys.path:
        sys.path.insert(0, str(ESPALIER_PATH))


def discover_segment_files(data_root: Path, segment: str) -> Tuple[Path, Path, Optional[Path]]:
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
    else:
        # fall back to global TreeGrafting/refs if present
        global_ref = Path("Data") / "TreeGrafting" / "refs" / f"Puumala_{segment}_Ref.fasta"
        if global_ref.exists():
            ref_path = global_ref

    return fasta_path, metadata_path, ref_path


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader]


def summarize_metadata(rows: List[Dict[str, str]], fields: Sequence[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    total = len(rows)
    for field in fields:
        non_null = sum(1 for row in rows if row.get(field))
        unique = len({row.get(field) for row in rows if row.get(field)})
        summary[field] = {
            "non_null": non_null,
            "unique": unique,
            "coverage_fraction": (non_null / total) if total else 0.0,
        }
    return summary


ACC_PATTERN = re.compile(r"([A-Za-z0-9_]+(?:\.\d+)?)")

SEGMENT_MAX_LENGTH = {
    "S": 2300,  # cap small segment lengths to avoid mislabeled long sequences
}

def clean_accession(value: str) -> str:
    if not value:
        return ""
    match = ACC_PATTERN.search(value)
    if not match:
        return value.strip()
    acc = match.group(1)
    return acc.split(".")[0]


DATE_PATTERN = re.compile(r"^(?P<year>\d{4})(?:[-/](?P<month>\d{1,2})(?:[-/](?P<day>\d{1,2}))?)?$")


def normalize_date(value: Optional[str]) -> Optional[NormalizedDate]:
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


TITLE_ISOLATE_PATTERN = re.compile(r"(?:strain|isolate)\s+([^,;]+)", re.IGNORECASE)
PUUV_PATTERN = re.compile(r"(PUUV[^,;]*)", re.IGNORECASE)


def derive_isolate_name(row: Dict[str, str], header: str) -> Optional[str]:
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
        match = PUUV_PATTERN.search(title)
        if match:
            candidates.append(match.group(1).strip())
    for part in header.split("|"):
        if "PUUV" in part.upper():
            candidates.append(part.strip())
    acc = clean_accession(row.get("Accession", ""))
    if acc:
        candidates.append(acc)
    for cand in candidates:
        if cand:
            return cand
    return None


def canonicalize_isolate(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    trimmed = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return trimmed or "isolate"


def slugify_isolate(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    if not cleaned:
        cleaned = fallback
    return cleaned


def read_fasta(path: Path) -> List[Tuple[str, str]]:
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


def _find_duplicates(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {value: count for value, count in counts.items() if count > 1}


def _raise_if_duplicate_labels(
    labels: Iterable[str],
    *,
    context: str,
    label_name: str,
) -> None:
    dupes = _find_duplicates(labels)
    if not dupes:
        return
    examples = ", ".join(f"{key} (x{count})" for key, count in sorted(dupes.items())[:10])
    raise ValueError(
        f"Duplicate {label_name} detected in {context}: {examples}. "
        "Resolve duplicates before running Augur/Espalier."
    )


def preflight_fasta_header_uniqueness(
    records: Sequence[Tuple[str, str]],
    *,
    source: Path,
    normalizer: Optional[Callable[[str], str]] = None,
    normalizer_name: Optional[str] = None,
) -> None:
    headers = [header for header, _ in records]
    label_name = "FASTA header"
    if normalizer is not None:
        headers = [normalizer(header) for header in headers]
        label_name = f"FASTA header ({normalizer_name or 'normalized'})"
    _raise_if_duplicate_labels(headers, context=str(source), label_name=label_name)


def select_longest(existing: Optional[SequenceEntry], incoming: SequenceEntry) -> SequenceEntry:
    if existing is None or incoming.length > existing.length:
        return incoming
    return existing


def process_segment(
    data_root: Path,
    segment: str,
    length_threshold: float,
) -> SegmentResult:
    fasta_path, metadata_path, ref_path = discover_segment_files(data_root, segment)
    LOGGER.info("Processing segment %s (FASTA=%s, metadata=%s)", segment, fasta_path.name, metadata_path.name)
    metadata_rows = read_tsv(metadata_path)
    metadata_summary = summarize_metadata(
        metadata_rows,
        fields=["Accession", "Isolate", "GenBank_Title", "Collection_Date"],
    )
    metadata_by_acc = {}
    for row in metadata_rows:
        acc = clean_accession(row.get("Accession", ""))
        if acc:
            metadata_by_acc[acc] = row
    fasta_records = read_fasta(fasta_path)
    if not fasta_records:
        raise ValueError(f"No sequences found in {fasta_path}")
    preflight_fasta_header_uniqueness(fasta_records, source=fasta_path)
    preflight_fasta_header_uniqueness(
        fasta_records,
        source=fasta_path,
        normalizer=lambda header: "_".join(header.split()),
        normalizer_name="whitespace-sanitized",
    )
    max_length = max(len(seq) for _, seq in fasta_records)
    cap = SEGMENT_MAX_LENGTH.get(segment.upper())
    effective_max = min(max_length, cap) if cap else max_length
    min_length = effective_max * length_threshold
    kept: Dict[str, SequenceEntry] = {}
    dropped: List[Dict[str, str]] = []
    for header, sequence in fasta_records:
        seq_len = len(sequence)
        if seq_len < min_length:
            dropped.append({"header": header, "reason": "short_sequence", "length": str(seq_len)})
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
        metadata_stats=metadata_summary,
        max_length=max_length,
        min_length_threshold=min_length,
    )


def choose_display(displays: Iterable[str], fallback: str) -> str:
    valid = [d for d in displays if d]
    if not valid:
        return fallback
    valid.sort(key=lambda item: (-len(item), item))
    return valid[0]


def choose_date(dates: Iterable[NormalizedDate]) -> NormalizedDate:
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


def build_tip_assignments(segment_results: Sequence[SegmentResult]) -> Dict[str, TipAssignment]:
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
    prelim: Dict[str, Dict[str, object]] = {}
    tip_to_canonicals: Dict[str, List[str]] = {}
    for canonical, bucket in isolates.items():
        display = choose_display(bucket["displays"], fallback=canonical.upper())
        date = choose_date(bucket["dates"])
        slug = slugify_isolate(display, fallback=canonical.upper())
        base_tip_name = f"{slug}|{date.iso}"
        prelim[canonical] = {
            "display": display,
            "date": date,
            "slug": slug,
            "base_tip_name": base_tip_name,
            "present_segments": sorted(set(bucket["segments"])),
        }
        tip_to_canonicals.setdefault(base_tip_name, []).append(canonical)

    assignments: Dict[str, TipAssignment] = {}
    for canonical, data in prelim.items():
        date = data["date"]
        assert isinstance(date, NormalizedDate)
        base_tip_name = str(data["base_tip_name"])
        canonicals = tip_to_canonicals[base_tip_name]
        tip_name = base_tip_name
        if len(canonicals) > 1:
            # Deterministic collision disambiguation to keep Espalier/RAxML keys unique.
            stable_suffix = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
            tip_name = f"{base_tip_name}__{stable_suffix}"
        assignments[canonical] = TipAssignment(
            canonical_isolate=canonical,
            display_name=str(data["display"]),
            slug=str(data["slug"]),
            normalized_date=date,
            tip_name=tip_name,
            present_segments=list(data["present_segments"]),
        )

    duplicate_groups = {tip: canonicals for tip, canonicals in tip_to_canonicals.items() if len(canonicals) > 1}
    if duplicate_groups:
        LOGGER.warning(
            "Resolved %d duplicate tip label group(s) with deterministic suffixes.",
            len(duplicate_groups),
        )
    _raise_if_duplicate_labels(
        (assignment.tip_name for assignment in assignments.values()),
        context="tip assignments",
        label_name="tip_name",
    )
    return assignments


def write_fasta(path: Path, records: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i : i + 80] + "\n")


def convert_alignment_for_espalier(src: Path, dest: Path) -> Path:
    """
    Materialize an Espalier/RAxML-safe alignment with non-whitespace taxon names.

    We keep underscore-delimited labels (instead of converting to spaces) because
    RAxML tokenizes sequence names on whitespace, which can collapse distinct taxa
    into duplicate names (e.g. "KS19 1354" and "KS19 2195" both becoming "KS19").
    """
    records = []
    seen_headers: Dict[str, str] = {}
    for header, seq in read_fasta(src):
        normalized_header = "_".join(header.split())
        prior = seen_headers.get(normalized_header)
        if prior is not None and prior != header:
            raise ValueError(
                f"Duplicate normalized taxon label '{normalized_header}' after whitespace cleanup "
                f"while preparing {src}: '{prior}' vs '{header}'"
            )
        seen_headers[normalized_header] = header
        records.append((normalized_header, seq))
    write_fasta(dest, records)
    return dest


def convert_tree_for_espalier(src: Path, dest: Path) -> Path:
    """
    Materialize an Espalier-safe tree whose leaf labels use the same
    whitespace-to-underscore normalization as the sanitized alignments.
    """
    require_module(dendropy, "Espalier tree sanitization")
    tree = dendropy_tree_get(path=str(src), schema="newick")
    seen_labels: Dict[str, str] = {}
    for leaf in tree.leaf_node_iter():
        if not leaf.taxon or not leaf.taxon.label:
            continue
        original = leaf.taxon.label
        normalized = "_".join(original.split())
        prior = seen_labels.get(normalized)
        if prior is not None and prior != original:
            raise ValueError(
                f"Duplicate normalized taxon label '{normalized}' after whitespace cleanup "
                f"while preparing {src}: '{prior}' vs '{original}'"
            )
        seen_labels[normalized] = original
        leaf.taxon.label = normalized
    tree.write(
        path=str(dest),
        schema="newick",
        suppress_annotations=True,
        suppress_rooting=True,
    )
    return dest


def patch_espalier_reconciler_taxon_lookup() -> None:
    """
    Espalier's Reconciler normalizes every FASTA ID by replacing underscores
    with spaces before sequence lookup. That breaks hantavirus isolate names
    where underscores are semantically real, not whitespace substitutions.

    We patch the runtime class to accept both the raw FASTA IDs and the
    whitespace-expanded form.
    """
    import Espalier.Reconciler as Rmod  # type: ignore
    import Espalier.RAxML as RXmod  # type: ignore

    if getattr(Rmod.Reconciler, "_hanta_taxon_lookup_patched", False) and getattr(RXmod.RAxMLRunner, "_hanta_tree_sanitize_patched", False):
        return

    def _clamp_negative_edges(tree, min_edge_length: float = 1e-6):
        for edge in tree.postorder_edge_iter():
            if edge.length is not None and edge.length < min_edge_length:
                edge.length = min_edge_length

    def _sanitize_written_tree(path, min_edge_length: float = 1e-6):
        raw = Path(path).read_text()
        sanitized = re.sub(
            r":-(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            f":{min_edge_length}",
            raw,
        )
        if sanitized != raw:
            Path(path).write_text(sanitized)

    def _call(self, ref, alt, maf, seq_file, start_pos=None, end_pos=None):
        my_records = list(Rmod.SeqIO.parse(seq_file, "fasta"))
        seq_dict = {}
        for record in my_records:
            seq_dict[record.id] = record
            seq_dict.setdefault(record.id.replace("_", " "), record)
        total_length = len(seq_dict[next(iter(seq_dict))].seq)

        if not start_pos:
            start_pos = 0
        if not end_pos:
            end_pos = total_length

        left_dists = Rmod.np.abs(Rmod.np.arange(0, start_pos) - start_pos)
        right_dists = Rmod.np.abs(Rmod.np.arange(end_pos, total_length) - end_pos)
        maf = list(reversed(maf))

        Rmod.logging.info("Starting regrafting")
        Rmod.logging.info("Subtrees in MAF: %s", str(len(maf)))

        root_tree = maf[0].clone(depth=2)
        root = Rmod.ReconciledTree(parent=None, tree=root_tree, index=0)
        queue = [root]
        sampled_trees = []

        nodes_visited = 0
        ref_count = 0
        alt_count = 0
        while queue:
            parent = queue.pop()
            maf_index = parent.index + 1
            nodes_visited += 1
            if nodes_visited % 10 == 0:
                Rmod.logging.info("Processing next search node : %s", str(nodes_visited))

            if maf_index == len(maf):
                if parent.tree.__len__() != ref.__len__():
                    Rmod.logging.warning("Trees of unequal size found!!")
                parent.source = "Reconciled"
                sampled_trees.append(parent)
                Rmod.logging.debug("Reached tip node in search tree. Output tree #: %s", str(len(sampled_trees)))
            else:
                sub_tree = maf[maf_index]
                rec_tree = parent.tree
                Rmod.logging.debug("Finding attachment edges")
                attachment_edge_alt = Rmod.find_attachment_edge(rec_tree, alt, sub_tree)
                rec_tree_clone = rec_tree.clone(depth=2)
                attachment_edge_ref = Rmod.find_attachment_edge(rec_tree_clone, ref, sub_tree)

                Rmod.logging.debug("Attaching subtrees")
                sub_tree_clone = sub_tree.clone(depth=2)
                sub_tree_2 = sub_tree.clone(depth=2)
                rec_tree_ref = Rmod.attach_subtree(rec_tree_clone, sub_tree_clone, ref, attachment_edge_ref)
                rec_tree_alt = Rmod.attach_subtree(rec_tree, sub_tree_2, alt, attachment_edge_alt)

                if rec_tree_alt.__len__() > 3:
                    Rmod.logging.debug("Computing site likelihoods")
                    _clamp_negative_edges(rec_tree_alt)
                    _clamp_negative_edges(rec_tree_ref)
                    alt_tree_file = self.temp_dir + "temp_alt.tre"
                    rec_tree_alt.write(path=alt_tree_file, schema="newick", suppress_annotations=True, suppress_rooting=True)
                    _sanitize_written_tree(alt_tree_file)
                    ref_tree_file = self.temp_dir + "temp_ref.tre"
                    rec_tree_ref.write(path=ref_tree_file, schema="newick", suppress_annotations=True, suppress_rooting=True)
                    _sanitize_written_tree(ref_tree_file)
                    temp_seq_file = self.temp_dir + "temp_seqs.fasta"
                    temp_records = [seq_dict[nd.taxon.label] for nd in rec_tree_alt.leaf_node_iter()]
                    Rmod.SeqIO.write(temp_records, temp_seq_file, "fasta")
                    alt_site_likes = self.raxml.get_site_likelihoods(alt_tree_file, temp_seq_file)
                    ref_site_likes = self.raxml.get_site_likelihoods(ref_tree_file, temp_seq_file)

                    alt_local_like = Rmod.np.sum(alt_site_likes[start_pos:end_pos])
                    ref_local_like = Rmod.np.sum(ref_site_likes[start_pos:end_pos])

                    left_alt_site_likes = alt_site_likes[:start_pos] * Rmod.np.exp(-self.prior_gamma * left_dists)
                    right_alt_site_likes = alt_site_likes[end_pos:] * Rmod.np.exp(-self.prior_gamma * right_dists)
                    left_ref_site_likes = ref_site_likes[:start_pos] * Rmod.np.exp(-self.prior_gamma * left_dists)
                    right_ref_site_likes = ref_site_likes[end_pos:] * Rmod.np.exp(-self.prior_gamma * right_dists)

                    alt_global_like = Rmod.np.sum(left_alt_site_likes) + Rmod.np.sum(right_alt_site_likes)
                    ref_global_like = Rmod.np.sum(left_ref_site_likes) + Rmod.np.sum(right_ref_site_likes)
                else:
                    alt_local_like = 0
                    ref_local_like = 0
                    alt_global_like = 0
                    ref_global_like = 0

                alt_like = alt_local_like + alt_global_like
                ref_like = ref_local_like + ref_global_like
                Rmod.logging.debug("Updating search tree")

                if ref_like > alt_like:
                    child = Rmod.ReconciledTree(parent=parent, tree=rec_tree_ref, index=maf_index, like=ref_like)
                    queue.append(child)
                    ref_count += 1
                    ratio = Rmod.math.exp(alt_like - ref_like)
                    if ratio > self.lower_bound_ratio:
                        child = Rmod.ReconciledTree(parent=parent, tree=rec_tree_alt, index=maf_index, like=alt_like)
                        queue.append(child)
                        alt_count += 1
                else:
                    child = Rmod.ReconciledTree(parent=parent, tree=rec_tree_alt, index=maf_index, like=alt_like)
                    queue.append(child)
                    alt_count += 1
                    ratio = Rmod.math.exp(ref_like - alt_like)
                    if ratio > self.lower_bound_ratio:
                        child = Rmod.ReconciledTree(parent=parent, tree=rec_tree_ref, index=maf_index, like=ref_like)
                        queue.append(child)
                        ref_count += 1

        Rmod.logging.info("Alt / ref reattachments = " + f"{alt_count}" + " / " + f"{ref_count}")
        return sampled_trees

    if not getattr(Rmod.Reconciler, "_hanta_taxon_lookup_patched", False):
        Rmod.Reconciler.__call__ = _call
        Rmod.Reconciler._hanta_taxon_lookup_patched = True

    if not getattr(RXmod.RAxMLRunner, "_hanta_tree_sanitize_patched", False):
        _orig_get_site_likelihoods = RXmod.RAxMLRunner.get_site_likelihoods
        _orig_get_tree_likelihood = RXmod.RAxMLRunner.get_tree_likelihood

        def _patched_get_site_likelihoods(self, tree_file, seq_file):
            _sanitize_written_tree(tree_file)
            return _orig_get_site_likelihoods(self, tree_file, seq_file)

        def _patched_get_tree_likelihood(self, tree_file, seq_file):
            _sanitize_written_tree(tree_file)
            return _orig_get_tree_likelihood(self, tree_file, seq_file)

        RXmod.RAxMLRunner.get_site_likelihoods = _patched_get_site_likelihoods
        RXmod.RAxMLRunner.get_tree_likelihood = _patched_get_tree_likelihood
        RXmod.RAxMLRunner._hanta_tree_sanitize_patched = True


def write_dropped_table(path: Path, dropped: Sequence[Dict[str, str]]) -> None:
    if not dropped:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in dropped for key in row.keys()})
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in dropped:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def build_isolate_matrix(
    assignments: Dict[str, TipAssignment],
    segments: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    all_segments = list(segments)
    for canonical, assignment in sorted(assignments.items()):
        row: Dict[str, object] = {
            "canonical_isolate": canonical,
            "tip_name": assignment.tip_name,
            "display_name": assignment.display_name,
            "date": assignment.normalized_date.iso,
            "date_precision": assignment.normalized_date.precision,
        }
        for seg in all_segments:
            row[f"has_{seg}"] = "yes" if seg in assignment.present_segments else "no"
        present_fraction = len(assignment.present_segments) / max(1, len(all_segments))
        row["segment_completeness"] = f"{present_fraction:.2f}"
        rows.append(row)
    return rows


def write_isolate_matrix(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def partition_isolates(
    assignments: Dict[str, TipAssignment],
    required_segments: Sequence[str],
) -> Tuple[List[str], List[str]]:
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


def write_segment_fastas(
    output_root: Path,
    result: SegmentResult,
    assignments: Dict[str, TipAssignment],
    matched_isolates: Sequence[str],
    partial_isolates: Sequence[str],
) -> Dict[str, Path]:
    seg_dir = output_root / "segments" / result.segment
    seg_dir.mkdir(parents=True, exist_ok=True)
    filtered_records: List[Tuple[str, str]] = []
    matched_records: List[Tuple[str, str]] = []
    partial_records: List[Tuple[str, str]] = []
    for canonical, entry in sorted(result.sequences.items()):
        assignment = assignments[canonical]
        record = (assignment.tip_name, entry.sequence)
        filtered_records.append(record)
        if canonical in matched_isolates:
            matched_records.append(record)
        elif canonical in partial_isolates:
            partial_records.append(record)
    filtered_path = seg_dir / f"{result.segment}_filtered.fasta"
    matched_path = seg_dir / f"{result.segment}_matched.fasta"
    partial_path = seg_dir / f"{result.segment}_partial.fasta"
    write_fasta(filtered_path, filtered_records)
    write_fasta(matched_path, matched_records)
    if partial_records:
        write_fasta(partial_path, partial_records)
    write_dropped_table(seg_dir / f"{result.segment}_dropped.tsv", result.dropped)
    write_json(
        seg_dir / f"{result.segment}_summary.json",
        {
            "segment": result.segment,
            "n_raw_sequences": len(result.dropped) + len(result.sequences),
            "n_retained": len(result.sequences),
            "max_length": result.max_length,
            "min_length_threshold": result.min_length_threshold,
            "metadata_stats": result.metadata_stats,
        },
    )
    return {
        "filtered": filtered_path,
        "matched": matched_path,
        "partial": partial_path if partial_records else None,
    }


def write_matched_metadata(
    path: Path,
    segment: str,
    matched_isolates: Sequence[str],
    result: SegmentResult,
    assignments: Dict[str, TipAssignment],
) -> None:
    rows: List[Dict[str, str]] = []
    for canonical in matched_isolates:
        entry = result.sequences.get(canonical)
        if not entry:
            continue
        assignment = assignments[canonical]
        rows.append(
            {
                "strain": assignment.tip_name,
                "isolate_display": assignment.display_name,
                "canonical_isolate": canonical,
                "segment": segment,
                "accession": entry.accession,
                "date": assignment.normalized_date.iso,
                "date_precision": assignment.normalized_date.precision,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strain",
                "isolate_display",
                "canonical_isolate",
                "segment",
                "accession",
                "date",
                "date_precision",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def available_segments(segment_results: Sequence[SegmentResult]) -> List[str]:
    return [result.segment for result in segment_results]


def filter_alignment_to_taxa(alignment: Path, allowed_taxa: Iterable[str], output: Path) -> Optional[Path]:
    if not alignment.exists():
        LOGGER.warning("Alignment %s is missing; cannot filter to shared taxa.", alignment)
        return None
    allowed = set(allowed_taxa)
    records = [(header, seq) for header, seq in read_fasta(alignment) if header in allowed]
    if not records:
        LOGGER.warning("No records retained when filtering %s to %d taxa.", alignment, len(allowed))
        try:
            output.unlink()
        except FileNotFoundError:
            pass
        return None
    write_fasta(output, records)
    return output


def prepare_common_taxa(
    species_label: str,
    augur_outputs: Sequence[SegmentAugurOutput],
    species_output: Path,
    tree_source: str = "refined",
) -> Tuple[Dict[str, Path], Dict[str, Path], Dict[str, Path], Set[str]]:
    if not augur_outputs:
        return {}, {}, {}, set()
    require_module(dendropy, "Common tip pruning")
    tree_dir = (species_output / "espalier" / "trees").resolve()
    seq_dir = (species_output / "espalier" / "alignments").resolve()
    tree_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)
    tip_sets: Dict[str, Set[str]] = {}
    dend_trees: Dict[str, dendropy.Tree] = {}
    for out in augur_outputs:
        tree_path = out.raw_tree_newick if tree_source == "raw" else out.refined_tree_newick
        tree = dendropy_tree_get(path=str(tree_path), schema="newick")
        dend_trees[out.segment] = tree
        tip_sets[out.segment] = {leaf.taxon.label for leaf in tree.leaf_node_iter()}
    if not tip_sets:
        return {}, {}, {}, set()
    common_taxa = set.intersection(*(tips for tips in tip_sets.values()))
    if not common_taxa:
        LOGGER.warning("No shared taxa across %d segments for %s; skipping Espalier prep.", len(tip_sets), species_label)
        return {}, {}, {}, set()
    common_trees: Dict[str, Path] = {}
    binary_trees: Dict[str, Path] = {}
    common_alignments: Dict[str, Path] = {}
    for out in augur_outputs:
        extras = sorted(tip_sets[out.segment] - common_taxa)
        cleaned = dend_trees[out.segment].clone(depth=2)
        if extras:
            cleaned.prune_taxa_with_labels(extras)
        cleaned_path = (tree_dir / f"{out.segment}_tree_common.tre").resolve()
        cleaned.write(path=str(cleaned_path), schema="newick", suppress_annotations=True, suppress_rooting=True)
        binary_tree = cleaned.clone(depth=2)
        binary_tree.resolve_polytomies()
        binary_path = (tree_dir / f"{out.segment}_tree_common_binary.tre").resolve()
        binary_tree.write(path=str(binary_path), schema="newick", suppress_annotations=True, suppress_rooting=True)
        common_trees[out.segment] = cleaned_path
        binary_trees[out.segment] = binary_path
        filtered_alignment = seq_dir / f"{out.segment}_aligned_common.fasta"
        filtered = filter_alignment_to_taxa(out.aligned_fasta, common_taxa, filtered_alignment)
        if filtered:
            common_alignments[out.segment] = filtered.resolve()
    return common_alignments, common_trees, binary_trees, common_taxa


def compute_spr_distances(
    species_label: str,
    binary_trees: Dict[str, Path],
    report_dir: Path,
) -> Optional[Path]:
    if len(binary_trees) < 2:
        LOGGER.info("Not enough segments with data to compute SPR distances for %s.", species_label)
        return None
    require_module(dendropy, "SPR distance computation")
    if MAF is None:
        LOGGER.warning("Espalier MAF module unavailable; skipping SPR distances for %s.", species_label)
        return None
    taxa = dendropy.TaxonNamespace()
    records: List[Dict[str, object]] = []
    for (seg_a, path_a), (seg_b, path_b) in itertools.combinations(sorted(binary_trees.items()), 2):
        tree_a = dendropy_tree_get(path=str(path_a), schema="newick", taxon_namespace=taxa)
        tree_b = dendropy_tree_get(path=str(path_b), schema="newick", taxon_namespace=taxa)
        dist = MAF.get_spr_dist(tree_a.clone(depth=2), tree_b.clone(depth=2))
        records.append(
            {
                "species": species_label,
                "segment_a": seg_a,
                "segment_b": seg_b,
                "spr_distance": dist,
                "tree_a": str(path_a),
                "tree_b": str(path_b),
            }
        )
    if not records:
        return None
    report_dir.mkdir(parents=True, exist_ok=True)
    spr_path = report_dir / "spr_distances.tsv"
    if pd is not None:
        df = pd.DataFrame(records)
        df.to_csv(spr_path, sep="\t", index=False)
    else:
        fieldnames = list(records[0].keys())
        with spr_path.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in records:
                writer.writerow(row)
    return spr_path


def run_espalier_arg(
    species_label: str,
    species_slug: str,
    binary_trees: Dict[str, Path],
    common_alignments: Dict[str, Path],
    species_output: Path,
    rec_rate: float,
    lower_bound_ratio: float,
    prior_gamma: float,
    raxml_bin: str,
    threads: int,
) -> Optional[Path]:
    if len(binary_trees) < 2:
        LOGGER.info("Skipping Espalier ARG reconstruction for %s (need ≥2 segments).", species_label)
        return None
    ensure_espalier_path()
    require_module(dendropy, "Espalier ARG reconstruction")
    patch_espalier_reconciler_taxon_lookup()
    try:
        from Espalier.Reconciler import Reconciler  # type: ignore
        from Espalier.RAxML import RAxMLRunner  # type: ignore
        from Espalier.ARGBuilder import ARGBuilder  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires optional dependency
        raise RuntimeError("Espalier is not available but --run-espalier was requested.") from exc
    segments = sorted(set(binary_trees.keys()) & set(common_alignments.keys()))
    if len(segments) < 2:
        LOGGER.info(
            "Need at least two segments with shared alignments for %s; skipping Espalier ARG reconstruction.",
            species_label,
        )
        return None
    tree_paths = [str(binary_trees[seg]) for seg in segments]
    seq_paths: List[str] = []
    for seg in segments:
        align_path = common_alignments.get(seg)
        if not align_path or not align_path.exists():
            LOGGER.warning(
                "Missing filtered alignment for %s segment %s; skipping Espalier ARG reconstruction.",
                species_label,
                seg,
            )
            return None
        align_records = read_fasta(align_path)
        preflight_fasta_header_uniqueness(align_records, source=align_path)
        preflight_fasta_header_uniqueness(
            align_records,
            source=align_path,
            normalizer=lambda header: "_".join(header.split()),
            normalizer_name="whitespace-sanitized",
        )
        seq_paths.append(str(align_path))
    espalier_dir = (species_output / "espalier" / "arg").resolve()
    espalier_dir.mkdir(parents=True, exist_ok=True)
    esp_input_dir = espalier_dir / "espalier_inputs"
    esp_input_dir.mkdir(parents=True, exist_ok=True)
    esp_tree_paths: List[str] = []
    esp_seq_paths: List[str] = []
    for idx, seg in enumerate(segments):
        tree_out = esp_input_dir / f"{seg}_tree_sanitized.tre"
        seq_out = esp_input_dir / f"{seg}_aligned_sanitized.fasta"
        convert_tree_for_espalier(Path(tree_paths[idx]), tree_out)
        convert_alignment_for_espalier(Path(seq_paths[idx]), seq_out)
        esp_tree_paths.append(str(tree_out))
        esp_seq_paths.append(str(seq_out))
    ref_tree = dendropy_tree_get(path=esp_tree_paths[0], schema="newick")
    taxa = ref_tree.taxon_namespace
    raxml = RAxMLRunner(raxml_path=raxml_bin, threads=threads, temp_dir=str(espalier_dir) + os.sep)
    reconciler = Reconciler(raxml, lower_bound_ratio=lower_bound_ratio, prior_gamma=prior_gamma)
    argb = ARGBuilder(reconciler, raxml)
    with pushd(espalier_dir):
        tree_list = argb.reconstruct_ARG(esp_tree_paths, esp_seq_paths, ref_tree, rec_rate)
    local_dir = espalier_dir / "local_trees"
    local_dir.mkdir(parents=True, exist_ok=True)
    for idx, tree in enumerate(tree_list):
        out_path = local_dir / f"{species_slug}_ARG_local{idx}.tre"
        tree.taxon_namespace = taxa
        for leaf in tree.leaf_node_iter():
            if leaf.taxon and leaf.taxon.label:
                leaf.taxon.label = leaf.taxon.label.replace(" ", "_")
        tree.write(
            path=str(out_path),
            schema="newick",
            preserve_spaces=True,
            suppress_annotations=True,
            suppress_rooting=True,
        )
    return local_dir


def run_species_dataset(
    species_label: str,
    dataset_root: Path,
    slug: str,
    args: argparse.Namespace,
) -> SpeciesRunResult:
    LOGGER.info("=== Processing %s (%s) ===", species_label, dataset_root)
    species_output = (args.output_root / slug).resolve()
    species_output.mkdir(parents=True, exist_ok=True)
    report_dir = species_output / "reports"
    segment_results: List[SegmentResult] = []
    for segment in args.segments:
        try:
            result = process_segment(dataset_root, segment, args.length_threshold)
        except FileNotFoundError:
            LOGGER.warning("Segment %s missing for %s; skipping this segment.", segment, species_label)
            continue
        if not result.sequences:
            LOGGER.warning("No sequences passed filters for %s segment %s.", species_label, segment)
            continue
        segment_results.append(result)
    if not segment_results:
        LOGGER.warning("No usable segments for %s.", species_label)
        return SpeciesRunResult(
            species_label=species_label,
            slug=slug,
            dataset_root=dataset_root,
            output_dir=species_output,
            matched_isolates=0,
            partial_isolates=0,
            n_common_taxa=0,
            spr_table=None,
            arg_tree_dir=None,
        )
    assignments = build_tip_assignments(segment_results)
    _raise_if_duplicate_labels(
        (assignment.tip_name for assignment in assignments.values()),
        context=f"{species_label} tip assignments",
        label_name="tip_name",
    )
    isolate_rows = build_isolate_matrix(assignments, available_segments(segment_results))
    write_isolate_matrix(report_dir / "isolate_matrix.tsv", isolate_rows)
    available = available_segments(segment_results)
    matched_isolates, partial_isolates = partition_isolates(assignments, available)
    LOGGER.info(
        "%s matched isolates with segments %s; %s partial.",
        len(matched_isolates),
        ",".join(available),
        len(partial_isolates),
    )
    for result in segment_results:
        write_segment_fastas(
            output_root=species_output,
            result=result,
            assignments=assignments,
            matched_isolates=matched_isolates,
            partial_isolates=partial_isolates,
        )
    if len(matched_isolates) < args.min_complete_isolates:
        LOGGER.warning(
            "Only %d isolates with complete segment coverage for %s (requires >= %d); skipping downstream steps.",
            len(matched_isolates),
            species_label,
            args.min_complete_isolates,
        )
        return SpeciesRunResult(
            species_label=species_label,
            slug=slug,
            dataset_root=dataset_root,
            output_dir=species_output,
            matched_isolates=len(matched_isolates),
            partial_isolates=len(partial_isolates),
            n_common_taxa=0,
            spr_table=None,
            arg_tree_dir=None,
        )
    augur_outputs: List[SegmentAugurOutput] = []
    if not args.skip_augur:
        augur_outputs = run_augur_pipeline(
            output_root=species_output,
            segment_results=segment_results,
            matched_isolates=matched_isolates,
            assignments=assignments,
            threads=args.threads,
        )
    else:
        LOGGER.info("Skipping Augur execution per --skip-augur.")
    common_alignments: Dict[str, Path] = {}
    binary_trees: Dict[str, Path] = {}
    common_taxa: Set[str] = set()
    if augur_outputs:
        common_alignments, _, binary_trees, common_taxa = prepare_common_taxa(
            species_label=species_label,
            augur_outputs=augur_outputs,
            species_output=species_output,
            tree_source=args.common_tree_source,
        )
    spr_table = None
    if binary_trees:
        spr_table = compute_spr_distances(species_label, binary_trees, report_dir)
    arg_tree_dir = None
    if args.run_espalier and binary_trees and common_alignments:
        prior_gamma = args.prior_gamma if args.prior_gamma is not None else args.recombination_rate
        arg_tree_dir = run_espalier_arg(
            species_label=species_label,
            species_slug=slug,
            binary_trees=binary_trees,
            common_alignments=common_alignments,
            species_output=species_output,
            rec_rate=args.recombination_rate,
            lower_bound_ratio=args.lower_bound_ratio,
            prior_gamma=prior_gamma,
            raxml_bin=args.raxml_bin,
            threads=args.threads,
        )
    return SpeciesRunResult(
        species_label=species_label,
        slug=slug,
        dataset_root=dataset_root,
        output_dir=species_output,
        matched_isolates=len(matched_isolates),
        partial_isolates=len(partial_isolates),
        n_common_taxa=len(common_taxa),
        spr_table=spr_table,
        arg_tree_dir=arg_tree_dir,
    )


def write_species_summary(results: Sequence[SpeciesRunResult], output_root: Path) -> None:
    if not results:
        return
    summary_dir = output_root / "reports"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "species_summary.tsv"
    rows = [
        {
            "species": r.species_label,
            "slug": r.slug,
            "matched_isolates": r.matched_isolates,
            "partial_isolates": r.partial_isolates,
            "common_taxa": r.n_common_taxa,
            "spr_table": str(r.spr_table) if r.spr_table else "",
            "arg_tree_dir": str(r.arg_tree_dir) if r.arg_tree_dir else "",
        }
        for r in results
    ]
    if pd is not None:
        pd.DataFrame(rows).to_csv(summary_path, sep="\t", index=False)
    else:
        fieldnames = list(rows[0].keys())
        with summary_path.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    LOGGER.info("Species summary written to %s", summary_path)


class AugurRunner:
    def __init__(self, binary: str = "augur", threads: int = 4) -> None:
        self.binary = binary
        self.threads = threads
        self.retry_script = Path(__file__).resolve().parents[3] / "model_test" / "scripts" / "retry_augur_tree_dna_safe.py"

    def run(self, args: Sequence[str], cwd: Optional[Path] = None) -> None:
        cmd = [self.binary, *args]
        LOGGER.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

    def align(self, fasta: Path, output: Path, reference: Optional[Path]) -> None:
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

    def tree(self, alignment: Path, output_tree: Path) -> None:
        args = [
            "tree",
            "--alignment",
            str(alignment),
            "--output",
            str(output_tree),
            "--method",
            "iqtree",
            "--nthreads",
            str(self.threads),
            "--substitution-model",
            "GTR+G",
            "--tree-builder-args",
            "-st DNA",
        ]
        try:
            self.run(args)
        except subprocess.CalledProcessError:
            LOGGER.warning(
                "Augur tree failed for %s. Triggering deterministic DNA-safe retry.",
                alignment,
            )
            self.retry_tree_dna_safe(alignment, output_tree)

    def retry_tree_dna_safe(self, alignment: Path, output_tree: Path) -> None:
        if not self.retry_script.exists():
            raise RuntimeError(
                f"Augur tree retry helper is missing: {self.retry_script}"
            )
        cmd = [
            sys.executable,
            str(self.retry_script),
            "--alignment",
            str(alignment),
            "--output-tree",
            str(output_tree),
            "--threads",
            str(self.threads),
            "--augur-bin",
            self.binary,
        ]
        LOGGER.info("Running retry helper: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def refine(
        self,
        alignment: Path,
        tree: Path,
        metadata: Path,
        output_tree: Path,
        output_node_data: Path,
        ) -> None:
        args = [
            "refine",
            "--alignment",
            str(alignment),
            "--tree",
            str(tree),
            "--metadata",
            str(metadata),
            "--output-tree",
            str(output_tree),
            "--output-node-data",
            str(output_node_data),
            "--timetree",
            "--root",
            "best",
            "--date-confidence",
            "--stochastic-resolve",
            "--clock-filter-iqd",
            "3",
            "--clock-rate",
            '0.0008',
            "--clock-std-dev",
            "0.0004",
            "--max-iter",
            "5",
            "--precision",
            "3",
        ]
        try:
            self.run(args)
        except subprocess.CalledProcessError:
            LOGGER.warning(
                "Augur refine failed for %s. Falling back to the raw tree so common-tip Espalier prep can continue.",
                tree,
            )
            output_tree.write_text(tree.read_text())
            output_node_data.write_text('{"nodes": {}}\n')


def run_augur_pipeline(
    output_root: Path,
    segment_results: Sequence[SegmentResult],
    matched_isolates: Sequence[str],
    assignments: Dict[str, TipAssignment],
    threads: int,
) -> List[SegmentAugurOutput]:
    runner = AugurRunner(threads=threads)
    outputs: List[SegmentAugurOutput] = []
    for result in segment_results:
        seg_dir = output_root / "segments" / result.segment
        matched_fasta = seg_dir / f"{result.segment}_matched.fasta"
        if not matched_fasta.exists():
            LOGGER.warning("Matched FASTA for %s is missing, skipping Augur.", result.segment)
            continue
        matched_records = read_fasta(matched_fasta)
        preflight_fasta_header_uniqueness(matched_records, source=matched_fasta)
        preflight_fasta_header_uniqueness(
            matched_records,
            source=matched_fasta,
            normalizer=lambda header: "_".join(header.split()),
            normalizer_name="whitespace-sanitized",
        )
        seq_count = len(matched_records)
        if seq_count < 3:
            LOGGER.warning(
                "Segment %s has only %d sequences after filtering; skipping Augur tree building.",
                result.segment,
                seq_count,
            )
            continue
        augur_dir = output_root / "augur" / result.segment
        augur_dir.mkdir(parents=True, exist_ok=True)
        aligned_path = augur_dir / f"{result.segment}_aligned.fasta"
        raw_tree_path = augur_dir / f"{result.segment}_tree_raw.nwk"
        refined_tree_path = augur_dir / f"{result.segment}_tree_refined.nwk"
        node_data_path = augur_dir / f"{result.segment}_node_data.json"
        metadata_path = augur_dir / f"{result.segment}_matched_metadata.tsv"
        write_matched_metadata(metadata_path, result.segment, matched_isolates, result, assignments)
        runner.align(matched_fasta, aligned_path, result.reference_path)
        runner.tree(aligned_path, raw_tree_path)
        runner.refine(aligned_path, raw_tree_path, metadata_path, refined_tree_path, node_data_path)
        refined_newick = convert_nexus_to_newick(refined_tree_path, refined_tree_path.with_suffix(".newick"))
        outputs.append(
            SegmentAugurOutput(
                segment=result.segment,
                matched_fasta=matched_fasta.resolve(),
                aligned_fasta=aligned_path.resolve(),
                raw_tree_newick=raw_tree_path.resolve(),
                refined_tree_nexus=refined_tree_path.resolve(),
                refined_tree_newick=refined_newick.resolve(),
                metadata_tsv=metadata_path.resolve(),
            )
        )
    return outputs


def convert_nexus_to_newick(nexus_path: Path, out_path: Path) -> Path:
    """
    Convert Augur's Nexus tree into a Newick file.
    """
    raw_text = nexus_path.read_text().strip()
    if raw_text.startswith("(") and raw_text.endswith(";"):
        # Already Newick; just copy/normalize.
        out_path.write_text(raw_text + ("\n" if not raw_text.endswith("\n") else ""))
        return out_path
    if dendropy is not None:
        schema = "nexus"
        if raw_text.startswith("("):
            schema = "newick"
        tree = dendropy_tree_get(path=str(nexus_path), schema=schema)
        tree.write(path=str(out_path), schema="newick", suppress_annotations=True, suppress_rooting=True)
        return out_path
    match = re.search(
        r"tree\s+[^=]+=\s*(?:\[\&[^\]]+\]\s*)?(\(.+\));",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Could not locate tree string inside {nexus_path}")
    newick = match.group(1).strip()
    out_path.write_text(newick + ("\n" if not newick.endswith("\n") else ""))
    return out_path


def write_espalier_reference(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(ESPALIER_DOC_BLURB + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)
    configure_espalier_imports(args.espalier_path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    dataset_entries = enumerate_dataset_roots(args)
    results: List[SpeciesRunResult] = []
    for species_label, dataset_root, slug in dataset_entries:
        result = run_species_dataset(species_label, dataset_root, slug, args)
        results.append(result)
    write_species_summary(results, args.output_root)
    write_espalier_reference(args.output_root / "reports" / "espalier_reference.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Pipeline failed: %s", exc)
        sys.exit(1)
