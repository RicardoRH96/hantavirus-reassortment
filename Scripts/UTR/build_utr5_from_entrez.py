#!/usr/bin/env python3
"""
Build 5' UTR datasets from Entrez annotations and compute per-segment
all-to-all structure/sequence distance matrices.

CDS coordinate priority:
1) GFF3 CDS features
2) GenBank CDS features

5' UTR definition:
- '+' strand: sequence before CDS start
- '-' strand: sequence after CDS end, reverse-complemented to transcript orientation
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


LOGGER = logging.getLogger("utr5_builder")
SEGMENTS = ("S", "M", "L")
MANIFEST_RE = re.compile(r"^(?P<species>.+)_(?P<segment>[SML])_accessions\.txt$")


@dataclass
class CdsInfo:
    start: int
    end: int
    strand: str
    source: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build transcript-relative 5' UTR matrices from Entrez.")
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=repo_root / "model_test" / "runs" / "utr5" / "manifests",
        help="Folder with <species>_<segment>_accessions.txt manifest files.",
    )
    parser.add_argument(
        "--email",
        required=True,
        help="Entrez email.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("NCBI_API_KEY"),
        help="NCBI API key (optional, defaults to $NCBI_API_KEY).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "model_test" / "runs" / "utr5",
        help="Root output folder for per-species/per-segment results.",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=repo_root / "model_test" / "reports" / "utr5_coverage_summary.tsv",
        help="Coverage report output TSV.",
    )
    parser.add_argument(
        "--min-utr-length",
        type=int,
        default=15,
        help="Minimum 5' UTR length to retain.",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=0.34,
        help="Sleep duration after each Entrez request to reduce rate-limit errors.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logs.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path) -> List[str]:
    accessions: List[str] = []
    seen = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        acc = line.split(".")[0]
        if acc not in seen:
            seen.add(acc)
            accessions.append(acc)
    return accessions


def iter_manifests(root: Path) -> List[Tuple[str, str, Path]]:
    manifests: List[Tuple[str, str, Path]] = []
    for path in sorted(root.glob("*_accessions.txt")):
        match = MANIFEST_RE.match(path.name)
        if not match:
            continue
        manifests.append((match.group("species"), match.group("segment"), path))
    return manifests


def fetch_text_cached(
    *,
    db: str,
    accession: str,
    rettype: str,
    retmode: str,
    cache_path: Path,
    throttle_seconds: float,
) -> str:
    if cache_path.exists():
        return cache_path.read_text()
    LOGGER.debug("Downloading %s rettype=%s", accession, rettype)
    handle = Entrez.efetch(db=db, id=accession, rettype=rettype, retmode=retmode)
    text = handle.read()
    ensure_parent(cache_path)
    cache_path.write_text(text)
    if throttle_seconds > 0:
        time.sleep(throttle_seconds)
    return text


def fetch_fasta_cached(accession: str, cache_path: Path, throttle_seconds: float) -> SeqRecord:
    if cache_path.exists():
        return SeqIO.read(cache_path, "fasta")
    LOGGER.debug("Downloading %s fasta", accession)
    handle = Entrez.efetch(db="nuccore", id=accession, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    ensure_parent(cache_path)
    SeqIO.write(record, cache_path, "fasta")
    if throttle_seconds > 0:
        time.sleep(throttle_seconds)
    return record


def parse_gff3_cds(text: str) -> List[Tuple[int, int, str]]:
    cds: List[Tuple[int, int, str]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        feature_type = parts[2].strip().lower()
        if feature_type != "cds":
            continue
        try:
            start = int(parts[3])
            end = int(parts[4])
        except ValueError:
            continue
        strand = parts[6].strip()
        if strand not in {"+", "-"}:
            continue
        cds.append((start, end, strand))
    return cds


def parse_genbank_cds(record: SeqRecord) -> List[Tuple[int, int, str]]:
    cds: List[Tuple[int, int, str]] = []
    for feat in record.features:
        if feat.type != "CDS":
            continue
        strand_raw = getattr(feat.location, "strand", None)
        if strand_raw == 1:
            strand = "+"
        elif strand_raw == -1:
            strand = "-"
        else:
            continue
        start = int(feat.location.start) + 1
        end = int(feat.location.end)
        cds.append((start, end, strand))
    return cds


def choose_primary_cds(candidates: List[Tuple[int, int, str]], source: str) -> Tuple[Optional[CdsInfo], str]:
    if not candidates:
        return None, "no_cds_features"
    strands = {strand for _, _, strand in candidates}
    if len(strands) != 1:
        return None, "mixed_cds_strands"
    strand = next(iter(strands))
    if strand == "+":
        start, end, _ = min(candidates, key=lambda x: (x[0], x[1]))
    else:
        start, end, _ = max(candidates, key=lambda x: (x[1], x[0]))
    if end < start:
        return None, "invalid_cds_interval"
    return CdsInfo(start=start, end=end, strand=strand, source=source), ""


def extract_utr5(seq: Seq, cds: CdsInfo) -> Tuple[Optional[Seq], Optional[int], Optional[int], str]:
    seq_len = len(seq)
    if cds.strand == "+":
        utr_start = 1
        utr_end = cds.start - 1
        if utr_end < utr_start:
            return None, None, None, "no_5utr_before_cds"
        frag = seq[utr_start - 1 : utr_end]
    else:
        utr_start = cds.end + 1
        utr_end = seq_len
        if utr_end < utr_start:
            return None, None, None, "no_5utr_after_cds"
        frag = seq[utr_start - 1 : utr_end].reverse_complement()
    return frag, utr_start, utr_end, ""


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return levenshtein(a, b) / denom


def dotbracket_pairs(structure: str) -> set[Tuple[int, int]]:
    opens = {"(": ")", "[": "]", "{": "}", "<": ">"}
    stacks: Dict[str, List[int]] = {k: [] for k in opens}
    close_to_open = {v: k for k, v in opens.items()}
    pairs: set[Tuple[int, int]] = set()
    for idx, char in enumerate(structure):
        if char in opens:
            stacks[char].append(idx)
        elif char in close_to_open:
            opener = close_to_open[char]
            if stacks[opener]:
                left = stacks[opener].pop()
                pairs.add((left, idx))
    return pairs


def bp_distance(structure_a: str, structure_b: str) -> int:
    pairs_a = dotbracket_pairs(structure_a)
    pairs_b = dotbracket_pairs(structure_b)
    return len(pairs_a.symmetric_difference(pairs_b))


class FoldBackend:
    def __init__(self) -> None:
        self.backend = ""
        self.rna_module = None
        try:
            import RNA
        except Exception:
            RNA = None
        if RNA is not None:
            self.backend = "viennarna_python"
            self.rna_module = RNA
            return
        if shutil.which("RNAfold"):
            self.backend = "rnafold_cli"
            return
        raise RuntimeError(
            "ViennaRNA backend not available. Install Python RNA bindings or RNAfold CLI."
        )

    def fold(self, sequence_dna: str) -> Tuple[str, float]:
        seq_rna = sequence_dna.upper().replace("T", "U")
        if self.backend == "viennarna_python":
            structure, mfe = self.rna_module.fold(seq_rna)
            return str(structure), float(mfe)
        proc = subprocess.run(
            ["RNAfold", "--noPS"],
            input=seq_rna + "\n",
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if len(lines) < 2:
            raise RuntimeError(f"Unexpected RNAfold output: {proc.stdout!r}")
        second = lines[1]
        if "(" not in second or ")" not in second:
            raise RuntimeError(f"Cannot parse RNAfold result: {second!r}")
        structure = second.rsplit("(", 1)[0].strip()
        mfe_txt = second.rsplit("(", 1)[1].rstrip(")").strip()
        mfe = float(mfe_txt)
        return structure, mfe


def write_tsv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fasta(path: Path, records: List[Tuple[str, str]]) -> None:
    ensure_parent(path)
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i : i + 80] + "\n")


def write_matrix(path: Path, names: List[str], matrix: List[List[float]]) -> None:
    ensure_parent(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *[f"{value:.4f}" for value in row]])


def build_distance_matrix(items: List[Tuple[str, str]], metric) -> List[List[float]]:
    n = len(items)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(metric(items[i][1], items[j][1]))
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key

    manifest_root = args.manifest_root.resolve()
    output_root = args.output_root.resolve()
    coverage_output = args.coverage_output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_parent(coverage_output)

    manifests = iter_manifests(manifest_root)
    if not manifests:
        raise SystemExit(f"No manifest files found in {manifest_root}")

    cache_root = output_root / "cache"
    gff3_cache = cache_root / "gff3"
    gb_cache = cache_root / "genbank"
    fasta_cache = cache_root / "fasta"
    gff3_cache.mkdir(parents=True, exist_ok=True)
    gb_cache.mkdir(parents=True, exist_ok=True)
    fasta_cache.mkdir(parents=True, exist_ok=True)

    fold_backend = FoldBackend()
    LOGGER.info("Using fold backend: %s", fold_backend.backend)

    coverage_rows: List[Dict[str, object]] = []

    for species, segment, manifest_path in manifests:
        if segment not in SEGMENTS:
            continue
        LOGGER.info("Processing %s %s from %s", species, segment, manifest_path.name)

        accessions = read_manifest(manifest_path)
        species_dir = output_root / species / segment
        species_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows: List[Dict[str, object]] = []
        retained_records: List[Tuple[str, str]] = []

        n_cds_gff3 = 0
        n_cds_genbank = 0
        n_skipped = 0

        for accession in accessions:
            row: Dict[str, object] = {
                "species": species,
                "segment": segment,
                "accession": accession,
                "cds_source": "",
                "cds_start": "",
                "cds_end": "",
                "cds_strand": "",
                "utr5_start": "",
                "utr5_end": "",
                "utr5_length": "",
                "status": "skipped",
                "skip_reason": "",
            }
            try:
                fasta_record = fetch_fasta_cached(
                    accession,
                    fasta_cache / f"{accession}.fasta",
                    args.throttle_seconds,
                )
            except Exception as exc:
                row["skip_reason"] = f"fasta_fetch_failed:{exc}"
                metadata_rows.append(row)
                n_skipped += 1
                continue

            cds_info: Optional[CdsInfo] = None
            fallback_reason = ""

            try:
                gff_text = fetch_text_cached(
                    db="nuccore",
                    accession=accession,
                    rettype="gff3",
                    retmode="text",
                    cache_path=gff3_cache / f"{accession}.gff3",
                    throttle_seconds=args.throttle_seconds,
                )
                cds_info, fallback_reason = choose_primary_cds(parse_gff3_cds(gff_text), "gff3")
            except Exception as exc:
                fallback_reason = f"gff3_fetch_failed:{exc}"
                cds_info = None

            if cds_info is None:
                try:
                    gb_text = fetch_text_cached(
                        db="nuccore",
                        accession=accession,
                        rettype="gb",
                        retmode="text",
                        cache_path=gb_cache / f"{accession}.gb",
                        throttle_seconds=args.throttle_seconds,
                    )
                    with (gb_cache / f"{accession}.gb").open() as handle:
                        gb_record = SeqIO.read(handle, "genbank")
                    cds_info, gb_reason = choose_primary_cds(parse_genbank_cds(gb_record), "genbank")
                    if cds_info is None and not fallback_reason:
                        fallback_reason = gb_reason
                except Exception as exc:
                    if not fallback_reason:
                        fallback_reason = f"genbank_fetch_failed:{exc}"
                    cds_info = None

            if cds_info is None:
                row["skip_reason"] = fallback_reason or "cds_not_found"
                metadata_rows.append(row)
                n_skipped += 1
                continue

            utr_seq, utr_start, utr_end, utr_reason = extract_utr5(fasta_record.seq, cds_info)
            if utr_seq is None or utr_start is None or utr_end is None:
                row.update(
                    {
                        "cds_source": cds_info.source,
                        "cds_start": cds_info.start,
                        "cds_end": cds_info.end,
                        "cds_strand": cds_info.strand,
                        "skip_reason": utr_reason or "utr5_extraction_failed",
                    }
                )
                metadata_rows.append(row)
                n_skipped += 1
                continue

            utr_text = str(utr_seq).upper()
            if len(utr_text) < args.min_utr_length:
                row.update(
                    {
                        "cds_source": cds_info.source,
                        "cds_start": cds_info.start,
                        "cds_end": cds_info.end,
                        "cds_strand": cds_info.strand,
                        "utr5_start": utr_start,
                        "utr5_end": utr_end,
                        "utr5_length": len(utr_text),
                        "skip_reason": "utr5_too_short",
                    }
                )
                metadata_rows.append(row)
                n_skipped += 1
                continue

            record_id = accession
            retained_records.append((record_id, utr_text))

            row.update(
                {
                    "cds_source": cds_info.source,
                    "cds_start": cds_info.start,
                    "cds_end": cds_info.end,
                    "cds_strand": cds_info.strand,
                    "utr5_start": utr_start,
                    "utr5_end": utr_end,
                    "utr5_length": len(utr_text),
                    "status": "retained",
                    "skip_reason": "",
                }
            )
            metadata_rows.append(row)

            if cds_info.source == "gff3":
                n_cds_gff3 += 1
            else:
                n_cds_genbank += 1

        metadata_path = species_dir / "utr5_metadata.tsv"
        metadata_fields = [
            "species",
            "segment",
            "accession",
            "cds_source",
            "cds_start",
            "cds_end",
            "cds_strand",
            "utr5_start",
            "utr5_end",
            "utr5_length",
            "status",
            "skip_reason",
        ]
        write_tsv(metadata_path, metadata_rows, metadata_fields)

        structure_table_rows: List[Dict[str, object]] = []
        n_retained = len(retained_records)
        matrix_size = n_retained

        if n_retained > 0:
            write_fasta(species_dir / "utr5.fasta", retained_records)

            folded: List[Tuple[str, str, float, str]] = []
            for accession, seq in retained_records:
                structure, mfe = fold_backend.fold(seq)
                folded.append((accession, seq, mfe, structure))
                structure_table_rows.append(
                    {
                        "id": accession,
                        "species": species,
                        "segment": segment,
                        "length": len(seq),
                        "mfe_kcal": f"{mfe:.2f}",
                        "structure": structure,
                    }
                )

            write_tsv(
                species_dir / "utr5_structures.tsv",
                structure_table_rows,
                ["id", "species", "segment", "length", "mfe_kcal", "structure"],
            )

            seq_items = [(acc, seq) for acc, seq, _, _ in folded]
            struct_items = [(acc, struct) for acc, _, _, struct in folded]
            names = [acc for acc, _ in seq_items]

            seq_matrix = build_distance_matrix(seq_items, normalized_edit_distance)
            struct_matrix = build_distance_matrix(struct_items, bp_distance)

            write_matrix(species_dir / "utr5_sequence_distance.tsv", names, seq_matrix)
            write_matrix(species_dir / "utr5_structure_distance.tsv", names, struct_matrix)

        coverage_rows.append(
            {
                "species": species,
                "segment": segment,
                "manifest_path": str(manifest_path),
                "n_manifest_accessions": len(accessions),
                "n_retained": n_retained,
                "n_skipped": n_skipped,
                "n_cds_gff3": n_cds_gff3,
                "n_cds_genbank": n_cds_genbank,
                "matrix_size": matrix_size,
                "fold_backend": fold_backend.backend,
                "metadata_path": str(metadata_path),
                "species_segment_output": str(species_dir),
            }
        )

    coverage_fields = [
        "species",
        "segment",
        "manifest_path",
        "n_manifest_accessions",
        "n_retained",
        "n_skipped",
        "n_cds_gff3",
        "n_cds_genbank",
        "matrix_size",
        "fold_backend",
        "metadata_path",
        "species_segment_output",
    ]
    write_tsv(coverage_output, coverage_rows, coverage_fields)
    LOGGER.info("Wrote coverage summary: %s", coverage_output)


if __name__ == "__main__":
    main()
