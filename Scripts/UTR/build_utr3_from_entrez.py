#!/usr/bin/env python3
"""
Build 3' UTR datasets from Entrez annotations and compute per-segment
all-to-all structure/sequence distance matrices.

CDS coordinate priority:
1) GFF3 CDS features
2) GenBank CDS features

3' UTR definition:
- '+' strand: sequence after CDS end
- '-' strand: sequence before CDS start, reverse-complemented to transcript orientation
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Bio import Entrez, SeqIO
from Bio.Seq import Seq

from build_utr5_from_entrez import (
    CdsInfo,
    FoldBackend,
    SEGMENTS,
    build_distance_matrix,
    choose_primary_cds,
    configure_logging,
    ensure_parent,
    fetch_fasta_cached,
    fetch_text_cached,
    iter_manifests,
    normalized_edit_distance,
    parse_genbank_cds,
    parse_gff3_cds,
    read_manifest,
    write_fasta,
    write_matrix,
    write_tsv,
    bp_distance,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build transcript-relative 3' UTR matrices from Entrez.")
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
        default=repo_root / "model_test" / "runs" / "utr3",
        help="Root output folder for per-species/per-segment results.",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=repo_root / "model_test" / "reports" / "utr3_coverage_summary.tsv",
        help="Coverage report output TSV.",
    )
    parser.add_argument(
        "--min-utr-length",
        type=int,
        default=15,
        help="Minimum 3' UTR length to retain.",
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


def extract_utr3(seq: Seq, cds: CdsInfo) -> Tuple[Optional[Seq], Optional[int], Optional[int], str]:
    seq_len = len(seq)
    if cds.strand == "+":
        utr_start = cds.end + 1
        utr_end = seq_len
        if utr_end < utr_start:
            return None, None, None, "no_3utr_after_cds"
        frag = seq[utr_start - 1 : utr_end]
    else:
        utr_start = 1
        utr_end = cds.start - 1
        if utr_end < utr_start:
            return None, None, None, "no_3utr_before_cds"
        frag = seq[utr_start - 1 : utr_end].reverse_complement()
    return frag, utr_start, utr_end, ""


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
    coverage_rows: List[Dict[str, object]] = []

    for species, segment, manifest_path in manifests:
        if segment not in SEGMENTS:
            continue

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
                "utr3_start": "",
                "utr3_end": "",
                "utr3_length": "",
                "status": "skipped",
                "skip_reason": "",
            }
            try:
                fasta_record = fetch_fasta_cached(
                    accession,
                    fasta_cache / f"{accession}.fasta",
                    args.throttle_seconds,
                )
            except Exception as exc:  # pragma: no cover - network failure
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
            except Exception as exc:  # pragma: no cover - network failure
                fallback_reason = f"gff3_fetch_failed:{exc}"
                cds_info = None

            if cds_info is None:
                try:
                    fetch_text_cached(
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
                except Exception as exc:  # pragma: no cover - network failure
                    if not fallback_reason:
                        fallback_reason = f"genbank_fetch_failed:{exc}"
                    cds_info = None

            if cds_info is None:
                row["skip_reason"] = fallback_reason or "cds_not_found"
                metadata_rows.append(row)
                n_skipped += 1
                continue

            utr_seq, utr_start, utr_end, utr_reason = extract_utr3(fasta_record.seq, cds_info)
            if utr_seq is None or utr_start is None or utr_end is None:
                row.update(
                    {
                        "cds_source": cds_info.source,
                        "cds_start": cds_info.start,
                        "cds_end": cds_info.end,
                        "cds_strand": cds_info.strand,
                        "skip_reason": utr_reason or "utr3_extraction_failed",
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
                        "utr3_start": utr_start,
                        "utr3_end": utr_end,
                        "utr3_length": len(utr_text),
                        "skip_reason": "utr3_too_short",
                    }
                )
                metadata_rows.append(row)
                n_skipped += 1
                continue

            retained_records.append((accession, utr_text))
            row.update(
                {
                    "cds_source": cds_info.source,
                    "cds_start": cds_info.start,
                    "cds_end": cds_info.end,
                    "cds_strand": cds_info.strand,
                    "utr3_start": utr_start,
                    "utr3_end": utr_end,
                    "utr3_length": len(utr_text),
                    "status": "retained",
                    "skip_reason": "",
                }
            )
            metadata_rows.append(row)

            if cds_info.source == "gff3":
                n_cds_gff3 += 1
            else:
                n_cds_genbank += 1

        metadata_path = species_dir / "utr3_metadata.tsv"
        write_tsv(
            metadata_path,
            metadata_rows,
            [
                "species",
                "segment",
                "accession",
                "cds_source",
                "cds_start",
                "cds_end",
                "cds_strand",
                "utr3_start",
                "utr3_end",
                "utr3_length",
                "status",
                "skip_reason",
            ],
        )

        structure_table_rows: List[Dict[str, object]] = []
        n_retained = len(retained_records)
        matrix_size = n_retained

        if n_retained > 0:
            write_fasta(species_dir / "utr3.fasta", retained_records)

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
                species_dir / "utr3_structures.tsv",
                structure_table_rows,
                ["id", "species", "segment", "length", "mfe_kcal", "structure"],
            )

            seq_items = [(acc, seq) for acc, seq, _, _ in folded]
            struct_items = [(acc, struct) for acc, _, _, struct in folded]
            names = [acc for acc, _ in seq_items]

            seq_matrix = build_distance_matrix(seq_items, normalized_edit_distance)
            struct_matrix = build_distance_matrix(struct_items, bp_distance)

            write_matrix(species_dir / "utr3_sequence_distance.tsv", names, seq_matrix)
            write_matrix(species_dir / "utr3_structure_distance.tsv", names, struct_matrix)

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

    write_tsv(
        coverage_output,
        coverage_rows,
        [
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
        ],
    )


if __name__ == "__main__":
    main()
