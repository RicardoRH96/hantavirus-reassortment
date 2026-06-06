#!/usr/bin/env python3
"""
Fetch GFF3 annotations for a set of accessions, extract 5' and 3' UTRs,
filter out sequences without annotated UTRs, and compute pairwise distance
matrices for both secondary structure (ViennaRNA base-pair distance) and
sequence dissimilarity (normalized edit distance).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ViennaRNA import RNA


LOGGER = logging.getLogger("utr_fetch")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GFF3, extract UTRs, and compute distance matrices.")
    parser.add_argument(
        "--accessions",
        nargs="+",
        help="List of nucleotide accessions to download.",
    )
    parser.add_argument(
        "--accession-file",
        type=Path,
        help="Optional file containing one accession per line.",
    )
    parser.add_argument(
        "--email",
        required=True,
        help="Email address for NCBI Entrez.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("NCBI_API_KEY"),
        help="NCBI API key to increase rate limits (defaults to $NCBI_API_KEY).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data") / "UTR_downloads",
        help="Output directory for downloaded files and matrices.",
    )
    parser.add_argument(
        "--min-utr-length",
        type=int,
        default=15,
        help="Minimum length for each UTR; shorter ones are discarded (default: 15 nt).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    return args


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def load_accessions(args: argparse.Namespace) -> List[str]:
    accs: List[str] = []
    if args.accession_file and args.accession_file.exists():
        accs.extend([line.strip() for line in args.accession_file.read_text().splitlines() if line.strip()])
    if args.accessions:
        accs.extend(args.accessions)
    deduped = []
    seen = set()
    for acc in accs:
        if acc not in seen:
            deduped.append(acc)
            seen.add(acc)
    return deduped


def fetch_text(db: str, accession: str, rettype: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text()
    LOGGER.info("Downloading %s (%s)", accession, rettype)
    handle = Entrez.efetch(db=db, id=accession, rettype=rettype, retmode="text")
    text = handle.read()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def fetch_fasta(accession: str, cache_path: Path) -> SeqRecord:
    if cache_path.exists():
        return SeqIO.read(cache_path, "fasta")
    LOGGER.info("Downloading %s (fasta)", accession)
    handle = Entrez.efetch(db="nuccore", id=accession, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(record, cache_path, "fasta")
    return record


def normalize_feature_type(raw: str) -> str:
    cleaned = raw.lower().replace("'", "").replace("-", "_")
    return cleaned


def parse_gff_utrs(text: str) -> Dict[str, List[Tuple[int, int, str]]]:
    utrs: Dict[str, List[Tuple[int, int, str]]] = {"five": [], "three": []}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        ftype = normalize_feature_type(parts[2])
        start, end = int(parts[3]), int(parts[4])
        strand = parts[6]
        if "five_prime_utr" in ftype or ftype == "5utr":
            utrs["five"].append((start, end, strand))
        elif "three_prime_utr" in ftype or ftype == "3utr":
            utrs["three"].append((start, end, strand))
    return utrs


def select_longest(regions: List[Tuple[int, int, str]]) -> Optional[Tuple[int, int, str]]:
    if not regions:
        return None
    return sorted(regions, key=lambda item: (item[1] - item[0]), reverse=True)[0]


def slice_utr(seq: Seq, region: Tuple[int, int, str]) -> Seq:
    start, end, strand = region
    frag = seq[start - 1 : end]
    if strand == "-":
        frag = frag.reverse_complement()
    return frag


def extract_utrs(
    record: SeqRecord,
    utrs: Dict[str, List[Tuple[int, int, str]]],
    min_len: int,
) -> Optional[Tuple[Seq, Seq]]:
    five = select_longest(utrs.get("five", []))
    three = select_longest(utrs.get("three", []))
    if not five or not three:
        return None
    five_seq = slice_utr(record.seq, five)
    three_seq = slice_utr(record.seq, three)
    if len(five_seq) < min_len or len(three_seq) < min_len:
        return None
    return five_seq, three_seq


def fold_sequences(records: Sequence[SeqRecord]) -> List[Dict[str, object]]:
    folded: List[Dict[str, object]] = []
    for rec in records:
        struct, mfe = RNA.fold(str(rec.seq).upper().replace("T", "U"))
        folded.append(
            {
                "id": rec.id,
                "length": len(rec.seq),
                "mfe_kcal": float(mfe),
                "structure": struct,
            }
        )
    return folded


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


def build_distance_matrix(items: Sequence[Tuple[str, str]], dist_fn) -> List[List[float]]:
    n = len(items)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0.0
        for j in range(i + 1, n):
            val = float(dist_fn(items[i][1], items[j][1]))
            matrix[i][j] = matrix[j][i] = val
    return matrix


def bp_distance_matrix(structs: Sequence[Tuple[str, str]]) -> List[List[float]]:
    n = len(structs)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0.0
        for j in range(i + 1, n):
            max_len = max(len(structs[i][1]), len(structs[j][1]))
            s1 = structs[i][1].ljust(max_len, ".")
            s2 = structs[j][1].ljust(max_len, ".")
            matrix[i][j] = matrix[j][i] = float(RNA.bp_distance(s1, s2))
    return matrix


def write_matrix(path: Path, names: Sequence[str], matrix: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *[f"{val:.4f}" for val in row]])


def write_structure_table(path: Path, folded: Sequence[Dict[str, object]], utr_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "utr", "length", "mfe_kcal", "structure"])
        for entry in folded:
            writer.writerow([entry["id"], utr_label, entry["length"], f"{entry['mfe_kcal']:.2f}", entry["structure"]])


def run(args: argparse.Namespace) -> None:
    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key
    accessions = load_accessions(args)
    if not accessions:
        raise SystemExit("No accessions provided.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    utr5_records: List[SeqRecord] = []
    utr3_records: List[SeqRecord] = []

    for acc in accessions:
        gff_cache = args.output_dir / "gff3" / f"{acc}.gff3"
        fasta_cache = args.output_dir / "fasta" / f"{acc}.fasta"
        try:
            gff_text = fetch_text("nuccore", acc, "gff3", gff_cache)
            record = fetch_fasta(acc, fasta_cache)
        except Exception as exc:
            LOGGER.warning("Failed to download %s (%s); skipping.", acc, exc)
            continue
        utr_regions = parse_gff_utrs(gff_text)
        utr_pair = extract_utrs(record, utr_regions, args.min_utr_length)
        if not utr_pair:
            LOGGER.info("No usable UTR annotations for %s; discarded.", acc)
            continue
        five_seq, three_seq = utr_pair
        base_id = record.id.split()[0]
        utr5_records.append(SeqRecord(five_seq, id=f"{base_id}|5prime", description=""))
        utr3_records.append(SeqRecord(three_seq, id=f"{base_id}|3prime", description=""))

    if not utr5_records or not utr3_records:
        LOGGER.warning("No sequences with both UTRs available; nothing to do.")
        return

    utr_dir = args.output_dir / "utr_fastas"
    utr_dir.mkdir(parents=True, exist_ok=True)
    SeqIO.write(utr5_records, utr_dir / "utr5.fasta", "fasta")
    SeqIO.write(utr3_records, utr_dir / "utr3.fasta", "fasta")

    folded5 = fold_sequences(utr5_records)
    folded3 = fold_sequences(utr3_records)
    write_structure_table(args.output_dir / "utr5_structures.tsv", folded5, "5p")
    write_structure_table(args.output_dir / "utr3_structures.tsv", folded3, "3p")

    struct_names5 = [entry["id"] for entry in folded5]
    struct_names3 = [entry["id"] for entry in folded3]
    struct_pairs5 = [(entry["id"], str(entry["structure"])) for entry in folded5]
    struct_pairs3 = [(entry["id"], str(entry["structure"])) for entry in folded3]
    bp_mat_5 = bp_distance_matrix(struct_pairs5)
    bp_mat_3 = bp_distance_matrix(struct_pairs3)
    write_matrix(args.output_dir / "utr5_structure_distance.tsv", struct_names5, bp_mat_5)
    write_matrix(args.output_dir / "utr3_structure_distance.tsv", struct_names3, bp_mat_3)

    seq_pairs5 = [(rec.id, str(rec.seq)) for rec in utr5_records]
    seq_pairs3 = [(rec.id, str(rec.seq)) for rec in utr3_records]
    seq_mat_5 = build_distance_matrix(seq_pairs5, normalized_edit_distance)
    seq_mat_3 = build_distance_matrix(seq_pairs3, normalized_edit_distance)
    write_matrix(args.output_dir / "utr5_sequence_distance.tsv", [p[0] for p in seq_pairs5], seq_mat_5)
    write_matrix(args.output_dir / "utr3_sequence_distance.tsv", [p[0] for p in seq_pairs3], seq_mat_3)

    LOGGER.info(
        "Finished: %d accessions retained with UTRs (5': %d seqs, 3': %d seqs).",
        len(seq_pairs5),
        len(seq_pairs5),
        len(seq_pairs3),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)
    run(args)


if __name__ == "__main__":
    main()
