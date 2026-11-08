#!/usr/bin/env python3
"""
Inter-segment linkage disequilibrium (LD) calculator across polymorphic sites.

Computes classical biallelic LD statistics (D, D', ChiSq, ChiSq_df) along with
multi-allelic association measures:
- Cramer's V: sqrt(ChiSq / (n * (min(k1, k2) - 1)))
- Mutual information (MI) and normalized MI: MI / sqrt(H1 * H2)
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import AlignIO


def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def get_column(alignment: Sequence[str], idx: int) -> List[str]:
    return [row[idx] for row in alignment]


INVALID_AA = {"-", "?", "X"}
INVALID_NT_NON_BASE = set("RYSWKMBDHVN-")
VALID_BASES = {"A", "C", "G", "T"}
INVALID_3DI = {"-", "?"}


def cramers_v(chi2: float, n: int, k1: int, k2: int) -> float:
    """Compute Cramer's V: sqrt(chi2 / (n * (min(k1, k2) - 1)))."""
    if n == 0:
        return np.nan
    denom = n * (min(k1, k2) - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(max(chi2, 0.0) / denom))


def mutual_information(counts: List[int], k1: int, k2: int) -> Tuple[float, float]:
    """Compute mutual information (MI) and entropy-normalized MI (NMI) in [0, 1]."""
    total = float(sum(counts))
    if total <= 0:
        return (np.nan, np.nan)
    p_joint = np.array(counts, dtype=float) / total
    p1 = p_joint.reshape((k1, k2)).sum(axis=1)
    p2 = p_joint.reshape((k1, k2)).sum(axis=0)
    mi = 0.0
    for i in range(k1):
        for j in range(k2):
            pxy = p_joint[i * k2 + j]
            if pxy > 0 and p1[i] > 0 and p2[j] > 0:
                mi += pxy * np.log(pxy / (p1[i] * p2[j]))
    h1 = -np.sum([p * np.log(p) for p in p1 if p > 0])
    h2 = -np.sum([p * np.log(p) for p in p2 if p > 0])
    if h1 > 0 and h2 > 0:
        nmi = mi / np.sqrt(h1 * h2)
    else:
        nmi = np.nan
    return (mi, nmi)


def compute_ld_segment_pair(
    segA: str,
    segB: str,
    mode: str,
    input_dir: Path,
    pattern_aa: str,
    pattern_nt: str,
    pattern_3di: str,
    excluded_taxa: Iterable[str] | None = None,
    enforce_taxon_ids: bool = True,
) -> pd.DataFrame:
    """Compute pairwise site LD statistics between two aligned genomic segments."""
    excluded_taxa = set(excluded_taxa or [])

    if mode == 'aa':
        fileA = input_dir / pattern_aa.format(segment=segA)
        fileB = input_dir / pattern_aa.format(segment=segB)
    elif mode == 'nt':
        fileA = input_dir / pattern_nt.format(segment=segA)
        fileB = input_dir / pattern_nt.format(segment=segB)
    elif mode == '3di':
        fileA = input_dir / pattern_3di.format(segment=segA)
        fileB = input_dir / pattern_3di.format(segment=segB)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    alnA = AlignIO.read(fileA, "fasta")
    alnB = AlignIO.read(fileB, "fasta")

    recordsA = [(rec.id, str(rec.seq)) for rec in alnA]
    recordsB = [(rec.id, str(rec.seq)) for rec in alnB]

    strains: List[str] = []
    seqA: List[str] = []
    seqB: List[str] = []

    if enforce_taxon_ids:
        seq_mapB = {tax_id: seq for tax_id, seq in recordsB if tax_id not in excluded_taxa}
        for tax_id, seq in recordsA:
            if tax_id in excluded_taxa or tax_id not in seq_mapB:
                continue
            strains.append(tax_id)
            seqA.append(seq)
            seqB.append(seq_mapB[tax_id])
    else:
        n_pairs = min(len(recordsA), len(recordsB))
        if len(recordsA) != len(recordsB):
            print(
                f"Warning: segments {segA} and {segB} have {len(recordsA)} and {len(recordsB)} taxa respectively; "
                f"using the first {n_pairs} paired sequences."
            )
        for idx in range(n_pairs):
            taxA, seqA_str = recordsA[idx]
            taxB, seqB_str = recordsB[idx]
            if taxA in excluded_taxa or taxB in excluded_taxa:
                continue
            strains.append(f"{taxA}|{taxB}")
            seqA.append(seqA_str)
            seqB.append(seqB_str)

    n = len(strains)
    if n == 0:
        print("No taxa remaining after exclusions.")
        return pd.DataFrame()

    L1 = len(seqA[0])
    L2 = len(seqB[0])

    polyA: List[int] = []
    polyB: List[int] = []

    for i in range(L1):
        col = get_column(seqA, i)
        u = unique(col)
        if mode == 'aa':
            if len([x for x in u if x not in INVALID_AA]) <= 1:
                continue
        elif mode == 'nt':
            if len([x for x in u if x not in INVALID_NT_NON_BASE]) <= 1:
                continue
        elif mode == '3di':
            if len([x for x in u if x not in INVALID_3DI]) <= 1:
                continue
        if '-' in u:
            continue
        polyA.append(i)

    for j in range(L2):
        col = get_column(seqB, j)
        u = unique(col)
        if mode == 'aa':
            if len([x for x in u if x not in INVALID_AA]) <= 1:
                continue
        else:
            if len([x for x in u if x not in INVALID_NT_NON_BASE]) <= 1:
                continue
        if '-' in u:
            continue
        polyB.append(j)

    rows: List[Dict[str, float]] = []
    seen_cache: Dict[Tuple[int, ...], Tuple[float, float, float, float, float, float, float, float]] = {}

    if segA == segB:
        site_pairs = [(polyA[i], polyB[j]) for i in range(len(polyA)) for j in range(i + 1, len(polyB))]
    else:
        site_pairs = list(itertools.product(polyA, polyB))

    for i, j in site_pairs:
        haps: List[Tuple[str, str]] = []
        for a, b in zip(seqA, seqB):
            x, y = a[i], b[j]
            if mode == 'aa':
                if x in INVALID_AA or y in INVALID_AA:
                    continue
            else:
                if x not in VALID_BASES or y not in VALID_BASES:
                    continue
            haps.append((x, y))
        if len(haps) == 0:
            continue

        poly1 = unique([h[0] for h in haps])
        poly2 = unique([h[1] for h in haps])
        if len(poly1) <= 1 or len(poly2) <= 1:
            continue

        all_h = list(itertools.product(poly1, poly2))
        counts = Counter(haps)
        hapCount = [counts[h] for h in all_h]

        key = tuple(hapCount)
        if key in seen_cache:
            D, Dprime, chi, chisqdf, maf1, maf2, v, mi = seen_cache[key]
        else:
            total = sum(hapCount)
            p1 = [sum(counts.get((a, b), 0) for b in poly2) / total for a in poly1]
            p2 = [sum(counts.get((a, b), 0) for a in poly1) / total for b in poly2]
            p = [c / total for c in hapCount]
            D = np.nan
            Dprime = np.nan
            if len(poly1) == 2 and len(poly2) == 2:
                p11 = p[0]
                p1dot = p1[0]
                pdot1 = p2[0]
                D = p11 - p1dot * pdot1
                if D >= 0:
                    Dmax = min(p1dot * (1 - pdot1), (1 - p1dot) * pdot1)
                else:
                    Dmax = min((1 - p1dot) * (1 - pdot1), p1dot * pdot1)
                if Dmax > 0:
                    Dprime = abs(D / Dmax)
            chi = 0.0
            k1 = len(poly1)
            k2 = len(poly2)
            for a_idx in range(k1):
                for b_idx in range(k2):
                    obs = hapCount[a_idx * k2 + b_idx]
                    exp = p1[a_idx] * p2[b_idx] * total
                    if exp > 0:
                        chi += (obs - exp) ** 2 / exp
            chisqdf = chi / (total * (k1 - 1) * (k2 - 1))
            maf1 = min(p1)
            maf2 = min(p2)
            v = cramers_v(chi, total, k1, k2)
            mi, nmi = mutual_information(hapCount, k1, k2)
            seen_cache[key] = (D, Dprime, chi, chisqdf, maf1, maf2, v, mi)
        rows.append({
            "segmentA": segA,
            "segmentB": segB,
            "site1": i + 1,
            "site2": j + 1,
            "n_strains": n,
            "D": D,
            "Dprime": Dprime,
            "ChiSq": chi,
            "ChiSq_df": chisqdf,
            "MAF1": maf1,
            "MAF2": maf2,
            "CramerV": v,
            "MI": mi,
            "NMI": mutual_information(hapCount, len(poly1), len(poly2))[1],
        })

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute LD statistics including Cramer's V and MI.")
    ap.add_argument("--mode", choices=["aa", "nt", "3di"], default="aa")
    ap.add_argument("--segments", nargs="+", required=True, help="List of segment names.")
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory containing alignment files.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Directory to write results.")
    ap.add_argument("--pattern-aa", default="{segment}_aa.fa", help="Pattern for amino acid alignment files.")
    ap.add_argument("--pattern-nt", default="{segment}_nt.fa", help="Pattern for nucleotide alignment files.")
    ap.add_argument("--pattern-3di", default="{segment}_3di.fa", help="Pattern for 3D interleaved alignment files.")
    ap.add_argument("--exclude-taxa", type=Path, default=None, help="Optional file listing taxa to exclude.")
    ap.add_argument("--ignore-taxon-names", action="store_true", help="Do not enforce identical taxon IDs across segments.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    excluded = set()
    if args.exclude_taxa:
        with args.exclude_taxa.open() as fh:
            excluded = {line.strip() for line in fh if line.strip()}
    all_dfs: List[pd.DataFrame] = []
    for i, segA in enumerate(args.segments):
        for segB in args.segments[i:]:
            df = compute_ld_segment_pair(
                segA, segB,
                args.mode,
                args.input_dir,
                args.pattern_aa,
                args.pattern_nt,
                args.pattern_3di,
                excluded,
                enforce_taxon_ids=not args.ignore_taxon_names,
            )
            if not df.empty:
                out = args.output_dir / f"{segA}_{segB}_LD.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False)
                print(f"Wrote {out}")
            all_dfs.append(df)
    if all_dfs:
        big = pd.concat(all_dfs, ignore_index=True)
        big.to_csv(args.output_dir / "ALL_SEGMENT_PAIRS_LD.csv", index=False)


if __name__ == "__main__":
    main()