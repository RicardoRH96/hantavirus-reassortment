#!/usr/bin/env python3
import argparse, csv, multiprocessing as mp
from Bio import SeqIO
from ViennaRNA import RNA

# ───────── helper utilities ─────────
def to_rna(seq):
    """DNA → RNA (T→U)."""
    return str(seq).upper().replace('T', 'U')

def fold_record(args):
    """Worker: clip, fold, return (id, clipped_seq, structure, MFE)."""
    rec, utr_len = args
    clipped = rec.seq[:utr_len]
    struct, mfe = RNA.fold(to_rna(clipped))
    return rec.id, clipped, struct, mfe

# ───────── main routine ─────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fasta")
    parser.add_argument("output_fasta")
    parser.add_argument("mfe_table_tsv")
    parser.add_argument("-n", "--utr_len", type=int, default=43,
                        help="5'-UTR length to keep (default 43)")
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Worker processes (default 4)")
    args = parser.parse_args()

    records = list(SeqIO.parse(args.input_fasta, "fasta"))
    print(f"[INFO] {len(records)} records → clip {args.utr_len} nt")

    with mp.Pool(args.threads) as pool:
        results = pool.map(
            fold_record,
            [(rec, args.utr_len) for rec in records]   # tuple per worker
        )

    kept = dropped = 0

    species_map = {}
    for rec in records:
        # header looks like  "NC_077669.1 |...|Orthohantavirus sinnombreense"
        species = rec.description.split("|")[-1].strip()
        species_map[rec.id] = species.replace(" ", "_") 
    with open(args.output_fasta, "w") as fasta_out, \
        open(args.mfe_table_tsv, "w", newline="") as tsv_out:
        w = csv.writer(tsv_out, delimiter="\t")
        w.writerow(["id", "MFE_kcal", "structure"])

        for acc, clipped, struct, mfe in results:
            w.writerow([acc, f"{mfe:.2f}", struct])

            if mfe < -3.0:
                new_id  = f"{acc}|{species_map[acc]}"
                out_rec = SeqIO.SeqRecord(clipped, id=new_id, description="")
                SeqIO.write(out_rec, fasta_out, "fasta")
                kept += 1
            else:
                dropped += 1

    print(f"[INFO] kept {kept} (MFE < -3); dropped {dropped} (MFE ≥ -3)")

# ───────── bootstrap guard ─────────
if __name__ == "__main__":
    mp.freeze_support()   # safe on every OS
    main()