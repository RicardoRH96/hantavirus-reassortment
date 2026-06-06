#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-species Spearman correlation CSV files.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="SDM output root (contains ensemble-images/<species>/spearman_correlations_*.csv).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path for long-format output CSV (default: <out-dir>/spearman_all_species_long.csv).",
    )
    parser.add_argument(
        "--output-wide-csv",
        default=None,
        help="Path for wide-format output CSV (species x feature rho; default: <out-dir>/spearman_all_species_wide.csv).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    files = sorted(out_dir.glob("ensemble-images/*/spearman_correlations_*.csv"))
    if not files:
        raise FileNotFoundError(f"No Spearman files found under: {out_dir}/ensemble-images/*/")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        if "species" not in df.columns:
            df["species"] = fp.parent.name
        dfs.append(df)
    long_df = pd.concat(dfs, ignore_index=True)
    long_df = long_df.sort_values(["species", "feature"]).reset_index(drop=True)

    out_long = Path(args.output_csv) if args.output_csv else out_dir / "spearman_all_species_long.csv"
    out_long.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_long, index=False)

    wide = long_df.pivot_table(index="species", columns="feature", values="spearman_rho", aggfunc="first")
    out_wide = Path(args.output_wide_csv) if args.output_wide_csv else out_dir / "spearman_all_species_wide.csv"
    out_wide.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_wide)

    print(f"[OK] wrote: {out_long}")
    print(f"[OK] wrote: {out_wide}")
    print(f"[INFO] species={long_df['species'].nunique()} rows={len(long_df)} features={long_df['feature'].nunique()}")


if __name__ == "__main__":
    main()
