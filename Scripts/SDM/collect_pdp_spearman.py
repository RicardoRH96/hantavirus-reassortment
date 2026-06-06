#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
from scipy.stats import ConstantInputWarning


def _feature_map_from_importance(species_dir: Path) -> Dict[str, str]:
    """Map PDP csv basename -> canonical feature name using feature_importance file."""
    out: Dict[str, str] = {}
    fi_files = sorted(species_dir.glob("feature_importance_*.csv"))
    if not fi_files:
        return out
    try:
        fi = pd.read_csv(fi_files[0])
    except Exception:
        return out
    if "feature" not in fi.columns or "partial_dependence_csv" not in fi.columns:
        return out
    for _, row in fi.iterrows():
        rel = str(row.get("partial_dependence_csv", "") or "").strip()
        feat = str(row.get("feature", "") or "").strip()
        if not rel or not feat:
            continue
        out[Path(rel).name] = feat
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Spearman rho from existing PDP CSVs (feature_value vs partial_dependence)."
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="SDM output root containing ensemble-images/<species>/analysis/partial_dependence_*.csv",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Long-format output CSV path (default: <out-dir>/pdp_spearman_all_species_long.csv).",
    )
    parser.add_argument(
        "--output-wide-csv",
        default=None,
        help="Wide-format output CSV path (default: <out-dir>/pdp_spearman_all_species_wide.csv).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    species_dirs = sorted((out_dir / "ensemble-images").glob("*"))
    if not species_dirs:
        raise FileNotFoundError(f"No species folders found under: {out_dir}/ensemble-images")

    rows = []
    for species_dir in species_dirs:
        if not species_dir.is_dir():
            continue
        species = species_dir.name
        analysis_dir = species_dir / "analysis"
        if not analysis_dir.exists():
            continue

        fmap = _feature_map_from_importance(species_dir)
        pdp_files = sorted(analysis_dir.glob("partial_dependence_*.csv"))
        for pdp_csv in pdp_files:
            try:
                df = pd.read_csv(pdp_csv)
            except Exception:
                continue
            if "feature_value" not in df.columns or "partial_dependence" not in df.columns:
                continue

            x = pd.to_numeric(df["feature_value"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df["partial_dependence"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            n_valid = int(valid.sum())
            rho = np.nan
            p = np.nan
            status = "ok"
            if n_valid >= 3:
                xv = x[valid]
                yv = y[valid]
                if np.nanstd(xv) == 0:
                    status = "constant_feature_grid"
                elif np.nanstd(yv) == 0:
                    status = "constant_pdp"
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConstantInputWarning)
                            rho_val, p_val = spearmanr(xv, yv, nan_policy="omit")
                        rho = float(rho_val) if np.isfinite(rho_val) else np.nan
                        p = float(p_val) if np.isfinite(p_val) else np.nan
                    except Exception:
                        status = "spearman_failed"
            else:
                status = "too_few_points"

            inferred_feat = pdp_csv.stem.replace("partial_dependence_", "", 1)
            feature = fmap.get(pdp_csv.name, inferred_feat)
            rows.append(
                {
                    "species": species,
                    "feature": feature,
                    "pdp_csv": str(pdp_csv),
                    "spearman_rho": rho,
                    "p_value": p,
                    "n_points": n_valid,
                    "status": status,
                }
            )

    if not rows:
        raise FileNotFoundError("No valid PDP CSV files were found to compute Spearman correlations.")

    long_df = pd.DataFrame(rows).sort_values(["species", "feature"]).reset_index(drop=True)
    out_long = Path(args.output_csv) if args.output_csv else out_dir / "pdp_spearman_all_species_long.csv"
    out_long.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_long, index=False)

    wide = long_df.pivot_table(index="species", columns="feature", values="spearman_rho", aggfunc="first")
    out_wide = Path(args.output_wide_csv) if args.output_wide_csv else out_dir / "pdp_spearman_all_species_wide.csv"
    out_wide.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_wide)

    print(f"[OK] wrote: {out_long}")
    print(f"[OK] wrote: {out_wide}")
    print(f"[INFO] species={long_df['species'].nunique()} rows={len(long_df)} features={long_df['feature'].nunique()}")


if __name__ == "__main__":
    main()
