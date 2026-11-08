#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SENTINEL_ABS_DEFAULT = 1e30


def _latest_matching(cache_dir: Path, pattern: str) -> Path:
    candidates = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} under {cache_dir}")
    return candidates[0]


def _impute_median(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        median = out[col].median(skipna=True)
        out[col] = out[col].fillna(median)
    return out


def _univariate_auc(y: np.ndarray, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(x)):
        return float("nan")
    x = np.where(np.isfinite(x), x, np.nan)
    median = np.nanmedian(x)
    x = np.where(np.isnan(x), median, x)
    auc = float(roc_auc_score(y, x))
    return max(auc, 1.0 - auc)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize feature quality and suggest feature-selection candidates.")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing features_pres_*.parquet and features_abs_*.parquet.",
    )
    ap.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Absolute correlation threshold above which to flag feature pairs (default: 0.95).",
    )
    ap.add_argument(
        "--sentinel-abs",
        type=float,
        default=SENTINEL_ABS_DEFAULT,
        help="Treat |value| >= sentinel_abs as a sentinel/extreme (default: 1e30).",
    )
    ap.add_argument(
        "--top-pairs",
        type=int,
        default=30,
        help="Max number of correlated pairs to print (default: 30).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to write per-feature summary CSV.",
    )
    args = ap.parse_args()

    cache_dir: Path = args.cache_dir
    pres_path = _latest_matching(cache_dir, "features_pres_*.parquet")
    abs_path = _latest_matching(cache_dir, "features_abs_*.parquet")

    X_pres = pd.read_parquet(pres_path)
    X_abs = pd.read_parquet(abs_path)

    if list(X_pres.columns) != list(X_abs.columns):
        raise ValueError("Presence/absence feature columns differ; cannot compare.")

    features = list(X_pres.columns)
    y = np.concatenate([np.ones(len(X_pres), dtype=int), np.zeros(len(X_abs), dtype=int)])
    X_all = pd.concat([X_pres, X_abs], axis=0, ignore_index=True)

    sent = float(args.sentinel_abs)
    num = X_all.apply(pd.to_numeric, errors="coerce")
    extreme_mask = (num.abs() >= sent) & np.isfinite(num.to_numpy())
    extreme_counts = pd.Series(extreme_mask.sum(axis=0), index=features)

    summary_rows: List[Dict[str, object]] = []
    for feat in features:
        col = num[feat]
        is_extreme = col.abs() >= sent
        safe = col.mask(is_extreme, np.nan)
        summary_rows.append(
            {
                "feature": feat,
                "n_total": int(len(col)),
                "n_nan": int(col.isna().sum()),
                "n_extreme": int(is_extreme.sum()),
                "nan_frac": float(col.isna().mean()),
                "extreme_frac": float(is_extreme.mean()),
                "min_nonextreme": float(np.nanmin(safe.to_numpy())) if safe.notna().any() else np.nan,
                "max_nonextreme": float(np.nanmax(safe.to_numpy())) if safe.notna().any() else np.nan,
                "n_unique_nonextreme": int(safe.nunique(dropna=True)),
                "auc_effect": _univariate_auc(y, col.to_numpy()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("auc_effect", ascending=False)

    print(f"[INFO] cache_dir = {cache_dir}")
    print(f"[INFO] pres = {pres_path.name} shape={X_pres.shape}")
    print(f"[INFO] abs  = {abs_path.name} shape={X_abs.shape}")
    print()
    print("[INFO] Top features by univariate AUC (effect size, max(auc,1-auc)):")
    print(summary.head(15).to_string(index=False))

    corr_threshold = float(args.corr_threshold)
    X_imp = _impute_median(num)
    corr = X_imp.corr().abs()
    np.fill_diagonal(corr.values, 0.0)

    pairs: List[Tuple[float, str, str]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            v = float(corr.iat[i, j])
            if np.isfinite(v) and v >= corr_threshold:
                pairs.append((v, a, b))

    pairs.sort(reverse=True)
    print()
    print(f"[INFO] Correlated pairs |corr| >= {corr_threshold:.2f}: {len(pairs)}")
    for v, a, b in pairs[: int(args.top_pairs)]:
        print(f"  {v:0.3f}  {a}  ~  {b}")

    auc_map = dict(zip(summary["feature"], summary["auc_effect"]))
    keep = set(features)
    dropped: List[Tuple[str, str, float]] = []
    for v, a, b in pairs:
        if a not in keep or b not in keep:
            continue
        a_score = float(auc_map.get(a, 0.5))
        b_score = float(auc_map.get(b, 0.5))
        if a_score >= b_score:
            keep.remove(b)
            dropped.append((b, a, v))
        else:
            keep.remove(a)
            dropped.append((a, b, v))

    if dropped:
        print()
        print("[INFO] Greedy drop suggestions (drop -> keep, corr):")
        for drop_feat, keep_feat, v in dropped[:50]:
            print(f"  drop {drop_feat}  (keep {keep_feat}, corr={v:0.3f})")
        print()
        print(f"[INFO] Suggested keep set size: {len(keep)}/{len(features)}")
        print("  " + ", ".join(sorted(keep)))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_csv, index=False)
        print(f"\n[OK] Wrote summary CSV: {args.out_csv}")


if __name__ == "__main__":
    main()

