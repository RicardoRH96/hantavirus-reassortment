#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def _count_pattern_in_file(path: Path, pattern: bytes, chunk_size: int = 8 * 1024 * 1024) -> int:
    """Count byte-pattern occurrences using chunked reads (memory-safe for very large files)."""
    count = 0
    overlap = max(0, len(pattern) - 1)
    tail = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data = tail + chunk
            count += data.count(pattern)
            tail = data[-overlap:] if overlap else b""
    return count


def _count_any_pattern_in_file(path: Path, patterns: list[bytes], chunk_size: int = 8 * 1024 * 1024) -> int:
    """Count occurrences across multiple byte-pattern variants."""
    return sum(_count_pattern_in_file(path, p, chunk_size=chunk_size) for p in patterns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count GBIF datapoints per species from per-species GeoJSON files."
    )
    parser.add_argument(
        "--inputs-dir",
        default="Inputs",
        help="Directory containing per-species GeoJSON files (default: Inputs).",
    )
    parser.add_argument(
        "--output-csv",
        default="Outputs/gbif_points_per_species.csv",
        help="Output CSV path (default: Outputs/gbif_points_per_species.csv).",
    )
    args = parser.parse_args()

    inputs_dir = Path(args.inputs_dir).resolve()
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Inputs directory not found: {inputs_dir}")

    geojson_files = sorted(inputs_dir.glob("*.geojson"))
    if not geojson_files:
        raise FileNotFoundError(f"No .geojson files found in: {inputs_dir}")

    rows = []
    for fp in geojson_files:
        species = fp.stem
        try:
            # Most files are compact single-line GeoJSON; chunked byte counting avoids OOM.
            n_records = _count_any_pattern_in_file(
                fp,
                [b'"type":"Feature"', b'"type": "Feature"'],
            )
            n_null_geom = _count_any_pattern_in_file(
                fp,
                [b'"geometry":null', b'"geometry": null'],
            )
            n_non_null_geom = max(0, n_records - n_null_geom)
            rows.append(
                {
                    "species": species,
                    "n_records": n_records,
                    "n_non_null_geometry": n_non_null_geom,
                    "source_file": str(fp),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "species": species,
                    "n_records": pd.NA,
                    "n_non_null_geometry": pd.NA,
                    "source_file": str(fp),
                    "error": str(exc),
                }
            )

    out_df = pd.DataFrame(rows).sort_values("species").reset_index(drop=True)
    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[OK] wrote: {out_path}")
    print(f"[INFO] species files scanned: {len(out_df)}")


if __name__ == "__main__":
    main()
