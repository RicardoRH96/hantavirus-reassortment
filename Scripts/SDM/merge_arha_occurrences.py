#!/usr/bin/env python3
"""
Merge ArHa occurrences with existing GBIF-derived GeoJSON inputs.

For each species GeoJSON under the GBIF input directory, the script appends
presence points from the ArHa dataset (if available), annotating records with
their source and useful provenance fields. The combined GeoJSON files are
written to an output directory, leaving the originals untouched.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

DEFAULT_GBIF_DIR = Path(__file__).resolve().parents[1] / "Inputs"
DEFAULT_ARHA_CSV = Path(__file__).resolve().parents[1] / "ArHa_Test" / "host_data.cleaned.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Inputs_with_ArHa"


def canonical_species_key(name: str) -> Tuple[str, str]:
    """
    Produce a pair (slug, label) from a raw species name.

    slug  -> 'Genus_species' (for filenames)
    label -> 'Genus species' (for readability)
    """
    cleaned = name.strip()
    label = cleaned
    slug = cleaned.replace(" ", "_")
    return slug, label


def load_arha_occurrences(
    csv_path: Path,
    min_hosts: float,
    require_valid_coord: bool,
) -> Dict[str, gpd.GeoDataFrame]:
    df = pd.read_csv(csv_path)

    if require_valid_coord and "coord_status" in df.columns:
        df = df[df["coord_status"].str.lower() == "valid"]

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

    if "number_of_hosts" in df.columns:
        df = df[df["number_of_hosts"].fillna(0) >= min_hosts]

    species_col = "species_clean" if "species_clean" in df.columns else "host_species"
    df = df.dropna(subset=[species_col])

    geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")

    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, raw_name in gdf[species_col].items():
        slug, label = canonical_species_key(raw_name)
        gdf.at[idx, "species_label"] = label
        gdf.at[idx, "species_slug"] = slug
        buckets[slug].append(idx)

    grouped: Dict[str, gpd.GeoDataFrame] = {}
    for slug, indices in buckets.items():
        grouped[slug] = gdf.loc[indices].copy()

    return grouped


def read_geojson(path: Path) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(path)
    except ValueError as exc:
        raise RuntimeError(f"Failed to read GeoJSON: {path}") from exc

    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    gdf = gdf.copy()
    gdf["source"] = gdf.get("source", "GBIF")
    gdf["source"] = gdf["source"].fillna("GBIF")
    return gdf


def harmonize_columns(gdf: gpd.GeoDataFrame, columns: Iterable[str]) -> gpd.GeoDataFrame:
    for col in columns:
        if col not in gdf.columns:
            gdf[col] = pd.NA
    return gdf


def merge_species_records(
    gbif_gdf: gpd.GeoDataFrame,
    arha_gdf: gpd.GeoDataFrame | None,
    drop_duplicate_coords: bool,
) -> gpd.GeoDataFrame:
    records: List[gpd.GeoDataFrame] = []
    if not gbif_gdf.empty:
        gbif = gbif_gdf.copy()
        gbif["source"] = "GBIF"
        records.append(gbif)

    if arha_gdf is not None and not arha_gdf.empty:
        subset_cols = {
            "host_record_id",
            "study_id",
            "species_label",
            "species_slug",
            "number_of_hosts",
            "trap_nights_clean",
            "coord_status",
        } & set(arha_gdf.columns)
        keep_cols = list(subset_cols) + ["geometry"]
        arha_subset = arha_gdf[keep_cols].copy()
        arha_subset["source"] = "ArHa"
        records.append(arha_subset)

    if not records:
        return gpd.GeoDataFrame(columns=gbif_gdf.columns, geometry="geometry", crs="EPSG:4326")

    combined = gpd.GeoDataFrame(
        pd.concat(records, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )

    if drop_duplicate_coords and not combined.empty:
        combined["_lon"] = combined.geometry.x.round(6)
        combined["_lat"] = combined.geometry.y.round(6)
        combined = combined.drop_duplicates(subset=["_lon", "_lat", "source"])
        combined = combined.drop(columns=["_lon", "_lat"])

    return combined


def process_species_files(
    gbif_dir: Path,
    arha_by_species: Dict[str, gpd.GeoDataFrame],
    output_dir: Path,
    include_arha_only: bool,
    drop_duplicate_coords: bool,
) -> Dict[str, Dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, int]] = {}

    for path in sorted(gbif_dir.glob("*.geojson")):
        slug = path.stem
        label = slug.replace("_", " ")

        gbif_gdf = read_geojson(path)
        arha_gdf = arha_by_species.get(slug)

        combined = merge_species_records(gbif_gdf, arha_gdf, drop_duplicate_coords=drop_duplicate_coords)

        if combined.empty:
            combined = gbif_gdf

        out_path = output_dir / path.name
        combined.to_file(out_path, driver="GeoJSON")

        summary[slug] = {
            "gbif": len(gbif_gdf),
            "arha": len(arha_gdf) if arha_gdf is not None else 0,
            "combined": len(combined),
        }

    if include_arha_only:
        existing_slugs = {p.stem for p in gbif_dir.glob("*.geojson")}
        for slug, gdf in arha_by_species.items():
            if slug in existing_slugs:
                continue
            out_path = output_dir / f"{slug}.geojson"
            combined = merge_species_records(
                gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326"),
                gdf,
                drop_duplicate_coords=drop_duplicate_coords,
            )
            combined.to_file(out_path, driver="GeoJSON")
            summary[slug] = {
                "gbif": 0,
                "arha": len(gdf),
                "combined": len(combined),
            }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ArHa occurrences into GBIF GeoJSON inputs."
    )
    parser.add_argument(
        "--gbif-dir",
        type=Path,
        default=DEFAULT_GBIF_DIR,
        help=f"Directory containing GBIF GeoJSON files (default: {DEFAULT_GBIF_DIR})",
    )
    parser.add_argument(
        "--arha-csv",
        type=Path,
        default=DEFAULT_ARHA_CSV,
        help=f"Path to ArHa cleaned CSV (default: {DEFAULT_ARHA_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write merged GeoJSON files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-hosts",
        type=float,
        default=1.0,
        help="Minimum number_of_hosts in ArHa records to include (default: 1)",
    )
    parser.add_argument(
        "--allow-zero-hosts",
        action="store_true",
        help="Include ArHa records even when number_of_hosts is 0 or missing.",
    )
    parser.add_argument(
        "--include-arha-only",
        action="store_true",
        help="Write GeoJSON files for species present only in ArHa.",
    )
    parser.add_argument(
        "--keep-duplicate-coords",
        action="store_true",
        help="Retain duplicate coordinate records (default drops duplicates per source).",
    )
    parser.add_argument(
        "--require-valid-coord",
        action="store_true",
        help="Only include ArHa records where coord_status == 'valid' (when column is present).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.allow_zero_hosts:
        min_hosts = -math.inf
    else:
        min_hosts = args.min_hosts

    arha_by_species = load_arha_occurrences(
        csv_path=args.arha_csv,
        min_hosts=min_hosts,
        require_valid_coord=args.require_valid_coord,
    )

    summary = process_species_files(
        gbif_dir=args.gbif_dir,
        arha_by_species=arha_by_species,
        output_dir=args.output_dir,
        include_arha_only=args.include_arha_only,
        drop_duplicate_coords=not args.keep_duplicate_coords,
    )

    print(f"[INFO] Wrote merged GeoJSONs to: {args.output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
