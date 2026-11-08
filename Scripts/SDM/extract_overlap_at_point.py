#!/usr/bin/env python3
"""Utilities and CLI for sampling the SDM overlap raster at point coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import rasterio
    from rasterio.warp import transform
except ImportError:
    rasterio = None
    transform = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RASTER = REPO_ROOT / "Data" / "SDM" / "OverlapRasters" / "overlap_richness_threshold_0p9.tif"
DEFAULT_INPUT_CRS = "EPSG:4326"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract overlap richness values from the SDM overlap raster at one or more point coordinates."
        )
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=DEFAULT_RASTER,
        help=f"Path to overlap raster. Default: {DEFAULT_RASTER}",
    )
    parser.add_argument(
        "--lon",
        type=float,
        help="Longitude / x coordinate for a single point.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        help="Latitude / y coordinate for a single point.",
    )
    parser.add_argument(
        "--input-crs",
        default=DEFAULT_INPUT_CRS,
        help=f"CRS for input coordinates. Default: {DEFAULT_INPUT_CRS}.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV of points to sample.",
    )
    parser.add_argument(
        "--x-col",
        default="lon",
        help="CSV column for x/longitude. Default: lon.",
    )
    parser.add_argument(
        "--y-col",
        default="lat",
        help="CSV column for y/latitude. Default: lat.",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Optional CSV column to preserve as a record identifier.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format. Default: json.",
    )
    return parser.parse_args()


def _require_rasterio(task: str) -> None:
    if rasterio is None or transform is None:
        raise RuntimeError(f"{task} requires rasterio, which is not available in this environment.")


def to_raster_xy(x: float, y: float, input_crs: str, dataset) -> Tuple[float, float]:
    _require_rasterio("Raster overlap sampling")
    if not dataset.crs or input_crs == dataset.crs.to_string():
        return x, y
    xs, ys = transform(input_crs, dataset.crs, [x], [y])
    return xs[0], ys[0]


def sample_overlap(dataset, x: float, y: float, input_crs: str = DEFAULT_INPUT_CRS, record_id=None):
    raster_x, raster_y = to_raster_xy(x, y, input_crs, dataset)
    bounds = dataset.bounds
    in_bounds = (
        bounds.left <= raster_x <= bounds.right
        and bounds.bottom <= raster_y <= bounds.top
    )

    if not in_bounds:
        return {
            "id": record_id,
            "input_x": x,
            "input_y": y,
            "input_crs": input_crs,
            "raster_crs": dataset.crs.to_string() if dataset.crs else None,
            "overlap_count": None,
            "has_overlap": None,
            "is_nodata": True,
            "in_bounds": False,
        }

    sample = next(dataset.sample([(raster_x, raster_y)], masked=True))
    value = sample[0]
    is_nodata = bool(getattr(value, "mask", False))
    overlap_count = None if is_nodata else int(value.item())

    return {
        "id": record_id,
        "input_x": x,
        "input_y": y,
        "input_crs": input_crs,
        "raster_crs": dataset.crs.to_string() if dataset.crs else None,
        "overlap_count": overlap_count,
        "has_overlap": None if overlap_count is None else overlap_count > 0,
        "is_nodata": is_nodata,
        "in_bounds": True,
    }


def sample_raster_at_coordinates(
    raster_path: Path,
    latitudes: pd.Series,
    longitudes: pd.Series,
    input_crs: str = DEFAULT_INPUT_CRS,
) -> pd.Series:
    """Return overlap-count values aligned to the input index."""
    _require_rasterio("Raster overlap sampling")

    lat = pd.to_numeric(latitudes, errors="coerce")
    lon = pd.to_numeric(longitudes, errors="coerce")
    out = pd.Series(pd.NA, index=latitudes.index, dtype="Float64")
    valid_mask = lat.notna() & lon.notna()
    if not valid_mask.any():
        return out

    with rasterio.open(raster_path) as dataset:
        for idx, x, y in zip(
            lat.index[valid_mask],
            lon.loc[valid_mask].astype(float),
            lat.loc[valid_mask].astype(float),
        ):
            row = sample_overlap(dataset, float(x), float(y), input_crs=input_crs, record_id=idx)
            overlap_count = row["overlap_count"]
            if overlap_count is not None:
                out.at[idx] = float(overlap_count)

    return out


def load_points(args) -> Iterable[Tuple[float, float, Optional[str]]]:
    if args.csv:
        with args.csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield (
                    float(row[args.x_col]),
                    float(row[args.y_col]),
                    row.get(args.id_col) if args.id_col else None,
                )
        return

    if args.lon is None or args.lat is None:
        raise SystemExit("Provide --lon/--lat for a single point, or use --csv.")

    yield args.lon, args.lat, None


def emit_json(rows: List[dict]) -> None:
    print(json.dumps(rows, indent=2))


def emit_csv(rows: List[dict]) -> None:
    fieldnames = [
        "id",
        "input_x",
        "input_y",
        "input_crs",
        "raster_crs",
        "overlap_count",
        "has_overlap",
        "is_nodata",
        "in_bounds",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    _require_rasterio("Raster overlap sampling")
    with rasterio.open(args.raster) as dataset:
        rows = [
            sample_overlap(dataset, x, y, args.input_crs, record_id)
            for x, y, record_id in load_points(args)
        ]

    if args.format == "csv":
        emit_csv(rows)
        return
    emit_json(rows)


if __name__ == "__main__":
    main()
