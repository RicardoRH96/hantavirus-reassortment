#!/usr/bin/env python3

"""
Download higher resolution WorldClim layers and replace existing 5 arc-minute
rasters in the Covariates directory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List

WORLDCLIM_URLS = [
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_bio.zip",
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_elev.zip",
]


def download_file(url: str, destination: Path) -> None:
    """Download a file to the given destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def extract_zip(archive: Path, destination: Path) -> List[Path]:
    """Extract a zip archive and return all TIFF files contained within."""
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
    return list(destination.rglob("*.tif"))


def remove_existing_rasters(target_dir: Path, prefix: str = "wc2.1_5m") -> List[str]:
    """Remove rasters in target_dir that start with the given prefix."""
    removed: List[str] = []
    for raster in target_dir.glob(f"{prefix}*.tif"):
        raster.unlink()
        removed.append(raster.name)
    return removed


def move_new_rasters(rasters: Iterable[Path], destination: Path) -> List[str]:
    """Move rasters into destination, replacing if they already exist."""
    moved: List[str] = []
    for raster in rasters:
        target = destination / raster.name
        if target.exists():
            target.unlink()
        shutil.move(str(raster), target)
        moved.append(target.name)
    return moved


def parse_args() -> argparse.Namespace:
    default_covariates_dir = (
        Path(__file__).resolve().parents[1] / "Covariates"
    )
    parser = argparse.ArgumentParser(
        description="Download WorldClim 30s rasters and replace 5m rasters."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_covariates_dir,
        help="Directory containing covariate rasters (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    covariates_dir: Path = args.output_dir.expanduser().resolve()
    covariates_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmpdir = Path(tmp_str)
        extracted_rasters: List[Path] = []

        for url in WORLDCLIM_URLS:
            archive_name = url.split("/")[-1]
            archive_path = tmpdir / archive_name
            download_file(url, archive_path)

            extract_dir = tmpdir / archive_name.replace(".zip", "")
            extract_dir.mkdir(exist_ok=True)
            extracted_rasters.extend(extract_zip(archive_path, extract_dir))

        if not extracted_rasters:
            print("No rasters were extracted; aborting.", file=sys.stderr)
            return 1

        removed = remove_existing_rasters(covariates_dir, prefix="wc2.1_5m")
        moved = move_new_rasters(extracted_rasters, covariates_dir)

    print(f"Removed {len(removed)} existing rasters:")
    for name in removed:
        print(f"  - {name}")

    print(f"Added {len(moved)} new rasters:")
    for name in moved:
        print(f"  - {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
