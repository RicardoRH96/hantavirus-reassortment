"""Clip environmental raster layers by an IUCN (or any) species range polygon.

Features
--------
* Iterates over all rasters in an input directory (default: ../Covariates)
* Reads a vector layer (GeoJSON/Shapefile/any format supported by GeoPandas)
* Optional attribute filter (--where) to subset features
* Optional buffer in kilometers applied in projected space (EPSG:3857)
* Reprojects geometries to match each raster automatically
* Writes out masked rasters with updated metadata
* Skips rasters if an already clipped version exists unless --overwrite
* Optional parallel execution (--workers)

Usage (examples)
---------------
python clipRaster_IUCN.py \
	--vector ../Inputs/Abrothrix_longipilis.geojson \
	--rasters-dir ../Covariates \
	--out-dir ../Outputs/Abrothrix_longipilis/clipped \
	--buffer-km 50

Filter a single species out of a multi-species file:
python clipRaster_IUCN.py \
	--vector ranges_all_species.geojson \
	--where "binomial == 'Abrothrix longipilis'" \
	--out-dir ../Outputs/Abrothrix_longipilis/clipped

Dependencies
------------
Required: geopandas, rasterio, shapely, pyproj
Optional: tqdm (for nice progress bar)

Install (if needed):
pip install geopandas rasterio shapely pyproj tqdm

Notes
-----
Buffering on geographic CRS is handled by temporarily reprojecting to EPSG:3857
(Web Mercator). For large areas or polar regions you may prefer a custom equal-
area projection; adapt `BUFFER_CRS` constant accordingly.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.errors import RasterioIOError
from shapely.geometry import mapping, GeometryCollection
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry
import shapely
from pyproj import Transformer

try:  # optional
	from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	tqdm = None  # type: ignore


BUFFER_CRS = "EPSG:3857"  # used for buffering in meters


@dataclass
class ClipJobResult:
	raster: Path
	output: Path
	success: bool
	message: str


def read_vector(vector_path: Path, where: Optional[str] = None) -> gpd.GeoDataFrame:
	if not vector_path.exists():
		raise FileNotFoundError(f"Vector file not found: {vector_path}")
	gdf = gpd.read_file(vector_path)
	if gdf.empty:
		raise ValueError(f"Vector layer is empty: {vector_path}")
	if where:
		try:
			gdf = gdf.query(where)
		except Exception as e:  # provide clearer error
			raise ValueError(f"Failed evaluating --where expression '{where}': {e}")
		if gdf.empty:
			raise ValueError(f"Attribute filter produced empty selection: {where}")
	gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
	if gdf.empty:
		raise ValueError("All geometries empty or null after filtering.")
	# Dissolve into a single geometry (unary_union) to speed up masking
	union_geom = unary_union(gdf.geometry.tolist())
	return gpd.GeoDataFrame(index=[0], geometry=[union_geom], crs=gdf.crs)


def buffer_km(gdf: gpd.GeoDataFrame, buffer_km: float) -> gpd.GeoDataFrame:
	if buffer_km <= 0:
		return gdf
	if gdf.crs is None:
		raise ValueError("Input vector has no CRS; cannot buffer reliably.")
	transformer_to = Transformer.from_crs(gdf.crs, BUFFER_CRS, always_xy=True)
	transformer_back = Transformer.from_crs(BUFFER_CRS, gdf.crs, always_xy=True)
	buffered_geoms: List[BaseGeometry] = []
	for geom in gdf.geometry:
		if geom.is_empty:
			continue
		# project to buffer CRS
		projected = shapely.ops.transform(lambda x, y, z=None: transformer_to.transform(x, y), geom)  # type: ignore
		buffered = projected.buffer(buffer_km * 1000.0)
		restored = shapely.ops.transform(lambda x, y, z=None: transformer_back.transform(x, y), buffered)  # type: ignore
		buffered_geoms.append(restored)
	if not buffered_geoms:
		raise ValueError("No geometries remained after buffering.")
	union_geom = unary_union(buffered_geoms)
	return gpd.GeoDataFrame(index=[0], geometry=[union_geom], crs=gdf.crs)


def buffer_degrees(gdf: gpd.GeoDataFrame, buffer_deg: float, work_crs: str = BUFFER_CRS) -> gpd.GeoDataFrame:
	"""
	Buffer the geometry by an approximate degree distance.
	The geometries are reprojected to `work_crs` (default: metric), buffered by
	buffer_deg converted to meters, then reprojected back.
	"""
	if buffer_deg <= 0:
		return gdf
	if gdf.crs is None:
		raise ValueError("Input vector has no CRS; cannot buffer reliably.")
	target = gdf
	if work_crs is None:
		work_crs = BUFFER_CRS
	temp = target.to_crs(work_crs)
	meters = float(buffer_deg) * 111_000.0
	temp.geometry = temp.geometry.buffer(meters)
	return temp.to_crs(target.crs)


def load_species_range(vector_path: Path, species_name: str, buffer_deg: float = 0.0) -> gpd.GeoDataFrame:
	"""
	Load and dissolve the IUCN range for the requested species name.

	Parameters
	----------
	vector_path : Path
		Path to the vector dataset (shapefile/GeoPackage/etc).
	species_name : str
		Species name. Underscores are converted to spaces for matching `sci_name`.
	buffer_deg : float
		Optional buffer (in degrees) applied in a geographic CRS (default EPSG:4326).

	Returns
	-------
	GeoDataFrame with a single, dissolved geometry representing the species range.

	Raises
	------
	FileNotFoundError
		If the vector_path does not exist.
	ValueError
		If the species cannot be found in the dataset.
	"""
	vector_path = Path(vector_path)
	if not vector_path.exists():
		raise FileNotFoundError(f"IUCN vector dataset not found: {vector_path}")

	target_name = species_name.replace("_", " ").strip()
	gdf = gpd.read_file(vector_path)
	if "sci_name" not in gdf.columns:
		raise ValueError(f"Expected column 'sci_name' not found in {vector_path}")

	matches = gdf.loc[
		gdf["sci_name"].str.lower() == target_name.lower(),
		["geometry"]
	].copy()
	if matches.empty:
		raise ValueError(f"Species '{target_name}' not found in {vector_path.name}")

	matches.geometry = matches.geometry.buffer(0)  # fix minor topology issues
	union_geom = unary_union([geom for geom in matches.geometry if geom and not geom.is_empty])
	if union_geom.is_empty:
		raise ValueError(f"Species '{target_name}' range geometry is empty after union.")

	out = gpd.GeoDataFrame(index=[0], geometry=[union_geom], crs=gdf.crs)
	if buffer_deg > 0:
		out = buffer_degrees(out, buffer_deg)
	out.geometry = out.geometry.buffer(0)  # clean up self-intersections
	return out


def ensure_out_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def list_rasters(rasters_dir: Path, extensions: Sequence[str] = (".tif", ".tiff")) -> List[Path]:
	out: List[Path] = []
	for ext in extensions:
		out.extend(rasters_dir.glob(f"*{ext}"))
	return sorted(out)


def clip_single_raster(raster_path: Path, geom_gdf: gpd.GeoDataFrame, out_dir: Path, overwrite: bool = False) -> ClipJobResult:
	out_path = out_dir / raster_path.name
	if out_path.exists() and not overwrite:
		return ClipJobResult(raster=raster_path, output=out_path, success=True, message="exists (skipped)")
	try:
		with rasterio.open(raster_path) as src:
			target_crs = src.crs
			if target_crs is None:
				return ClipJobResult(raster_path, out_path, False, "Raster has no CRS")
			if geom_gdf.crs != target_crs:
				geom_proj = geom_gdf.to_crs(target_crs)
			else:
				geom_proj = geom_gdf
			geom = geom_proj.geometry.iloc[0]
			if geom.is_empty:
				return ClipJobResult(raster_path, out_path, False, "Geometry empty")
			geoms = [mapping(geom)]
			out_image, out_transform = mask(src, geoms, crop=True)
			out_meta = src.meta.copy()
			out_meta.update({
				"driver": "GTiff",
				"height": out_image.shape[1],
				"width": out_image.shape[2],
				"transform": out_transform,
				"compress": "LZW",
			})
		ensure_out_dir(out_dir)
		with rasterio.open(out_path, "w", **out_meta) as dst:
			dst.write(out_image)
		return ClipJobResult(raster_path, out_path, True, "clipped")
	except RasterioIOError as e:
		return ClipJobResult(raster_path, out_path, False, f"RasterioIOError: {e}")
	except Exception as e:  # broad catch to continue other rasters
		return ClipJobResult(raster_path, out_path, False, f"Error: {e}")


def clip_rasters(
	rasters: Sequence[Path],
	geom_gdf: gpd.GeoDataFrame,
	out_dir: Path,
	overwrite: bool = False,
	workers: int = 1,
) -> List[ClipJobResult]:
	results: List[ClipJobResult] = []
	iterator: Iterable[Path] = rasters
	progress_wrapper = (lambda x: tqdm(x, desc="Clipping", unit="rst")) if tqdm else (lambda x: x)
	if workers and workers > 1:
		with futures.ThreadPoolExecutor(max_workers=workers) as ex:
			fut_map = {ex.submit(clip_single_raster, r, geom_gdf, out_dir, overwrite): r for r in rasters}
			for fut in progress_wrapper(futures.as_completed(fut_map)):
				try:
					results.append(fut.result())
				except Exception as e:  # improbable due to internal catching
					r = fut_map[fut]
					results.append(ClipJobResult(r, out_dir / r.name, False, f"Unhandled future exception: {e}"))
	else:
		for r in progress_wrapper(iterator):
			results.append(clip_single_raster(r, geom_gdf, out_dir, overwrite))
	return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Clip rasters by a species range polygon (IUCN).")
	p.add_argument("--vector", required=True, type=Path, help="Path to species range vector file (GeoJSON/Shapefile).")
	p.add_argument("--rasters-dir", type=Path, default=Path("../Covariates"), help="Directory containing input rasters.")
	p.add_argument("--out-dir", type=Path, required=True, help="Output directory for clipped rasters.")
	p.add_argument("--where", type=str, default=None, help="Pandas query string to filter vector attributes.")
	p.add_argument("--buffer-km", type=float, default=0.0, help="Buffer distance in kilometers applied to geometry.")
	p.add_argument("--overwrite", action="store_true", help="Overwrite existing clipped rasters.")
	p.add_argument("--workers", type=int, default=1, help="Number of worker threads (>=2 enables parallel clipping).")
	p.add_argument("--list-only", action="store_true", help="Only list raster files that would be processed and exit.")
	return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)
	rasters_dir: Path = args.rasters_dir
	if not rasters_dir.exists():
		raise SystemExit(f"Rasters dir not found: {rasters_dir}")
	raster_paths = list_rasters(rasters_dir)
	if not raster_paths:
		raise SystemExit(f"No raster files found in {rasters_dir}")
	if args.list_only:
		print("Rasters that would be processed:")
		for r in raster_paths:
			print(r)
		return 0
	geom_gdf = read_vector(args.vector, where=args.where)
	if args.buffer_km > 0:
		geom_gdf = buffer_km(geom_gdf, args.buffer_km)
	results = clip_rasters(
		raster_paths,
		geom_gdf,
		args.out_dir,
		overwrite=args.overwrite,
		workers=max(1, args.workers),
	)
	# Summary
	success = sum(1 for r in results if r.success)
	skipped = sum(1 for r in results if r.message.startswith("exists"))
	failed = [r for r in results if not r.success and not r.message.startswith("exists")]
	print(f"Processed {len(results)} rasters | success: {success} | skipped(existing): {skipped} | failed: {len(failed)}")
	if failed:
		print("Failures:")
		for f in failed:
			print(f" - {f.raster.name}: {f.message}")
	return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())
