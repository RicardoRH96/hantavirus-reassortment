#!/usr/bin/env python3

import argparse
import os
import hashlib
from pathlib import Path
from joblib import dump, load, parallel_backend
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from contextlib import ExitStack

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
from shapely.ops import unary_union
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from clipRaster_IUCN import load_species_range

from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.inspection import partial_dependence
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)

#Bayesian search
from skopt import BayesSearchCV
from skopt.space import Real, Integer

import time
import json
from contextlib import contextmanager

# ----------------------- CONFIG  -----------------------
THIN_DISTANCE_M = 25_000      # presence thinning radius (meters)
SPATIAL_BLOCK_KM = 150        # spatial CV block size
CV_FOLDS = 5                  # number of GroupKFold folds
RNG_SEED = 72724              # reproducibility for pseudo-absences
IUCN_BUFFER_DEGREES = 0.5
DEFAULT_IUCN_VECTOR = Path(__file__).resolve().parents[1] / "IUCN_mammals" / "MAMMALS_TERRESTRIAL_ONLY.shp"
PD_GRID_RESOLUTION = 20
OUTPUT_NODATA = -9999.0
DEFAULT_TILE_SIZE = 512
DEFAULT_FEATURE_CORR_THRESHOLD = 0.95
DEFAULT_FEATURE_MAX_NAN_FRAC = 0.3
DEFAULT_FEATURE_MIN_FEATURES = 10
DEFAULT_THIN_MIN_PRES = 30
DEFAULT_THIN_MIN_DISTANCE_M = 1_000
DEFAULT_THIN_ADAPT_ITERS = 6
# --------------------------------------------------------


class IdentityCalibrator:
    """Fallback calibrator that returns probabilities unchanged."""

    def predict(self, values):
        return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


class ProbabilityCalibratedEstimator(BaseEstimator, ClassifierMixin):
    """Wrap a fitted estimator and apply a post-hoc probability calibrator."""

    def __init__(self, base_estimator, calibrator):
        self.base_estimator = base_estimator
        self.calibrator = calibrator

    def __sklearn_is_fitted__(self):
        if getattr(self, "_force_fitted", False):
            return True
        if hasattr(self.base_estimator, "__sklearn_is_fitted__"):
            try:
                return bool(self.base_estimator.__sklearn_is_fitted__())
            except Exception:
                pass
        try:
            check_is_fitted(self.base_estimator)
            return True
        except Exception:
            return False

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        raw = self.base_estimator.predict_proba(X)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError("Base estimator must provide binary class probabilities.")
        pos = raw[:, 1]
        calibrated = np.clip(self.calibrator.predict(pos), 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def named_steps(self):
        return getattr(self.base_estimator, "named_steps", None)

    def __getattr__(self, attr):
        return getattr(self.base_estimator, attr)


class AveragingEnsemble(BaseEstimator, ClassifierMixin):
    """Average class probabilities from multiple estimators."""

    def __init__(self, estimators: Sequence[Tuple[str, BaseEstimator]]):
        if not estimators:
            raise ValueError("AveragingEnsemble requires at least one estimator.")
        self.estimators = list(estimators)
        self.fitted_estimators_: List[Tuple[str, BaseEstimator]] = []

    def fit(self, X, y):
        self.fitted_estimators_ = []
        self.classes_ = np.unique(y)
        for name, estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(X, y)
            self.fitted_estimators_.append((name, fitted))
        if self.fitted_estimators_ and hasattr(self.fitted_estimators_[0][1], "n_features_in_"):
            self.n_features_in_ = self.fitted_estimators_[0][1].n_features_in_
        return self

    def predict_proba(self, X):
        if not self.fitted_estimators_:
            raise ValueError("AveragingEnsemble must be fit before predicting.")
        probs = [est.predict_proba(X) for _, est in self.fitted_estimators_]
        return np.mean(probs, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_probability_calibrator(y_true, oof_prob):
    """
    Fit an isotonic regression calibrator on out-of-fold probabilities.
    Returns either the calibrated model or an identity calibrator if not feasible.
    """
    y_true = np.asarray(y_true, dtype=float)
    oof_prob = np.clip(np.asarray(oof_prob, dtype=float), 0.0, 1.0)
    if y_true.size == 0 or oof_prob.size == 0:
        raise ValueError("Cannot calibrate with empty arrays.")
    if np.unique(oof_prob).size < 2:
        return IdentityCalibrator()
    try:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_prob, y_true)
        return iso
    except Exception as exc:
        warnings.warn(f"Falling back to identity calibrator due to: {exc}", RuntimeWarning)
        return IdentityCalibrator()


def safe_filename(value: str) -> str:
    """Sanitize a string for safe filesystem usage while retaining readability."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value)

def load_rasters(covariate_dir):
    """Return sorted .tif paths; without copying to output_dir""" 
    raster_files = sorted(Path(covariate_dir).glob('*.tif'))
    if not raster_files:
        raise FileNotFoundError(f"No .tif files found in directory: {covariate_dir}")
    return [str(p) for p in raster_files]

def short_name(path_str: str) -> str:
    """Stable, readable column name per raster band set"""
    p = Path(path_str)
    return p.stem #e.g., "bio1" from "/path/to/bio1.tif"

def get_cpus_per_task(default: int = 1) -> int:
    """Best-effort CPU count for thread control inside SLURM/containers."""
    for key in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        val = os.environ.get(key)
        if val:
            try:
                parsed = int(val)
                if parsed > 0:
                    return parsed
            except ValueError:
                continue
    return default

def _window_from_bounds_clipped(bounds, dataset) -> Window:
    """Convert bounds (in dataset CRS) to an integer Window clipped to dataset extent."""
    left, bottom, right, top = bounds
    win_f = window_from_bounds(left, bottom, right, top, transform=dataset.transform)
    col_off = int(np.floor(win_f.col_off))
    row_off = int(np.floor(win_f.row_off))
    col_end = int(np.ceil(win_f.col_off + win_f.width))
    row_end = int(np.ceil(win_f.row_off + win_f.height))

    col_off = max(0, col_off)
    row_off = max(0, row_off)
    col_end = min(dataset.width, col_end)
    row_end = min(dataset.height, row_end)

    if col_end <= col_off or row_end <= row_off:
        raise ValueError("Projection window collapsed after clipping; check bounds/CRS.")
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)

def _reproject_geom(geom, src_crs, dst_crs):
    if geom is None:
        return None
    if src_crs is None or dst_crs is None or src_crs == dst_crs:
        return geom
    try:
        return gpd.GeoSeries([geom], crs=src_crs).to_crs(dst_crs).iloc[0]
    except Exception:
        return geom

def _select_projection_bounds(
    project_mode: str,
    pres_pts: gpd.GeoDataFrame,
    domain_geom,
    species_range_geom_pres,
    raster_ds,
    project_buffer_deg: float = 0.0,
):
    raster_crs = raster_ds.crs
    raster_bounds = raster_ds.bounds
    raster_bounds_geom = box(*raster_bounds)

    if project_mode == "global":
        geom = raster_bounds_geom
    elif project_mode == "iucn_bbox" and species_range_geom_pres is not None:
        geom = _reproject_geom(species_range_geom_pres, pres_pts.crs, raster_crs)
    elif project_mode == "pres_bbox":
        bbox = box(*pres_pts.to_crs(raster_crs).total_bounds)
        geom = bbox
    else:  # domain_bbox
        geom = _reproject_geom(domain_geom, pres_pts.crs, raster_crs)

    if geom is None:
        geom = raster_bounds_geom

    geom = geom.intersection(raster_bounds_geom)
    if geom.is_empty:
        geom = raster_bounds_geom

    minx, miny, maxx, maxy = geom.bounds
    if project_buffer_deg and raster_crs and raster_crs.to_epsg() == 4326:
        minx -= project_buffer_deg
        miny -= project_buffer_deg
        maxx += project_buffer_deg
        maxy += project_buffer_deg

    # Clip to dataset bounds
    left, bottom, right, top = raster_bounds
    minx = max(left, minx)
    maxx = min(right, maxx)
    miny = max(bottom, miny)
    maxy = min(top, maxy)
    return (minx, miny, maxx, maxy)

def _align_raster_stack(raster_files: Sequence[str]) -> Tuple[dict, List[float]]:
    """Ensure rasters share grid/CRS; return reference profile and nodata list."""
    if not raster_files:
        raise ValueError("No rasters provided.")
    with rasterio.open(raster_files[0]) as ref:
        ref_profile = ref.profile.copy()
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width, ref_height = ref.width, ref.height
        nodatas = [ref.nodata]

    for rp in raster_files[1:]:
        with rasterio.open(rp) as src:
            nodatas.append(src.nodata)
            if src.crs != ref_crs:
                raise ValueError(f"Raster CRS mismatch: {rp} ({src.crs}) != {ref_crs}")
            if (src.width, src.height) != (ref_width, ref_height):
                raise ValueError(f"Raster shape mismatch: {rp} ({src.width}x{src.height}) != ({ref_width}x{ref_height})")
            try:
                same_transform = src.transform.almost_equals(ref_transform)
            except Exception:
                same_transform = (src.transform == ref_transform)
            if not same_transform:
                raise ValueError(f"Raster transform mismatch: {rp}")

    return ref_profile, nodatas

def _stack_features_for_window(datasets, window: Window) -> Tuple[np.ndarray, np.ndarray]:
    """Read all rasters for a window into (H, W, F) float32 and valid mask (H, W)."""
    bands = []
    valid_mask = None
    for ds in datasets:
        arr = ds.read(1, window=window, masked=True)
        arr_f = arr.astype("float32", copy=False)
        data = arr_f.filled(np.nan)
        m = ~np.ma.getmaskarray(arr)
        m &= np.isfinite(data)
        valid_mask = m if valid_mask is None else (valid_mask & m)
        bands.append(data)
    feat = np.stack(bands, axis=-1)  # (H, W, F)
    if valid_mask is None:
        valid_mask = np.zeros(feat.shape[:2], dtype=bool)
    return feat, valid_mask

def project_probability_rasters(
    raster_files: Sequence[str],
    estimator: BaseEstimator,
    feature_names: Sequence[str],
    out_dir: Path,
    safe_species: str,
    base_window: Window,
    tile_size: int = DEFAULT_TILE_SIZE,
    nodata: float = OUTPUT_NODATA,
    make_styled: bool = True,
) -> Dict[str, Path]:
    """
    Chunked projection over a raster stack. Avoids building a giant target_xs matrix in memory.

    Writes:
      - probability_0_{species}.tif
      - probability_1_{species}.tif
      - certainty_{species}.tif
      - probability_1_{species}_styled.tif (uint8, optional)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_size = max(16, int(tile_size))

    prob0_path = out_dir / f"probability_0_{safe_species}.tif"
    prob1_path = out_dir / f"probability_1_{safe_species}.tif"
    cert_path = out_dir / f"certainty_{safe_species}.tif"
    styled_path = out_dir / f"probability_1_{safe_species}_styled.tif"
    tmp_prob0_path = out_dir / f".probability_0_{safe_species}.incomplete"
    tmp_prob1_path = out_dir / f".probability_1_{safe_species}.incomplete"
    tmp_cert_path = out_dir / f".certainty_{safe_species}.incomplete"
    tmp_styled_path = out_dir / f".probability_1_{safe_species}_styled.incomplete"

    # Clean stale outputs (partial runs) to avoid confusing downstream steps.
    for p in (prob0_path, prob1_path, cert_path, styled_path, tmp_prob0_path, tmp_prob1_path, tmp_cert_path, tmp_styled_path):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        with ExitStack() as stack:
            datasets = [stack.enter_context(rasterio.open(p)) for p in raster_files]
            ref = datasets[0]

            out_transform = window_transform(base_window, ref.transform)
            out_height = int(base_window.height)
            out_width = int(base_window.width)
            if out_height <= 0 or out_width <= 0:
                raise ValueError("Projection window has non-positive dimensions.")

            base_profile = ref.profile.copy()
            base_profile.update(
                height=out_height,
                width=out_width,
                transform=out_transform,
                dtype="float32",
                count=1,
                nodata=nodata,
            )

            # Preserve compression if present, otherwise pick a sensible default.
            base_profile.setdefault("compress", "lzw")

            styled_profile = base_profile.copy()
            styled_profile.update(dtype="uint8")
            styled_profile.pop("nodata", None)

            prob0_ds = stack.enter_context(rasterio.open(tmp_prob0_path, "w", **base_profile))
            prob1_ds = stack.enter_context(rasterio.open(tmp_prob1_path, "w", **base_profile))
            cert_ds = stack.enter_context(rasterio.open(tmp_cert_path, "w", **base_profile))
            styled_ds = None
            if make_styled:
                styled_ds = stack.enter_context(rasterio.open(tmp_styled_path, "w", **styled_profile))
                cmap = cm.get_cmap("viridis", 256)
                color_map = {i: tuple(int(round(255 * c)) for c in cmap(i / 255.0)[:3]) for i in range(256)}
                color_map[0] = (255, 255, 255)
                styled_ds.write_colormap(1, color_map)

            base_col0 = int(base_window.col_off)
            base_row0 = int(base_window.row_off)
            base_col1 = base_col0 + out_width
            base_row1 = base_row0 + out_height

            print(f"[INFO] Projecting raster window: {out_width}x{out_height} (~{out_width*out_height:,} px) from global grid.")
            print(f"[INFO] Using {len(raster_files)} rasters / {len(feature_names)} features for projection.")

            for row in range(base_row0, base_row1, tile_size):
                h = min(tile_size, base_row1 - row)
                for col in range(base_col0, base_col1, tile_size):
                    w = min(tile_size, base_col1 - col)
                    src_win = Window(col, row, w, h)
                    dst_win = Window(col - base_col0, row - base_row0, w, h)

                    feat, valid_mask = _stack_features_for_window(datasets, src_win)
                    valid_flat = valid_mask.reshape(-1)
                    n_pixels = valid_flat.size
                    n_feat = feat.shape[2]
                    if n_feat != len(feature_names):
                        raise ValueError(f"Feature count mismatch: window has {n_feat} but expected {len(feature_names)}")

                    prob1 = np.full(n_pixels, nodata, dtype="float32")
                    if valid_flat.any():
                        X = feat.reshape(-1, n_feat)[valid_flat]
                        prob1_valid = estimator.predict_proba(X)[:, 1].astype("float32", copy=False)
                        prob1[valid_flat] = np.clip(prob1_valid, 0.0, 1.0)

                    prob1_img = prob1.reshape(h, w)
                    valid_img = valid_mask

                    prob0_img = np.full((h, w), nodata, dtype="float32")
                    cert_img = np.full((h, w), nodata, dtype="float32")
                    if valid_img.any():
                        p = prob1_img[valid_img]
                        prob0_img[valid_img] = 1.0 - p
                        cert_img[valid_img] = np.maximum(p, 1.0 - p)

                    prob0_ds.write(prob0_img, 1, window=dst_win)
                    prob1_ds.write(prob1_img, 1, window=dst_win)
                    cert_ds.write(cert_img, 1, window=dst_win)

                    if styled_ds is not None:
                        styled = np.zeros((h, w), dtype="uint8")
                        if valid_img.any():
                            styled_vals = np.round(np.clip(prob1_img[valid_img], 0.0, 1.0) * 255).astype("uint8")
                            styled[valid_img] = styled_vals
                        styled_ds.write(styled, 1, window=dst_win)
    except Exception:
        for p in (tmp_prob0_path, tmp_prob1_path, tmp_cert_path, tmp_styled_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    os.replace(tmp_prob0_path, prob0_path)
    os.replace(tmp_prob1_path, prob1_path)
    os.replace(tmp_cert_path, cert_path)
    if make_styled:
        os.replace(tmp_styled_path, styled_path)

    outputs = {"prob0": prob0_path, "prob1": prob1_path, "certainty": cert_path}
    if make_styled:
        outputs["styled"] = styled_path
    return outputs


def _hash_signature(raster_files, bbox, species_name, crs_epsg, n_pres, range_digest=None):
    h = hashlib.sha1()
    for p in sorted(raster_files):
        h.update(Path(p).name.encode())
        if Path(p).exists():
            h.update(str(Path(p).stat().st_mtime_ns).encode())
    # Bump when feature-cache semantics change (e.g., nodata handling, sampling logic).
    h.update(b"|features=v2")
    h.update(",".join(f"{x:.5f}" for x in bbox).encode())
    h.update(str(crs_epsg).encode())
    h.update(str(n_pres).encode())
    h.update(species_name.encode())
    if range_digest:
        h.update(range_digest.encode())
    return h.hexdigest()[:12]

def to_local_meters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Choose a local UTM CRS using the median lon/lat of valid POINT geometries.
    Avoids union_all() so it works even with mixed/invalid rows.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS")

    # Keep only valid, non-empty Points for computing the centroid
    g_valid = gdf.loc[
        gdf.geometry.notnull()
        & gdf.geometry.is_valid
        & ~gdf.geometry.is_empty
        & (gdf.geometry.geom_type == "Point")
    ].copy()

    if g_valid.empty:
        raise ValueError("No valid Point geometries to determine a local CRS")

    g_ll = g_valid.to_crs(4326)
    lon_med = float(np.nanmedian(g_ll.geometry.x.values))
    lat_med = float(np.nanmedian(g_ll.geometry.y.values))

    utm_zone = int((lon_med + 180) // 6) + 1
    epsg = 32600 + utm_zone if lat_med >= 0 else 32700 + utm_zone
    return gdf.to_crs(epsg)

def thin_points_distance(gdf: gpd.GeoDataFrame, min_dist_m: float, randomize: bool = True, rng_seed: int = RNG_SEED, n_switch: int = 2000) -> gpd.GeoDataFrame:
    """
    Distance-based thinning. Uses cKDTree for large n, vectorized loop for small n (faster at ~<=2000 pts).
    Randomized greedy: keep a point, remove all within min_dist_m.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS")

    gdf = gdf.loc[
        gdf.geometry.notnull()
        & gdf.geometry.is_valid
        & ~gdf.geometry.is_empty
        & (gdf.geometry.geom_type == "Point")
    ].copy()
    if gdf.empty:
        return gdf

    g = to_local_meters(gdf)
    coords = np.column_stack([g.geometry.x.values, g.geometry.y.values]).astype("float64")
    finite = np.isfinite(coords).all(axis=1)
    g, coords = g.iloc[finite].copy(), coords[finite]
    if len(g) == 0:
        return gdf.iloc[[]].copy()

    idx = np.arange(len(g))
    if randomize:
        rng = np.random.default_rng(rng_seed)
        rng.shuffle(idx)

    if len(g) < n_switch:
        # Small-n: vectorized O(n^2) masking (often faster than KD-tree at ~<= 2k)
        r2 = float(min_dist_m) * float(min_dist_m)
        taken = np.zeros(len(g), dtype=bool)
        kept = []
        for i in idx:
            if taken[i]:
                continue
            kept.append(i)
            dx = coords[:, 0] - coords[i, 0]
            dy = coords[:, 1] - coords[i, 1]
            taken |= (dx*dx + dy*dy) < r2
    else:
        # Large-n: KD-tree
        tree = cKDTree(coords)
        taken = np.zeros(len(g), dtype=bool)
        kept = []
        for i in idx:
            if taken[i]:
                continue
            kept.append(i)
            nbrs = tree.query_ball_point(coords[i], r=min_dist_m)
            taken[nbrs] = True

    out = g.iloc[sorted(kept)].copy().to_crs(gdf.crs)
    return out


def sample_rasters_at_points(gdf, raster_files, resampling=Resampling.bilinear):
    """
    Vectorized: for each raster, sample all points at once. Before it was sampling each point one by one.
    Handles CRS differences via WarpedVRT
    Returns a Dataframe of features aligned to gdf.index
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a defined CRS.")
    
    #Collect coordinates once
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    coords = np.vstack([xs,ys]).T # shape (n_points, 2)

    features = []
    colnames = []

    for rp in raster_files:
        with rasterio.open(rp) as src:
            #Reproject-on-the-fly so we can always pass point CRS
            with WarpedVRT(src, crs=gdf.crs, resampling=resampling) as vrt:
                # Sample yields arrays per point; use masked sampling so nodata becomes NaN.
                # Materialize the generator because numpy.ma.vstack iterates the input multiple times.
                samples = list(vrt.sample(coords, masked=True))
                if not samples:
                    raise ValueError(f"No samples returned for raster {rp}; check CRS/coords.")
                vals_ma = np.ma.vstack(samples)
                vals = vals_ma.astype("float32", copy=False).filled(np.nan)
                #if multiband, name them stem_b1, stem_b2, ...
                if vals.ndim == 1:
                    vals = vals[:, None]
                bnames = ([short_name(rp)] if vals.shape[1]==1
                          else [f"{short_name(rp)}_b{i}" for i in range(1, vals.shape[1]+1)])
                features.append(vals)
                colnames.extend(bnames)

    X = np.hstack(features) #[N, sum(bands)]
    df = pd.DataFrame(X, columns=colnames, index=gdf.index)
    return df

def make_valid_bbox_mask(raster_files, dst_crs):
    """
    Robust raster coverage mask in dst_crs: returns a single box covering
    all rasters. Avoids any union of many geometries (which can fail).
    """
    # Initialize with infinities
    minx = miny = float("+inf")
    maxx = maxy = float("-inf")

    for rp in raster_files:
        with rasterio.open(rp) as src:
            with WarpedVRT(src, crs=dst_crs, resampling=Resampling.nearest) as vrt:
                left, bottom, right, top = vrt.bounds
                if not np.isfinite([left, bottom, right, top]).all():
                    continue
                # expand global bounds
                if left   < minx: minx = left
                if bottom < miny: miny = bottom
                if right  > maxx: maxx = right
                if top    > maxy: maxy = top

    if not np.isfinite([minx, miny, maxx, maxy]).all() or (minx >= maxx) or (miny >= maxy):
        raise ValueError("Could not determine raster coverage bounds; check rasters and CRS.")

    from shapely.geometry import box
    return box(minx, miny, maxx, maxy)

def sample_pseudo_absences(n, region_poly, mask_poly, crs, rng=np.random.default_rng(RNG_SEED)):
    target = region_poly.intersection(mask_poly)
    try:
        target = target.buffer(0)
    except Exception:
        pass
    if target.is_empty:
        raise ValueError("Region ∩ raster mask is empty")
    minx, miny, maxx, maxy = target.bounds
    pts = []
    while len(pts) < n:
        x = rng.uniform(minx, maxx); y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if target.contains(p):
            pts.append(p)
    return gpd.GeoDataFrame(geometry=pts, crs=crs)

def sample_pseudo_absences_with_valid_covariates(
    n: int,
    domain_poly,
    crs,
    raster_files: Sequence[str],
    min_valid_frac: float,
    *,
    rng_seed: int = RNG_SEED,
    batch_multiplier: float = 5.0,
    max_rounds: int = 25,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Sample pseudo-absences inside domain_poly but require that a minimum fraction of
    covariates are non-missing after raster sampling.

    Returns (pseudo_absences_gdf, features_df) with matching row order.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    min_valid_frac = float(min_valid_frac)
    if not (0.0 < min_valid_frac <= 1.0):
        raise ValueError("min_valid_frac must be in (0, 1]")

    rng = np.random.default_rng(rng_seed)
    kept_pts: List[gpd.GeoDataFrame] = []
    kept_feats: List[pd.DataFrame] = []
    total_kept = 0

    for _round in range(int(max_rounds)):
        need = n - total_kept
        if need <= 0:
            break
        batch_n = max(need, int(np.ceil(need * float(batch_multiplier))))
        cand = sample_pseudo_absences(batch_n, domain_poly, domain_poly, crs=crs, rng=rng)
        X_cand = sample_rasters_at_points(cand, raster_files)
        valid_frac = X_cand.notna().mean(axis=1)
        keep_mask = valid_frac >= min_valid_frac
        if not keep_mask.any():
            continue

        cand_keep = cand.loc[keep_mask].copy()
        X_keep = X_cand.loc[keep_mask].copy()
        if len(cand_keep) > need:
            cand_keep = cand_keep.iloc[:need].copy()
            X_keep = X_keep.iloc[:need].copy()

        kept_pts.append(cand_keep.reset_index(drop=True))
        kept_feats.append(X_keep.reset_index(drop=True))
        total_kept += len(cand_keep)

    if total_kept < n:
        raise ValueError(
            f"Could not sample {n} pseudo-absences with >= {min_valid_frac:.2f} valid covariates "
            f"after {max_rounds} rounds; got {total_kept}."
        )

    pseudo_abs = pd.concat(kept_pts, axis=0, ignore_index=True)
    X_abs = pd.concat(kept_feats, axis=0, ignore_index=True)
    return pseudo_abs, X_abs

def make_spatial_cv(n_splits=CV_FOLDS, geom=None, cell_km=SPATIAL_BLOCK_KM, y=None, random_state=RNG_SEED):
    if geom is None: raise ValueError("geom required to make spatial CV groups")
    groups = make_spatial_groups(geom, cell_km)
    n_splits = int(n_splits)
    n_samples = len(groups)
    n_groups = int(pd.Series(groups).nunique())

    # Cap splits to what is feasible for group CV.
    n_splits_eff = min(n_splits, n_samples, n_groups)
    if y is not None:
        y_arr = np.asarray(y, dtype=int)
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        # Stratified variants require at least one sample per class in each fold.
        n_splits_eff = min(n_splits_eff, n_pos, n_neg)
    if n_splits_eff < 2:
        # Not enough unique spatial groups; fall back to stratified CV if possible.
        if y is not None:
            try:
                from sklearn.model_selection import StratifiedKFold

                n_splits_strat = min(n_splits, n_samples, n_pos, n_neg)
                if n_splits_strat >= 2:
                    return StratifiedKFold(n_splits=n_splits_strat, shuffle=True, random_state=random_state), None
            except Exception:
                pass
        return None, None
    try:
        if y is not None:
            from sklearn.model_selection import StratifiedGroupKFold
            cv = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
            return cv, groups
    except ImportError:
        pass
    return GroupKFold(n_splits=n_splits_eff), groups

def make_spatial_groups(gdf: gpd.GeoDataFrame, cell_km: float = SPATIAL_BLOCK_KM) -> np.ndarray:
    """Group label per point based on ~cell_km bins in lat/lon."""
    g = gdf.to_crs(4326)
    step = cell_km / 111.0
    latb = (g.geometry.y / step).astype(int)
    lonb = (g.geometry.x / step).astype(int)
    return (latb.astype(str) + "_" + lonb.astype(str)).values

def print_metrics(y_true, y_prob, label=""):
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"[{label}] ROC-AUC={auc:.3f} | PR-AUC={ap:.3f} | Brier={brier:.3f} | "
          f"Confusion @0.5: TN={tn} FP={fp} FN={fn} TP={tp}")

def _auc_effect(y: np.ndarray, x: np.ndarray) -> float:
    """
    Univariate effect size proxy: max(auc, 1-auc) after median-imputing non-finite values.
    Returns 0.5 for degenerate cases.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return 0.5
    x2 = x.copy()
    med = float(np.nanmedian(x2[finite]))
    x2[~finite] = med
    try:
        auc = float(roc_auc_score(y, x2))
    except Exception:
        return 0.5
    return max(auc, 1.0 - auc)

def select_feature_subset(
    feature_matrix: pd.DataFrame,
    y: np.ndarray,
    *,
    corr_threshold: float = DEFAULT_FEATURE_CORR_THRESHOLD,
    max_nan_frac: float = DEFAULT_FEATURE_MAX_NAN_FRAC,
    min_features: int = DEFAULT_FEATURE_MIN_FEATURES,
) -> Tuple[pd.DataFrame, List[str], Dict[str, object]]:
    """
    Per-species feature selection:
      1) Drop features with too much missingness.
      2) Drop constant features.
      3) Prune highly correlated pairs, keeping the feature with larger univariate AUC effect.

    Returns (selected_feature_matrix, selected_feature_cols, report_dict).
    """
    if feature_matrix.empty:
        raise ValueError("feature_matrix is empty; cannot select features.")

    if y is None or len(y) != len(feature_matrix):
        raise ValueError("y must align with feature_matrix rows.")

    corr_threshold = float(corr_threshold)
    max_nan_frac = float(max_nan_frac)
    min_features = int(min_features)
    if min_features < 1:
        min_features = 1
    if not (0.0 <= max_nan_frac < 1.0):
        max_nan_frac = DEFAULT_FEATURE_MAX_NAN_FRAC

    feature_cols = list(feature_matrix.columns)
    nan_frac = feature_matrix.isna().mean(axis=0)
    nunique = feature_matrix.nunique(dropna=True)

    dropped_missing = [c for c in feature_cols if float(nan_frac.get(c, 0.0)) > max_nan_frac]
    dropped_constant = [c for c in feature_cols if int(nunique.get(c, 0)) <= 1]

    keep = [c for c in feature_cols if c not in set(dropped_missing) and c not in set(dropped_constant)]
    if len(keep) == 0:
        # As a last resort, keep any non-constant feature.
        keep = [c for c in feature_cols if c not in set(dropped_constant)]

    kept_before_corr = list(keep)
    dropped_correlated: List[Dict[str, object]] = []

    # Correlation pruning (only if we have more than min_features)
    if corr_threshold > 0 and len(keep) > max(min_features, 1):
        X = feature_matrix[keep].copy()

        # Median impute for correlation computation
        for c in X.columns:
            med = X[c].median(skipna=True)
            if not np.isfinite(med):
                med = 0.0
            X[c] = X[c].fillna(med)

        corr = X.corr().abs()
        np.fill_diagonal(corr.values, 0.0)

        # Precompute scores for tie-breaking
        scores: Dict[str, float] = {c: _auc_effect(y, feature_matrix[c].to_numpy()) for c in keep}
        variances: Dict[str, float] = {c: float(np.nanvar(feature_matrix[c].to_numpy(dtype=float))) for c in keep}

        pairs: List[Tuple[float, str, str]] = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for j in range(i + 1, len(cols)):
                b = cols[j]
                v = float(corr.iat[i, j])
                if np.isfinite(v) and v >= corr_threshold:
                    pairs.append((v, a, b))
        pairs.sort(reverse=True)

        keep_set = set(keep)
        for v, a, b in pairs:
            if len(keep_set) <= min_features:
                break
            if a not in keep_set or b not in keep_set:
                continue
            a_score = float(scores.get(a, 0.5))
            b_score = float(scores.get(b, 0.5))
            if a_score > b_score:
                drop, keep_feat = b, a
            elif b_score > a_score:
                drop, keep_feat = a, b
            else:
                # tie-break: prefer lower missingness, then higher variance
                a_nan = float(nan_frac.get(a, 0.0))
                b_nan = float(nan_frac.get(b, 0.0))
                if a_nan != b_nan:
                    drop, keep_feat = (a, b) if a_nan > b_nan else (b, a)
                else:
                    a_var = float(variances.get(a, 0.0))
                    b_var = float(variances.get(b, 0.0))
                    drop, keep_feat = (a, b) if a_var < b_var else (b, a)

            if drop in keep_set and len(keep_set) > min_features:
                keep_set.remove(drop)
                dropped_correlated.append({"drop": drop, "keep": keep_feat, "corr": float(v)})

        keep = [c for c in feature_cols if c in keep_set]

    # Final guard: ensure we don't end up with 0 features
    if not keep:
        keep = kept_before_corr[:1]

    selected = feature_matrix[keep].copy()

    report: Dict[str, object] = {
        "enabled": True,
        "corr_threshold": corr_threshold,
        "max_nan_frac": max_nan_frac,
        "min_features": min_features,
        "n_features_in": len(feature_cols),
        "n_features_kept": len(keep),
        "dropped_missing": dropped_missing,
        "dropped_constant": dropped_constant,
        "dropped_correlated": dropped_correlated,
        "selected_features": keep,
    }
    return selected, keep, report


def main(input_file, covariate_dir, output_dir, species_name, thin=False, tune=False, iucn_path=None):
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine IUCN range dataset to use
    iucn_candidate = None
    if iucn_path:
        iucn_candidate = Path(iucn_path)
    else:
        if DEFAULT_IUCN_VECTOR.exists():
            iucn_candidate = DEFAULT_IUCN_VECTOR

    species_range_gdf = None
    species_range_geom_native = None
    species_range_digest = None
    if iucn_candidate is not None:
        try:
            species_range_gdf = load_species_range(iucn_candidate, species_name, buffer_deg=IUCN_BUFFER_DEGREES)
            species_range_geom_native = species_range_gdf.geometry.iloc[0]
            species_range_digest = hashlib.sha1(species_range_geom_native.wkb).hexdigest()[:16]
            print(f"[INFO] Loaded IUCN range for {species_name} from {iucn_candidate} with {IUCN_BUFFER_DEGREES}° buffer.")
        except FileNotFoundError:
            print(f"[WARN] IUCN range file not found at {iucn_candidate}; proceeding without range constraint.")
        except ValueError as exc:
            print(f"[WARN] {exc}; proceeding without range constraint.")
        except Exception as exc:
            print(f"[WARN] Failed to load IUCN range from {iucn_candidate}: {exc}; proceeding without range constraint.")

    # Load data and rasters
    data = gpd.read_file(input_file)
    raster_files = load_rasters(covariate_dir)
    print(f"There are {len(raster_files)} raster files in the folder")

    # Valid presences & thin
    pres_all = data[data.geometry.notnull() & data.geometry.is_valid].copy()
    if pres_all.empty:
        raise ValueError("No valid presence points found.")

    species_range_geom_pres = None
    if species_range_gdf is not None:
        if pres_all.crs is None:
            pres_all = pres_all.set_crs(species_range_gdf.crs)
        else:
            pres_all = pres_all.to_crs(species_range_gdf.crs)
        species_range_geom_pres = species_range_gdf.geometry.iloc[0]
        covers_mask = pres_all.geometry.apply(species_range_geom_pres.covers)
        removed = int(len(pres_all) - covers_mask.sum())
        if removed > 0:
            print(f"[INFO] Removed {removed} presences outside buffered IUCN range; kept {covers_mask.sum()}.")
        pres_all = pres_all.loc[covers_mask].copy()
        if pres_all.empty:
            raise ValueError("All presence points fell outside the buffered IUCN range.")

    if thin:
        # Adaptive thinning: avoid collapsing small datasets to too few points.
        # Controls:
        #   SDM_THIN_DISTANCE_M (float, meters)
        #   SDM_THIN_MIN_PRES (int)
        #   SDM_THIN_MIN_FRAC (float in (0,1], optional)
        #   SDM_THIN_MIN_DISTANCE_M (float, meters)
        #   SDM_THIN_ADAPT_ITERS (int)
        try:
            thin_dist_m = float(os.environ.get("SDM_THIN_DISTANCE_M", str(THIN_DISTANCE_M)))
        except ValueError:
            thin_dist_m = float(THIN_DISTANCE_M)
        try:
            thin_min_pres = int(os.environ.get("SDM_THIN_MIN_PRES", str(DEFAULT_THIN_MIN_PRES)))
        except ValueError:
            thin_min_pres = DEFAULT_THIN_MIN_PRES
        try:
            thin_min_frac = float(os.environ.get("SDM_THIN_MIN_FRAC", "0") or 0.0)
        except ValueError:
            thin_min_frac = 0.0
        try:
            thin_min_dist_m = float(os.environ.get("SDM_THIN_MIN_DISTANCE_M", str(DEFAULT_THIN_MIN_DISTANCE_M)))
        except ValueError:
            thin_min_dist_m = float(DEFAULT_THIN_MIN_DISTANCE_M)
        try:
            thin_adapt_iters = int(os.environ.get("SDM_THIN_ADAPT_ITERS", str(DEFAULT_THIN_ADAPT_ITERS)))
        except ValueError:
            thin_adapt_iters = DEFAULT_THIN_ADAPT_ITERS

        n0 = len(pres_all)
        min_keep = min(
            n0,
            max(thin_min_pres, int(np.ceil(thin_min_frac * n0)) if thin_min_frac > 0 else 0),
        )
        min_keep = max(1, int(min_keep))

        if n0 <= min_keep:
            pres = pres_all
            print(f"[INFO] Skipping thinning (n_pres={n0} <= min_keep={min_keep}).")
        else:
            dist = max(thin_min_dist_m, thin_dist_m)
            pres_thin = thin_points_distance(pres_all, dist)
            # If too aggressive, progressively reduce distance until we keep enough points.
            if len(pres_thin) < min_keep and thin_adapt_iters > 0:
                best = pres_thin
                best_dist = dist
                for _ in range(thin_adapt_iters):
                    if dist <= thin_min_dist_m:
                        break
                    dist = max(thin_min_dist_m, dist * 0.7)
                    cand = thin_points_distance(pres_all, dist)
                    best, best_dist = cand, dist
                    if len(cand) >= min_keep:
                        break
                pres_thin = best
                dist = best_dist

            if len(pres_thin) < min_keep:
                pres = pres_all
                print(
                    f"[WARN] Thinning would keep {len(pres_thin)}/{n0} (< min_keep={min_keep}); using unthinned presences."
                )
            else:
                pres = pres_thin
                print(f"[INFO] Thinned presences: {len(pres)} (from {n0}) at {dist/1000:.1f} km")
    else:
        pres = pres_all
        print(f"Presences not thinned: {len(pres)}")

    if species_range_gdf is not None:
        if pres.crs != species_range_gdf.crs:
            species_range_geom_pres = species_range_gdf.to_crs(pres.crs).geometry.iloc[0]
        else:
            species_range_geom_pres = species_range_gdf.geometry.iloc[0]

    # ---- Region for pseudo-absences from bounds (robust, no union_all) ----
    pres_pts = pres.loc[
        pres.geometry.notnull()
        & pres.geometry.is_valid
        & ~pres.geometry.is_empty
        & (pres.geometry.geom_type == "Point")
    ].copy()
    if pres_pts.empty:
        raise ValueError("No valid presence points after filtering.")

    minx, miny, maxx, maxy = pres_pts.total_bounds
    env = box(minx, miny, maxx, maxy)

    if species_range_geom_pres is not None:
        region_poly = species_range_geom_pres
        print(f"[INFO] Using buffered IUCN range to bound pseudo-absence sampling.")
    elif pres_pts.crs and pres_pts.crs.to_epsg() == 4326:
        region_poly = env.buffer(0.5)  # ~0.5° pad
    else:
        env_gdf = gpd.GeoDataFrame(geometry=[env], crs=pres_pts.crs)
        env_m = to_local_meters(env_gdf).geometry.iloc[0]
        region_poly = gpd.GeoSeries([env_m.buffer(50_000)],
                                    crs=to_local_meters(pres_pts).crs).to_crs(pres_pts.crs).iloc[0]

    # Raster coverage mask (same CRS)
    raster_mask = make_valid_bbox_mask(raster_files, pres_pts.crs)

    # Tolerant sampling domain with fallbacks
    domain = region_poly.intersection(raster_mask)
    if domain.is_empty:
        domain = raster_mask
    if domain.is_empty:
        if pres_pts.crs and pres_pts.crs.to_epsg() == 4326:
            domain = raster_mask.buffer(0.5)
        else:
            mask_gdf = gpd.GeoDataFrame(geometry=[raster_mask], crs=pres_pts.crs)
            mask_m = to_local_meters(mask_gdf).geometry.iloc[0]
            domain = gpd.GeoSeries([mask_m.buffer(50_000)],
                                   crs=to_local_meters(pres_pts).crs).to_crs(pres_pts.crs).iloc[0]
        if domain.is_empty:
            raise ValueError("Sampling domain is empty after fallbacks (check CRS & extents).")

    # Pseudo-absences strictly inside the domain
    n_abs = len(pres_pts)
    min_valid_abs_frac = float(os.environ.get("SDM_ABSENCE_MIN_VALID_FRAC", "0") or 0.0)
    if min_valid_abs_frac > 0:
        try:
            pseudo_abs, X_abs = sample_pseudo_absences_with_valid_covariates(
                n_abs,
                domain,
                pres_pts.crs,
                raster_files,
                min_valid_abs_frac,
            )
            print(f"[INFO] Sampled pseudo-absences with >= {min_valid_abs_frac:.2f} valid covariates.")
        except Exception as exc:
            print(f"[WARN] Validated pseudo-absence sampling failed ({exc}); falling back to unvalidated sampling.")
            pseudo_abs = sample_pseudo_absences(n_abs, domain, domain, crs=pres_pts.crs)
            X_abs = None
    else:
        pseudo_abs = sample_pseudo_absences(n_abs, domain, domain, crs=pres_pts.crs)
        X_abs = None

    # Feature sampling cache signature
    bbox = np.array(pres.total_bounds)
    crs_epsg = pres.crs.to_epsg() if pres.crs is not None else 0
    sig = _hash_signature(raster_files, bbox, species_name, crs_epsg, len(pres), range_digest=species_range_digest)
    bust_cache = os.environ.get("SDM_BUST_CACHE", "0") == "1"
    pres_cache = cache_dir / f"features_pres_{sig}.parquet"
    abs_cache  = cache_dir / f"features_abs_{sig}.parquet"

    def _looks_ok(df: pd.DataFrame, min_rows=10, max_nan_frac=0.5):
        if df is None or df.empty or len(df) < min_rows:
            return False
        feat_df = df.select_dtypes(include=[np.number])
        if feat_df.empty:
            return False
        # Guard against nodata sentinels (e.g., -3.4e38 from WorldClim) leaking into caches.
        try:
            min_val = float(feat_df.min(skipna=True).min())
            max_val = float(feat_df.max(skipna=True).max())
            if np.isfinite(min_val) and min_val < -1e30:
                return False
            if np.isfinite(max_val) and max_val > 1e30:
                return False
        except Exception:
            return False
        nan_frac = feat_df.isna().mean().mean()
        return nan_frac <= max_nan_frac

    if (pres_cache.exists() and abs_cache.exists() and not bust_cache):
        X_pres = pd.read_parquet(pres_cache)
        X_abs  = pd.read_parquet(abs_cache)
        if not (_looks_ok(X_pres) and _looks_ok(X_abs) and (X_pres.shape[1] == X_abs.shape[1])):
            print("Cache failed integrity check — resampling features.")
            X_pres = sample_rasters_at_points(pres, raster_files)
            X_abs  = sample_rasters_at_points(pseudo_abs, raster_files)
            X_pres.to_parquet(pres_cache, index=True)
            X_abs.to_parquet(abs_cache, index=True)
            print("Sampled features and cached.")
        else:
            print("Loaded cached features.")
    else:
        X_pres = sample_rasters_at_points(pres, raster_files)
        if X_abs is None:
            X_abs = sample_rasters_at_points(pseudo_abs, raster_files)
        X_pres.to_parquet(pres_cache, index=True)
        X_abs.to_parquet(abs_cache, index=True)
        print("Sampled features and cached.")

    # Prepare combined dataset
    pres_df = pd.concat([X_pres, pd.Series(1, index=X_pres.index, name="presence")], axis=1)
    abs_df  = pd.concat([X_abs,  pd.Series(0, index=X_abs.index,  name="presence")], axis=1)
    combined = pd.concat([pres_df, abs_df], axis=0, ignore_index=True)

    # Geometry for spatial CV (kept for grouping)
    pres_coords = pres.geometry.reset_index(drop=True)
    abs_coords  = pseudo_abs.geometry.reset_index(drop=True)
    combined_geom = gpd.GeoDataFrame(geometry=pd.concat([pres_coords, abs_coords], ignore_index=True), crs=pres.crs)

    # Clean feature matrix: drop columns/rows that are entirely NaN/inf
    feature_cols = [c for c in combined.columns if c != "presence"]
    feature_matrix = combined[feature_cols].apply(pd.to_numeric, errors="coerce")
    feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)

    col_keep_mask = feature_matrix.notna().any(axis=0)
    if not col_keep_mask.all():
        dropped_cols = [feature_cols[i] for i, keep in enumerate(col_keep_mask) if not keep]
        print(f"[WARN] Dropping {len(dropped_cols)} feature columns with no valid observations: {dropped_cols}")
    feature_matrix = feature_matrix.loc[:, col_keep_mask]
    feature_cols = feature_matrix.columns.tolist()

    row_keep_mask = feature_matrix.notna().any(axis=1)
    if not row_keep_mask.all():
        dropped_rows = (~row_keep_mask).sum()
        print(f"[WARN] Dropping {dropped_rows} samples with no valid covariate values.")
    feature_matrix = feature_matrix.loc[row_keep_mask].reset_index(drop=True)
    combined = combined.loc[row_keep_mask].reset_index(drop=True)
    combined_geom = combined_geom.loc[row_keep_mask].reset_index(drop=True)
    corr_feature_matrix = feature_matrix.copy()
    corr_y = combined["presence"].to_numpy(dtype=int)

    if combined["presence"].sum() == 0 or combined["presence"].sum() == len(combined):
        raise ValueError("Class imbalance after filtering removed all presences or pseudo-absences.")

    # Optional per-species feature selection
    feature_selection_report: Dict[str, object] = {"enabled": False}
    if os.environ.get("SDM_FEATURE_SELECT", "1") != "0":
        corr_thr = float(os.environ.get("SDM_FEATURE_CORR_THRESHOLD", str(DEFAULT_FEATURE_CORR_THRESHOLD)))
        max_nan = float(os.environ.get("SDM_FEATURE_MAX_NAN_FRAC", str(DEFAULT_FEATURE_MAX_NAN_FRAC)))
        min_feats = int(os.environ.get("SDM_FEATURE_MIN_FEATURES", str(DEFAULT_FEATURE_MIN_FEATURES)))
        try:
            feature_matrix, feature_cols, feature_selection_report = select_feature_subset(
                feature_matrix,
                combined["presence"].to_numpy(dtype=int),
                corr_threshold=corr_thr,
                max_nan_frac=max_nan,
                min_features=min_feats,
            )
            print(f"[INFO] Feature selection kept {len(feature_cols)}/{len([c for c in combined.columns if c != 'presence'])} covariates.")
            if feature_selection_report.get("dropped_correlated"):
                dropped_n = len(feature_selection_report["dropped_correlated"])  # type: ignore[index]
                print(f"[INFO] Dropped {dropped_n} covariates due to correlation >= {corr_thr:.2f}.")
        except Exception as exc:
            print(f"[WARN] Feature selection failed ({exc}); proceeding with all covariates.")
            feature_selection_report = {"enabled": False, "error": str(exc)}

    # Train matrices (single build; use NumPy for consistency)
    train_xs = feature_matrix.to_numpy(dtype="float32", copy=False)
    train_y  = combined["presence"].to_numpy(dtype=int)

    # CV with spatial groups
    cv, groups = make_spatial_cv(CV_FOLDS, combined_geom, SPATIAL_BLOCK_KM, y=train_y)

    # Models (quiet)
    n_jobs = get_cpus_per_task(default=1)
    xgb_base = XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        tree_method='hist', n_jobs=n_jobs, random_state=RNG_SEED, verbosity=0
    )
    lgbm_base = LGBMClassifier(class_weight='balanced', n_jobs=n_jobs, random_state=RNG_SEED, verbosity=-1)

    def cv_score(estimator, X, y, cv, groups, scorer_name):
        # Keep folds serial to avoid duplicating models/data in memory; models use threads via n_jobs.
        y_prob = cross_val_predict(estimator, X, y, cv=cv, groups=groups, method='predict_proba', n_jobs=1)[:, 1]
        if scorer_name == 'roc_auc':
            return roc_auc_score(y, y_prob), y_prob
        else:
            return -brier_score_loss(y, y_prob), y_prob

    def fold_class_health(cv, y, groups):
        """Return (train_ok, test_ok) booleans for class presence in each fold."""
        train_ok = True
        test_ok = True
        for train_idx, test_idx in cv.split(np.zeros_like(y), y, groups=groups):
            if len(np.unique(y[train_idx])) < 2:
                train_ok = False
            if len(np.unique(y[test_idx])) < 2:
                test_ok = False
        return train_ok, test_ok

    cv_mode = "spatial"
    if cv is None:
        train_ok, test_ok = True, True
        cv_mode = "none"
    else:
        train_ok, test_ok = fold_class_health(cv, train_y, groups)

    if cv is not None and not train_ok:
        # Spatial grouping can yield single-class training folds for small/clustered ranges.
        # Fall back to standard stratified CV so model fitting doesn't crash (e.g., XGBoost base_score).
        from sklearn.model_selection import StratifiedKFold

        n_pos = int(train_y.sum())
        n_neg = int((train_y == 0).sum())
        n_splits = min(CV_FOLDS, len(train_y), n_pos, n_neg)
        if n_splits >= 2:
            print(f"[WARN] Spatial CV produced single-class training folds; falling back to StratifiedKFold(n_splits={n_splits}).")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
            groups = None
            cv_mode = "stratified"
            train_ok, test_ok = fold_class_health(cv, train_y, groups)
        else:
            print("[WARN] Too few samples per class for CV; fitting without cross-validation/calibration.")
            cv = None
            groups = None
            cv_mode = "none"

    scorer = 'roc_auc'
    if cv is not None and not test_ok:
        scorer = 'neg_brier_score'
        print("[WARN] Some CV test folds are single-class; using 'neg_brier_score' for tuning/metrics.")
    best_oof_probs = None
    best_cv = None
    y_prob_xgb = None
    y_prob_lgb = None
    ensemble_members: List[Tuple[str, BaseEstimator]] = []

    # ---- tuning vs no-tuning ----
    if tune and cv is None:
        print("[WARN] Tuning requested but CV is disabled; proceeding without tuning.")
        tune = False
    if tune:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer

        xgb_search_space = {
            'model__n_estimators': Integer(300, 1600),
            'model__learning_rate': Real(1e-3, 0.2, prior='log-uniform'),
            'model__max_depth': Integer(3, 9),
            'model__subsample': Real(0.5, 1.0),
            'model__colsample_bytree': Real(0.5, 1.0),
            'model__reg_lambda': Real(1e-3, 10.0, prior='log-uniform'),
        }
        lgbm_search_space = {
            'model__n_estimators': Integer(400, 2000),
            'model__learning_rate': Real(1e-3, 0.2, prior='log-uniform'),
            'model__num_leaves': Integer(15, 255),
            'model__subsample': Real(0.5, 1.0),
            'model__colsample_bytree': Real(0.5, 1.0),
            'model__min_child_samples': Integer(5, 200),
            'model__reg_lambda': Real(1e-3, 10.0, prior='log-uniform'),
        }

        def bayes_tune_pipeline(base_model, search_space, X, y, groups, cv, scoring='roc_auc',
                                n_iter=40, n_points=1, n_jobs=1, random_state=RNG_SEED):
            pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', base_model)])
            opt = BayesSearchCV(
                estimator=pipe,
                search_spaces=search_space,
                n_iter=n_iter,
                n_points=n_points,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                refit=True,
                random_state=random_state,
                verbose=0,
                error_score='raise'
            )
            opt.fit(X, y, groups=groups)
            return opt

        with parallel_backend('threading'):
            xgb_opt = bayes_tune_pipeline(xgb_base, xgb_search_space, train_xs, train_y, groups, cv, scoring=scorer)
            lgbm_opt = bayes_tune_pipeline(lgbm_base, lgbm_search_space, train_xs, train_y, groups, cv, scoring=scorer)

        with parallel_backend('threading'):
            y_prob_xgb = cross_val_predict(
                xgb_opt.best_estimator_,
                train_xs,
                train_y,
                cv=cv,
                groups=groups,
                method='predict_proba',
                n_jobs=1
            )[:, 1]
            print_metrics(train_y, y_prob_xgb, label="xgb tuned (spatial CV)")

            y_prob_lgb = cross_val_predict(
                lgbm_opt.best_estimator_,
                train_xs,
                train_y,
                cv=cv,
                groups=groups,
                method='predict_proba',
                n_jobs=1
            )[:, 1]
            print_metrics(train_y, y_prob_lgb, label="lgbm tuned (spatial CV)")

        ensemble_members = [
            ("xgb", xgb_opt.best_estimator_.named_steps['model']),
            ("lgbm", lgbm_opt.best_estimator_.named_steps['model'])
        ]

    else:
        xgb_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', xgb_base)])
        lgbm_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', lgbm_base)])

        if cv is None:
            xgb_pipe.fit(train_xs, train_y)
            y_prob_xgb = xgb_pipe.predict_proba(train_xs)[:, 1]
            print_metrics(train_y, y_prob_xgb, label="xgb default (in-sample)")
        else:
            xgb_score, y_prob_xgb = cv_score(xgb_pipe, train_xs, train_y, cv, groups, scorer)
            print_metrics(train_y, y_prob_xgb, label=f"xgb default ({cv_mode} CV)")

        if cv is None:
            lgbm_pipe.fit(train_xs, train_y)
            y_prob_lgb = lgbm_pipe.predict_proba(train_xs)[:, 1]
            print_metrics(train_y, y_prob_lgb, label="lgbm default (in-sample)")
        else:
            lgbm_score, y_prob_lgb = cv_score(lgbm_pipe, train_xs, train_y, cv, groups, scorer)
            print_metrics(train_y, y_prob_lgb, label=f"lgbm default ({cv_mode} CV)")
        ensemble_members = [("xgb", xgb_base), ("lgbm", lgbm_base)]

    if y_prob_xgb is None or y_prob_lgb is None:
        raise RuntimeError("Failed to compute out-of-fold predictions for ensemble members.")

    ensemble_probs = np.clip((y_prob_xgb + y_prob_lgb) / 2.0, 0.0, 1.0)
    print_metrics(train_y, ensemble_probs, label="ensemble (spatial CV)")
    if scorer == 'neg_brier_score':
        best_cv = -brier_score_loss(train_y, ensemble_probs)
        print(f"[ensemble] CV Brier={-best_cv:.4f}")
    else:
        best_cv = roc_auc_score(train_y, ensemble_probs)
        print(f"[ensemble] CV ROC-AUC={best_cv:.3f}")

    best_oof_probs = ensemble_probs
    winner_name = "ensemble"
    ensemble_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', AveragingEnsemble(ensemble_members))
    ])
    ensemble_pipeline.fit(train_xs, train_y)
    best_pipeline = ensemble_pipeline

    feature_importance_df = None
    model_step = None
    if hasattr(best_pipeline, "named_steps"):
        model_step = best_pipeline.named_steps.get('model')  # type: ignore[attr-defined]
    # Feature importances: for ensembles, average member importances when available.
    importances_list = []
    if model_step is not None and hasattr(model_step, "fitted_estimators_"):
        for member_name, member in getattr(model_step, "fitted_estimators_", []):
            if hasattr(member, "feature_importances_"):
                imp = np.asarray(getattr(member, "feature_importances_", None), dtype=float)
                if imp.shape[0] == len(feature_cols):
                    importances_list.append(imp)
    if importances_list:
        importances = np.mean(np.stack(importances_list, axis=0), axis=0)
        feature_importance_df = (
            pd.DataFrame({"feature": feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    else:
        print(f"[WARN] Feature importances unavailable for {winner_name}; skipping export.")

    # --- Probability calibration ---
    calibrator = IdentityCalibrator()
    if best_oof_probs is None:
        print("[WARN] Skipping probability calibration because no out-of-fold predictions were collected.")
    else:
        try:
            calibrator = build_probability_calibrator(train_y, best_oof_probs)
            raw_brier = brier_score_loss(train_y, np.clip(best_oof_probs, 0.0, 1.0))
            calibrated_probs = np.clip(calibrator.predict(best_oof_probs), 0.0, 1.0)
            cal_brier = brier_score_loss(train_y, calibrated_probs)
            print(f"[INFO] Probability calibration Brier: raw={raw_brier:.4f} -> calibrated={cal_brier:.4f}")
        except Exception as exc:
            print(f"[WARN] Probability calibration failed ({exc}); using raw probabilities.")
            calibrator = IdentityCalibrator()

    projection_estimator = ProbabilityCalibratedEstimator(best_pipeline, calibrator)

    # --- Make per-species output dir to avoid cross-task collisions ---
    safe_species = safe_filename(species_name)
    model_output_dir = Path(output_dir) / f"{winner_name}-images" / safe_species
    model_output_dir.mkdir(parents=True, exist_ok=True)
    spearman_path = model_output_dir / f"spearman_correlations_{safe_species}.csv"
    write_species_spearman_outputs(
        corr_feature_matrix,
        corr_y,
        species_name=species_name,
        out_csv=spearman_path,
    )

    # Persist trained models in repo-local Models/ (not scratch), so they survive scratch expiry.
    models_root = Path(__file__).resolve().parents[1] / "Models"
    models_dir = models_root / winner_name / safe_species
    models_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path = models_dir / f"{safe_species}_model.joblib"
    model_metadata_path = models_dir / f"{safe_species}_model.json"
    interpreted_cv = None
    if best_cv is not None:
        interpreted_cv = float(-best_cv if scorer == 'neg_brier_score' else best_cv)
    model_bundle = {
        "pipeline": best_pipeline,
        "calibrator": calibrator,
        "feature_names": feature_cols,
        "feature_selection": feature_selection_report,
        "winner": winner_name,
        "scorer": scorer,
        "cv_score_raw": float(best_cv) if best_cv is not None else None,
        "cv_score_interpreted": interpreted_cv,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        dump(model_bundle, model_artifact_path)
        print(f"[INFO] saved model artifact: {model_artifact_path}")
    except Exception as exc:
        print(f"[WARN] Failed to save model artifact ({exc}).")

    metadata = {
        "winner": winner_name,
        "scorer": scorer,
        "cv_score_raw": float(best_cv) if best_cv is not None else None,
        "cv_score_interpreted": interpreted_cv,
        "feature_names": feature_cols,
        "feature_selection": feature_selection_report,
        "model_artifact": model_artifact_path.name,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rasters_outdir": str(model_output_dir),
    }
    try:
        with model_metadata_path.open("w", encoding="utf-8") as meta_fp:
            json.dump(metadata, meta_fp, indent=2)
        print(f"[INFO] wrote model metadata: {model_metadata_path}")
    except Exception as exc:
        print(f"[WARN] Failed to write model metadata ({exc}).")

    analysis_dir = model_output_dir / "analysis"
    pd_artifacts: Dict[str, Dict[str, str]] = {}
    do_analysis = os.environ.get("SDM_NO_ANALYSIS", "0") != "1"
    if feature_importance_df is not None and do_analysis:
        # Use imputed X for grid generation so PDPs don't get polluted by NaNs.
        X_for_pd = train_xs
        try:
            imputer = getattr(best_pipeline, "named_steps", {}).get("imputer")
            if imputer is not None and hasattr(imputer, "transform"):
                X_for_pd = imputer.transform(train_xs)
        except Exception:
            X_for_pd = train_xs

        analysis_dir.mkdir(parents=True, exist_ok=True)
        pd_estimator = projection_estimator if os.environ.get("SDM_PDP_CALIBRATED", "1") == "1" else best_pipeline
        for feat_idx, feat_name in enumerate(feature_cols):
            safe_feat = safe_filename(feat_name)
            try:
                pd_res = partial_dependence(
                    pd_estimator,
                    X_for_pd,
                    features=[feat_idx],
                    grid_resolution=PD_GRID_RESOLUTION,
                )
            except Exception as exc:
                print(f"[WARN] Partial dependence failed for '{feat_name}': {exc}")
                continue
            if "values" in pd_res:
                values = pd_res["values"][0]
            elif "grid_values" in pd_res:
                values = pd_res["grid_values"][0]
            else:
                print(f"[WARN] Partial dependence missing grid values for '{feat_name}'; skipping.")
                continue
            averaged = pd_res["average"][0]
            pd_df = pd.DataFrame({"feature_value": values, "partial_dependence": averaged})
            csv_path = analysis_dir / f"partial_dependence_{safe_feat}.csv"
            try:
                pd_df.to_csv(csv_path, index=False)
            except Exception as exc:
                print(f"[WARN] Failed writing partial dependence CSV for '{feat_name}': {exc}")
                continue
            fig, ax = plt.subplots(figsize=(4.5, 3.0))
            ax.plot(values, averaged, color="#1f77b4")
            ax.set_xlabel(feat_name)
            ax.set_ylabel("Partial dependence (calibrated p)" if pd_estimator is projection_estimator else "Partial dependence")
            ax.set_title(f"{species_name} · {feat_name}")
            ax.grid(True, color="#e0e0e0", linewidth=0.5, linestyle='--', alpha=0.3)
            plot_path = analysis_dir / f"partial_dependence_{safe_feat}.png"
            try:
                fig.tight_layout()
                fig.savefig(plot_path, dpi=150)
            except Exception as exc:
                print(f"[WARN] Failed writing partial dependence plot for '{feat_name}': {exc}")
            finally:
                plt.close(fig)
            pd_artifacts[feat_name] = {
                "csv": str(Path("analysis") / csv_path.name),
                "plot": str(Path("analysis") / plot_path.name),
            }

        fi_path = model_output_dir / f"feature_importance_{safe_species}.csv"
        try:
            fi_to_write = feature_importance_df.copy()
            if pd_artifacts:
                fi_to_write["partial_dependence_csv"] = fi_to_write["feature"].map(lambda f: pd_artifacts.get(f, {}).get("csv", ""))
                fi_to_write["partial_dependence_plot"] = fi_to_write["feature"].map(lambda f: pd_artifacts.get(f, {}).get("plot", ""))
            fi_to_write.to_csv(fi_path, index=False)
            print(f"[INFO] wrote feature importances: {fi_path}")
        except Exception as exc:
            print(f"[WARN] Failed to write feature importances ({exc}).")
    elif feature_importance_df is not None:
        fi_path = model_output_dir / f"feature_importance_{safe_species}.csv"
        try:
            feature_importance_df.to_csv(fi_path, index=False)
            print(f"[INFO] wrote feature importances: {fi_path}")
        except Exception as exc:
            print(f"[WARN] Failed to write feature importances ({exc}).")

    # -------- Projection (windowed; avoids OOM on 30-arcsec globals) --------
    # Use only the rasters that correspond to the training feature columns, in the same order.
    name_to_path = {short_name(p): p for p in raster_files}
    missing = [f for f in feature_cols if f not in name_to_path]
    if missing:
        raise ValueError(f"Missing rasters for trained features: {missing}")
    proj_rasters = [name_to_path[f] for f in feature_cols]

    _align_raster_stack(proj_rasters)
    with rasterio.open(proj_rasters[0]) as ref_ds:
        project_mode = os.environ.get("SDM_PROJECT_MODE", "global")
        print(f"[INFO] Projection mode: {project_mode}")
        bounds = _select_projection_bounds(
            project_mode=project_mode,
            pres_pts=pres_pts,
            domain_geom=domain,
            species_range_geom_pres=species_range_geom_pres,
            raster_ds=ref_ds,
            project_buffer_deg=float(os.environ.get("SDM_PROJECT_BUFFER_DEG", "0") or 0.0),
        )
        base_window = _window_from_bounds_clipped(bounds, ref_ds)

    make_styled = os.environ.get("SDM_NO_STYLED", "0") != "1"
    tile_size = int(os.environ.get("SDM_TILE_SIZE", str(DEFAULT_TILE_SIZE)))
    outputs = project_probability_rasters(
        proj_rasters,
        projection_estimator,
        feature_cols,
        model_output_dir,
        safe_species,
        base_window=base_window,
        tile_size=tile_size,
        nodata=OUTPUT_NODATA,
        make_styled=make_styled,
    )

    print(f"[OK] wrote: {outputs['prob0']}")
    print(f"[OK] wrote: {outputs['prob1']}")
    print(f"[OK] wrote: {outputs['certainty']}")
    if "styled" in outputs:
        print(f"[INFO] wrote styled probability raster: {outputs['styled']}")


def write_partial_dependence_outputs(
    estimator,
    X_for_pd,
    feature_cols: Sequence[str],
    analysis_dir: Path,
    species_name: str,
    *,
    y_label: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    pd_artifacts: Dict[str, Dict[str, str]] = {}
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for feat_idx, feat_name in enumerate(feature_cols):
        safe_feat = safe_filename(feat_name)
        values = None
        averaged = None
        try:
            pd_res = partial_dependence(
                estimator,
                X_for_pd,
                features=[feat_idx],
                grid_resolution=PD_GRID_RESOLUTION,
            )
            if "values" in pd_res:
                values = pd_res["values"][0]
            elif "grid_values" in pd_res:
                values = pd_res["grid_values"][0]
            else:
                raise ValueError("Partial dependence missing grid values.")
            averaged = pd_res["average"][0]
        except Exception as exc:
            msg = str(exc)
            if "must be a fitted regressor or classifier" in msg:
                print(f"[WARN] Sklearn PDP rejected estimator for '{feat_name}'; using manual PDP fallback.")
                try:
                    X_work = np.asarray(X_for_pd, dtype="float32", order="C").copy()
                    col = X_work[:, feat_idx]
                    finite_mask = np.isfinite(col)
                    if not finite_mask.any():
                        raise ValueError("No finite values for feature column.")
                    p5, p95 = np.nanpercentile(col[finite_mask], [5, 95])
                    if not np.isfinite(p5) or not np.isfinite(p95):
                        raise ValueError("Invalid percentile bounds for feature column.")
                    if p5 == p95:
                        values = np.array([p5], dtype="float32")
                    else:
                        values = np.linspace(p5, p95, PD_GRID_RESOLUTION, dtype="float32")
                    original = X_work[:, feat_idx].copy()
                    averaged_list = []
                    for v in values:
                        X_work[:, feat_idx] = v
                        probs = estimator.predict_proba(X_work)[:, 1]
                        averaged_list.append(float(np.nanmean(probs)))
                    averaged = np.array(averaged_list, dtype="float32")
                    X_work[:, feat_idx] = original
                except Exception as inner_exc:
                    print(f"[WARN] Manual PDP failed for '{feat_name}': {inner_exc}")
                    continue
            else:
                print(f"[WARN] Partial dependence failed for '{feat_name}': {exc}")
                continue
        if values is None or averaged is None:
            print(f"[WARN] Partial dependence missing results for '{feat_name}'; skipping.")
            continue
        pd_df = pd.DataFrame({"feature_value": values, "partial_dependence": averaged})
        csv_path = analysis_dir / f"partial_dependence_{safe_feat}.csv"
        try:
            pd_df.to_csv(csv_path, index=False)
        except Exception as exc:
            print(f"[WARN] Failed writing partial dependence CSV for '{feat_name}': {exc}")
            continue
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        ax.plot(values, averaged, color="#1f77b4")
        ax.set_xlabel(feat_name)
        ax.set_ylabel(f"Partial dependence ({y_label})" if y_label else "Partial dependence")
        ax.set_title(f"{species_name} · {feat_name}")
        ax.grid(True, color="#e0e0e0", linewidth=0.5, linestyle='--', alpha=0.3)
        plot_path = analysis_dir / f"partial_dependence_{safe_feat}.png"
        try:
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
        except Exception as exc:
            print(f"[WARN] Failed writing partial dependence plot for '{feat_name}': {exc}")
        finally:
            plt.close(fig)
        pd_artifacts[feat_name] = {
            "csv": str(Path("analysis") / csv_path.name),
            "plot": str(Path("analysis") / plot_path.name),
        }
    return pd_artifacts


def write_species_spearman_outputs(
    feature_matrix: pd.DataFrame,
    y_presence: np.ndarray,
    species_name: str,
    out_csv: Path,
) -> Optional[pd.DataFrame]:
    """Write per-feature Spearman correlation against presence/absence for one species."""
    if feature_matrix is None or feature_matrix.empty:
        print("[WARN] Skipping Spearman export: empty feature matrix.")
        return None
    if y_presence is None or len(y_presence) != len(feature_matrix):
        print("[WARN] Skipping Spearman export: mismatched y and feature matrix lengths.")
        return None

    rows: List[Dict[str, object]] = []
    y_vec = pd.Series(y_presence).astype(float).to_numpy()
    for feat in feature_matrix.columns:
        x = pd.to_numeric(feature_matrix[feat], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y_vec)
        n_valid = int(valid.sum())
        if n_valid < 3:
            rows.append(
                {
                    "species": species_name,
                    "feature": feat,
                    "spearman_rho": np.nan,
                    "p_value": np.nan,
                    "n_valid": n_valid,
                    "n_present": int(np.sum(y_vec[valid] == 1.0)),
                    "n_absent": int(np.sum(y_vec[valid] == 0.0)),
                }
            )
            continue
        try:
            rho, p = spearmanr(x[valid], y_vec[valid], nan_policy="omit")
        except Exception:
            rho, p = np.nan, np.nan
        rows.append(
            {
                "species": species_name,
                "feature": feat,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "n_valid": n_valid,
                "n_present": int(np.sum(y_vec[valid] == 1.0)),
                "n_absent": int(np.sum(y_vec[valid] == 0.0)),
            }
        )

    if not rows:
        print("[WARN] Spearman export produced no rows.")
        return None
    out_df = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] wrote Spearman correlations: {out_csv}")
    return out_df


def pdp_from_saved_model(
    input_file: str,
    covariate_dir: str,
    output_dir: str,
    species_name: str,
    *,
    model_artifact: Optional[str] = None,
    iucn_path: Optional[str] = None,
    thin: bool = False,
):
    """
    Generate PDP CSV/PNGs using an already-trained model artifact (no training, no projection).

    Requires the original presence GeoJSON + covariate rasters to rebuild (or reuse cached)
    the feature matrix. This will typically hit the existing feature parquet cache under
    `<output_dir>/cache/` and will not refit any model.
    """
    # Load model bundle (trained pipeline + calibrator + feature list)
    safe_species = safe_filename(species_name)
    models_root = Path(__file__).resolve().parents[1] / "Models" / "ensemble" / safe_species
    default_artifact = models_root / f"{safe_species}_model.joblib"
    model_path = Path(model_artifact) if model_artifact else default_artifact
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    model_bundle = load(model_path)
    best_pipeline = model_bundle.get("pipeline")
    calibrator = model_bundle.get("calibrator", IdentityCalibrator())
    trained_features = list(model_bundle.get("feature_names") or [])
    if best_pipeline is None or not trained_features:
        raise ValueError("Model bundle missing required keys: 'pipeline' and/or 'feature_names'.")

    projection_estimator = ProbabilityCalibratedEstimator(best_pipeline, calibrator)
    projection_estimator._force_fitted = True
    projection_estimator.fitted_ = True
    pd_estimator = projection_estimator if os.environ.get("SDM_PDP_CALIBRATED", "1") == "1" else best_pipeline
    pd_label = "calibrated p" if pd_estimator is projection_estimator else None

    # Determine where to write analysis outputs.
    # Default to current output_dir; optionally allow model metadata path if requested.
    meta_path = models_root / f"{safe_species}_model.json"
    model_output_dir = Path(output_dir) / "ensemble-images" / safe_species
    use_metadata_out = os.environ.get("SDM_PDP_USE_METADATA_OUT", "0") == "1"
    using_metadata_out = False
    if use_metadata_out and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rasters_outdir = meta.get("rasters_outdir")
            if rasters_outdir:
                candidate = Path(rasters_outdir)
                try:
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    model_output_dir = candidate
                    using_metadata_out = True
                except Exception as exc:
                    print(f"[WARN] Ignoring rasters_outdir from metadata (unwritable: {candidate}): {exc}")
        except Exception as exc:
            print(f"[WARN] Failed reading model metadata outdir ({exc}); using output_dir.")
    analysis_dir = model_output_dir / "analysis"
    print(f"[INFO] PDP output directory: {model_output_dir} (source={'metadata' if using_metadata_out else 'output_dir'})")

    # Rebuild feature matrix (cached) from the original inputs/rasters
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    iucn_candidate = Path(iucn_path) if iucn_path else (DEFAULT_IUCN_VECTOR if DEFAULT_IUCN_VECTOR.exists() else None)
    species_range_gdf = None
    species_range_digest = None
    if iucn_candidate is not None:
        try:
            species_range_gdf = load_species_range(iucn_candidate, species_name, buffer_deg=IUCN_BUFFER_DEGREES)
            species_range_geom_native = species_range_gdf.geometry.iloc[0]
            species_range_digest = hashlib.sha1(species_range_geom_native.wkb).hexdigest()[:16]
        except Exception:
            species_range_gdf = None
            species_range_digest = None

    data = gpd.read_file(input_file)
    raster_files = load_rasters(covariate_dir)

    pres_all = data[data.geometry.notnull() & data.geometry.is_valid].copy()
    if pres_all.empty:
        raise ValueError("No valid presence points found.")

    species_range_geom_pres = None
    if species_range_gdf is not None:
        if pres_all.crs is None:
            pres_all = pres_all.set_crs(species_range_gdf.crs)
        else:
            pres_all = pres_all.to_crs(species_range_gdf.crs)
        species_range_geom_pres = species_range_gdf.geometry.iloc[0]
        covers_mask = pres_all.geometry.apply(species_range_geom_pres.covers)
        pres_all = pres_all.loc[covers_mask].copy()
        if pres_all.empty:
            raise ValueError("All presence points fell outside the buffered IUCN range.")

    pres = pres_all
    if thin:
        try:
            thin_dist_m = float(os.environ.get("SDM_THIN_DISTANCE_M", str(THIN_DISTANCE_M)))
        except ValueError:
            thin_dist_m = float(THIN_DISTANCE_M)
        try:
            thin_min_pres = int(os.environ.get("SDM_THIN_MIN_PRES", str(DEFAULT_THIN_MIN_PRES)))
        except ValueError:
            thin_min_pres = DEFAULT_THIN_MIN_PRES
        try:
            thin_min_frac = float(os.environ.get("SDM_THIN_MIN_FRAC", "0") or 0.0)
        except ValueError:
            thin_min_frac = 0.0
        try:
            thin_min_dist_m = float(os.environ.get("SDM_THIN_MIN_DISTANCE_M", str(DEFAULT_THIN_MIN_DISTANCE_M)))
        except ValueError:
            thin_min_dist_m = float(DEFAULT_THIN_MIN_DISTANCE_M)
        try:
            thin_adapt_iters = int(os.environ.get("SDM_THIN_ADAPT_ITERS", str(DEFAULT_THIN_ADAPT_ITERS)))
        except ValueError:
            thin_adapt_iters = DEFAULT_THIN_ADAPT_ITERS

        n0 = len(pres_all)
        min_keep = min(
            n0,
            max(thin_min_pres, int(np.ceil(thin_min_frac * n0)) if thin_min_frac > 0 else 0),
        )
        min_keep = max(1, int(min_keep))
        if n0 > min_keep:
            dist = max(thin_min_dist_m, thin_dist_m)
            pres_thin = thin_points_distance(pres_all, dist)
            if len(pres_thin) < min_keep and thin_adapt_iters > 0:
                best = pres_thin
                for _ in range(thin_adapt_iters):
                    if dist <= thin_min_dist_m:
                        break
                    dist = max(thin_min_dist_m, dist / 2.0)
                    cand = thin_points_distance(pres_all, dist)
                    if len(cand) > len(best):
                        best = cand
                    if len(cand) >= min_keep:
                        best = cand
                        break
                pres_thin = best
            if len(pres_thin) >= min_keep:
                pres = pres_thin

    if pres.empty:
        raise ValueError("No valid presence points after filtering.")

    pres_pts = pres
    env = box(*pres.total_bounds)
    region_poly = species_range_geom_pres if species_range_geom_pres is not None else env.buffer(0.5)

    raster_mask = make_valid_bbox_mask(raster_files, pres_pts.crs)
    domain = region_poly.intersection(raster_mask)
    if domain.is_empty:
        domain = raster_mask
    if domain.is_empty:
        raise ValueError("Sampling domain is empty after fallbacks (check CRS & extents).")

    n_abs = len(pres_pts)
    min_valid_abs_frac = float(os.environ.get("SDM_ABSENCE_MIN_VALID_FRAC", "0") or 0.0)
    X_abs = None
    if min_valid_abs_frac > 0:
        try:
            pseudo_abs, X_abs = sample_pseudo_absences_with_valid_covariates(
                n_abs,
                domain,
                pres_pts.crs,
                raster_files,
                min_valid_abs_frac,
            )
        except Exception:
            pseudo_abs = sample_pseudo_absences(n_abs, domain, domain, crs=pres_pts.crs)
            X_abs = None
    else:
        pseudo_abs = sample_pseudo_absences(n_abs, domain, domain, crs=pres_pts.crs)
        X_abs = None

    bbox = np.array(pres.total_bounds)
    crs_epsg = pres.crs.to_epsg() if pres.crs is not None else 0
    sig = _hash_signature(raster_files, bbox, species_name, crs_epsg, len(pres), range_digest=species_range_digest)
    bust_cache = os.environ.get("SDM_BUST_CACHE", "0") == "1"
    pres_cache = cache_dir / f"features_pres_{sig}.parquet"
    abs_cache = cache_dir / f"features_abs_{sig}.parquet"

    def _looks_ok(df: pd.DataFrame, min_rows=10, max_nan_frac=0.5):
        if df is None or df.empty or len(df) < min_rows:
            return False
        feat_df = df.select_dtypes(include=[np.number])
        if feat_df.empty:
            return False
        try:
            min_val = float(feat_df.min(skipna=True).min())
            max_val = float(feat_df.max(skipna=True).max())
            if np.isfinite(min_val) and min_val < -1e30:
                return False
            if np.isfinite(max_val) and max_val > 1e30:
                return False
        except Exception:
            return False
        nan_frac = feat_df.isna().mean().mean()
        return nan_frac <= max_nan_frac

    if (pres_cache.exists() and abs_cache.exists() and not bust_cache):
        X_pres = pd.read_parquet(pres_cache)
        X_abs_cache = pd.read_parquet(abs_cache)
        if not (_looks_ok(X_pres) and _looks_ok(X_abs_cache) and (X_pres.shape[1] == X_abs_cache.shape[1])):
            X_pres = sample_rasters_at_points(pres, raster_files)
            X_abs_cache = sample_rasters_at_points(pseudo_abs, raster_files)
            X_pres.to_parquet(pres_cache, index=True)
            X_abs_cache.to_parquet(abs_cache, index=True)
        X_abs = X_abs_cache
    else:
        X_pres = sample_rasters_at_points(pres, raster_files)
        if X_abs is None:
            X_abs = sample_rasters_at_points(pseudo_abs, raster_files)
        X_pres.to_parquet(pres_cache, index=True)
        X_abs.to_parquet(abs_cache, index=True)

    pres_df = pd.concat([X_pres, pd.Series(1, index=X_pres.index, name="presence")], axis=1)
    abs_df = pd.concat([X_abs, pd.Series(0, index=X_abs.index, name="presence")], axis=1)
    combined = pd.concat([pres_df, abs_df], axis=0, ignore_index=True)

    feature_cols = [c for c in combined.columns if c != "presence"]
    feature_matrix = combined[feature_cols].apply(pd.to_numeric, errors="coerce")
    feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    col_keep_mask = feature_matrix.notna().any(axis=0)
    feature_matrix = feature_matrix.loc[:, col_keep_mask]
    feature_cols = feature_matrix.columns.tolist()
    row_keep_mask = feature_matrix.notna().any(axis=1)
    feature_matrix = feature_matrix.loc[row_keep_mask].reset_index(drop=True)
    corr_y = combined.loc[row_keep_mask, "presence"].to_numpy(dtype=int)
    spearman_path = model_output_dir / f"spearman_correlations_{safe_species}.csv"
    write_species_spearman_outputs(
        feature_matrix,
        corr_y,
        species_name=species_name,
        out_csv=spearman_path,
    )

    missing = [f for f in trained_features if f not in feature_cols]
    if missing:
        raise ValueError(f"Trained features not present in sampled covariates: {missing}")
    feature_matrix = feature_matrix[trained_features]

    X_for_pd = feature_matrix.to_numpy(dtype="float32", copy=False)
    try:
        imputer = getattr(best_pipeline, "named_steps", {}).get("imputer")
        if imputer is not None and hasattr(imputer, "transform"):
            X_for_pd = imputer.transform(X_for_pd)
    except Exception:
        pass

    pd_artifacts = write_partial_dependence_outputs(
        estimator=pd_estimator,
        X_for_pd=X_for_pd,
        feature_cols=trained_features,
        analysis_dir=analysis_dir,
        species_name=species_name,
        y_label=pd_label,
    )

    fi_path = model_output_dir / f"feature_importance_{safe_species}.csv"
    if fi_path.exists() and pd_artifacts:
        try:
            fi_df = pd.read_csv(fi_path)
            if "feature" in fi_df.columns:
                fi_df["partial_dependence_csv"] = fi_df["feature"].map(lambda f: pd_artifacts.get(f, {}).get("csv", ""))
                fi_df["partial_dependence_plot"] = fi_df["feature"].map(lambda f: pd_artifacts.get(f, {}).get("plot", ""))
                fi_df.to_csv(fi_path, index=False)
        except Exception as exc:
            print(f"[WARN] Failed to augment feature importances with PDP paths ({exc}).")

    print(f"[OK] wrote PDPs to: {analysis_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process geospatial data for species distribution modeling.")
    parser.add_argument('input_file', type=str, help='Path to the GeoJSON file containing the species data.')
    parser.add_argument('covariate_dir', type=str, help='Directory containing covariate raster files.')
    parser.add_argument('output_dir', type=str, help='Output directory to store results and processed rasters.')
    parser.add_argument('species_name', type=str, help='Name of the species being modeled.')
    parser.add_argument('--thin', dest='thin', action='store_true', help='Apply spatial thinning to presences (default: off)')
    parser.add_argument('--tune', dest='tune', action='store_true', help='Enable Bayesian hyperparameter optimization (default: off)')
    parser.add_argument('--iucn-shp', dest='iucn_shp', type=str, default=None,
                        help='Path to IUCN range shapefile (defaults to repo copy if omitted).')
    parser.add_argument('--pdp-only', dest='pdp_only', action='store_true',
                        help='Generate partial dependence CSV/PNGs from a saved model artifact (no retraining, no projection).')
    parser.add_argument('--model-artifact', dest='model_artifact', type=str, default=None,
                        help='Path to saved model joblib (defaults to Models/ensemble/<species>/<species>_model.joblib).')
    args = parser.parse_args()

    if args.pdp_only:
        pdp_from_saved_model(
            args.input_file,
            args.covariate_dir,
            args.output_dir,
            args.species_name,
            model_artifact=args.model_artifact,
            iucn_path=args.iucn_shp,
            thin=args.thin,
        )
    else:
        main(
            args.input_file,
            args.covariate_dir,
            args.output_dir,
            args.species_name,
            thin=args.thin,
            tune=args.tune,
            iucn_path=args.iucn_shp,
        )
