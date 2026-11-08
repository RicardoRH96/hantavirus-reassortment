#!/usr/bin/env python3
import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds


def _collect_probability_rasters(ensemble_dir: Path) -> List[Path]:
    rasters = sorted(ensemble_dir.glob("*/probability_1_*.tif"))
    if not rasters:
        raise FileNotFoundError(f"No probability_1 rasters found under: {ensemble_dir}")
    return rasters


def _validate_probability_rasters(paths: List[Path]) -> None:
    bad = []
    for p in paths:
        try:
            with rasterio.open(p) as ds:
                _ = ds.width, ds.height, ds.crs, ds.transform
                read_any = False
                for _, win in ds.block_windows(1):
                    ds.read(1, window=win, masked=True)
                    read_any = True
                    break
                if not read_any:
                    ds.read(1, window=Window(0, 0, min(1, ds.width), min(1, ds.height)), masked=True)
        except Exception as exc:
            bad.append((p, exc))

    if bad:
        msg = "\n".join(f"  - {p}: {exc}" for p, exc in bad[:10])
        raise RuntimeError(
            "Found unreadable probability rasters. This usually means overlap started while "
            "reprojection jobs were still writing files.\n"
            f"{msg}"
        )


def _union_grid(paths: List[Path]) -> Tuple[Affine, int, int, str]:
    with rasterio.open(paths[0]) as ref:
        crs = ref.crs
        xres = ref.transform.a
        yres = ref.transform.e
        inv_ref = ~ref.transform

    min_col = None
    min_row = None
    max_col = None
    max_row = None
    for p in paths:
        with rasterio.open(p) as ds:
            if str(ds.crs) != str(crs):
                raise ValueError(f"CRS mismatch for {p}: {ds.crs} vs {crs}")
            if not np.isclose(ds.transform.a, xres) or not np.isclose(ds.transform.e, yres):
                raise ValueError(f"Resolution mismatch for {p}.")
            left, bottom, right, top = ds.bounds
            c0, r0 = inv_ref * (left, top)
            c1, r1 = inv_ref * (right, bottom)
            cmin = int(np.floor(min(c0, c1)))
            cmax = int(np.ceil(max(c0, c1)))
            rmin = int(np.floor(min(r0, r1)))
            rmax = int(np.ceil(max(r0, r1)))
            min_col = cmin if min_col is None else min(min_col, cmin)
            min_row = rmin if min_row is None else min(min_row, rmin)
            max_col = cmax if max_col is None else max(max_col, cmax)
            max_row = rmax if max_row is None else max(max_row, rmax)

    assert min_col is not None and min_row is not None and max_col is not None and max_row is not None
    width = max_col - min_col
    height = max_row - min_row
    out_transform = Affine(
        xres, 0.0, ref.transform.c + min_col * xres,
        0.0, yres, ref.transform.f + min_row * yres,
    )
    return out_transform, width, height, crs


def _rounded_window_for_bounds(bounds, transform, width, height) -> Window:
    win = from_bounds(*bounds, transform=transform)
    col_off = int(np.floor(win.col_off))
    row_off = int(np.floor(win.row_off))
    col_end = int(np.ceil(win.col_off + win.width))
    row_end = int(np.ceil(win.row_off + win.height))
    col_off = max(0, col_off)
    row_off = max(0, row_off)
    col_end = min(width, col_end)
    row_end = min(height, row_end)
    return Window(col_off=col_off, row_off=row_off, width=max(0, col_end - col_off), height=max(0, row_end - row_off))


def _apply_nodata_where_uncovered(target_ds, uncovered_mask_ds, nodata_value) -> None:
    for _, win in target_ds.block_windows(1):
        target = target_ds.read(1, window=win)
        uncovered = uncovered_mask_ds.read(1, window=win) == 1
        target[uncovered] = nodata_value
        target_ds.write(target, 1, window=win)


def _threshold_tag(threshold: float) -> str:
    return f"{threshold:g}".replace("-", "m").replace(".", "p")


def _stack_source_raster(
    src,
    sum_ds,
    count_ds,
    rich_datasets,
    out_transform,
    out_width,
    out_height,
) -> bool:
    base_out_win = _rounded_window_for_bounds(src.bounds, out_transform, out_width, out_height)
    if base_out_win.width <= 0 or base_out_win.height <= 0:
        return False

    base_col_off = int(base_out_win.col_off)
    base_row_off = int(base_out_win.row_off)
    max_width = min(int(base_out_win.width), src.width)
    max_height = min(int(base_out_win.height), src.height)
    if max_width <= 0 or max_height <= 0:
        return False

    wrote_any = False
    for _, src_win in src.block_windows(1):
        src_col_off = int(src_win.col_off)
        src_row_off = int(src_win.row_off)
        if src_col_off >= max_width or src_row_off >= max_height:
            continue
        block_width = min(int(src_win.width), max_width - src_col_off)
        block_height = min(int(src_win.height), max_height - src_row_off)
        if block_width <= 0 or block_height <= 0:
            continue

        src_block_win = Window(src_col_off, src_row_off, block_width, block_height)
        data = src.read(1, window=src_block_win, masked=True).astype("float32")
        arr = np.asarray(data.filled(np.nan), dtype="float32")
        valid = (~np.ma.getmaskarray(data)) & np.isfinite(arr)
        if not valid.any():
            continue

        dst_win = Window(
            col_off=base_col_off + src_col_off,
            row_off=base_row_off + src_row_off,
            width=block_width,
            height=block_height,
        )
        sum_block = sum_ds.read(1, window=dst_win)
        count_block = count_ds.read(1, window=dst_win)

        sum_block[valid] += arr[valid]
        count_block[valid] += 1

        sum_ds.write(sum_block, 1, window=dst_win)
        count_ds.write(count_block, 1, window=dst_win)
        for threshold, rich_ds in rich_datasets:
            rich_block = rich_ds.read(1, window=dst_win)
            rich_block[valid & (arr >= threshold)] += 1
            rich_ds.write(rich_block, 1, window=dst_win)
        wrote_any = True

    return wrote_any


def _write_overlap_png(
    raster_path: Path,
    png_path: Path,
    title: str,
    colorbar_label: str,
    max_dim: int = 4000,
) -> None:
    with rasterio.open(raster_path) as ds:
        scale = max(ds.width / max_dim, ds.height / max_dim, 1.0)
        out_width = max(1, int(np.ceil(ds.width / scale)))
        out_height = max(1, int(np.ceil(ds.height / scale)))
        arr = ds.read(
            1,
            out_shape=(out_height, out_width),
            masked=True,
            resampling=Resampling.nearest,
        )

    # Keep nodata and zero-overlap pixels white in the visualization.
    positive = np.ma.masked_less_equal(arr, 0)

    fig_height = max(4.0, 12.0 * (positive.shape[0] / max(positive.shape[1], 1)))
    fig, ax = plt.subplots(figsize=(12, fig_height), facecolor="white")
    ax.set_facecolor("white")

    if positive.count() > 0:
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")
        vmax = float(positive.max())
        im = ax.imshow(positive, origin="upper", cmap=cmap, vmin=1, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label)
    else:
        blank = np.ones(positive.shape, dtype="float32")
        ax.imshow(blank, origin="upper", cmap="gray", vmin=0, vmax=1)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stack per-species suitability rasters into overlap maps using only valid projected pixels."
    )
    parser.add_argument("--ensemble-dir", required=True, help="Path to rasters/ensemble-images directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory for overlap rasters.")
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=[0.5],
        help="One or more suitability thresholds for richness counts (default: 0.5).",
    )
    parser.add_argument("--compress", default="DEFLATE", help="GeoTIFF compression (default: DEFLATE).")
    parser.add_argument("--png-max-dim", type=int, default=4000, help="Maximum width or height for the preview PNG (default: 4000).")
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = list(dict.fromkeys(float(t) for t in args.threshold))

    rasters = _collect_probability_rasters(ensemble_dir)
    print("[INFO] validating probability rasters", flush=True)
    _validate_probability_rasters(rasters)
    out_transform, out_width, out_height, out_crs = _union_grid(rasters)
    print(f"[INFO] stacking {len(rasters)} rasters over grid {out_width}x{out_height} ({out_crs})")
    print(f"[INFO] thresholds={', '.join(f'{t:g}' for t in thresholds)}", flush=True)

    profile = {
        "driver": "GTiff",
        "width": out_width,
        "height": out_height,
        "count": 1,
        "crs": out_crs,
        "transform": out_transform,
        "tiled": True,
        "compress": args.compress,
        "BIGTIFF": "IF_SAFER",
        "blockxsize": 512,
        "blockysize": 512,
    }
    float_nodata = -9999.0
    int_nodata = -1

    sum_path = out_dir / "overlap_sum_probability.tif"
    count_path = out_dir / "overlap_coverage_count.tif"
    mean_path = out_dir / "overlap_mean_probability.tif"
    coverage_png_path = out_dir / "overlap_coverage_count.png"
    rich_outputs = []
    for threshold in thresholds:
        tag = _threshold_tag(threshold)
        if len(thresholds) == 1:
            rich_path = out_dir / "overlap_richness_threshold.tif"
            rich_png_path = out_dir / "overlap_richness_threshold.png"
        else:
            rich_path = out_dir / f"overlap_richness_threshold_{tag}.tif"
            rich_png_path = out_dir / f"overlap_richness_threshold_{tag}.png"
        rich_outputs.append((threshold, rich_path, rich_png_path))

    uncovered_mask_path = out_dir / "overlap_uncovered_mask.tif"

    with ExitStack() as stack:
        sum_ds = stack.enter_context(rasterio.open(sum_path, "w+", dtype="float32", nodata=float_nodata, **profile))
        count_ds = stack.enter_context(rasterio.open(count_path, "w+", dtype="int32", nodata=int_nodata, **profile))
        uncovered_ds = stack.enter_context(rasterio.open(uncovered_mask_path, "w+", dtype="uint8", nodata=0, **profile))
        rich_datasets = [
            (threshold, stack.enter_context(rasterio.open(rich_path, "w+", dtype="int32", nodata=int_nodata, **profile)))
            for threshold, rich_path, _ in rich_outputs
        ]

        for i, rp in enumerate(rasters, 1):
            with rasterio.open(rp) as src:
                _stack_source_raster(
                    src,
                    sum_ds,
                    count_ds,
                    rich_datasets,
                    out_transform,
                    out_width,
                    out_height,
                )

            if i % 25 == 0 or i == len(rasters):
                print(f"[INFO] processed {i}/{len(rasters)} rasters", flush=True)

        print("[INFO] finalizing uncovered mask", flush=True)
        for _, win in uncovered_ds.block_windows(1):
            coverage = count_ds.read(1, window=win)
            uncovered = (coverage == 0).astype("uint8")
            uncovered_ds.write(uncovered, 1, window=win)

        _apply_nodata_where_uncovered(sum_ds, uncovered_ds, float_nodata)
        _apply_nodata_where_uncovered(count_ds, uncovered_ds, int_nodata)
        for _, rich_ds in rich_datasets:
            _apply_nodata_where_uncovered(rich_ds, uncovered_ds, int_nodata)

    print("[INFO] computing mean suitability raster", flush=True)
    with rasterio.open(sum_path) as sum_ds, rasterio.open(count_path) as count_ds:
        mean_profile = sum_ds.profile.copy()
        mean_profile["dtype"] = "float32"
        mean_profile["nodata"] = float_nodata
        with rasterio.open(mean_path, "w", **mean_profile) as mean_ds:
            for _, win in mean_ds.block_windows(1):
                s = sum_ds.read(1, window=win).astype("float32")
                c = count_ds.read(1, window=win).astype("float32")
                m = np.full_like(s, float_nodata, dtype="float32")
                mask = c > 0
                m[mask] = s[mask] / c[mask]
                mean_ds.write(m, 1, window=win)

    print(f"[INFO] rendering overlap PNG previews (max_dim={args.png_max_dim})", flush=True)
    _write_overlap_png(
        count_path,
        coverage_png_path,
        title="Projection coverage count",
        colorbar_label="species coverage count",
        max_dim=args.png_max_dim,
    )
    for threshold, rich_path, rich_png_path in rich_outputs:
        _write_overlap_png(
            rich_path,
            rich_png_path,
            title=f"Overlap richness (threshold >= {threshold:g})",
            colorbar_label="species overlap count",
            max_dim=args.png_max_dim,
        )

    print(f"[OK] wrote: {sum_path}")
    print(f"[OK] wrote: {count_path}")
    print(f"[OK] wrote: {mean_path}")
    print(f"[OK] wrote: {coverage_png_path}")
    for _, rich_path, rich_png_path in rich_outputs:
        print(f"[OK] wrote: {rich_path}")
        print(f"[OK] wrote: {rich_png_path}")
    print(f"[OK] wrote: {uncovered_mask_path}")


if __name__ == "__main__":
    main()
