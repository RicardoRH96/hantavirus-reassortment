#!/usr/bin/env python3
"""Exploratory ecology x molecular response surfaces from the integrated decomposition.

This helper is intentionally descriptive. It uses the existing module-level
decomposition output from the integrated explanatory figure and renders smoothed
2D surfaces over selected ecology x molecular pairs.

These are additive visual slices, not formal interaction estimates. If a pair
looks promising here, the next step is an explicit interaction refit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.linalg import LinAlgError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODULE_TABLE = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "final_hierarchical_block_comparison"
    / "figures"
    / "manuscript_utr3_integrated"
    / "integrated_module_contributions.tsv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures"


PAIR_SPECS = [
    ("Overlap opportunity", "LD burden", "Overlap opportunity × LD burden"),
    ("Host gestation length", "UTR5 L structure", "Host gestation length × UTR5 L structure"),
    ("Host litter size", "Significant LD fraction", "Host litter size × significant LD fraction"),
    ("Host dispersal", "Tree discordance", "Host dispersal × tree discordance"),
]

FALLBACK_PAIR_SPECS = [
    ("Host gestation length", "LD burden", "Host gestation length × LD burden"),
    ("Host litter size", "LD burden", "Host litter size × LD burden"),
    ("Host body mass", "LD burden", "Host body mass × LD burden"),
    ("Overlap opportunity", "Tree discordance", "Overlap opportunity × tree discordance"),
    ("Host dispersal", "LD burden", "Host dispersal × LD burden"),
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-table", type=Path, default=DEFAULT_MODULE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="ecology_molecular_interaction_surfaces.png")
    parser.add_argument("--grid-size", type=int, default=60)
    return parser.parse_args(argv)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.03, label.lower(), transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def pretty_label(label: str) -> str:
    replacements = {
        "Overlap opportunity": "Overlap opportunity contribution (logit)",
        "LD burden": "LD / compatibility contribution (logit)",
        "Tree discordance": "Tree discordance contribution (logit)",
        "UTR5 L structure": "UTR5 L structure contribution (logit)",
        "UTR5 M structure": "UTR5 M structure contribution (logit)",
        "Significant LD fraction": "Significant LD fraction contribution (logit)",
        "Host gestation length": "Host gestation contribution (logit)",
        "Host litter size": "Host litter-size contribution (logit)",
        "Host dispersal": "Host dispersal contribution (logit)",
        "Host body mass": "Host body-mass contribution (logit)",
    }
    return replacements.get(label, f"{label} contribution (logit)")


def load_module_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    needed = {"target", "probability"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df.copy()


def choose_pair_specs(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    chosen: list[tuple[str, str, str]] = []
    for pair in PAIR_SPECS:
        if pair[0] in df.columns and pair[1] in df.columns:
            chosen.append(pair)
    for pair in FALLBACK_PAIR_SPECS:
        if len(chosen) >= 4:
            break
        if pair in chosen:
            continue
        if pair[0] in df.columns and pair[1] in df.columns:
            chosen.append(pair)
    if not chosen:
        raise ValueError("No eligible ecology × molecular/phylogenetic pair specs found in module table")
    return chosen[:4]


def smooth_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    # Use the full observed range so the surface covers the same domain as the
    # overlaid points. Quantile clipping made large white regions that looked
    # like missing data even though they were only outside the interpolation box.
    x_lo, x_hi = float(np.min(x)), float(np.max(x))
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    x_pad = max((x_hi - x_lo) * 0.03, 1e-6)
    y_pad = max((y_hi - y_lo) * 0.03, 1e-6)
    x_lo -= x_pad
    x_hi += x_pad
    y_lo -= y_pad
    y_hi += y_pad
    grid_x, grid_y = np.meshgrid(np.linspace(x_lo, x_hi, grid_size), np.linspace(y_lo, y_hi, grid_size))
    try:
        rbf = RBFInterpolator(np.column_stack([x, y]), z, smoothing=0.45, kernel="thin_plate_spline")
        grid_z = rbf(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    except (ValueError, LinAlgError, np.linalg.LinAlgError):
        # Conservative fallback: bilinear plane fit.
        design = np.column_stack([np.ones_like(x), x, y])
        coef, *_ = np.linalg.lstsq(design, z, rcond=None)
        grid_z = coef[0] + coef[1] * grid_x + coef[2] * grid_y
    return grid_x, grid_y, np.clip(grid_z, 0.0, 1.0)


def pair_summary(df: pd.DataFrame, xcol: str, ycol: str) -> dict[str, float]:
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    p = df["probability"].to_numpy(dtype=float)
    return {
        "x_col": xcol,
        "y_col": ycol,
        "x_mean": float(np.mean(x)),
        "y_mean": float(np.mean(y)),
        "x_sd": float(np.std(x)),
        "y_sd": float(np.std(y)),
        "probability_mean": float(np.mean(p)),
        "probability_sd": float(np.std(p)),
        "spearman_xy": float(pd.Series(x).corr(pd.Series(y), method="spearman")),
        "spearman_x_p": float(pd.Series(x).corr(pd.Series(p), method="spearman")),
        "spearman_y_p": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
    }


def plot_pairs(df: pd.DataFrame, outpath: Path, grid_size: int, pair_specs: list[tuple[str, str, str]]) -> pd.DataFrame:
    n_panels = len(pair_specs)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 3.0 * nrows), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, wspace=0.20, hspace=0.26)
    axes = np.atleast_1d(axes).ravel()
    cmap = mpl.colormaps["viridis"]
    summaries = []
    panel_letters = [f"{chr(65 + i)})" for i in range(n_panels)]

    for ax, (xcol, ycol, title), letter in zip(axes, pair_specs, panel_letters):
        grid_x, grid_y, grid_z = smooth_surface(df[xcol], df[ycol], df["probability"], grid_size=grid_size)
        summaries.append(pair_summary(df, xcol, ycol))

        levels = np.linspace(0.0, 1.0, 11)
        cf = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.contour(grid_x, grid_y, grid_z, levels=np.linspace(0.1, 0.9, 5), colors="white", linewidths=0.5, alpha=0.65)

        neg = df["target"].to_numpy(dtype=int) == 0
        pos = ~neg
        ax.scatter(df.loc[neg, xcol], df.loc[neg, ycol], s=10, c="#7f7f7f", alpha=0.32, edgecolors="none")
        ax.scatter(df.loc[pos, xcol], df.loc[pos, ycol], s=12, c="#c44e52", alpha=0.65, edgecolors="white", linewidths=0.2)

        ax.set_xlabel(pretty_label(xcol))
        ax.set_ylabel(pretty_label(ycol))
        ax.text(0.03, 0.96, letter.lower(), transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n_panels:]:
        ax.axis("off")

    cbar = fig.colorbar(cf, ax=axes.tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("Smoothed model probability")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summaries)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_module_table(args.module_table)
    pair_specs = choose_pair_specs(df)
    outpath = args.output_dir / args.output_name
    summary = plot_pairs(df, outpath, args.grid_size, pair_specs)
    summary.to_csv(outpath.with_suffix(".tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
