#!/usr/bin/env python3
"""
Aggregate partial dependence outputs across species and produce
multi-species summaries (faceted ribbons, density contours, and PCA clustering).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SENTINEL_FLOAT = np.float32(-3.3999999521443642e38)
GRID_POINTS = 50


@dataclass
class PartialDependenceRecord:
    species: str
    model: str
    feature: str
    values: np.ndarray
    dependence: np.ndarray
    csv_path: Path
    plot_path: Path


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value)


def discover_pd_records(outputs_dir: Path) -> List[PartialDependenceRecord]:
    records: List[PartialDependenceRecord] = []
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    for model_dir in outputs_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for species_dir in model_dir.iterdir():
            if not species_dir.is_dir():
                continue
            species_name = species_dir.name
            analysis_dir = species_dir / "analysis"
            if not analysis_dir.exists():
                continue
            for csv_path in analysis_dir.glob("partial_dependence_*.csv"):
                feature_name = csv_path.stem.replace("partial_dependence_", "")
                plot_path = analysis_dir / f"partial_dependence_{feature_name}.png"
                try:
                    df = pd.read_csv(csv_path)
                    if df.shape[1] < 2:
                        print(f"[WARN] Unexpected format in {csv_path}; skipping")
                        continue
                    values = df.iloc[:, 0].to_numpy(dtype=float)
                    pd_vals = df.iloc[:, 1].to_numpy(dtype=float)
                    if values.size == 0:
                        continue
                except Exception as exc:
                    print(f"[WARN] Failed reading {csv_path}: {exc}")
                    continue
                records.append(
                    PartialDependenceRecord(
                        species=species_name,
                        model=model_name,
                        feature=feature_name,
                        values=values,
                        dependence=pd_vals,
                        csv_path=csv_path,
                        plot_path=plot_path,
                    )
                )
    if not records:
        raise RuntimeError("No partial dependence files discovered under outputs directory.")
    return records


def build_long_dataframe(records: Iterable[PartialDependenceRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        rows.append(
            pd.DataFrame(
                {
                    "species": rec.species,
                    "model": rec.model,
                    "feature": rec.feature,
                    "feature_value": rec.values,
                    "partial_dependence": rec.dependence,
                }
            )
        )
    return pd.concat(rows, axis=0, ignore_index=True)


def clean_long_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["feature_value"] = pd.to_numeric(out["feature_value"], errors="coerce")
    out["partial_dependence"] = pd.to_numeric(out["partial_dependence"], errors="coerce")

    valid = np.isfinite(out["feature_value"]) & np.isfinite(out["partial_dependence"])
    sentinel_mask = np.isclose(out["feature_value"], SENTINEL_FLOAT)
    large_mask = np.abs(out["feature_value"]) > 1e9
    valid &= ~sentinel_mask & ~large_mask
    out = out.loc[valid].copy()
    out.sort_values(["feature", "species", "feature_value"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def make_faceted_ribbons(df: pd.DataFrame, out_dir: Path, max_species: int = 12) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    features = sorted(df["feature"].unique())
    for feature in features:
        feature_df = df[df["feature"] == feature].copy()
        if feature_df.empty:
            continue

        selected_species = feature_df["species"].unique()[:max_species]
        feature_df = feature_df[feature_df["species"].isin(selected_species)]
        if feature_df.empty:
            continue

        feature_df = feature_df.sort_values(["species", "feature_value"])  # ensure monotonic lines

        g = sns.FacetGrid(
            feature_df,
            col="species",
            col_wrap=3,
            sharex=True,
            sharey=True,
            height=3.0,
            aspect=1.2,
        )
        g.map_dataframe(
            sns.lineplot,
            x="feature_value",
            y="partial_dependence",
            color="tab:blue",
        )
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels(feature, "Partial dependence")
        for ax in g.axes.flatten():
            ax.axhline(0.0, color="#cccccc", linewidth=0.8, linestyle="--", alpha=0.7)
        g.figure.suptitle(f"Partial dependence (subset) · {feature}", fontsize=14)
        g.figure.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = out_dir / f"ribbons_{safe_name(feature)}.png"
        g.savefig(out_path, dpi=150)
        plt.close(g.figure)
        print(f"[INFO] Faceted ribbons written: {out_path}")


def build_feature_curves(df: pd.DataFrame, grid_points: int = GRID_POINTS) -> Dict[str, Dict[str, np.ndarray]]:
    feature_curves: Dict[str, Dict[str, np.ndarray]] = {}
    feature_grids: Dict[str, np.ndarray] = {}

    for feature in sorted(df["feature"].unique()):
        feature_df = df[df["feature"] == feature]
        if feature_df.empty:
            continue
        min_val = feature_df["feature_value"].min()
        max_val = feature_df["feature_value"].max()
        if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val == min_val:
            continue
        grid = np.linspace(min_val, max_val, grid_points)
        curves: Dict[str, np.ndarray] = {}
        for species, sub in feature_df.groupby("species"):
            sub = sub.sort_values("feature_value").drop_duplicates("feature_value")
            if sub.shape[0] < 2:
                continue
            y_interp = np.interp(
                grid,
                sub["feature_value"],
                sub["partial_dependence"],
                left=sub["partial_dependence"].iloc[0],
                right=sub["partial_dependence"].iloc[-1],
            )
            curves[species] = y_interp
        if curves:
            feature_curves[feature] = curves
            feature_grids[feature] = grid

    return {f: {"grid": feature_grids[f], "curves": curves} for f, curves in feature_curves.items()}


def make_density_contours(feature_data: Dict[str, Dict[str, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("white")

    for feature, data in feature_data.items():
        grid = data["grid"]
        curves = data["curves"]
        if not curves:
            continue

        species_list = list(curves.keys())
        x_bins = np.linspace(grid.min(), grid.max(), 60)
        y_min = min(curve.min() for curve in curves.values())
        y_max = max(curve.max() for curve in curves.values())
        if y_max == y_min:
            continue
        y_bins = np.linspace(y_min, y_max, 60)

        count_grid = np.zeros((len(x_bins) - 1, len(y_bins) - 1), dtype=float)
        for species in species_list:
            curve = curves[species]
            hist, _, _ = np.histogram2d(grid, curve, bins=[x_bins, y_bins])
            count_grid += (hist > 0).astype(float)

        contour_path = out_dir / f"contour_{safe_name(feature)}.png"
        fig, ax = plt.subplots(figsize=(6, 4.5))
        pcm = ax.imshow(
            count_grid.T,
            origin="lower",
            aspect="auto",
            extent=(x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]),
            cmap="viridis",
        )
        fig.colorbar(pcm, ax=ax, label="Species count")
        ax.set_title(f"Species coverage density · {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Partial dependence")
        fig.tight_layout()
        fig.savefig(contour_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Density contour written: {contour_path}")


def make_pca(feature_data: Dict[str, Dict[str, np.ndarray]], out_dir: Path, n_components: int = 2) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not feature_data:
        print("[WARN] PCA skipped: no feature data available.")
        return

    feature_names = sorted(feature_data.keys())
    species_sets = [set(data["curves"].keys()) for data in feature_data.values() if data["curves"]]
    if not species_sets:
        print("[WARN] PCA skipped: empty feature curves.")
        return

    common_species = set.intersection(*species_sets)
    if len(common_species) < 2:
        print("[WARN] PCA skipped: fewer than two species with curves across all features.")
        return

    species_list = sorted(common_species)
    vectors = []
    for species in species_list:
        parts = []
        for feature in feature_names:
            curve = feature_data[feature]["curves"].get(species)
            if curve is None:
                break
            parts.append(curve)
        else:
            vectors.append(np.concatenate(parts))
            continue
        # species missing data for some feature: skip
    if not vectors:
        print("[WARN] PCA skipped: no species with complete curve coverage.")
        return

    matrix = np.vstack(vectors)
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(matrix_scaled)
    explained = pca.explained_variance_ratio_

    pca_df = pd.DataFrame(
        {
            "species": species_list[: components.shape[0]],
            "PC1": components[:, 0],
            "PC2": components[:, 1] if components.shape[1] >= 2 else 0.0,
        }
    )

    pca_plot = out_dir / "pca_species_scatter.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", ax=ax)
    for _, row in pca_df.iterrows():
        ax.text(row["PC1"], row["PC2"], row["species"], fontsize=7, alpha=0.7)
    ax.set_title("Species clustering by partial dependence (PCA)")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    fig.tight_layout()
    fig.savefig(pca_plot, dpi=150)
    plt.close(fig)
    print(f"[INFO] PCA scatter written: {pca_plot}")

    out_scores = out_dir / "pca_scores.csv"
    scores = pca_df.copy()
    for idx, var in enumerate(explained, start=1):
        scores[f"explained_variance_PC{idx}"] = var
    scores.to_csv(out_scores, index=False)
    print(f"[INFO] PCA scores written: {out_scores}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize partial dependence outputs across species.")
    parser.add_argument("outputs_dir", type=Path, help="Path to SDM outputs directory (e.g., SDM/Outputs/withMoreCovariates)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory to store summary plots (default: outputs_dir/partial_dep_summary)")
    parser.add_argument(
        "--max-ribbon-species",
        type=int,
        default=12,
        help="Number of species to show per feature in the ribbon facet plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir: Path = args.outputs_dir
    summary_dir = args.out_dir or (outputs_dir / "partial_dep_summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    records = discover_pd_records(outputs_dir)
    df_raw = build_long_dataframe(records)
    df = clean_long_dataframe(df_raw)
    if df.empty:
        raise RuntimeError("All partial dependence rows filtered out; check data integrity.")

    feature_data = build_feature_curves(df)

    ribbons_dir = summary_dir / "ribbons"
    make_faceted_ribbons(df, ribbons_dir, max_species=args.max_ribbon_species)

    contour_dir = summary_dir / "contours"
    make_density_contours(feature_data, contour_dir)

    pca_dir = summary_dir / "pca"
    make_pca(feature_data, pca_dir)


if __name__ == "__main__":
    main()
