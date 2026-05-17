#!/usr/bin/env python3
"""Species-varying-slope response model on ecology and molecular block scores.

This model treats ecology and molecular block scores as model-derived summary
axes and asks whether species differ in how these blocks shift effective
reassortment probability. Response surfaces are exported in a Figure 6-like
format for per-species visualization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import arviz as az
import bambi as bmb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "species_varying_slope_block_model"
)
DEFAULT_FIGURE_PATH = REPO_ROOT / "results" / "figures" / "figure6_species_block_surface_facets.png"

ECOLOGY_MODULES_ALL = [
    "Host gestation length",
    "Host litter size",
    "Host dispersal",
    "Host body mass",
    "Overlap opportunity",
    "Local host prevalence",
]
MOLECULAR_MODULES_ALL = [
    "LD burden",
    "Significant LD fraction",
    "UTR5 S structure",
    "UTR5 M structure",
    "UTR5 L structure",
    "UTR3 S structure",
    "UTR3 M structure",
    "UTR3 L structure",
]

SPECIES_LABELS = {
    "orthohantavirus_andesense": "Andes virus",
    "orthohantavirus_dobravaense": "Dobrava-Belgrade virus",
    "orthohantavirus_hantanense": "Hantaan virus",
    "orthohantavirus_puumalaense": "Puumala virus",
    "orthohantavirus_seoulense": "Seoul virus",
    "orthohantavirus_sinnombreense": "Sin Nombre virus",
    "orthohantavirus_tulaense": "Tula virus",
}

COMMON_PRIORS = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1.5),
    "common": bmb.Prior("Normal", mu=0, sigma=1.0),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-table", type=Path, default=DEFAULT_MODULE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--tune", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=30)
    return parser.parse_args(argv)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Helvetica Neue",
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.5,
        }
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_analysis_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    ecology_modules = [c for c in ECOLOGY_MODULES_ALL if c in df.columns]
    molecular_modules = [c for c in MOLECULAR_MODULES_ALL if c in df.columns]
    needed = {"tip_name", "species", "target", "probability"} | set(ecology_modules) | set(molecular_modules)
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if not ecology_modules:
        raise ValueError(f"No ecology modules found in {path}")
    if not molecular_modules:
        raise ValueError(f"No molecular modules found in {path}")

    out = df.copy()
    out["ecology_score_raw"] = out[ecology_modules].sum(axis=1)
    out["molecular_score_raw"] = out[molecular_modules].sum(axis=1)
    for raw_col, z_col in [
        ("ecology_score_raw", "ecology_score_z"),
        ("molecular_score_raw", "molecular_score_z"),
    ]:
        mean = float(out[raw_col].mean())
        std = float(out[raw_col].std(ddof=0))
        out[z_col] = 0.0 if std == 0 else (out[raw_col] - mean) / std
    return out[
        [
            "tip_name",
            "species",
            "target",
            "probability",
            "ecology_score_raw",
            "molecular_score_raw",
            "ecology_score_z",
            "molecular_score_z",
        ]
    ].copy()


def fit_model(df: pd.DataFrame, args: argparse.Namespace) -> tuple[bmb.Model, az.InferenceData, str]:
    formula = "target ~ ecology_score_z * molecular_score_z + (1 + ecology_score_z + molecular_score_z | species)"
    model = bmb.Model(formula, df, family="bernoulli", priors=COMMON_PRIORS)
    idata = model.fit(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        random_seed=args.random_seed,
        target_accept=args.target_accept,
        idata_kwargs={"log_likelihood": True},
    )
    return model, idata, formula


def species_order(df: pd.DataFrame) -> list[str]:
    return df.groupby("species")["target"].mean().sort_values().index.tolist()


def make_prediction_grid(df: pd.DataFrame, grid_size: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    eco_vals = np.linspace(
        float(np.quantile(df["ecology_score_z"], 0.05)),
        float(np.quantile(df["ecology_score_z"], 0.95)),
        grid_size,
    )
    mol_vals = np.linspace(
        float(np.quantile(df["molecular_score_z"], 0.05)),
        float(np.quantile(df["molecular_score_z"], 0.95)),
        grid_size,
    )
    species_list = species_order(df)
    rows = []
    for species in species_list:
        for eco in eco_vals:
            for mol in mol_vals:
                rows.append(
                    {
                        "species": species,
                        "ecology_score_z": eco,
                        "molecular_score_z": mol,
                    }
                )
    grid_df = pd.DataFrame(rows)
    return grid_df, eco_vals, mol_vals


def summarize_surface_predictions(
    model: bmb.Model,
    idata: az.InferenceData,
    grid_df: pd.DataFrame,
    eco_vals: np.ndarray,
    mol_vals: np.ndarray,
) -> pd.DataFrame:
    pred = model.predict(idata, data=grid_df, kind="response_params", inplace=False)
    p = np.asarray(pred.posterior["p"])
    p_draws = p.reshape(-1, p.shape[-1])
    summary = grid_df.copy()
    summary["predicted_mean"] = p_draws.mean(axis=0)
    summary["predicted_hdi_3"] = np.quantile(p_draws, 0.03, axis=0)
    summary["predicted_hdi_97"] = np.quantile(p_draws, 0.97, axis=0)
    summary["eco_rank"] = np.tile(np.arange(len(eco_vals)), len(summary) // len(eco_vals))
    summary["mol_rank"] = np.repeat(np.arange(len(mol_vals)), len(summary) // len(mol_vals)).reshape(-1, 1)
    return summary


def compute_species_metrics(surface_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for species, sub in surface_df.groupby("species", sort=False):
        pivot = (
            sub.pivot_table(
                index="molecular_score_z",
                columns="ecology_score_z",
                values="predicted_mean",
                aggfunc="mean",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        z = pivot.to_numpy()
        eco_axis = pivot.columns.to_numpy(dtype=float)
        mol_axis = pivot.index.to_numpy(dtype=float)
        eco_grad = np.abs(np.gradient(z, eco_axis, axis=1)).mean()
        mol_grad = np.abs(np.gradient(z, mol_axis, axis=0)).mean()
        low_idx = 0
        high_idx = -1
        eco_effect_low_compat = float(z[low_idx, high_idx] - z[low_idx, low_idx])
        eco_effect_high_compat = float(z[high_idx, high_idx] - z[high_idx, low_idx])
        mol_effect_low_ecology = float(z[high_idx, low_idx] - z[low_idx, low_idx])
        mol_effect_high_ecology = float(z[high_idx, high_idx] - z[low_idx, high_idx])
        rows.append(
            {
                "species": species,
                "min_predicted_probability": float(np.nanmin(z)),
                "max_predicted_probability": float(np.nanmax(z)),
                "surface_range": float(np.nanmax(z) - np.nanmin(z)),
                "mean_abs_ecology_gradient": float(eco_grad),
                "mean_abs_molecular_gradient": float(mol_grad),
                "molecular_to_ecology_gradient_ratio": float(mol_grad / eco_grad) if eco_grad > 0 else np.nan,
                "ecology_effect_low_compatibility": eco_effect_low_compat,
                "ecology_effect_high_compatibility": eco_effect_high_compat,
                "ecology_threshold_delta": float(eco_effect_high_compat - eco_effect_low_compat),
                "molecular_effect_low_ecology": mol_effect_low_ecology,
                "molecular_effect_high_ecology": mol_effect_high_ecology,
                "molecular_context_delta": float(mol_effect_high_ecology - mol_effect_low_ecology),
            }
        )
    out = pd.DataFrame(rows)
    out["species_label"] = out["species"].map(lambda s: SPECIES_LABELS.get(s, s))
    out["dominant_block"] = np.where(
        out["mean_abs_molecular_gradient"] > out["mean_abs_ecology_gradient"],
        "molecular",
        "ecology",
    )
    out["flat_near_zero_flag"] = (out["max_predicted_probability"] < 0.15) & (out["surface_range"] < 0.10)
    return out.sort_values("species_label").reset_index(drop=True)


def export_species_effects(idata: az.InferenceData, outdir: Path) -> pd.DataFrame:
    summary = az.summary(idata, var_names=list(idata.posterior.data_vars), round_to=4).reset_index().rename(
        columns={"index": "term"}
    )
    summary.to_csv(outdir / "posterior_summary.tsv", sep="\t", index=False)
    rows = []
    for term in summary["term"]:
        if "|species[" not in term:
            continue
        base, species = term.split("[", 1)
        species = species.rstrip("]")
        effect = base.replace("|species", "").strip()
        effect = effect if effect else "intercept"
        row = summary.loc[summary["term"] == term].iloc[0].to_dict()
        row["species"] = species
        row["effect"] = effect
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "species_varying_effects.tsv", sep="\t", index=False)
    return out


def plot_species_surfaces(
    data_df: pd.DataFrame,
    surface_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    outpath: Path,
) -> None:
    set_style()
    species_list = species_order(data_df)
    panel_letters = list("abcdefg")
    cmap = mpl.colormaps["viridis"]

    fig = plt.figure(figsize=(13.8, 7.6))
    gs = fig.add_gridspec(2, 4, left=0.04, right=0.98, top=0.95, bottom=0.06, wspace=0.10, hspace=0.20)

    for idx, species in enumerate(species_list):
        ax = fig.add_subplot(gs[idx // 4, idx % 4], projection="3d")
        sub_surface = surface_df[surface_df["species"] == species].copy()
        pivot = (
            sub_surface.pivot_table(
                index="molecular_score_z",
                columns="ecology_score_z",
                values="predicted_mean",
                aggfunc="mean",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        eco = pivot.columns.to_numpy(dtype=float)
        mol = pivot.index.to_numpy(dtype=float)
        gx, gy = np.meshgrid(eco, mol)
        gz = pivot.to_numpy()

        ax.plot_surface(gx, gy, gz, cmap=cmap, linewidth=0, antialiased=True, alpha=0.92)
        ax.contour(gx, gy, gz, zdir="z", offset=0.0, levels=5, colors="white", linewidths=0.65)

        sub_data = data_df[data_df["species"] == species].copy()
        pos = sub_data["target"].to_numpy() == 1
        ax.scatter(
            sub_data.loc[~pos, "ecology_score_z"],
            sub_data.loc[~pos, "molecular_score_z"],
            sub_data.loc[~pos, "probability"],
            s=9,
            c="#8f8f8f",
            alpha=0.32,
            depthshade=False,
        )
        ax.scatter(
            sub_data.loc[pos, "ecology_score_z"],
            sub_data.loc[pos, "molecular_score_z"],
            sub_data.loc[pos, "probability"],
            s=11,
            c="#bc4749",
            alpha=0.68,
            depthshade=False,
        )

        ax.set_xlabel("Ecology score (z)", labelpad=0)
        ax.set_ylabel("Molecular score (z)", labelpad=0)
        ax.set_zlabel("Predicted probability", labelpad=0)
        ax.set_zlim(0, 1)
        ax.view_init(elev=27, azim=-58)
        ax.tick_params(pad=0)
        ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1))
        ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
        metric_row = metrics_df.loc[metrics_df["species"] == species].iloc[0]
        dom = "Mol." if metric_row["dominant_block"] == "molecular" else "Eco."
        flat = " flat" if bool(metric_row["flat_near_zero_flag"]) else ""
        ax.text2D(
            0.03,
            0.98,
            f"{panel_letters[idx]}\n{SPECIES_LABELS.get(species, species)}\n{dom}{flat}",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
        )

    if len(species_list) < 8:
        fig.add_subplot(gs[1, 3]).axis("off")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_run_info(args: argparse.Namespace, formula: str, analysis_df: pd.DataFrame, outdir: Path) -> None:
    info = {
        "module_table": str(args.module_table),
        "formula": formula,
        "rows": int(len(analysis_df)),
        "positives": int(analysis_df["target"].sum()),
        "positive_rate": float(analysis_df["target"].mean()),
        "species_counts": analysis_df["species"].value_counts().to_dict(),
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "random_seed": args.random_seed,
    }
    (outdir / "run_info.json").write_text(json.dumps(info, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_path.parent.mkdir(parents=True, exist_ok=True)

    analysis_df = load_analysis_frame(args.module_table)
    analysis_df.to_csv(args.output_dir / "analysis_dataset.tsv", sep="\t", index=False)

    model, idata, formula = fit_model(analysis_df, args)
    idata.to_netcdf(args.output_dir / "posterior.nc")
    (args.output_dir / "formula.txt").write_text(formula + "\n")
    export_species_effects(idata, args.output_dir)

    grid_df, eco_vals, mol_vals = make_prediction_grid(analysis_df, args.grid_size)
    surface_df = summarize_surface_predictions(model, idata, grid_df, eco_vals, mol_vals)
    surface_df.to_csv(args.output_dir / "species_response_surface.tsv", sep="\t", index=False)

    metrics_df = compute_species_metrics(surface_df)
    metrics_df.to_csv(args.output_dir / "species_response_metrics.tsv", sep="\t", index=False)

    plot_species_surfaces(analysis_df, surface_df, metrics_df, args.figure_path)
    write_run_info(args, formula, analysis_df, args.output_dir)

    print(metrics_df.to_string(index=False))
    print(args.figure_path)


if __name__ == "__main__":
    main()
