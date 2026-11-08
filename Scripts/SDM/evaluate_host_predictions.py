#!/usr/bin/env python3
"""Evaluate species raster predictions against capture records."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns


@dataclass
class EvalResult:
    species: str
    safe_species: str
    record_id: str
    latitude: float
    longitude: float
    probability: float
    predicted_presence: bool
    number_of_hosts: float
    raster_path: Path


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value)


def find_probability_raster(outputs_dir: Path, safe_species: str) -> Optional[Path]:
    candidates = [
        outputs_dir / "lgbm-images" / safe_species / f"probability_1_{safe_species}.tif",
        outputs_dir / "xgb-images" / safe_species / f"probability_1_{safe_species}.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(outputs_dir.glob(f"**/{safe_species}/probability_1_{safe_species}.tif"))
    if matches:
        return matches[0]
    return None


def sample_probability(raster_path: Path, lon: float, lat: float) -> float:
    with rasterio.open(raster_path) as src:
        sample = next(src.sample([(lon, lat)]))
        value = float(sample[0])
    if (not np.isfinite(value)) or value < 0.0 or value > 1.0:
        return float('nan')
    return value


def evaluate(csv_path: Path, outputs_dir: Path, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    captures = pd.read_csv(csv_path)
    required = {"host_species", "latitude", "longitude", "number_of_hosts"}
    missing = required - set(captures.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    captures['number_of_hosts'] = pd.to_numeric(captures['number_of_hosts'], errors='coerce')
    captures = captures.dropna(subset=['number_of_hosts']).copy()
    if captures.empty:
        raise RuntimeError("No records remaining after dropping rows with missing number_of_hosts.")

    results: List[EvalResult] = []
    skipped: List[str] = []

    captures['species_clean'] = captures['host_species'].astype(str).str.strip()
    captures['safe_species'] = captures['species_clean'].apply(safe_name)

    for safe_species, group in captures.groupby('safe_species'):
        if not safe_species:
            continue
        species_name = group['species_clean'].iloc[0]
        raster_path = find_probability_raster(outputs_dir, safe_species)
        if raster_path is None:
            skipped.append(species_name)
            continue
        coords = list(zip(group['longitude'].astype(float), group['latitude'].astype(float)))
        with rasterio.open(raster_path) as src:
            samples = list(src.sample(coords))
        for (lon, lat), sample, (_, row) in zip(coords, samples, group.iterrows()):
            prob = float(sample[0]) if sample.size else float('nan')
            if not np.isfinite(prob):
                predicted = False
            else:
                predicted = bool(prob >= threshold)
            weight = float(row['number_of_hosts']) if np.isfinite(row['number_of_hosts']) else 1.0
            record_id = str(row.get('host_record_id', row.get('record_id', '')))
            results.append(EvalResult(
                species=species_name,
                safe_species=safe_species,
                record_id=record_id,
                latitude=float(row['latitude']),
                longitude=float(row['longitude']),
                probability=prob,
                predicted_presence=predicted,
                number_of_hosts=weight,
                raster_path=raster_path,
            ))

    if not results:
        raise RuntimeError("No evaluation records were produced. Check species names and outputs.")

    records_df = pd.DataFrame([result.__dict__ for result in results])
    records_df['threshold'] = threshold
    records_df['actual_presence'] = (records_df['number_of_hosts'] > 0).astype(int)
    records_df['predicted_label'] = records_df['predicted_presence'].astype(int)

    # Weighted confusion counts including true absences (number_of_hosts == 0)
    confusion_df = (
        records_df
        .groupby(['actual_presence', 'predicted_label'])
        .agg(
            count_weighted=('number_of_hosts', 'sum'),
            count_records=('record_id', 'size'),
        )
        .reset_index()
    )
    label_map = {1: 'presence', 0: 'absence'}
    confusion_df['actual_label'] = confusion_df['actual_presence'].map(label_map)
    confusion_df['predicted_label'] = confusion_df['predicted_label'].map(label_map)
    confusion_df = confusion_df[['actual_label', 'predicted_label', 'count_weighted', 'count_records']]

    species_summary = (
        records_df.groupby('species', group_keys=False)
        .apply(lambda grp: pd.Series({
            'n_records': len(grp),
            'weight_total': grp['number_of_hosts'].sum(),
            'predicted_presence_records': grp['predicted_presence'].sum(),
            'predicted_presence_weight': grp.loc[grp['predicted_presence'], 'number_of_hosts'].sum(),
            'predicted_absence_records': (~grp['predicted_presence']).sum(),
            'predicted_absence_weight': grp.loc[~grp['predicted_presence'], 'number_of_hosts'].sum(),
            'mean_probability': grp['probability'].mean(),
            'median_probability': grp['probability'].median(),
            'min_probability': grp['probability'].min(),
            'max_probability': grp['probability'].max(),
        }))
        .reset_index()
    )

    if skipped:
        unique_skipped = sorted(set(skipped))
        print(f"[WARN] Missing raster for {len(unique_skipped)} species: {unique_skipped}", file=sys.stderr)

    return records_df, species_summary, confusion_df


def make_plots(records_df: pd.DataFrame, out_dir: Path) -> None:
    sns.set_style("whitegrid")

    plot_df = records_df[np.isfinite(records_df['probability'])]
    plot_df = plot_df[(plot_df['probability'] >= 0.0) & (plot_df['probability'] <= 1.0)]

    scatter_path = out_dir / 'hosts_vs_probability_scatter.png'
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(plot_df['number_of_hosts'], plot_df['probability'], alpha=0.6, s=15)
    ax.set_xlabel('Number of hosts (log scale)')
    ax.set_ylabel('Predicted probability')
    ax.set_title('Probability vs. host capture counts')
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    boxplot_path = out_dir / 'hosts_by_prediction_boxplot.png'
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=plot_df, x='predicted_label', y='number_of_hosts', ax=ax)
    sns.stripplot(data=plot_df, x='predicted_label', y='number_of_hosts', color='black', alpha=0.3, ax=ax, jitter=0.3, size=3)
    ax.set_yscale('log')
    ax.set_xlabel('Predicted class (0 = absence, 1 = presence)')
    ax.set_ylabel('Number of hosts (log scale)')
    ax.set_title('Host counts by predicted class')
    fig.tight_layout()
    fig.savefig(boxplot_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted rasters at capture locations.")
    parser.add_argument('csv_path', type=Path, help='Path to host_data cleaned CSV')
    parser.add_argument('outputs_dir', type=Path, help='Path to SDM outputs directory (contains lgbm-images/xgb-images)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for presence (default: 0.5)')
    parser.add_argument('--out-dir', type=Path, default=None, help='Directory to write evaluation CSV summaries')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.outputs_dir / 'evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)

    records_df, species_summary, confusion_df = evaluate(args.csv_path, args.outputs_dir, args.threshold)

    records_path = out_dir / 'capture_predictions.csv'
    species_path = out_dir / 'capture_summary_by_species.csv'
    confusion_path = out_dir / 'confusion_summary.csv'

    records_df.to_csv(records_path, index=False)
    species_summary.to_csv(species_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)

    make_plots(records_df, out_dir)

    print(f"[INFO] Written per-record predictions to {records_path}")
    print(f"[INFO] Written per-species summary to {species_path}")
    print(f"[INFO] Written confusion summary to {confusion_path}")


if __name__ == '__main__':
    main()
