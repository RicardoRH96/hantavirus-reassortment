#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "Outputs" / "misc_figures" / "training_metrics_scatter.png"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot XGBoost, LightGBM, and Ensemble training metrics against the number of "
            "presence records."
        ),
        epilog=(
            "Examples:\n"
            "  python src/plotTrainingMetrics.py training_summary_21498643_tuned.csv\n"
            "  python src/plotTrainingMetrics.py training_summary.csv Outputs/misc_figures/training_metrics_job21498643.png\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "summary_csv",
        nargs="?",
        default="training_summary.csv",
        help="CSV produced by src/gatherModelStats.py.",
    )
    parser.add_argument(
        "output_png",
        nargs="?",
        default=str(DEFAULT_OUTPUT),
        help=f"Output PNG path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--trend",
        choices=("none", "kernel"),
        default="none",
        help="Optional exploratory smoother to overlay on top of the scatter plot.",
    )
    parser.add_argument(
        "--trend-models",
        nargs="+",
        choices=("xgb", "lgbm", "ensemble"),
        default=None,
        help="Model prefixes to smooth when --trend is enabled. Default: all available models.",
    )
    parser.add_argument(
        "--trend-bandwidth",
        type=float,
        default=0.35,
        help="Gaussian-kernel bandwidth on the log10(n) axis for the exploratory trend line.",
    )
    return parser.parse_args()


def gaussian_kernel_smooth(x: np.ndarray, y: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    weights = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bandwidth) ** 2)
    denom = weights.sum(axis=1)
    return (weights @ y) / np.clip(denom, 1e-12, None)


def main() -> int:
    import matplotlib.pyplot as plt

    args = parse_args()
    summary_csv = Path(args.summary_csv).expanduser()
    output_png = Path(args.output_png).expanduser()

    if not summary_csv.is_file():
        print(f"[ERROR] Summary CSV does not exist: {summary_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(summary_csv)
    required_columns = [
        "presences_not_thinned",
        "xgb_roc_auc",
        "xgb_pr_auc",
        "xgb_brier",
        "lgbm_roc_auc",
        "lgbm_pr_auc",
        "lgbm_brier",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(
            f"[ERROR] Missing required columns in {summary_csv}: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    df["n_points"] = pd.to_numeric(df["presences_not_thinned"], errors="coerce")
    for col in [c for c in df.columns if c.endswith(("_roc_auc", "_pr_auc", "_brier"))]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    plot_df = df.dropna(subset=["n_points"]).copy()
    if plot_df.empty:
        print(f"[ERROR] No complete metric rows were found in {summary_csv}", file=sys.stderr)
        return 1

    model_specs = [
        ("XGBoost", "xgb", "#d62728"),
        ("LightGBM", "lgbm", "#1f77b4"),
        ("Ensemble", "ensemble", "#2ca02c"),
    ]
    available_models = [
        (label, prefix, color)
        for label, prefix, color in model_specs
        if all(f"{prefix}_{suffix}" in plot_df.columns for suffix in ("roc_auc", "pr_auc", "brier"))
    ]
    if not available_models:
        print(f"[ERROR] No model metric columns were found in {summary_csv}", file=sys.stderr)
        return 1

    trend_model_prefixes = {prefix for _, prefix, _ in available_models}
    if args.trend_models is not None:
        trend_model_prefixes &= set(args.trend_models)

    metrics = [
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Brier", "brier"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for ax, (metric_label, metric_suffix) in zip(axes, metrics):
        plotted_any = False
        for model_label, model_prefix, color in available_models:
            metric_col = f"{model_prefix}_{metric_suffix}"
            model_df = plot_df.dropna(subset=[metric_col])
            if model_df.empty:
                continue
            plotted_any = True
            ax.scatter(
                model_df["n_points"],
                model_df[metric_col],
                color=color,
                marker="o",
                alpha=0.65,
                edgecolor="k",
                linewidth=0.3,
                s=28,
                label=model_label,
            )
            if args.trend == "kernel" and model_prefix in trend_model_prefixes and len(model_df) >= 5:
                x = np.log10(model_df["n_points"].to_numpy(dtype=float))
                y = model_df[metric_col].to_numpy(dtype=float)
                order = np.argsort(x)
                x = x[order]
                y = y[order]
                grid = np.linspace(x.min(), x.max(), 200)
                trend = gaussian_kernel_smooth(x, y, grid, args.trend_bandwidth)
                ax.plot(
                    10 ** grid,
                    trend,
                    color=color,
                    linewidth=2.0,
                    alpha=0.95,
                    zorder=4,
                )
        if not plotted_any:
            print(
                f"[ERROR] No non-null values found for metric '{metric_label}' in {summary_csv}",
                file=sys.stderr,
            )
            return 1
        ax.set_title(metric_label)
        ax.set_xscale("log")
        ax.set_xlabel("Presence records (log10)")
        ax.set_ylabel(metric_label)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="k",
            markeredgewidth=0.3,
            label=label,
            markersize=8,
        )
        for label, _, color in available_models
    ]
    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(handles)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    print(f"[OK] Wrote plot to {output_png.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
