#!/usr/bin/env python3
"""Summarize final LD-vs-epistasis equivalent comparisons across model layers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "model_test" / "runs" / "intersegment_epistasis_final_direct553"
DEFAULT_EGT_ROOT = REPO_ROOT / "specific_evolutionary_analyses" / "outputs_epistasis_final_direct553_egt"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--egt-root", type=Path, default=DEFAULT_EGT_ROOT)
    return parser.parse_args(argv)


def metric_rows(
    layer: str,
    comparison: str,
    ld_model: str,
    epistasis_model: str,
    metrics: Iterable[tuple[str, float, float]],
    notes: str,
) -> list[dict[str, object]]:
    rows = []
    for metric, ld_value, epi_value in metrics:
        rows.append(
            {
                "layer": layer,
                "comparison": comparison,
                "ld_model": ld_model,
                "epistasis_model": epistasis_model,
                "metric": metric,
                "ld_value": ld_value,
                "epistasis_value": epi_value,
                "epistasis_minus_ld": epi_value - ld_value,
                "notes": notes,
            }
        )
    return rows


def summarize_lightgbm(root: Path) -> list[dict[str, object]]:
    path = root / "model_suite_lgbm_full_equivalent_ld_vs_epistasis" / "summary.tsv"
    df = pd.read_csv(path, sep="\t")
    rows: list[dict[str, object]] = []
    comparisons = [
        (
            "summary-only",
            "lineage_context_ld_summary",
            "lineage_context_epistasis_time_div",
            "LD species summaries vs species-level time/div epistasis summaries.",
        ),
        (
            "full-equivalent",
            "lineage_context_ld_full_equivalent",
            "lineage_context_epistasis_full_equivalent",
            "LD summaries plus LD pair states vs epistasis summaries plus selected pair-state/numeric features.",
        ),
    ]
    metric_names = [
        "roc_auc_mean",
        "average_precision_mean",
        "oof_roc_auc",
        "oof_average_precision",
        "average_precision_lift_vs_prevalence",
    ]
    for cv_scheme, sub in df.groupby("cv_scheme", sort=True):
        by_name = sub.set_index("feature_set")
        for comparison, ld_model, epi_model, notes in comparisons:
            if ld_model not in by_name.index or epi_model not in by_name.index:
                continue
            metrics = [
                (metric, float(by_name.loc[ld_model, metric]), float(by_name.loc[epi_model, metric]))
                for metric in metric_names
            ]
            rows.extend(
                metric_rows(
                    "lightgbm",
                    f"{cv_scheme}:{comparison}",
                    ld_model,
                    epi_model,
                    metrics,
                    notes,
                )
            )
    return rows


def summarize_bayesian(root: Path) -> list[dict[str, object]]:
    path = root / "bayesian_ld_epistasis_comparison" / "block_comparison_summary.tsv"
    df = pd.read_csv(path, sep="\t").set_index("model")
    rows: list[dict[str, object]] = []
    comparisons = [
        (
            "block-summary",
            "ld_summary_block",
            "epistasis_summary_block",
            "Species-level LD summaries vs species-level time/div/FDR epistasis summaries.",
        ),
        (
            "integrated-summary",
            "integrated_ld_summary",
            "integrated_epistasis_summary",
            "Ecology plus LD summaries vs ecology plus epistasis summaries.",
        ),
        (
            "integrated-full-numeric",
            "integrated_ld_summary",
            "integrated_epistasis_summary_pairnumeric",
            "Ecology plus LD summaries vs ecology plus epistasis summaries and selected-pair numeric burdens.",
        ),
        (
            "overlap-interaction",
            "integrated_overlap_ld_interaction",
            "integrated_overlap_epistasis_interaction",
            "Overlap-opportunity interaction using LD burden vs selected epistasis burden.",
        ),
        (
            "tree-sensitivity",
            "integrated_ld_tree_sensitivity",
            "integrated_epistasis_tree_sensitivity",
            "Integrated LD vs epistasis models with tree-discordance sensitivity term.",
        ),
    ]
    metric_names = [
        "loo_elpd",
        "waic_elpd",
        "species_sigma_mean",
        "sigma_reduction_fraction_vs_species_only",
        "n_divergences",
    ]
    for comparison, ld_model, epi_model, notes in comparisons:
        if ld_model not in df.index or epi_model not in df.index:
            continue
        metrics = [(metric, float(df.loc[ld_model, metric]), float(df.loc[epi_model, metric])) for metric in metric_names]
        rows.extend(metric_rows("bayesian", comparison, ld_model, epi_model, metrics, notes))
    return rows


def coefficient_value(coefs: pd.DataFrame, model: str, term: str, metric: str) -> float:
    row = coefs[(coefs["model"] == model) & (coefs["term"] == term)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][metric])


def summarize_egt(egt_root: Path) -> list[dict[str, object]]:
    report_dir = egt_root / "reports"
    comp = pd.read_csv(report_dir / "13_egt_ld_epistasis_model_comparison.tsv", sep="\t").set_index("model")
    coefs = pd.read_csv(report_dir / "13_egt_ld_epistasis_establishment_coefficients.tsv", sep="\t")
    ld_model = "establishment_ld_molecular_permissiveness"
    epi_model = "establishment_epistasis_molecular_permissiveness"
    rows = metric_rows(
        "egt",
        "establishment-molecular-swap",
        ld_model,
        epi_model,
        [
            ("loo_elpd", float(comp.loc[ld_model, "loo_elpd"]), float(comp.loc[epi_model, "loo_elpd"])),
            ("loo_p", float(comp.loc[ld_model, "loo_p"]), float(comp.loc[epi_model, "loo_p"])),
            ("n_divergences", float(comp.loc[ld_model, "n_divergences"]), float(comp.loc[epi_model, "n_divergences"])),
            (
                "molecular_permissiveness_mean",
                coefficient_value(coefs, ld_model, "molecular_permissiveness_ld_z", "mean"),
                coefficient_value(coefs, epi_model, "molecular_permissiveness_epistasis_z", "mean"),
            ),
            (
                "molecular_permissiveness_p_positive",
                coefficient_value(coefs, ld_model, "molecular_permissiveness_ld_z", "posterior_prob_positive"),
                coefficient_value(coefs, epi_model, "molecular_permissiveness_epistasis_z", "posterior_prob_positive"),
            ),
            (
                "ecology_x_molecular_mean",
                coefficient_value(coefs, ld_model, "ecology_x_molecular_permissiveness_ld_z", "mean"),
                coefficient_value(coefs, epi_model, "ecology_x_molecular_permissiveness_epistasis_z", "mean"),
            ),
            (
                "ecology_x_molecular_p_positive",
                coefficient_value(coefs, ld_model, "ecology_x_molecular_permissiveness_ld_z", "posterior_prob_positive"),
                coefficient_value(coefs, epi_model, "ecology_x_molecular_permissiveness_epistasis_z", "posterior_prob_positive"),
            ),
        ],
        "Matched establishment model with fixed origin/control construction; only molecular permissiveness block is swapped.",
    )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = args.root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(summarize_lightgbm(args.root))
    rows.extend(summarize_bayesian(args.root))
    rows.extend(summarize_egt(args.egt_root))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "final_ld_vs_epistasis_equivalent_comparison.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
