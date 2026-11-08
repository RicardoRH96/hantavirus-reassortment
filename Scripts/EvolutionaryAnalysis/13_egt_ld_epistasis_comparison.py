#!/usr/bin/env python3
"""EGT-style LD-vs-epistasis establishment comparison.

This companion keeps the 08 fixation/establishment origin-control construction
fixed, then fits two matched-event establishment models:

* LD molecular permissiveness: original LD + UTR compact molecular index.
* Epistasis molecular permissiveness: final epistasis burden + UTR compact
  molecular index.

The goal is to test whether epistasis can replace LD in the establishment
surface without changing the event construction or ecological opportunity term.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(
    os.environ.get(
        "HANTA_EVO_DATASET",
        REPO_ROOT
        / "model_test"
        / "runs"
        / "intersegment_epistasis_final_direct553"
        / "datasets"
        / "sequence_level_final_dataset_direct553_epistasis_final.tsv",
    )
).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("HANTA_EVO_OUTPUT_DIR", SCRIPT_DIR / "outputs_epistasis_final_direct553_egt")
).expanduser()

BASE_EVENT_MEAN_FEATURES = [
    "overlap_opportunity_local",
    "arha_prev_local",
    "species_feat_mean_cramers_v_overall",
    "utr5_structural_distance",
    "utr3_structural_distance",
    "terminal_utr5_length_disparity",
    "terminal_utr3_length_disparity",
    "tree_discordance_score",
    "graph_neighbor_overlap",
    "log_sampling_effort",
    "species_feat_epi_time_mean_score",
    "species_feat_epi_time_max_score",
    "species_feat_epi_div_mean_score",
    "species_feat_epi_div_max_score",
    "species_feat_epi_final_time_selected_pair_per_100k_site_pairs",
    "species_feat_epi_final_time_min_fdr_neglog10",
    "species_feat_epi_final_time_best_robust_positive_fraction_neglog10p",
    "species_feat_epi_final_div_min_fdr_neglog10",
    "epi_selected_time_aa_rare_state_fraction",
    "epi_selected_time_aa_major_state_fraction",
    "epi_selected_time_aa_problem_state_fraction",
]
COMPACT_FEATURES = [
    "ecological_opportunity",
    "ld_compatibility",
    "epistasis_compatibility",
    "species_utr_compatibility",
    "terminal_utr_regime",
    "molecular_permissiveness_ld",
    "molecular_permissiveness_epistasis",
    "ecology_x_molecular_permissiveness_ld",
    "ecology_x_molecular_permissiveness_epistasis",
]


@dataclass(frozen=True)
class EgtSpec:
    name: str
    molecular_feature: str
    interaction_feature: str


SPECS = (
    EgtSpec(
        name="establishment_ld_molecular_permissiveness",
        molecular_feature="molecular_permissiveness_ld",
        interaction_feature="ecology_x_molecular_permissiveness_ld",
    ),
    EgtSpec(
        name="establishment_epistasis_molecular_permissiveness",
        molecular_feature="molecular_permissiveness_epistasis",
        interaction_feature="ecology_x_molecular_permissiveness_epistasis",
    ),
)
COMMON_PRIORS = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1.5),
    "common": bmb.Prior("Normal", mu=0, sigma=1.0),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tree-root", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=900)
    parser.add_argument("--tune", type=int, default=900)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.97)
    parser.add_argument("--random-seed", type=int, default=20260605)
    parser.add_argument("--force", action="store_true", help="Refit models even when posterior.nc already exists.")
    return parser.parse_args(argv)


def load_model08(args: argparse.Namespace):
    os.environ["HANTA_EVO_DATASET"] = str(args.dataset)
    os.environ["HANTA_EVO_OUTPUT_DIR"] = str(args.output_dir)
    if args.tree_root is not None:
        os.environ["HANTA_EVO_TREE_ROOT"] = str(args.tree_root)
    if args.labels is not None:
        os.environ["HANTA_EVO_MAIN_LABELS"] = str(args.labels)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    path = SCRIPT_DIR / "08_fixation_establishment_model.py"
    spec = importlib.util.spec_from_file_location("egt_model08_for_epistasis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_tip_set(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return set()
    return {str(v) for v in parsed}


def standardize_columns(df: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    out = df.copy()
    ref: Dict[str, Dict[str, float]] = {}
    for feature in features:
        values = pd.to_numeric(out[feature], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median)
        std = float(filled.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        mean = float(filled.mean())
        out[f"{feature}_z"] = (filled - mean) / std
        out[f"{feature}_missing"] = values.isna().astype(int)
        ref[feature] = {"median": median, "mean": mean, "std": std}
    return out, ref


def zmean(df: pd.DataFrame, columns: Sequence[str], signs: Optional[Sequence[float]] = None) -> pd.Series:
    signs = list(signs or [1.0] * len(columns))
    values = []
    for column, sign in zip(columns, signs):
        values.append(sign * pd.to_numeric(df[column], errors="coerce"))
    return pd.concat(values, axis=1).mean(axis=1)


def add_event_means(events: pd.DataFrame, sequence_df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = events.copy()
    seq_by_tip = sequence_df.set_index("tip_name", drop=False)
    rows: List[Dict[str, float]] = []
    for event in out.itertuples(index=False):
        tips = parse_tip_set(event.tip_set_json)
        sub = seq_by_tip.loc[seq_by_tip.index.intersection(sorted(tips))]
        row: Dict[str, float] = {}
        for feature in features:
            if feature in out.columns and not str(feature).startswith("species_feat_epi_") and not str(feature).startswith("epi_selected_"):
                continue
            if feature in sub.columns and not sub.empty:
                row[feature] = float(pd.to_numeric(sub[feature], errors="coerce").mean())
            elif feature not in out.columns:
                row[feature] = np.nan
        rows.append(row)
    mean_df = pd.DataFrame(rows)
    for col in mean_df.columns:
        out[col] = mean_df[col].to_numpy()
    return out


def add_compact_features(events: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    out, base_ref = standardize_columns(events, BASE_EVENT_MEAN_FEATURES)
    out["ecological_opportunity"] = zmean(out, ["overlap_opportunity_local_z", "arha_prev_local_z"])
    out["ld_compatibility"] = -pd.to_numeric(out["species_feat_mean_cramers_v_overall_z"], errors="coerce")
    out["epistasis_compatibility"] = zmean(
        out,
        [
            "species_feat_epi_time_mean_score_z",
            "species_feat_epi_time_max_score_z",
            "species_feat_epi_final_time_selected_pair_per_100k_site_pairs_z",
            "species_feat_epi_final_time_min_fdr_neglog10_z",
            "epi_selected_time_aa_rare_state_fraction_z",
        ],
    )
    out["species_utr_compatibility"] = -zmean(out, ["utr5_structural_distance_z", "utr3_structural_distance_z"])
    out["terminal_utr_regime"] = zmean(
        out,
        ["terminal_utr5_length_disparity_z", "terminal_utr3_length_disparity_z"],
        signs=[-1.0, 1.0],
    )
    out["molecular_permissiveness_ld"] = zmean(
        out,
        ["ld_compatibility", "species_utr_compatibility", "terminal_utr_regime"],
    )
    out["molecular_permissiveness_epistasis"] = zmean(
        out,
        ["epistasis_compatibility", "species_utr_compatibility", "terminal_utr_regime"],
    )
    out["ecology_x_molecular_permissiveness_ld"] = (
        pd.to_numeric(out["ecological_opportunity"], errors="coerce")
        * pd.to_numeric(out["molecular_permissiveness_ld"], errors="coerce")
    )
    out["ecology_x_molecular_permissiveness_epistasis"] = (
        pd.to_numeric(out["ecological_opportunity"], errors="coerce")
        * pd.to_numeric(out["molecular_permissiveness_epistasis"], errors="coerce")
    )
    out, compact_ref = standardize_columns(out, COMPACT_FEATURES)
    ref = {
        **{f"base::{key}": value for key, value in base_ref.items()},
        **{f"compact::{key}": value for key, value in compact_ref.items()},
    }
    return out, ref


def build_matched_events(model08) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df, lineage_df, origin_summary = model08.build_base_dataframe()
    df = model08.add_workhorse_terminal_utr_features(df)
    ref = model08.numeric_reference(df)
    analysis = model08.apply_numeric_reference(df, ref)
    graph_nodes, graph_edges = model08.lineage_graph_tables(analysis)
    denested_origin_summary, denesting_manifest = model08.denest_origin_summary(origin_summary)
    origin_events = model08.positive_origin_events(analysis, denested_origin_summary, graph_nodes)
    all_controls = model08.nonreassortant_internal_nodes(analysis, graph_nodes)
    matched_controls = model08.match_controls(origin_events, all_controls)
    origin_events = origin_events.assign(established=1, matched_origin_id=pd.NA, match_score=0.0)
    matched_controls = matched_controls.assign(established=0)
    matched = pd.concat([origin_events, matched_controls], ignore_index=True, sort=False)
    matched = add_event_means(matched, df, BASE_EVENT_MEAN_FEATURES)
    matched, _ = add_compact_features(matched)
    return matched, origin_events, all_controls, denesting_manifest


def fit_establishment(df: pd.DataFrame, spec: EgtSpec, args: argparse.Namespace) -> tuple[bmb.Model, az.InferenceData, str, List[str]]:
    terms = [
        "ecological_opportunity_z",
        f"{spec.molecular_feature}_z",
        f"{spec.interaction_feature}_z",
        "tree_discordance_score_z",
        "graph_neighbor_overlap_z",
        "log_sampling_effort_z",
    ]
    formula = "established ~ " + " + ".join(terms) + " + (1|species) + (1|dominant_lineage_id)"
    model = bmb.Model(formula, df, family="bernoulli", priors=COMMON_PRIORS)
    idata = model.fit(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
        idata_kwargs={"log_likelihood": True},
    )
    return model, idata, formula, terms


def sign_probability(idata: az.InferenceData, term: str) -> float:
    values = np.asarray(idata.posterior[term]).reshape(-1)
    return float((values > 0).mean())


def model_diagnostics(idata: az.InferenceData) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        loo = az.loo(idata)
        out["loo_elpd"] = float(loo.elpd_loo)
        out["loo_p"] = float(loo.p_loo)
        out["loo_se"] = float(loo.se)
    except Exception:
        out["loo_elpd"] = np.nan
        out["loo_p"] = np.nan
        out["loo_se"] = np.nan
    if "diverging" in idata.sample_stats:
        out["n_divergences"] = int(np.asarray(idata.sample_stats["diverging"]).sum())
    else:
        out["n_divergences"] = np.nan
    return out


def summarize_terms(idata: az.InferenceData, terms: Sequence[str], model_name: str) -> pd.DataFrame:
    summary = az.summary(idata, var_names=list(terms), kind="stats", hdi_prob=0.94)
    rows = []
    for term in terms:
        if term not in summary.index:
            continue
        rows.append(
            {
                "model": model_name,
                "term": term,
                "mean": float(summary.loc[term, "mean"]),
                "sd": float(summary.loc[term, "sd"]),
                "hdi_3%": float(summary.loc[term, "hdi_3%"]),
                "hdi_97%": float(summary.loc[term, "hdi_97%"]),
                "posterior_prob_positive": sign_probability(idata, term),
            }
        )
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.output_dir / "datasets"
    report_dir = args.output_dir / "reports"
    dataset_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    model08 = load_model08(args)
    matched, origin_events, controls, denesting_manifest = build_matched_events(model08)
    matched.to_csv(dataset_dir / "13_egt_ld_epistasis_event_dataset.tsv", sep="\t", index=False)
    origin_events.to_csv(report_dir / "13_origin_events.tsv", sep="\t", index=False)
    controls.to_csv(report_dir / "13_background_control_candidates.tsv", sep="\t", index=False)
    denesting_manifest.to_csv(report_dir / "13_denesting_manifest.tsv", sep="\t", index=False)

    comparison_rows: List[Dict[str, object]] = []
    coefficient_frames: List[pd.DataFrame] = []
    formulas: Dict[str, str] = {}
    for spec in SPECS:
        formula_terms = [
            "ecological_opportunity_z",
            f"{spec.molecular_feature}_z",
            f"{spec.interaction_feature}_z",
            "tree_discordance_score_z",
            "graph_neighbor_overlap_z",
            "log_sampling_effort_z",
        ]
        formula = "established ~ " + " + ".join(formula_terms) + " + (1|species) + (1|dominant_lineage_id)"
        formulas[spec.name] = formula
        model_dir = report_dir / spec.name
        model_dir.mkdir(exist_ok=True)
        (model_dir / "formula.txt").write_text(formula + "\n")
        posterior_path = model_dir / "posterior.nc"
        if posterior_path.exists() and not args.force:
            idata = az.from_netcdf(posterior_path)
            terms = formula_terms
        else:
            _, idata, formula, terms = fit_establishment(matched, spec, args)
            idata.to_netcdf(posterior_path)
        az.summary(idata, var_names=list(idata.posterior.data_vars), round_to=4).to_csv(
            model_dir / "posterior_summary.tsv",
            sep="\t",
        )
        coefficient_frames.append(summarize_terms(idata, terms, spec.name))
        comparison_rows.append(
            {
                "model": spec.name,
                "molecular_feature": spec.molecular_feature,
                "interaction_feature": spec.interaction_feature,
                "formula": formula,
                "rows": int(len(matched)),
                "established_events": int(pd.to_numeric(matched["established"], errors="coerce").sum()),
                **model_diagnostics(idata),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    comparison.to_csv(report_dir / "13_egt_ld_epistasis_model_comparison.tsv", sep="\t", index=False)
    coefficients.to_csv(report_dir / "13_egt_ld_epistasis_establishment_coefficients.tsv", sep="\t", index=False)
    (report_dir / "13_egt_ld_epistasis_formulas.json").write_text(json.dumps(formulas, indent=2, sort_keys=True) + "\n")
    run_info = {
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "draws": int(args.draws),
        "tune": int(args.tune),
        "chains": int(args.chains),
        "cores": int(args.cores),
        "rows": int(len(matched)),
        "established_events": int(pd.to_numeric(matched["established"], errors="coerce").sum()),
        "comparison_models": [spec.__dict__ for spec in SPECS],
    }
    (report_dir / "13_egt_ld_epistasis_run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
