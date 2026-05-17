#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_SCRIPT_DIR = REPO_ROOT / "scripts" / "figures"
for path in (SCRIPT_DIR, FIGURE_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_sequence_model_suite import ModelSpec, fit_pipeline  # noqa: E402
from plot_integrated_explanation_figure import (  # noqa: E402
    build_module_matrix,
    class_rank_shift,
    cluster_top_features,
    load_labeled_dataset,
    mean_feature_ranks,
)
from sequence_model_suite_explanatory_utils import (  # noqa: E402
    FEATURE_SETS,
    apply_ld_encoding,
    build_feature_frame,
    detect_feature_blocks,
)


DEFAULT_DATASET = (
    REPO_ROOT
    / "data"
    / "processed"
    / "final_sequence_dataset.tsv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "modeling"
    / "integrated_explanation_bundle"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ld-encoding", choices=["full", "interaction_only"], default="full")
    parser.add_argument("--top-features", type=int, default=28)
    parser.add_argument("--top-clusters", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_labeled_dataset(args.dataset)
    feature_set = next(fs for fs in FEATURE_SETS if fs.name == "ecology_molecular_lineage_core")
    blocks = apply_ld_encoding(detect_feature_blocks(df), args.ld_encoding)
    X, numeric, categorical = build_feature_frame(df, feature_set, blocks)
    y = df["target"].to_numpy(dtype=int)

    model_spec = ModelSpec(name="lightgbm", kind="lightgbm")
    bundle = fit_pipeline(model_spec, X, y, numeric, categorical, args.random_state)
    x_t = bundle.preprocessor.transform(X)
    model = bundle.model
    try:
        shap_matrix = np.asarray(model.predict(x_t, pred_contrib=True))
    except TypeError:
        shap_matrix = np.asarray(model.booster_.predict(x_t, pred_contrib=True))
    contributions = shap_matrix[:, :-1]
    bias = shap_matrix[:, -1]
    logits = contributions.sum(axis=1) + bias
    probabilities = bundle.predict_proba(X)[:, 1]
    feature_names = np.asarray(bundle.feature_names)

    contrib_df = pd.DataFrame(contributions, columns=feature_names)
    contrib_df.insert(0, "probability", probabilities)
    contrib_df.insert(0, "logit", logits)
    contrib_df.insert(0, "target", y)
    contrib_df.insert(0, "species", df["species"].to_numpy())
    contrib_df.insert(0, "tip_name", df["tip_name"].to_numpy())
    contrib_df.to_csv(args.output_dir / "integrated_exact_feature_contributions.tsv", sep="\t", index=False)

    module_df = build_module_matrix(feature_names, contributions)
    module_out = module_df.copy()
    module_out.insert(0, "probability", probabilities)
    module_out.insert(0, "target", y)
    module_out.insert(0, "species", df["species"].to_numpy())
    module_out.insert(0, "tip_name", df["tip_name"].to_numpy())
    module_out.to_csv(args.output_dir / "integrated_module_contributions.tsv", sep="\t", index=False)

    rank_df = mean_feature_ranks(feature_names, contributions)
    rank_df.to_csv(args.output_dir / "integrated_feature_rank_summary.tsv", sep="\t", index=False)

    shift_df = class_rank_shift(feature_names, contributions, y)
    shift_df.to_csv(args.output_dir / "integrated_feature_rank_shift.tsv", sep="\t", index=False)

    cluster_df = cluster_top_features(feature_names, contributions, args.top_features, args.top_clusters)
    cluster_df.to_csv(args.output_dir / "integrated_top_feature_clusters.tsv", sep="\t", index=False)

    manifest = {
        "dataset": str(args.dataset),
        "feature_set": "ecology_molecular_lineage_core",
        "model": "lightgbm",
        "rows": int(len(df)),
        "positives": int(y.sum()),
        "feature_count": int(len(feature_names)),
        "ld_encoding": args.ld_encoding,
        "top_features": args.top_features,
        "top_clusters": args.top_clusters,
        "note": "Notebook-compatible Figure 3VB data bundle using the biology-only repeated-stratified feature set.",
    }
    (args.output_dir / "integrated_explanation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(args.output_dir)


if __name__ == "__main__":
    main()
