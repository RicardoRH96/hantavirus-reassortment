from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import arviz as az
import bambi as bmb
import matplotlib as mpl
import numpy as np
import pandas as pd
import tskit
from Bio import Phylo, SeqIO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "results" / "evolutionary_robustness"
DATASET_DIR = OUTPUT_DIR / "datasets"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"

FINAL_SEQUENCE_DATASET_PATH = REPO_ROOT / "data" / "processed" / "final_sequence_dataset.tsv"
LABEL_DIR = REPO_ROOT / "data" / "processed" / "labels"
MAIN_LABELS_PATH = LABEL_DIR / "sequence_true_labels.tsv"
MULTI_FAST_LABELS_PATH = LABEL_DIR / "alternative_multispecies_fast_sequence_true_labels.tsv"
MULTI_FULL_LABELS_PATH = LABEL_DIR / "alternative_multispecies_full_sequence_true_labels.tsv"
ANDES_LABELS_PATH = LABEL_DIR / "andes_sequence_true_labels.tsv"
SNV_LABELS_PATH = LABEL_DIR / "sin_nombre_sequence_true_labels.tsv"

SPECIES_ORDER = [
    "orthohantavirus_andesense",
    "orthohantavirus_dobravaense",
    "orthohantavirus_hantanense",
    "orthohantavirus_puumalaense",
    "orthohantavirus_seoulense",
    "orthohantavirus_sinnombreense",
    "orthohantavirus_tulaense",
]
SPECIES_LABELS = {
    "orthohantavirus_andesense": "Andes",
    "orthohantavirus_dobravaense": "Dobrava",
    "orthohantavirus_hantanense": "Hantaan",
    "orthohantavirus_puumalaense": "Puumala",
    "orthohantavirus_seoulense": "Seoul",
    "orthohantavirus_sinnombreense": "Sin Nombre",
    "orthohantavirus_tulaense": "Tula",
}
TREE_RUN_ROOTS = {
    species: REPO_ROOT / "data" / "processed" / "trees" / species
    for species in SPECIES_ORDER
}
TRUE_RUN_ROOTS = {
    species: REPO_ROOT / "data" / "processed" / "arg" / species
    for species in SPECIES_ORDER
}
SEGMENTS = ("S", "M", "L")
PASTEL = {
    "Andes": "#d8c6f2",
    "Dobrava": "#f2c6d8",
    "Hantaan": "#c6def1",
    "Puumala": "#d4f0c0",
    "Seoul": "#f7d6b6",
    "Sin Nombre": "#fde2e4",
    "Tula": "#e2ece9",
}
KEY_FEATURES = [
    "overlap_opportunity_local",
    "arha_prev_local",
    "species_feat_hostdiv_faith_pd",
    "species_feat_mean_cramers_v_overall",
    "species_feat_prop_significant_overall",
    "species_feat_utr5_S_structdist_mean",
    "species_feat_utr5_M_structdist_mean",
    "species_feat_utr5_L_structdist_mean",
    "tree_discordance_score",
]
UTR3_FEATURES = [
    "species_feat_utr3_S_structdist_mean",
    "species_feat_utr3_M_structdist_mean",
    "species_feat_utr3_L_structdist_mean",
]
KEY_FEATURES_PLUS_UTR3 = KEY_FEATURES[:-1] + UTR3_FEATURES + [KEY_FEATURES[-1]]
ECOLOGY_FEATURES = [
    "overlap_opportunity_local",
    "arha_prev_local",
    "species_feat_hostdiv_faith_pd",
]
MOLECULAR_FEATURES = [
    "species_feat_mean_cramers_v_overall",
    "species_feat_prop_significant_overall",
    "species_feat_utr5_S_structdist_mean",
    "species_feat_utr5_M_structdist_mean",
    "species_feat_utr5_L_structdist_mean",
    "tree_discordance_score",
]
MOLECULAR_FEATURES_PLUS_UTR3 = MOLECULAR_FEATURES[:-1] + UTR3_FEATURES + [MOLECULAR_FEATURES[-1]]
COMMON_PRIORS = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1.5),
    "common": bmb.Prior("Normal", mu=0, sigma=1.0),
}
DISPLAY_NAMES = {
    "overlap_opportunity_local": "Local overlap",
    "arha_prev_local": "Local prevalence",
    "species_feat_hostdiv_faith_pd": "Host phylo diversity",
    "species_feat_mean_cramers_v_overall": "LD burden",
    "species_feat_prop_significant_overall": "LD support",
    "species_feat_utr5_S_structdist_mean": "UTR5 S distance",
    "species_feat_utr5_M_structdist_mean": "UTR5 M distance",
    "species_feat_utr5_L_structdist_mean": "UTR5 L distance",
    "species_feat_utr3_S_structdist_mean": "UTR3 S distance",
    "species_feat_utr3_M_structdist_mean": "UTR3 M distance",
    "species_feat_utr3_L_structdist_mean": "UTR3 L distance",
    "tree_discordance_score": "Tree discordance",
}


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, DATASET_DIR, REPORT_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["legend.fontsize"] = 9


def species_color(species: str) -> str:
    return PASTEL[SPECIES_LABELS[species]]


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def format_value(value: object, precision: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if abs(float(value) - round(float(value))) < 1e-9:
            return str(int(round(float(value))))
        return f"{float(value):.{precision}f}"
    return str(value)


def render_dataframe_markdown(title: str, df: pd.DataFrame, precision: int = 3) -> str:
    display = df.copy()
    cols = list(display.columns)
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(v, precision=precision) for v in row) + " |")
    lines.append("")
    return "\n".join(lines)


def stable_hash(values: Sequence[str]) -> str:
    text = "||".join(values)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def load_final_sequence_dataset() -> pd.DataFrame:
    df = pd.read_csv(FINAL_SEQUENCE_DATASET_PATH, sep="\t", low_memory=False)
    df = df[df["target_source"] == "sequence_true_label"].copy()
    df["tip_name"] = df["tip_name"].astype(str)
    df["species"] = df["species"].astype(str)
    df["species_label"] = df["species"].map(SPECIES_LABELS)
    df["country"] = df["country"].fillna("Unknown").astype(str)
    df["host"] = df["host"].fillna("Unknown").astype(str)
    df["site_key"] = df["site_key"].fillna("Unknown").astype(str)
    df["event_key"] = df["event_key"].fillna("Unknown").astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in KEY_FEATURES + ["is_reassortant_final"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_label_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns={"sequence_id": "tip_name", "is_reassortant": "response"})
    df["tip_name"] = df["tip_name"].astype(str)
    df["species"] = df["species"].astype(str)
    df["response"] = pd.to_numeric(df["response"], errors="coerce").fillna(0).astype(int)
    if "descriptive_distance_to_reassort_node" in df.columns:
        df["descriptive_distance_to_reassort_node"] = pd.to_numeric(df["descriptive_distance_to_reassort_node"], errors="coerce")
    return df


def build_combined_label_table(main_multi_path: Path) -> pd.DataFrame:
    multi = load_label_table(main_multi_path)
    andes = load_label_table(ANDES_LABELS_PATH)
    snv = load_label_table(SNV_LABELS_PATH)
    combined = pd.concat([multi, andes, snv], ignore_index=True)
    return combined[["species", "tip_name", "response"]].drop_duplicates(subset=["species", "tip_name"], keep="last")


def alternative_label_sets() -> Dict[str, pd.DataFrame]:
    main = load_label_table(MAIN_LABELS_PATH)
    output = {
        "current_main": main[["species", "tip_name", "response"]],
        "backfill_fast": build_combined_label_table(MULTI_FAST_LABELS_PATH),
        "backfill_full": build_combined_label_table(MULTI_FULL_LABELS_PATH),
    }
    if "descriptive_distance_to_reassort_node" in main.columns:
        dist = main["descriptive_distance_to_reassort_node"]
        for threshold in [1, 3]:
            alt = main[["species", "tip_name"]].copy()
            alt["response"] = ((dist <= threshold) & dist.notna()).astype(int)
            output[f"distance_leq_{threshold}"] = alt
    return output


def prune_tree_to_tips(tree, keep_tips: set[str]):
    for term in list(tree.get_terminals()):
        if term.name not in keep_tips:
            tree.prune(term)
    return tree


def load_species_tree(species: str):
    tree_path = TREE_RUN_ROOTS[species] / "augur" / "S" / "S_tree_refined.newick"
    return Phylo.read(str(tree_path), "newick")


def lineage_assignments(df: pd.DataFrame, min_size: int = 8, max_size: int = 40) -> pd.DataFrame:
    rows: List[dict] = []
    for species in SPECIES_ORDER:
        sp = df[df["species"] == species].copy()
        if sp.empty:
            continue
        keep_tips = set(sp["tip_name"].astype(str))
        tree = prune_tree_to_tips(load_species_tree(species), keep_tips)
        parent_map = {}
        leaf_sizes = {}
        leaf_keys = {}
        for clade in tree.find_clades(order="level"):
            leafset = tuple(sorted(t.name for t in clade.get_terminals() if t.name))
            leaf_sizes[clade] = len(leafset)
            leaf_keys[clade] = leafset
            for child in getattr(clade, "clades", []):
                parent_map[child] = clade

        tip_nodes = {tip.name: tip for tip in tree.get_terminals() if tip.name}
        for tip_name, tip in tip_nodes.items():
            candidates = []
            cur = tip
            while cur in parent_map:
                cur = parent_map[cur]
                size = leaf_sizes[cur]
                if min_size <= size <= max_size:
                    candidates.append(cur)
            if candidates:
                chosen = candidates[0]
                source = "bounded_clade"
            else:
                cur = tip
                chosen = None
                while cur in parent_map:
                    cur = parent_map[cur]
                    if leaf_sizes[cur] > 1:
                        chosen = cur
                        break
                source = "nearest_internal" if chosen is not None else "singleton"

            if chosen is None:
                lineage_id = f"{species}|singleton|{tip_name}"
                lineage_size = 1
            else:
                leafset = leaf_keys[chosen]
                lineage_id = f"{species}|lin_{stable_hash(leafset)}"
                lineage_size = len(leafset)
            rows.append(
                {
                    "species": species,
                    "tip_name": tip_name,
                    "lineage_id": lineage_id,
                    "lineage_size": lineage_size,
                    "lineage_source": source,
                }
            )
    return pd.DataFrame(rows)


def build_segment_spans(species: str) -> tuple[Dict[str, tuple[float, float]], int]:
    spans: Dict[str, tuple[float, float]] = {}
    left = 0.0
    genome_length = 0
    for seg in SEGMENTS:
        aln_path = TREE_RUN_ROOTS[species] / "augur" / seg / f"{seg}_aligned.fasta"
        first = next(SeqIO.parse(str(aln_path), "fasta"), None)
        if first is None:
            raise RuntimeError(f"Empty alignment: {aln_path}")
        seg_len = len(first.seq)
        spans[seg] = (left, left + float(seg_len))
        left += float(seg_len)
        genome_length += seg_len
    return spans, genome_length


def midpoint(span: tuple[float, float], sequence_length: float) -> float:
    left, right = span
    mid = left + ((right - left) / 2.0)
    return min(mid, max(sequence_length - 1e-6, 0.0))


def origin_cluster_assignments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_labels = load_label_table(MAIN_LABELS_PATH)[["species", "tip_name", "response"]]
    positives = set(main_labels.loc[main_labels["response"] == 1, "tip_name"].astype(str))
    cluster_candidates: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    cluster_rows: list[dict] = []

    for species in SPECIES_ORDER:
        species_labels = load_label_table(MAIN_LABELS_PATH)
        species_labels = species_labels[species_labels["species"] == species].sort_values("tip_name").reset_index(drop=True)
        if species_labels.empty:
            continue
        ts = tskit.load(str(TRUE_RUN_ROOTS[species] / "arg_em.trees"))
        recomb_flag = getattr(tskit, "NODE_IS_RECOMBINATION", 131072)
        spans, _ = build_segment_spans(species)
        sample_ids = list(ts.samples())
        sample_to_tip = dict(zip(sample_ids, species_labels["tip_name"].astype(str)))
        seen_cluster_ids = set()

        for node_id, node in enumerate(ts.nodes()):
            if not (node.flags & recomb_flag):
                continue
            for seg in SEGMENTS:
                tree = ts.at(midpoint(spans[seg], ts.sequence_length))
                if node_id not in tree.nodes():
                    continue
                descendants = sorted(sample_to_tip[s] for s in tree.samples(node_id) if s in sample_to_tip)
                if len(descendants) < 2:
                    continue
                positive_desc = sorted(tip for tip in descendants if tip in positives)
                if not positive_desc:
                    continue
                cluster_id = f"{species}|origin_{stable_hash(positive_desc)}"
                if cluster_id not in seen_cluster_ids:
                    cluster_rows.append(
                        {
                            "origin_cluster_id": cluster_id,
                            "species": species,
                            "species_label": SPECIES_LABELS[species],
                            "positive_tip_n": len(positive_desc),
                            "all_descendant_n": len(descendants),
                            "arg_node_id": node_id,
                            "arg_time": float(node.time),
                            "tip_set_json": json.dumps(positive_desc),
                        }
                    )
                    seen_cluster_ids.add(cluster_id)
                for tip in positive_desc:
                    cluster_candidates[tip].append((len(positive_desc), float(node.time), cluster_id))

    assignments = []
    for row in df.itertuples(index=False):
        tip = str(row.tip_name)
        if int(row.response) == 1 and tip in cluster_candidates:
            _, _, cluster_id = sorted(cluster_candidates[tip], key=lambda x: (x[0], x[1], x[2]))[0]
        elif int(row.response) == 1:
            cluster_id = f"{row.species}|origin_singleton|{tip}"
        else:
            cluster_id = f"{row.species}|nonreassortant|{tip}"
        assignments.append({"species": row.species, "tip_name": tip, "origin_cluster_id": cluster_id})
    return pd.DataFrame(assignments), pd.DataFrame(cluster_rows).drop_duplicates(subset=["origin_cluster_id"])


def build_base_dataframe() -> pd.DataFrame:
    base = load_final_sequence_dataset()
    current_labels = load_label_table(MAIN_LABELS_PATH)[["species", "tip_name", "response"]]
    df = base.drop(columns=["is_reassortant_final"]).merge(current_labels, on=["species", "tip_name"], how="left")
    df["response"] = df["response"].fillna(0).astype(int)
    lineage_df = lineage_assignments(df)
    origin_df, origin_summary = origin_cluster_assignments(df)
    df = df.merge(lineage_df, on=["species", "tip_name"], how="left")
    df = df.merge(origin_df, on=["species", "tip_name"], how="left")
    df["species_country"] = df["species_label"] + " / " + df["country"]
    df["site_year_host_key"] = (
        df["species"].astype(str)
        + "|"
        + df["site_key"].astype(str)
        + "|"
        + df["year"].fillna(-1).astype(int).astype(str)
        + "|"
        + df["host"].astype(str)
    )
    return df, lineage_df, origin_summary


def numeric_reference(df: pd.DataFrame, features: Sequence[str] = KEY_FEATURES) -> Dict[str, dict]:
    ref = {}
    for feature in features:
        series = pd.to_numeric(df[feature], errors="coerce")
        filled = series.fillna(series.median())
        std = float(filled.std(ddof=0))
        ref[feature] = {
            "median": float(series.median()) if series.notna().any() else 0.0,
            "mean": float(filled.mean()),
            "std": std if std > 0 else 1.0,
        }
    return ref


def apply_numeric_reference(df: pd.DataFrame, ref: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for feature in ref:
        values = pd.to_numeric(out[feature], errors="coerce")
        out[f"{feature}_missing"] = values.isna().astype(int)
        filled = values.fillna(ref[feature]["median"])
        out[f"{feature}_z"] = (filled - ref[feature]["mean"]) / ref[feature]["std"]
    return out


def active_feature_terms(df: pd.DataFrame, features: Sequence[str]) -> list[str]:
    terms: list[str] = []
    for feature in features:
        z_col = f"{feature}_z"
        miss_col = f"{feature}_missing"
        if z_col in df.columns and pd.to_numeric(df[z_col], errors="coerce").nunique(dropna=True) > 1:
            terms.append(z_col)
        if miss_col in df.columns and pd.to_numeric(df[miss_col], errors="coerce").nunique(dropna=True) > 1:
            terms.append(miss_col)
    return terms


def build_bayesian_formula(df: pd.DataFrame, features: Sequence[str], group_effects: Sequence[str] | None = None) -> tuple[str, list[str]]:
    fixed_terms = active_feature_terms(df, features)
    rhs = list(fixed_terms)
    for group in group_effects or []:
        rhs.append(f"(1|{group})")
    if not rhs:
        rhs = ["1"]
    return f"response ~ {' + '.join(rhs)}", fixed_terms


def fit_bayesian_logistic(
    df: pd.DataFrame,
    features: Sequence[str],
    group_effects: Sequence[str] | None = None,
    draws: int = 300,
    tune: int = 300,
    chains: int = 2,
    cores: int = 2,
    target_accept: float = 0.95,
    random_seed: int = 42,
):
    formula, fixed_terms = build_bayesian_formula(df, features, group_effects=group_effects)
    model = bmb.Model(formula, df, family="bernoulli", priors=COMMON_PRIORS)
    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=random_seed,
        idata_kwargs={"log_likelihood": True},
    )
    return model, idata, formula, fixed_terms


def posterior_sign_probability(idata, term: str) -> float:
    values = np.asarray(idata.posterior[term]).reshape(-1)
    return float((values > 0).mean())


def summarize_bayesian_terms(idata, terms: Sequence[str], model_name: str) -> pd.DataFrame:
    rows = []
    if not terms:
        return pd.DataFrame(columns=["model", "term", "mean", "sd", "hdi_3%", "hdi_97%", "posterior_prob_positive"])
    summary = az.summary(idata, var_names=list(terms), kind="stats", hdi_prob=0.94)
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
                "posterior_prob_positive": posterior_sign_probability(idata, term),
            }
        )
    return pd.DataFrame(rows)


def design_matrix(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    category_levels: Mapping[str, Sequence[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    frames = []
    names: list[str] = []
    for feature in numeric_features:
        z_col = f"{feature}_z"
        miss_col = f"{feature}_missing"
        frames.append(df[[z_col]].astype(float))
        names.append(feature)
        if df[miss_col].nunique(dropna=False) > 1:
            frames.append(df[[miss_col]].astype(float))
            names.append(f"{feature}_missing")
    categorical_features = list(categorical_features or [])
    category_levels = dict(category_levels or {})
    for feature in categorical_features:
        cats = category_levels.get(feature)
        series = pd.Series(
            pd.Categorical(df[feature].astype(str), categories=cats if cats is not None else None),
            index=df.index,
        )
        dummies = pd.get_dummies(series, prefix=feature, drop_first=True, dtype=float)
        if not dummies.empty:
            frames.append(dummies)
            names.extend(dummies.columns.tolist())
    X = pd.concat(frames, axis=1) if frames else pd.DataFrame({"__constant__": np.zeros(len(df), dtype=float)}, index=df.index)
    y = df["response"].astype(int).to_numpy()
    return X.to_numpy(dtype=float), y, names


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    penalty_strength: float = 0.1,
) -> dict:
    model = LogisticRegression(
        penalty="l2",
        C=penalty_strength,
        solver="lbfgs",
        max_iter=5000,
        fit_intercept=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    null_p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
    null_loglik = float(np.sum(y * np.log(null_p) + (1 - y) * np.log(1 - null_p)))
    model_loglik = float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))
    coefs = {name: float(value) for name, value in zip(feature_names, model.coef_[0])}
    return {
        "model": model,
        "coefficients": coefs,
        "intercept": float(model.intercept_[0]),
        "probabilities": probs,
        "loglik": model_loglik,
        "null_loglik": null_loglik,
        "pseudo_r2_mcfadden": float(1.0 - (model_loglik / null_loglik)),
        "log_loss": float(log_loss(y, probs)),
    }


def bootstrap_coefficients(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    category_levels: Mapping[str, Sequence[str]],
    resample_ids: Sequence[str],
    cluster_col: str,
    focal_terms: Sequence[str],
    iterations: int = 200,
    penalty_strength: float = 1.0,
    sample_one_per_positive_cluster: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    positive_clusters = set(df.loc[df["response"] == 1, cluster_col].astype(str))
    unique_resample_ids = list(dict.fromkeys(str(x) for x in resample_ids))
    all_cluster_map = {cluster: frame.copy() for cluster, frame in df.groupby(cluster_col, sort=False)}

    for iteration in range(iterations):
        if sample_one_per_positive_cluster:
            sampled_parts = []
            for cluster in positive_clusters:
                frame = all_cluster_map[cluster]
                sampled_parts.append(frame.sample(n=1, random_state=int(rng.integers(1_000_000))))
            sampled_parts.append(df[df["response"] == 0].copy())
            sample = pd.concat(sampled_parts, ignore_index=True)
        else:
            sampled_clusters = rng.choice(np.array(unique_resample_ids), size=len(unique_resample_ids), replace=True)
            sampled_parts = [all_cluster_map[cluster] for cluster in sampled_clusters]
            sample = pd.concat(sampled_parts, ignore_index=True)
        if sample["response"].nunique() < 2:
            continue
        X, y, names = design_matrix(sample, numeric_features, categorical_features, category_levels)
        fit = fit_logistic(X, y, names, penalty_strength=penalty_strength)
        for term in focal_terms:
            rows.append(
                {
                    "iteration": iteration,
                    "term": term,
                    "coefficient": fit["coefficients"].get(term, math.nan),
                    "scheme": "one_tip_per_origin" if sample_one_per_positive_cluster else "cluster_bootstrap",
                }
            )
    return pd.DataFrame(rows)


def coefficient_interval_table(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(list(group_cols), dropna=False)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n"] = len(sub)
        row["mean"] = float(sub["coefficient"].mean())
        row["q05"] = float(sub["coefficient"].quantile(0.05))
        row["q50"] = float(sub["coefficient"].quantile(0.50))
        row["q95"] = float(sub["coefficient"].quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def top_countries(df: pd.DataFrame, min_n: int = 20, top_n: int = 5) -> list[str]:
    counts = df["country"].value_counts()
    return counts[counts >= min_n].head(top_n).index.tolist()
