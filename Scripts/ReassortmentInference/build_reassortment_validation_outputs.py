#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import baltic as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tskit
from Bio import Phylo, SeqIO


REPO_ROOT = Path(__file__).resolve().parents[2]
DIRECT516_PATH = Path(
    os.environ.get(
        "HANTA_EVO_DATASET",
        REPO_ROOT
        / "model_test"
        / "runs"
        / "model_site"
        / "datasets"
        / "sequence_level_final_dataset_combined_relaxed_andesense_all_backfills_full_geooverlap_manual_plus_tula_coords_direct516.tsv",
    )
).expanduser()
SEQUENCE_TRUE_LABELS_PATH = Path(
    os.environ.get(
        "HANTA_EVO_MAIN_LABELS",
        REPO_ROOT / "model_test" / "runs" / "model_sequence" / "datasets" / "sequence_true_labels.tsv",
    )
).expanduser()
REPORT_DIR = REPO_ROOT / "model_test" / "reports"
FIGURE_DIR = REPO_ROOT / "Notebooks" / "figures"

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
_TREE_ROOT_OVERRIDE = os.environ.get("HANTA_EVO_TREE_ROOT", "").strip()
_TRUE_ROOT_OVERRIDE = os.environ.get("HANTA_EVO_TRUE_ROOT", "").strip()


def _species_root(env_root: str, default: Path, species: str) -> Path:
    return Path(env_root).expanduser() / species if env_root else default


TREE_RUN_ROOTS = {
    "orthohantavirus_andesense": _species_root(_TREE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "andes_backfill_espalier_full" / "orthohantavirus_andesense", "orthohantavirus_andesense"),
    "orthohantavirus_dobravaense": _species_root(_TREE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "multi_species_backfill_espalier" / "orthohantavirus_dobravaense", "orthohantavirus_dobravaense"),
    "orthohantavirus_hantanense": _species_root(_TREE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "multi_species_backfill_espalier" / "orthohantavirus_hantanense", "orthohantavirus_hantanense"),
    "orthohantavirus_puumalaense": _species_root(_TREE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "multi_species_backfill_espalier" / "orthohantavirus_puumalaense", "orthohantavirus_puumalaense"),
    "orthohantavirus_seoulense": _species_root(_TREE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "multi_species_backfill_espalier" / "orthohantavirus_seoulense", "orthohantavirus_seoulense"),
    "orthohantavirus_sinnombreense": _species_root(_TREE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "snv_backfill_espalier_full" / "orthohantavirus_sinnombreense", "orthohantavirus_sinnombreense"),
    "orthohantavirus_tulaense": _species_root(_TREE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "multi_species_backfill_espalier" / "orthohantavirus_tulaense", "orthohantavirus_tulaense"),
}
TRUE_RUN_ROOTS = {
    "orthohantavirus_andesense": _species_root(_TRUE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "andes_backfill_true_reassortment" / "orthohantavirus_andesense", "orthohantavirus_andesense"),
    "orthohantavirus_dobravaense": _species_root(_TRUE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "multi_species_backfill_true_reassortment_full" / "orthohantavirus_dobravaense", "orthohantavirus_dobravaense"),
    "orthohantavirus_hantanense": _species_root(_TRUE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "multi_species_backfill_true_reassortment_full" / "orthohantavirus_hantanense", "orthohantavirus_hantanense"),
    "orthohantavirus_puumalaense": _species_root(_TRUE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "multi_species_backfill_true_reassortment_full" / "orthohantavirus_puumalaense", "orthohantavirus_puumalaense"),
    "orthohantavirus_seoulense": _species_root(_TRUE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "multi_species_backfill_true_reassortment_full" / "orthohantavirus_seoulense", "orthohantavirus_seoulense"),
    "orthohantavirus_sinnombreense": _species_root(_TRUE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "snv_backfill_true_reassortment" / "orthohantavirus_sinnombreense", "orthohantavirus_sinnombreense"),
    "orthohantavirus_tulaense": _species_root(_TRUE_ROOT_OVERRIDE, REPO_ROOT / "model_test" / "runs" / "multi_species_backfill_true_reassortment_full" / "orthohantavirus_tulaense", "orthohantavirus_tulaense"),
}
PALETTE = ["#1b9e77", "#d95f02", "#7570b3"]
BLACK = "#111111"


def build_segment_spans(seg_order: Sequence[str], aln_paths: Sequence[Path]) -> tuple[Dict[str, tuple[float, float]], int]:
    spans: Dict[str, tuple[float, float]] = {}
    left = 0.0
    genome_length = 0
    for seg, aln_path in zip(seg_order, aln_paths):
        first = next(SeqIO.parse(str(aln_path), "fasta"), None)
        if first is None:
            raise RuntimeError(f"Empty alignment: {aln_path}")
        seg_len = len(first.seq)
        right = left + float(seg_len)
        spans[seg] = (left, right)
        left = right
        genome_length += seg_len
    return spans, genome_length


def segment_midpoint(span: tuple[float, float], sequence_length: float) -> float:
    left, right = span
    midpoint = left + ((right - left) / 2.0)
    max_position = max(sequence_length - 1e-6, 0.0)
    return min(midpoint, max_position)


def clade_leafset_biophylo(clade) -> frozenset[str]:
    return frozenset(t.name for t in clade.get_terminals() if t.name)


def load_direct516() -> pd.DataFrame:
    df = pd.read_csv(DIRECT516_PATH, sep="\t", low_memory=False)
    return df[df["target_source"] == "sequence_true_label"].copy()


def load_sequence_true_labels() -> pd.DataFrame:
    return pd.read_csv(SEQUENCE_TRUE_LABELS_PATH, sep="\t", low_memory=False)


def format_value(x: object) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (float, np.floating)) and math.isinf(float(x)):
        return "Inf"
    if isinstance(x, (float, np.floating)):
        if abs(float(x) - round(float(x))) < 1e-9:
            return str(int(round(float(x))))
        return f"{float(x):.2f}"
    return str(x)


def write_markdown_table(path: Path, frame: pd.DataFrame, title: str) -> None:
    display = frame.copy()
    for col in display.columns:
        display[col] = display[col].map(format_value)
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)]
    text = f"## {title}\n\n" + "\n".join([header, sep] + rows) + "\n"
    path.write_text(text)


def build_descriptive_tables() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_direct516()
    df["sequence_true_descriptive_distance_to_reassort_node"] = pd.to_numeric(
        df["sequence_true_descriptive_distance_to_reassort_node"], errors="coerce"
    )
    df["is_reassortant_final"] = pd.to_numeric(df["is_reassortant_final"], errors="coerce")
    df = df[df["is_reassortant_final"].notna()].copy()
    df["label"] = df["is_reassortant_final"].map({0.0: "non_reassortant", 1.0: "reassortant"})
    df["is_inf_distance"] = df["sequence_true_descriptive_distance_to_reassort_node"].map(np.isinf)
    finite = df[~df["is_inf_distance"]].copy()

    overall = (
        df.groupby("label", dropna=False)
        .agg(
            n=("label", "size"),
            finite_n=("is_inf_distance", lambda s: int((~s).sum())),
            inf_n=("is_inf_distance", lambda s: int(s.sum())),
        )
        .reset_index()
    )
    finite_overall = (
        finite.groupby("label", dropna=False)["sequence_true_descriptive_distance_to_reassort_node"]
        .agg(finite_min="min", finite_p25=lambda s: s.quantile(0.25), finite_median="median", finite_mean="mean", finite_p75=lambda s: s.quantile(0.75), finite_max="max")
        .reset_index()
    )
    overall = overall.merge(finite_overall, on="label", how="left")

    species = (
        df.groupby(["species", "label"], dropna=False)
        .agg(
            n=("label", "size"),
            finite_n=("is_inf_distance", lambda s: int((~s).sum())),
            inf_n=("is_inf_distance", lambda s: int(s.sum())),
        )
        .reset_index()
    )
    finite_species = (
        finite.groupby(["species", "label"], dropna=False)["sequence_true_descriptive_distance_to_reassort_node"]
        .agg(finite_min="min", finite_p25=lambda s: s.quantile(0.25), finite_median="median", finite_mean="mean", finite_p75=lambda s: s.quantile(0.75), finite_max="max")
        .reset_index()
    )
    species = species.merge(finite_species, on=["species", "label"], how="left")
    species["species_label"] = species["species"].map(SPECIES_LABELS)
    species = species[
        [
            "species_label",
            "label",
            "n",
            "finite_n",
            "inf_n",
            "finite_min",
            "finite_p25",
            "finite_median",
            "finite_mean",
            "finite_p75",
            "finite_max",
        ]
    ].sort_values(["species_label", "label"])

    overall_out = REPORT_DIR / "reassortment_distance_validation_overall.tsv"
    species_out = REPORT_DIR / "reassortment_distance_validation_by_species_and_label.tsv"
    overall.to_csv(overall_out, sep="\t", index=False)
    species.to_csv(species_out, sep="\t", index=False)

    md_path = REPORT_DIR / "reassortment_distance_validation_for_notion.md"
    intro = (
        "## Reassortment Distance Validation\n\n"
        "Distances are computed from each direct Espalier-labeled tip to the nearest inferred reassortment node along the tip-to-root path. "
        "Finite values indicate a reassortment ancestor was encountered; `Inf` means no reassortment node occurs anywhere on that ancestral path.\n\n"
    )
    md_path.write_text(intro)
    write_markdown_table(REPORT_DIR / "_tmp_overall.md", overall, "Overall by label")
    write_markdown_table(REPORT_DIR / "_tmp_species.md", species, "By species and label")
    md_path.write_text(
        intro
        + (REPORT_DIR / "_tmp_overall.md").read_text()
        + "\n"
        + (REPORT_DIR / "_tmp_species.md").read_text()
    )
    (REPORT_DIR / "_tmp_overall.md").unlink(missing_ok=True)
    (REPORT_DIR / "_tmp_species.md").unlink(missing_ok=True)


def build_recomb_clades_for_species(species: str) -> Dict[str, Set[frozenset[str]]]:
    true_root = TRUE_RUN_ROOTS[species]
    tree_root = TREE_RUN_ROOTS[species]
    ts = tskit.load(str(true_root / "arg_em.trees"))
    recomb_flag = getattr(tskit, "NODE_IS_RECOMBINATION", 131072)
    recomb_nodes = [i for i, n in enumerate(ts.nodes()) if n.flags & recomb_flag]
    seg_order = ["S", "M", "L"]
    aln_paths = [tree_root / "augur" / seg / f"{seg}_aligned.fasta" for seg in seg_order]
    segment_spans, _ = build_segment_spans(seg_order, aln_paths)

    seq = load_sequence_true_labels()
    tip_labels = sorted(seq.loc[seq["species"] == species, "sequence_id"].astype(str).tolist())
    sample_ids = list(ts.samples())
    if len(tip_labels) != len(sample_ids):
        raise RuntimeError(f"Tip/sample mismatch for {species}: tips={len(tip_labels)} samples={len(sample_ids)}")
    sample_to_label = dict(zip(sample_ids, tip_labels))

    recomb_clades_by_seg: Dict[str, Set[frozenset[str]]] = {seg: set() for seg in seg_order}
    for seg in seg_order:
        t = ts.at(segment_midpoint(segment_spans[seg], ts.sequence_length))
        present_nodes = set(t.nodes())
        for rn in recomb_nodes:
            if rn not in present_nodes:
                continue
            desc = [sample_to_label[s] for s in t.samples(rn) if s in sample_to_label]
            if len(desc) < 2:
                continue
            recomb_clades_by_seg[seg].add(frozenset(desc))
    return recomb_clades_by_seg


def prune_tree_to_labeled_tips(tree_path: Path, keep_tips: Set[str]) -> Path:
    tree = Phylo.read(str(tree_path), "newick")
    for term in list(tree.get_terminals()):
        if term.name not in keep_tips:
            tree.prune(term)
    handle = tempfile.NamedTemporaryFile("w+", suffix=".nwk", delete=False)
    handle.close()
    Phylo.write(tree, handle.name, "newick")
    return Path(handle.name)


def assign_reassortment_states(tree: bt.tree, reassort_leafsets: Set[frozenset[str]]) -> Set[int]:
    node_by_leafset: Dict[frozenset[str], object] = {}
    for obj in tree.Objects:
        if getattr(obj, "branchType", None) == "node":
            node_by_leafset[frozenset(getattr(obj, "leaves", set()))] = obj
    reassort_nodes = {node_by_leafset[ls] for ls in reassort_leafsets if ls in node_by_leafset}

    def walk(node: object, hits_before: int) -> None:
        node.traits["reassort_hits_before"] = hits_before
        node.traits["branch_colour"] = BLACK if hits_before == 0 else PALETTE[(hits_before - 1) % len(PALETTE)]
        node.traits["is_reassort_node"] = int(node in reassort_nodes)
        node.traits["reassort_diamond_colour"] = PALETTE[hits_before % len(PALETTE)]
        hits_after = hits_before + (1 if node in reassort_nodes else 0)
        for child in getattr(node, "children", []):
            walk(child, hits_after)

    walk(tree.root, 0)
    return {obj.index for obj in reassort_nodes}


def species_label_tip_set(species: str) -> Set[str]:
    seq = load_sequence_true_labels()
    sub = seq[seq["species"] == species].copy()
    return set(sub["sequence_id"].astype(str))


def plot_species_panel(ax, species: str) -> Tuple[int, int, Dict[str, int]]:
    keep_tips = species_label_tip_set(species)
    tree_path = TREE_RUN_ROOTS[species] / "augur" / "S" / "S_tree_refined.newick"
    tmp_tree = prune_tree_to_labeled_tips(tree_path, keep_tips)
    tree = bt.loadNewick(str(tmp_tree), absoluteTime=False)
    tree.sortBranches()
    reassort_by_seg = build_recomb_clades_for_species(species)
    reassort_leafsets: Set[frozenset[str]] = set().union(*reassort_by_seg.values())
    reassort_node_indices = assign_reassortment_states(tree, reassort_leafsets)

    tree.plotTree(
        ax,
        colour=lambda k: k.traits.get("branch_colour", BLACK),
        width=lambda k: 1.0 if getattr(k, "branchType", None) == "leaf" else 1.2,
        zorder=1,
    )
    tree.plotPoints(
        ax,
        target=lambda k: getattr(k, "branchType", None) == "leaf",
        size=lambda k: 16,
        colour=lambda k: k.traits.get("branch_colour", BLACK),
        outline=False,
        zorder=3,
    )

    diamonds = [obj for obj in tree.Objects if obj.index in reassort_node_indices]
    if diamonds:
        ax.scatter(
            [k.x for k in diamonds],
            [k.y for k in diamonds],
            s=28,
            marker="D",
            c=[k.traits.get("reassort_diamond_colour", PALETTE[0]) for k in diamonds],
            edgecolors=BLACK,
            linewidths=0.4,
            zorder=4,
        )

    ax.set_title(f"{SPECIES_LABELS[species]}  (n={len(keep_tips)})", fontsize=11, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    tmp_tree.unlink(missing_ok=True)
    seg_counts = {seg: len(reassort_by_seg.get(seg, set())) for seg in ["S", "M", "L"]}
    return len(keep_tips), len(diamonds), seg_counts


def build_supplemental_figure() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
    axes = axes.flatten()

    meta_rows: List[Dict[str, object]] = []
    for ax, species in zip(axes[:7], SPECIES_ORDER):
        n_tips, n_reassort_nodes, seg_counts = plot_species_panel(ax, species)
        meta_rows.append(
            {
                "species": species,
                "species_label": SPECIES_LABELS[species],
                "n_tips": n_tips,
                "n_reassortment_nodes_S_tree": n_reassort_nodes,
                "n_reassortment_clades_S": seg_counts.get("S", 0),
                "n_reassortment_clades_M": seg_counts.get("M", 0),
                "n_reassortment_clades_L": seg_counts.get("L", 0),
            }
        )

    legend_ax = axes[7]
    legend_ax.axis("off")
    y0 = 0.92
    legend_ax.text(0.0, y0, "Path-state legend", fontsize=12, fontweight="bold", transform=legend_ax.transAxes)
    legend_ax.plot([0.02, 0.18], [y0 - 0.10, y0 - 0.10], color=BLACK, lw=2, transform=legend_ax.transAxes)
    legend_ax.scatter([0.18], [y0 - 0.10], s=30, color=BLACK, transform=legend_ax.transAxes)
    legend_ax.text(0.23, y0 - 0.10, "No reassortment node on the root-to-tip path", va="center", fontsize=10, transform=legend_ax.transAxes)
    for idx, color in enumerate(PALETTE, start=1):
        yy = y0 - 0.20 - ((idx - 1) * 0.10)
        legend_ax.plot([0.02, 0.18], [yy, yy], color=color, lw=2, transform=legend_ax.transAxes)
        legend_ax.scatter([0.10], [yy], s=36, color=color, marker="D", edgecolors=BLACK, linewidths=0.4, transform=legend_ax.transAxes)
        legend_ax.text(0.23, yy, f"Branches after reassortment encounter {idx}", va="center", fontsize=10, transform=legend_ax.transAxes)
    png = FIGURE_DIR / "supplement_reassortment_path_states.png"
    svg = FIGURE_DIR / "supplement_reassortment_path_states.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(meta_rows).to_csv(REPORT_DIR / "supplement_reassortment_path_states_manifest.tsv", sep="\t", index=False)


def main() -> None:
    build_descriptive_tables()
    build_supplemental_figure()


if __name__ == "__main__":
    main()
