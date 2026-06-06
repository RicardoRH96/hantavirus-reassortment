from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.gridspec import GridSpec

from evo_utils import FIGURE_DIR, PASTEL, REPORT_DIR, SPECIES_ORDER, SPECIES_LABELS, set_plot_style


NODE_PATH = REPORT_DIR / "08_species_lineage_graph_nodes.tsv"
EDGE_PATH = REPORT_DIR / "08_species_lineage_graph_edges.tsv"
ORIGIN_PATH = REPORT_DIR / "08_reassortment_origin_establishment_events.tsv"
BACKBONE_PATH = REPORT_DIR / "09_species_lineage_graph_backbone_edges.tsv"
SUMMARY_PATH = REPORT_DIR / "09_species_lineage_graph_supplement_summary.tsv"
REPORT_PATH = REPORT_DIR / "09_species_lineage_graph_supplement_report.md"
PNG_PATH = FIGURE_DIR / "09_species_lineage_graph_supplement.png"
SVG_PATH = FIGURE_DIR / "09_species_lineage_graph_supplement.svg"

EDGE_TOP_K_PER_NODE = 2
EDGE_QUANTILE = 0.90
NODE_MIN_TIPS = 1.0
NODE_MIN_AREA = 3.0
NODE_MAX_AREA = 430.0


def node_marker_area(sampled_tips: np.ndarray | float, max_tips: float) -> np.ndarray | float:
    values = np.asarray(sampled_tips, dtype=float)
    denom = max(float(max_tips) - NODE_MIN_TIPS, 1e-9)
    normalized = np.clip((values - NODE_MIN_TIPS) / denom, 0.0, 1.0)
    scaled = NODE_MIN_AREA + normalized * (NODE_MAX_AREA - NODE_MIN_AREA)
    if np.isscalar(sampled_tips):
        return float(scaled)
    return scaled


def _stable_seed(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % (2**32 - 1)
    return int(value)


def _lighten(hex_color: str, amount: float = 0.5) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(hex_color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


def _darken(hex_color: str, amount: float = 0.35) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(hex_color), dtype=float)
    return tuple(rgb * (1.0 - amount))


def load_graph_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [str(path) for path in [NODE_PATH, EDGE_PATH, ORIGIN_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing 08 graph inputs. Run specific_evolutionary_analyses/08_fixation_establishment_model.py first:\n"
            + "\n".join(missing)
        )
    nodes = pd.read_csv(NODE_PATH, sep="\t")
    edges = pd.read_csv(EDGE_PATH, sep="\t")
    origins = pd.read_csv(ORIGIN_PATH, sep="\t")
    for col in ["n_tips", "graph_neighbor_overlap", "mean_overlap", "positive_fraction"]:
        nodes[col] = pd.to_numeric(nodes[col], errors="coerce")
    edges["edge_weight"] = pd.to_numeric(edges["edge_weight"], errors="coerce")
    origins["descendant_tip_n"] = pd.to_numeric(origins["descendant_tip_n"], errors="coerce")
    return nodes, edges, origins


def add_origin_counts(nodes: pd.DataFrame, origins: pd.DataFrame) -> pd.DataFrame:
    origin_counts = (
        origins.groupby("dominant_lineage_id", as_index=False)
        .agg(
            origin_event_n=("event_id", "nunique"),
            max_origin_descendant_tips=("descendant_tip_n", "max"),
        )
        .rename(columns={"dominant_lineage_id": "lineage_id"})
    )
    out = nodes.merge(origin_counts, on="lineage_id", how="left")
    out["origin_event_n"] = out["origin_event_n"].fillna(0).astype(int)
    out["max_origin_descendant_tips"] = out["max_origin_descendant_tips"].fillna(0)
    return out


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def build_backbone_edges(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for species, species_nodes in nodes.groupby("species", sort=False):
        species_edges = edges[edges["species"] == species].copy()
        if species_edges.empty:
            continue
        graph = nx.Graph()
        graph.add_nodes_from(species_nodes["lineage_id"].astype(str))
        for row in species_edges.itertuples(index=False):
            graph.add_edge(
                str(row.left_lineage_id),
                str(row.right_lineage_id),
                weight=float(row.edge_weight),
            )
        mst_keys: set[tuple[str, str]] = set()
        if graph.number_of_edges() > 0:
            mst = nx.maximum_spanning_tree(graph, weight="weight")
            mst_keys = {_edge_key(u, v) for u, v in mst.edges()}
        strongest_keys: set[tuple[str, str]] = set()
        for lineage_id in species_nodes["lineage_id"].astype(str):
            incident = species_edges[
                (species_edges["left_lineage_id"] == lineage_id) | (species_edges["right_lineage_id"] == lineage_id)
            ].sort_values("edge_weight", ascending=False)
            for row in incident.head(EDGE_TOP_K_PER_NODE).itertuples(index=False):
                strongest_keys.add(_edge_key(row.left_lineage_id, row.right_lineage_id))
        threshold = float(species_edges["edge_weight"].quantile(EDGE_QUANTILE))
        quantile_keys = {
            _edge_key(row.left_lineage_id, row.right_lineage_id)
            for row in species_edges[species_edges["edge_weight"] >= threshold].itertuples(index=False)
        }
        selected = mst_keys | strongest_keys | quantile_keys
        for row in species_edges.itertuples(index=False):
            key = _edge_key(row.left_lineage_id, row.right_lineage_id)
            if key not in selected:
                continue
            record = row._asdict()
            record["backbone_edge"] = True
            record["in_maximum_spanning_tree"] = key in mst_keys
            record["in_top_k_per_node"] = key in strongest_keys
            record["in_species_top_quantile"] = key in quantile_keys
            record["species_edge_weight_q90"] = threshold
            records.append(record)
    return pd.DataFrame(records).sort_values(["species_label", "edge_weight"], ascending=[True, False])


def layout_species_network(species_nodes: pd.DataFrame, backbone_edges: pd.DataFrame) -> dict[str, np.ndarray]:
    graph = nx.Graph()
    graph.add_nodes_from(species_nodes["lineage_id"].astype(str))
    for row in backbone_edges.itertuples(index=False):
        graph.add_edge(str(row.left_lineage_id), str(row.right_lineage_id), weight=float(row.edge_weight))
    if graph.number_of_nodes() == 1:
        only = next(iter(graph.nodes()))
        return {only: np.array([0.0, 0.0])}
    layout = nx.spring_layout(
        graph,
        weight="weight",
        seed=_stable_seed(str(species_nodes["species"].iloc[0])),
        iterations=900,
        k=1.25 / math.sqrt(max(graph.number_of_nodes(), 2)),
    )
    values = np.array(list(layout.values()), dtype=float)
    center = values.mean(axis=0)
    span = np.ptp(values, axis=0)
    span[span == 0] = 1.0
    return {node: (coord - center) / max(span) for node, coord in layout.items()}


def _edge_segments(edge_frame: pd.DataFrame, pos: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    segments = []
    for row in edge_frame.itertuples(index=False):
        left = str(row.left_lineage_id)
        right = str(row.right_lineage_id)
        if left in pos and right in pos:
            segments.append((pos[left], pos[right]))
    return segments


def draw_species_panel(
    ax: plt.Axes,
    species_nodes: pd.DataFrame,
    raw_edges: pd.DataFrame,
    backbone_edges: pd.DataFrame,
    opportunity_norm: Normalize,
    opportunity_cmap: LinearSegmentedColormap,
    max_tips: float,
    node_size_scale: float = 1.0,
    species_fontsize: float = 10.5,
    edge_width_scale: float = 1.0,
    axis_limit_scale: float = 1.0,
) -> None:
    species_label = str(species_nodes["species_label"].iloc[0])
    species_color = PASTEL.get(species_label, "#cccccc")
    pos = layout_species_network(species_nodes, backbone_edges)

    raw_segments = _edge_segments(raw_edges, pos)
    if raw_segments:
        ax.add_collection(
            LineCollection(
                raw_segments,
                colors=[(*_darken(species_color, 0.15), 0.075)],
                linewidths=0.28,
                zorder=1,
            )
        )

    if not backbone_edges.empty:
        weights = backbone_edges["edge_weight"].to_numpy(dtype=float)
        denom = max(weights.max() - weights.min(), 1e-9)
        widths = (0.45 + 2.2 * (weights - weights.min()) / denom) * edge_width_scale
        alphas = 0.34 + 0.46 * (weights - weights.min()) / denom
        for (_, row), width, alpha in zip(backbone_edges.iterrows(), widths, alphas):
            left = str(row["left_lineage_id"])
            right = str(row["right_lineage_id"])
            if left not in pos or right not in pos:
                continue
            ax.plot(
                [pos[left][0], pos[right][0]],
                [pos[left][1], pos[right][1]],
                color=(*_darken(species_color, 0.45), alpha),
                linewidth=float(width),
                solid_capstyle="round",
                zorder=2,
            )

    x = np.array([pos[str(node)][0] for node in species_nodes["lineage_id"]])
    y = np.array([pos[str(node)][1] for node in species_nodes["lineage_id"]])
    size_values = species_nodes["n_tips"].clip(lower=1).to_numpy(dtype=float)
    node_sizes = node_marker_area(size_values, max_tips=max_tips) * node_size_scale
    colors = opportunity_cmap(opportunity_norm(species_nodes["graph_neighbor_overlap"].to_numpy(dtype=float)))

    ax.scatter(
        x,
        y,
        s=node_sizes + np.maximum(4.0, node_sizes * 0.08),
        c="white",
        edgecolors="white",
        linewidths=1.8,
        zorder=3,
    )
    ax.scatter(
        x,
        y,
        s=node_sizes,
        c=colors,
        edgecolors=_darken(species_color, 0.55),
        linewidths=0.75,
        zorder=4,
    )
    origin_nodes = species_nodes["origin_event_n"].to_numpy(dtype=int) > 0
    if origin_nodes.any():
        origin_sizes = node_sizes[origin_nodes] + np.maximum(8.0, node_sizes[origin_nodes] * 0.12)
        ax.scatter(
            x[origin_nodes],
            y[origin_nodes],
            s=origin_sizes,
            facecolors="none",
            edgecolors="#b7657b",
            linewidths=1.35,
            zorder=5,
        )

    ax.text(
        0.03,
        0.96,
        species_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=species_fontsize,
        fontweight="bold",
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.6},
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.66 * axis_limit_scale, 0.66 * axis_limit_scale)
    ax.set_ylim(-0.62 * axis_limit_scale, 0.62 * axis_limit_scale)
    ax.axis("off")


def draw_legend_panel(
    ax: plt.Axes,
    opportunity_norm: Normalize,
    opportunity_cmap: LinearSegmentedColormap,
    max_tips: float,
) -> None:
    ax.axis("off")
    cax = ax.inset_axes([0.05, 0.875, 0.82, 0.045])
    sm = ScalarMappable(norm=opportunity_norm, cmap=opportunity_cmap)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=2, pad=1)
    ax.text(0.05, 0.95, "Graph-neighborhood opportunity", transform=ax.transAxes, fontsize=8.0, color="#444444")

    ax.text(0.05, 0.745, "Lineage sample size", transform=ax.transAxes, fontsize=8.0, color="#444444")
    example_sizes = [1, max(2, int(round(max_tips / 3))), int(max_tips)]
    xs = [0.14, 0.33, 0.56]
    for xval, tips in zip(xs, example_sizes):
        size = node_marker_area(float(tips), max_tips=max_tips)
        ax.scatter(xval, 0.655, s=size, color="#d7ece2", edgecolors="#66706f", linewidths=0.75, transform=ax.transAxes)
        ax.text(xval, 0.535, str(tips), transform=ax.transAxes, ha="center", va="top", fontsize=7.3, color="#555555")

    ax.text(0.05, 0.435, "Backbone edge weight", transform=ax.transAxes, fontsize=8.0, color="#444444")
    for yi, lw, alpha, label in [(0.385, 0.8, 0.42, "lower"), (0.335, 2.1, 0.62, "middle"), (0.285, 3.2, 0.84, "higher")]:
        ax.plot([0.08, 0.34], [yi, yi], color=(0.36, 0.39, 0.41, alpha), lw=lw, solid_capstyle="round", transform=ax.transAxes)
        ax.text(0.39, yi, label, transform=ax.transAxes, va="center", fontsize=7.4, color="#555555")
    ax.plot([0.08, 0.34], [0.235, 0.235], color="#b7b7b7", lw=0.9, alpha=0.34, solid_capstyle="round", transform=ax.transAxes)
    ax.text(0.39, 0.235, "raw scaffold", transform=ax.transAxes, va="center", fontsize=7.4, color="#555555")

    ax.text(0.05, 0.145, "Lineage status", transform=ax.transAxes, fontsize=8.0, color="#444444")
    ax.scatter(0.11, 0.095, s=105, color="#d7ece2", edgecolors="#66706f", linewidths=0.75, transform=ax.transAxes)
    ax.text(0.19, 0.095, "Lineage node", transform=ax.transAxes, va="center", fontsize=7.6, color="#555555")
    ax.scatter(0.11, 0.045, s=125, facecolors="none", edgecolors="#b7657b", linewidths=1.45, transform=ax.transAxes)
    ax.text(0.19, 0.045, "ARG-origin lineage", transform=ax.transAxes, va="center", fontsize=7.6, color="#555555")


def render_summary(nodes: pd.DataFrame, edges: pd.DataFrame, backbone: pd.DataFrame) -> pd.DataFrame:
    summary = (
        nodes.groupby(["species", "species_label"], as_index=False)
        .agg(
            lineage_nodes=("lineage_id", "nunique"),
            sampled_tips=("n_tips", "sum"),
            origin_lineages=("origin_event_n", lambda s: int((s > 0).sum())),
            mean_neighbor_overlap=("graph_neighbor_overlap", "mean"),
        )
        .merge(
            edges.groupby(["species", "species_label"], as_index=False).agg(raw_edges=("edge_weight", "size"), median_edge_weight=("edge_weight", "median")),
            on=["species", "species_label"],
            how="left",
        )
        .merge(
            backbone.groupby(["species", "species_label"], as_index=False).agg(backbone_edges=("edge_weight", "size"), median_backbone_weight=("edge_weight", "median")),
            on=["species", "species_label"],
            how="left",
        )
    )
    summary["backbone_edges"] = summary["backbone_edges"].fillna(0).astype(int)
    summary["backbone_fraction"] = summary["backbone_edges"] / summary["raw_edges"].replace(0, np.nan)
    order_map = {species: idx for idx, species in enumerate(SPECIES_ORDER)}
    summary["_order"] = summary["species"].map(order_map).fillna(99)
    return summary.sort_values("_order").drop(columns="_order")


def render_report(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for col in ["mean_neighbor_overlap", "median_edge_weight", "median_backbone_weight", "backbone_fraction"]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    lines = [
        "# Species Lineage Graph Supplement",
        "",
        "This figure visualizes the lineage graph used to calculate graph-neighborhood opportunity in the fixation-inspired establishment analysis. The raw graph is almost complete within species, so the plotted network emphasizes a reproducible backbone: the maximum-spanning tree, each lineage's two strongest links, and species-specific top-decile edge weights. The full raw graph remains visible as a faint scaffold.",
        "",
        "Node marker area is min-max scaled from the observed lineage sample-size range with a very small singleton area, so one-tip lineages remain visually distinct from larger sampled clades.",
        "",
        f"- Input nodes: `{NODE_PATH}`",
        f"- Input edges: `{EDGE_PATH}`",
        f"- Backbone edges: `{BACKBONE_PATH}`",
        f"- Figure PNG: `{PNG_PATH}`",
        f"- Figure SVG: `{SVG_PATH}`",
        "",
        "| Species | Lineages | Sampled tips | Origin lineages | Raw edges | Backbone edges | Backbone fraction | Mean neighbor overlap | Median edge weight | Median backbone weight |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.species_label} | {row.lineage_nodes} | {int(row.sampled_tips)} | {int(row.origin_lineages)} | "
            f"{int(row.raw_edges)} | {int(row.backbone_edges)} | {row.backbone_fraction} | "
            f"{row.mean_neighbor_overlap} | {row.median_edge_weight} | {row.median_backbone_weight} |"
        )
    return "\n".join(lines) + "\n"


def plot_lineage_graph_supplement(nodes: pd.DataFrame, edges: pd.DataFrame, backbone: pd.DataFrame) -> None:
    set_plot_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.6,
        }
    )
    opportunity_values = nodes["graph_neighbor_overlap"].dropna().to_numpy(dtype=float)
    max_tips = float(nodes["n_tips"].max())
    norm = Normalize(vmin=float(np.quantile(opportunity_values, 0.04)), vmax=float(np.quantile(opportunity_values, 0.96)), clip=True)
    cmap = LinearSegmentedColormap.from_list(
        "pastel_opportunity",
        ["#f8f6ef", "#d9efe4", "#a8d8d2", "#f2cbd3", "#c98396"],
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.9, 7.3), constrained_layout=False)
    gs = GridSpec(2, 4, figure=fig, left=0.03, right=0.985, top=0.965, bottom=0.045, wspace=0.05, hspace=0.06)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]

    ordered_species = [species for species in SPECIES_ORDER if species in set(nodes["species"])]
    for ax, species in zip(axes[:7], ordered_species):
        species_nodes = nodes[nodes["species"] == species].copy()
        species_edges = edges[edges["species"] == species].copy()
        species_backbone = backbone[backbone["species"] == species].copy()
        draw_species_panel(ax, species_nodes, species_edges, species_backbone, norm, cmap, max_tips=max_tips)

    draw_legend_panel(axes[7], norm, cmap, max_tips)
    for label, ax in zip(list("ABCDEFGH"), axes):
        ax.text(0.0, 1.015, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold", color="#2f2f2f")

    fig.savefig(PNG_PATH, dpi=450, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    nodes, edges, origins = load_graph_tables()
    nodes = add_origin_counts(nodes, origins)
    backbone = build_backbone_edges(nodes, edges)
    summary = render_summary(nodes, edges, backbone)
    backbone.to_csv(BACKBONE_PATH, sep="\t", index=False)
    summary.to_csv(SUMMARY_PATH, sep="\t", index=False)
    REPORT_PATH.write_text(render_report(summary))
    plot_lineage_graph_supplement(nodes, edges, backbone)
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
