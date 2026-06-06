from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evo_utils import DATASET_DIR, FIGURE_DIR, PASTEL, REPORT_DIR, SPECIES_LABELS, SPECIES_ORDER, set_plot_style


BOTTOM_LEGEND_PNG = FIGURE_DIR / "10_combined_strategy_graphs_bottom_legend.png"
BOTTOM_LEGEND_JPG = FIGURE_DIR / "10_combined_strategy_graphs_bottom_legend.jpg"
BOTTOM_LEGEND_PDF = FIGURE_DIR / "10_combined_strategy_graphs_bottom_legend.pdf"
BOTTOM_LEGEND_SVG = FIGURE_DIR / "10_combined_strategy_graphs_bottom_legend.svg"
SIDE_LEGEND_PNG = FIGURE_DIR / "10_combined_strategy_graphs_side_legend.png"
SIDE_LEGEND_JPG = FIGURE_DIR / "10_combined_strategy_graphs_side_legend.jpg"
SIDE_LEGEND_PDF = FIGURE_DIR / "10_combined_strategy_graphs_side_legend.pdf"
SIDE_LEGEND_SVG = FIGURE_DIR / "10_combined_strategy_graphs_side_legend.svg"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model08 = load_module("fixation_establishment_08", SCRIPT_DIR / "08_fixation_establishment_model.py")
graph09 = load_module("lineage_graph_09", SCRIPT_DIR / "09_lineage_graph_supplement.py")


def apply_style() -> None:
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


def load_strategy_inputs():
    required = [
        DATASET_DIR / "08_fixation_establishment_model_dataset.tsv",
        REPORT_DIR / "08_reassortment_origin_establishment_events.tsv",
        REPORT_DIR / "08_arg_origin_denesting_manifest.tsv",
        REPORT_DIR / "08_rarefaction_sensitivity.tsv",
        REPORT_DIR / "08_hurdle_establishment_idata.nc",
        REPORT_DIR / "08_event_predictor_standardization.tsv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing 08 outputs. Run the 08 notebook or script first:\n" + "\n".join(missing))
    matched_events = pd.read_csv(DATASET_DIR / "08_fixation_establishment_model_dataset.tsv", sep="\t")
    origin_events = pd.read_csv(REPORT_DIR / "08_reassortment_origin_establishment_events.tsv", sep="\t")
    denesting_manifest = pd.read_csv(REPORT_DIR / "08_arg_origin_denesting_manifest.tsv", sep="\t")
    rarefaction_summary = pd.read_csv(REPORT_DIR / "08_rarefaction_sensitivity.tsv", sep="\t")
    establishment_idata = az.from_netcdf(REPORT_DIR / "08_hurdle_establishment_idata.nc")
    standardization = pd.read_csv(REPORT_DIR / "08_event_predictor_standardization.tsv", sep="\t")
    event_ref = {
        row["feature"]: {"median": float(row["median"]), "mean": float(row["mean"]), "std": float(row["std"])}
        for _, row in standardization.iterrows()
    }
    return origin_events, matched_events, denesting_manifest, rarefaction_summary, establishment_idata, event_ref


def load_network_inputs():
    nodes, edges, origins = graph09.load_graph_tables()
    nodes = graph09.add_origin_counts(nodes, origins)
    backbone = graph09.build_backbone_edges(nodes, edges)
    opportunity_values = nodes["graph_neighbor_overlap"].dropna().to_numpy(dtype=float)
    norm = Normalize(
        vmin=float(np.quantile(opportunity_values, 0.04)),
        vmax=float(np.quantile(opportunity_values, 0.96)),
        clip=True,
    )
    cmap = LinearSegmentedColormap.from_list(
        "pastel_opportunity",
        ["#f8f6ef", "#d9efe4", "#a8d8d2", "#f2cbd3", "#c98396"],
    )
    max_tips = float(nodes["n_tips"].max())
    return nodes, edges, backbone, norm, cmap, max_tips


def draw_strategy_panels(
    fig: plt.Figure,
    subplot_spec,
    origin_events: pd.DataFrame,
    matched_events: pd.DataFrame,
    denesting_manifest: pd.DataFrame,
    rarefaction_summary: pd.DataFrame,
    establishment_idata,
    event_ref: dict[str, dict[str, float]],
) -> None:
    surface_cmap = LinearSegmentedColormap.from_list(
        "payoff_surface",
        ["#f4f7f8", "#dbe8f1", "#ded7ee", "#efc2ca", "#d8879c"],
    )
    div_cmap = LinearSegmentedColormap.from_list(
        "soft_diverging",
        ["#8fb8cc", "#f3f4f2", "#d8879c"],
    )
    gs = GridSpecFromSubplotSpec(
        3,
        3,
        subplot_spec=subplot_spec,
        width_ratios=[1.15, 1.15, 1.15],
        height_ratios=[1.08, 1.08, 0.86],
        wspace=0.44,
        hspace=0.42,
    )
    ax_surface = fig.add_subplot(gs[0:2, 0:2])
    ax_curves = fig.add_subplot(gs[0:2, 2])
    ax_denest = fig.add_subplot(gs[2, 0:2])
    ax_rare = fig.add_subplot(gs[2, 2])

    x = pd.to_numeric(matched_events["ecological_opportunity_z"], errors="coerce")
    y = pd.to_numeric(matched_events["molecular_permissiveness_z"], errors="coerce")
    xlim = np.nanpercentile(x, [2, 98])
    ylim = np.nanpercentile(y, [2, 98])
    xpad = max(0.15, 0.08 * float(xlim[1] - xlim[0]))
    ypad = max(0.15, 0.08 * float(ylim[1] - ylim[0]))
    xlim = np.array([xlim[0] - xpad, xlim[1] + xpad])
    ylim = np.array([ylim[0] - ypad, ylim[1] + ypad])

    xx = np.linspace(xlim[0], xlim[1], 90)
    yy = np.linspace(ylim[0], ylim[1], 90)
    grid_x, grid_y = np.meshgrid(xx, yy)
    surface, _, _ = model08.predict_establishment_fixed_effects(
        establishment_idata,
        grid_x.ravel(),
        grid_y.ravel(),
        event_ref,
    )
    surface = surface.reshape(grid_x.shape)
    levels = np.linspace(0.05, 0.95, 19)
    cf = ax_surface.contourf(grid_x, grid_y, surface, levels=levels, cmap=surface_cmap, extend="both")
    ax_surface.contour(grid_x, grid_y, surface, levels=[0.50], colors=["#676767"], linewidths=0.9, alpha=0.9)

    controls = matched_events[matched_events["established"].astype(int).eq(0)].copy()
    origins = matched_events[matched_events["established"].astype(int).eq(1)].copy()
    ax_surface.scatter(
        controls["ecological_opportunity_z"],
        controls["molecular_permissiveness_z"],
        s=18,
        facecolors="none",
        edgecolors="#9c9c9c",
        linewidths=0.45,
        alpha=0.42,
        zorder=3,
    )
    for species_label, sub in origins.groupby("species_label", sort=False):
        color = PASTEL.get(species_label, "#c7c7c7")
        sizes = 28.0 + 15.0 * np.log1p(pd.to_numeric(sub["descendant_tip_n"], errors="coerce").fillna(1.0))
        ax_surface.scatter(
            sub["ecological_opportunity_z"],
            sub["molecular_permissiveness_z"],
            s=sizes,
            color=color,
            edgecolors="#3f3f3f",
            linewidths=0.45,
            alpha=0.95,
            zorder=4,
        )
    ax_surface.set_xlabel("Ecological opportunity")
    ax_surface.set_ylabel("Molecular permissiveness")
    ax_surface.set_xlim(xlim)
    ax_surface.set_ylim(ylim)
    ax_surface.text(-0.07, 1.04, "A", transform=ax_surface.transAxes, fontsize=13, fontweight="bold", va="top")
    cbar = fig.colorbar(cf, ax=ax_surface, fraction=0.034, pad=0.018)
    cbar.set_label("Pr(establishment)")
    cbar.outline.set_visible(False)

    curve_x = np.linspace(xlim[0], xlim[1], 140)
    mol_levels = np.nanquantile(y, [0.15, 0.50, 0.85])
    curve_specs = [
        ("restrictive", mol_levels[0], "#8fb8cc"),
        ("median", mol_levels[1], "#8f8f8f"),
        ("permissive", mol_levels[2], "#d8879c"),
    ]
    for label, mol_value, color in curve_specs:
        median, lo, hi = model08.predict_establishment_fixed_effects(
            establishment_idata,
            curve_x,
            np.full_like(curve_x, mol_value),
            event_ref,
        )
        ax_curves.fill_between(curve_x, lo, hi, color=color, alpha=0.16, linewidth=0)
        ax_curves.plot(curve_x, median, color=color, linewidth=2.0, label=label)
    ax_curves.set_xlabel("Ecological opportunity")
    ax_curves.set_ylabel("Pr(establishment)")
    ax_curves.set_xlim(xlim)
    ax_curves.set_ylim(0, min(1.0, max(0.65, ax_curves.get_ylim()[1])))
    ax_curves.legend(frameon=False, loc="upper left", handlelength=1.6, borderpad=0.1)
    ax_curves.text(-0.18, 1.04, "B", transform=ax_curves.transAxes, fontsize=13, fontweight="bold", va="top")

    if not denesting_manifest.empty:
        denest = (
            denesting_manifest.groupby("species_label", as_index=False)
            .agg(candidates=("origin_cluster_id", "nunique"), retained=("denested_keep", "sum"))
        )
        order = [SPECIES_LABELS[s] for s in SPECIES_ORDER if SPECIES_LABELS[s] in set(denest["species_label"])]
        denest["species_label"] = pd.Categorical(denest["species_label"], categories=order, ordered=True)
        denest = denest.sort_values("species_label").reset_index(drop=True)
        ypos = np.arange(len(denest))
        for idx, row in denest.iterrows():
            color = PASTEL.get(str(row["species_label"]), "#c7c7c7")
            ax_denest.hlines(idx, row["retained"], row["candidates"], color="#b5b5b5", linewidth=1.0, zorder=1)
            ax_denest.scatter(row["candidates"], idx, s=42, facecolors="white", edgecolors="#7e7e7e", linewidths=0.9, zorder=2)
            ax_denest.scatter(row["retained"], idx, s=52, color=color, edgecolors="#3f3f3f", linewidths=0.45, zorder=3)
        ax_denest.set_yticks(ypos)
        ax_denest.set_yticklabels(denest["species_label"])
        ax_denest.invert_yaxis()
        ax_denest.set_xlabel("ARG origin clusters")
        ax_denest.set_xlim(0, max(denest["candidates"].max() + 3, 6))
    ax_denest.text(-0.07, 1.11, "C", transform=ax_denest.transAxes, fontsize=13, fontweight="bold", va="top")

    hurdle_rare = rarefaction_summary[
        rarefaction_summary["model"].eq("rarefied_hurdle_logistic_l2")
        & rarefaction_summary["term"].isin(
            [
                "ecological_opportunity_z",
                "molecular_permissiveness_z",
                "ecology_x_molecular_permissiveness_z",
                "tree_discordance_score_z",
            ]
        )
    ].copy()
    term_order = [
        "ecological_opportunity_z",
        "molecular_permissiveness_z",
        "ecology_x_molecular_permissiveness_z",
        "tree_discordance_score_z",
    ]
    term_labels = ["Ecology", "Molecular", "Interaction", "Discordance"]
    target_order = ["species_q25", "strict_species_min"]
    target_labels = ["q25", "strict"]
    if not hurdle_rare.empty:
        max_abs = max(0.1, float(np.nanmax(np.abs(hurdle_rare["coefficient_mean"]))))
        norm = Normalize(vmin=-max_abs, vmax=max_abs)
        for _, row in hurdle_rare.iterrows():
            xpos = term_order.index(row["term"])
            ypos = target_order.index(row["rarefaction_target"]) if row["rarefaction_target"] in target_order else 0
            stability = float(row["sign_stability_positive"])
            size = 85.0 + 210.0 * abs(stability - 0.5) * 2.0
            ax_rare.scatter(
                xpos,
                ypos,
                s=size,
                color=div_cmap(norm(float(row["coefficient_mean"]))),
                edgecolors="#464646",
                linewidths=0.45,
                alpha=0.95,
                zorder=3,
            )
        ax_rare.set_xticks(np.arange(len(term_order)))
        ax_rare.set_xticklabels(term_labels, rotation=35, ha="right")
        ax_rare.set_yticks(np.arange(len(target_order)))
        ax_rare.set_yticklabels(target_labels)
        ax_rare.set_xlim(-0.5, len(term_order) - 0.5)
        ax_rare.set_ylim(len(target_order) - 0.5, -0.5)
        for xpos in np.arange(-0.5, len(term_order), 1):
            ax_rare.axvline(xpos, color="#ececec", linewidth=0.7, zorder=0)
        for ypos in np.arange(-0.5, len(target_order), 1):
            ax_rare.axhline(ypos, color="#ececec", linewidth=0.7, zorder=0)
        sm = ScalarMappable(norm=norm, cmap=div_cmap)
        sm.set_array([])
        cbar2 = fig.colorbar(sm, ax=ax_rare, fraction=0.08, pad=0.02)
        cbar2.set_label("Coefficient")
        cbar2.outline.set_visible(False)
    ax_rare.text(-0.18, 1.11, "D", transform=ax_rare.transAxes, fontsize=13, fontweight="bold", va="top")

    for ax in [ax_surface, ax_curves, ax_denest, ax_rare]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, color="#777777")
    ax_surface.grid(False)
    ax_curves.yaxis.set_label_position("right")
    ax_curves.yaxis.tick_right()
    ax_curves.grid(axis="y", color="#eeeeee", linewidth=0.7)
    ax_curves.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)
    ax_curves.spines["left"].set_visible(False)
    ax_curves.spines["right"].set_visible(True)
    ax_denest.grid(axis="x", color="#eeeeee", linewidth=0.7)


def draw_network_row(
    fig: plt.Figure,
    subplot_spec,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    backbone: pd.DataFrame,
    norm: Normalize,
    cmap: LinearSegmentedColormap,
    max_tips: float,
    node_size_scale: float,
) -> list[plt.Axes]:
    gs = GridSpecFromSubplotSpec(1, 7, subplot_spec=subplot_spec, wspace=0.02)
    axes = [fig.add_subplot(gs[0, i]) for i in range(7)]
    ordered_species = [species for species in SPECIES_ORDER if species in set(nodes["species"])]
    for ax, species in zip(axes, ordered_species):
        species_nodes = nodes[nodes["species"] == species].copy()
        species_edges = edges[edges["species"] == species].copy()
        species_backbone = backbone[backbone["species"] == species].copy()
        graph09.draw_species_panel(
            ax,
            species_nodes,
            species_edges,
            species_backbone,
            norm,
            cmap,
            max_tips=max_tips,
            node_size_scale=node_size_scale,
            species_fontsize=7.2,
            edge_width_scale=0.68,
            axis_limit_scale=1.08,
        )
    axes[0].text(-0.05, 1.07, "E", transform=axes[0].transAxes, fontsize=13, fontweight="bold", va="top")
    return axes


def draw_horizontal_network_legend(
    ax: plt.Axes,
    norm: Normalize,
    cmap: LinearSegmentedColormap,
    max_tips: float,
) -> None:
    ax.axis("off")
    cax = ax.inset_axes([0.035, 0.34, 0.20, 0.18])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.2, length=2, pad=1)
    ax.text(0.035, 0.68, "Graph-neighborhood opportunity", transform=ax.transAxes, fontsize=8.0, color="#444444")

    ax.text(0.295, 0.68, "Lineage sample size", transform=ax.transAxes, fontsize=8.0, color="#444444")
    example_sizes = [1, max(2, int(round(max_tips / 3))), int(max_tips)]
    for xval, tips in zip([0.315, 0.375, 0.455], example_sizes):
        size = graph09.node_marker_area(float(tips), max_tips=max_tips) * 0.48
        ax.scatter(xval, 0.40, s=size, color="#d7ece2", edgecolors="#66706f", linewidths=0.65, transform=ax.transAxes)
        ax.text(xval, 0.16, str(tips), transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color="#555555")

    ax.text(0.545, 0.68, "Backbone edge weight", transform=ax.transAxes, fontsize=8.0, color="#444444")
    for x0, lw, alpha, label in [(0.555, 0.75, 0.42, "lower"), (0.665, 1.55, 0.62, "middle"), (0.775, 2.35, 0.84, "higher")]:
        ax.plot([x0, x0 + 0.055], [0.40, 0.40], color=(0.36, 0.39, 0.41, alpha), lw=lw, solid_capstyle="round", transform=ax.transAxes)
        ax.text(x0 + 0.065, 0.40, label, transform=ax.transAxes, va="center", fontsize=7.2, color="#555555")

    ax.text(0.895, 0.68, "Lineage status", transform=ax.transAxes, fontsize=8.0, color="#444444")
    ax.scatter(0.91, 0.40, s=72, color="#d7ece2", edgecolors="#66706f", linewidths=0.65, transform=ax.transAxes)
    ax.text(0.935, 0.40, "node", transform=ax.transAxes, va="center", fontsize=7.2, color="#555555")
    ax.scatter(0.91, 0.16, s=82, facecolors="none", edgecolors="#b7657b", linewidths=1.2, transform=ax.transAxes)
    ax.text(0.935, 0.16, "origin", transform=ax.transAxes, va="center", fontsize=7.2, color="#555555")


def draw_compact_side_network_legend(
    ax: plt.Axes,
    norm: Normalize,
    cmap: LinearSegmentedColormap,
    max_tips: float,
) -> None:
    ax.axis("off")
    cax = ax.inset_axes([0.08, 0.84, 0.82, 0.045])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.8, length=2, pad=1)
    ax.text(0.08, 0.93, "Graph-neighborhood opportunity", transform=ax.transAxes, fontsize=7.4, color="#444444")

    ax.text(0.08, 0.72, "Lineage sample size", transform=ax.transAxes, fontsize=7.4, color="#444444")
    example_sizes = [1, max(2, int(round(max_tips / 3))), int(max_tips)]
    for xval, tips in zip([0.17, 0.42, 0.74], example_sizes):
        size = graph09.node_marker_area(float(tips), max_tips=max_tips) * 0.44
        ax.scatter(xval, 0.61, s=size, color="#d7ece2", edgecolors="#66706f", linewidths=0.65, transform=ax.transAxes)
        ax.text(xval, 0.50, str(tips), transform=ax.transAxes, ha="center", va="top", fontsize=6.9, color="#555555")

    ax.text(0.08, 0.32, "Backbone edge weight", transform=ax.transAxes, fontsize=7.4, color="#444444")
    for yi, lw, alpha, label in [(0.25, 0.65, 0.42, "lower"), (0.18, 1.4, 0.62, "middle"), (0.11, 2.15, 0.84, "higher")]:
        ax.plot([0.10, 0.44], [yi, yi], color=(0.36, 0.39, 0.41, alpha), lw=lw, solid_capstyle="round", transform=ax.transAxes)
        ax.text(0.50, yi, label, transform=ax.transAxes, va="center", fontsize=6.9, color="#555555")

    ax.scatter(0.10, 0.045, s=70, color="#d7ece2", edgecolors="#66706f", linewidths=0.65, transform=ax.transAxes)
    ax.text(0.20, 0.045, "lineage node", transform=ax.transAxes, va="center", fontsize=6.9, color="#555555")
    ax.scatter(0.62, 0.045, s=80, facecolors="none", edgecolors="#b7657b", linewidths=1.2, transform=ax.transAxes)
    ax.text(0.73, 0.045, "origin", transform=ax.transAxes, va="center", fontsize=6.9, color="#555555")


def render_bottom_legend_layout(strategy_inputs, network_inputs) -> None:
    apply_style()
    origin_events, matched_events, denesting_manifest, rarefaction_summary, establishment_idata, event_ref = strategy_inputs
    nodes, edges, backbone, norm, cmap, max_tips = network_inputs
    fig = plt.figure(figsize=(13.4, 11.6), constrained_layout=False)
    outer = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[7.75, 2.15, 0.78],
        left=0.055,
        right=0.982,
        top=0.982,
        bottom=0.045,
        hspace=0.12,
    )
    draw_strategy_panels(
        fig,
        outer[0],
        origin_events,
        matched_events,
        denesting_manifest,
        rarefaction_summary,
        establishment_idata,
        event_ref,
    )
    draw_network_row(fig, outer[1], nodes, edges, backbone, norm, cmap, max_tips, node_size_scale=0.76)
    legend_ax = fig.add_subplot(outer[2])
    draw_horizontal_network_legend(legend_ax, norm, cmap, max_tips)
    fig.savefig(BOTTOM_LEGEND_PNG, dpi=450, bbox_inches="tight")
    fig.savefig(BOTTOM_LEGEND_JPG, dpi=450, bbox_inches="tight")
    fig.savefig(BOTTOM_LEGEND_PDF, bbox_inches="tight")
    fig.savefig(BOTTOM_LEGEND_SVG, bbox_inches="tight")
    plt.close(fig)


def render_side_legend_layout(strategy_inputs, network_inputs) -> None:
    apply_style()
    origin_events, matched_events, denesting_manifest, rarefaction_summary, establishment_idata, event_ref = strategy_inputs
    nodes, edges, backbone, norm, cmap, max_tips = network_inputs
    fig = plt.figure(figsize=(13.7, 12.0), constrained_layout=False)
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[7.7, 3.35],
        left=0.055,
        right=0.982,
        top=0.982,
        bottom=0.055,
        hspace=0.115,
    )
    draw_strategy_panels(
        fig,
        outer[0],
        origin_events,
        matched_events,
        denesting_manifest,
        rarefaction_summary,
        establishment_idata,
        event_ref,
    )
    bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[7.0, 1.08], wspace=0.035)
    draw_network_row(fig, bottom[0], nodes, edges, backbone, norm, cmap, max_tips, node_size_scale=0.76)
    legend_ax = fig.add_subplot(bottom[1])
    draw_compact_side_network_legend(legend_ax, norm, cmap, max_tips)
    fig.savefig(SIDE_LEGEND_PNG, dpi=450, bbox_inches="tight")
    fig.savefig(SIDE_LEGEND_JPG, dpi=450, bbox_inches="tight")
    fig.savefig(SIDE_LEGEND_PDF, bbox_inches="tight")
    fig.savefig(SIDE_LEGEND_SVG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    strategy_inputs = load_strategy_inputs()
    network_inputs = load_network_inputs()
    render_bottom_legend_layout(strategy_inputs, network_inputs)
    render_side_legend_layout(strategy_inputs, network_inputs)
    for path in [
        BOTTOM_LEGEND_PNG,
        BOTTOM_LEGEND_JPG,
        BOTTOM_LEGEND_PDF,
        BOTTOM_LEGEND_SVG,
        SIDE_LEGEND_PNG,
        SIDE_LEGEND_JPG,
        SIDE_LEGEND_PDF,
        SIDE_LEGEND_SVG,
    ]:
        print(path)


if __name__ == "__main__":
    main()
