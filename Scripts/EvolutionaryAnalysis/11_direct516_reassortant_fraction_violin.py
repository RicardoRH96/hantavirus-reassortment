from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from scipy.stats import gaussian_kde

from evo_utils import DIRECT516_PATH, FIGURE_DIR, PASTEL, REPORT_DIR, SPECIES_LABELS, SPECIES_ORDER, set_plot_style


PNG_PATH = FIGURE_DIR / "11_direct516_reassortant_fraction_violin.png"
SVG_PATH = FIGURE_DIR / "11_direct516_reassortant_fraction_violin.svg"
SUMMARY_PATH = REPORT_DIR / "11_direct516_reassortant_fraction_summary.tsv"
DRAW_PATH = REPORT_DIR / "11_direct516_reassortant_fraction_draws.tsv"

LABEL_COLUMN = "is_reassortant_final"
POSTERIOR_DRAWS = 25_000
RNG_SEED = 516


def hpd(data: np.ndarray, level: float) -> tuple[float, float]:
    values = np.sort(np.asarray(data, dtype=float))
    n_data = len(values)
    n_in = int(round(level * n_data))
    if n_in < 2:
        return float(values.min()), float(values.max())
    widths = values[n_in - 1 :] - values[: n_data - n_in + 1]
    idx = int(np.argmin(widths))
    return float(values[idx]), float(values[idx + n_in - 1])


def _darken(hex_color: str, amount: float = 0.35) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(hex_color), dtype=float)
    return tuple(rgb * (1.0 - amount))


def load_direct516_labels() -> pd.DataFrame:
    if not DIRECT516_PATH.exists():
        raise FileNotFoundError(f"Missing direct516 dataset: {DIRECT516_PATH}")
    df = pd.read_csv(DIRECT516_PATH, sep="\t", low_memory=False)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing required label column: {LABEL_COLUMN}")
    df = df[df[LABEL_COLUMN].notna()].copy()
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    df = df[df[LABEL_COLUMN].isin([0, 1])].copy()
    df["species_label"] = df["species"].map(SPECIES_LABELS).fillna(df["species"].astype(str))
    return df


def build_fraction_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "dataset": "Global",
            "species": "all_direct516",
            "reassortant_n": int(df[LABEL_COLUMN].sum()),
            "total_n": int(len(df)),
        }
    ]
    for species in SPECIES_ORDER:
        sub = df[df["species"].eq(species)]
        if sub.empty:
            continue
        rows.append(
            {
                "dataset": SPECIES_LABELS.get(species, species),
                "species": species,
                "reassortant_n": int(sub[LABEL_COLUMN].sum()),
                "total_n": int(len(sub)),
            }
        )
    out = pd.DataFrame(rows)
    out["nonreassortant_n"] = out["total_n"] - out["reassortant_n"]
    out["observed_fraction"] = out["reassortant_n"] / out["total_n"]
    return out


def posterior_fraction_draws(summary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    draw_frames = []
    for row in summary.itertuples(index=False):
        alpha = float(row.reassortant_n) + 0.5
        beta = float(row.nonreassortant_n) + 0.5
        draws = rng.beta(alpha, beta, size=POSTERIOR_DRAWS)
        draw_frames.append(
            pd.DataFrame(
                {
                    "dataset": row.dataset,
                    "species": row.species,
                    "fraction": draws,
                }
            )
        )
    return pd.concat(draw_frames, ignore_index=True)


def ordered_datasets(summary: pd.DataFrame) -> list[str]:
    medians = summary.set_index("dataset")["observed_fraction"]
    anchor = [label for label in ["Global", "Andes", "Sin Nombre"] if label in medians.index]
    rest = medians.drop(anchor, errors="ignore").sort_values().index.tolist()
    return anchor + rest


def add_draw_summaries(summary: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    draw_summary = (
        draws.groupby("dataset", as_index=False)["fraction"]
        .agg(
            posterior_median="median",
            posterior_q025=lambda s: float(np.quantile(s, 0.025)),
            posterior_q975=lambda s: float(np.quantile(s, 0.975)),
        )
    )
    return summary.merge(draw_summary, on="dataset", how="left")


def plot_violin(summary: pd.DataFrame, draws: pd.DataFrame) -> None:
    set_plot_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 1.5,
        }
    )
    colors = {"Global": "#f2b8b5"}
    colors.update({label: PASTEL.get(label, "#d0d0d0") for label in summary["dataset"] if label != "Global"})
    order = ordered_datasets(summary)
    observed = summary.set_index("dataset")["observed_fraction"].to_dict()

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)
    for xpos, dataset in enumerate(order):
        data = draws.loc[draws["dataset"].eq(dataset), "fraction"].dropna().to_numpy(dtype=float)
        if len(data) == 0:
            continue
        hpd_lo, hpd_hi = hpd(data, 0.95)
        if np.isclose(hpd_lo, hpd_hi):
            hpd_lo = max(0.0, hpd_lo - 1e-3)
            hpd_hi = min(1.0, hpd_hi + 1e-3)
        grid = np.linspace(hpd_lo, hpd_hi, 240)
        kde_values = gaussian_kde(data, bw_method=0.4)(grid)
        kde_values = kde_values / max(kde_values.max(), 1e-12) * 0.4
        color = colors.get(dataset, "#d0d0d0")

        ax.fill_betweenx(
            grid,
            xpos - kde_values,
            xpos + kde_values,
            facecolor=color,
            alpha=0.82,
            edgecolor=None,
            zorder=2,
        )
        ax.plot(xpos + kde_values, grid, color="#111111", lw=0.8, solid_capstyle="round", zorder=3)
        ax.plot(xpos - kde_values, grid, color="#111111", lw=0.8, solid_capstyle="round", zorder=3)
        ax.hlines(
            observed[dataset],
            xpos - 0.42,
            xpos + 0.42,
            colors="#111111",
            linewidth=2.0,
            capstyle="round",
            zorder=4,
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Reassortant sequences / total", labelpad=10)
    ax.set_xlabel("Dataset", labelpad=8)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(-0.025, 0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(length=3, color="#666666", width=0.8)
    ax.grid(False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.17, top=0.975)
    fig.savefig(PNG_PATH, dpi=450, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_direct516_labels()
    summary = build_fraction_summary(labels)
    draws = posterior_fraction_draws(summary)
    summary = add_draw_summaries(summary, draws)
    summary.to_csv(SUMMARY_PATH, sep="\t", index=False)
    draws.to_csv(DRAW_PATH, sep="\t", index=False)
    plot_violin(summary, draws)
    print(PNG_PATH)
    print(SVG_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
