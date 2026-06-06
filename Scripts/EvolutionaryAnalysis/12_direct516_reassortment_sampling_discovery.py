from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from evo_utils import DIRECT516_PATH, FIGURE_DIR, PASTEL, REPORT_DIR, SPECIES_LABELS, SPECIES_ORDER, set_plot_style


LABEL_COLUMN = "is_reassortant_final"
PNG_PATH = FIGURE_DIR / "12_direct516_reassortment_sampling_discovery_by_species.png"
SVG_PATH = FIGURE_DIR / "12_direct516_reassortment_sampling_discovery_by_species.svg"
CURVE_PATH = REPORT_DIR / "12_direct516_reassortment_sampling_discovery_curve.tsv"
FIT_PATH = REPORT_DIR / "12_direct516_reassortment_sampling_discovery_fits.tsv"
ENVELOPE_PATH = REPORT_DIR / "12_direct516_reassortment_sampling_discovery_permutation_envelope.tsv"
N_PERMUTATIONS = 10_000
RNG_SEED = 516
LOG_FIT_COLOR = "indianred"


def _darken(hex_color: str, amount: float = 0.28) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(hex_color), dtype=float)
    return tuple(np.clip(rgb * (1.0 - amount), 0, 1))


def greyed_color(color: str | tuple[float, float, float], frac_to_grey: float = 0.55) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(color), dtype=float)
    grey = np.array([0.5, 0.5, 0.5], dtype=float)
    return tuple(((1.0 - frac_to_grey) * rgb + frac_to_grey * grey).tolist())


def load_direct516() -> pd.DataFrame:
    if not DIRECT516_PATH.exists():
        raise FileNotFoundError(f"Missing direct516 dataset: {DIRECT516_PATH}")

    df = pd.read_csv(DIRECT516_PATH, sep="\t", low_memory=False)
    required = {"species", "date", "date_precision", LABEL_COLUMN, "canonical_isolate", "tip_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Direct516 dataset is missing required columns: {missing}")

    df = df[df[LABEL_COLUMN].notna()].copy()
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    df = df[df[LABEL_COLUMN].isin([0, 1])].copy()
    df["is_reassortant"] = df[LABEL_COLUMN].astype(int)
    df["collection_date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["collection_date"].isna().any():
        bad = df.loc[df["collection_date"].isna(), ["species", "canonical_isolate", "date"]]
        raise ValueError(f"Could not parse collection dates for {len(bad)} rows")

    df["species_label"] = df["species"].map(SPECIES_LABELS).fillna(df["species"].astype(str))
    return df


def build_curve_table(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for species in SPECIES_ORDER:
        sub = df[df["species"].eq(species)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            ["collection_date", "canonical_isolate", "tip_name"],
            kind="mergesort",
        ).reset_index(drop=True)
        sub["collection_year"] = sub["collection_date"].dt.year.astype(int)
        sub["cumulative_sampled_sequences"] = np.arange(1, len(sub) + 1)
        sub["cumulative_reassortant_tips"] = sub["is_reassortant"].cumsum()
        frames.append(
            sub[
                [
                    "species",
                    "species_label",
                    "canonical_isolate",
                    "tip_name",
                    "date",
                    "date_precision",
                    "collection_year",
                    "is_reassortant",
                    "cumulative_sampled_sequences",
                    "cumulative_reassortant_tips",
                ]
            ].copy()
        )
    return pd.concat(frames, ignore_index=True)


def log_least_squares_fit(curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve = curve.copy()
    fit_rows: list[dict[str, object]] = []

    for species in SPECIES_ORDER:
        mask = curve["species"].eq(species)
        sub = curve.loc[mask].copy()
        if sub.empty:
            continue

        x = sub["cumulative_sampled_sequences"].to_numpy(dtype=float)
        y = sub["cumulative_reassortant_tips"].to_numpy(dtype=float)
        X = np.column_stack([np.ones_like(x), np.log(x)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_fit = beta[0] + beta[1] * np.log(x)
        curve.loc[mask, "log_ls_fit"] = y_fit

        ss_res = float(np.sum((y - y_fit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot

        positive = sub[sub["is_reassortant"].eq(1)]
        first_positive = positive.iloc[0] if not positive.empty else None
        fit_rows.append(
            {
                "species": species,
                "species_label": SPECIES_LABELS.get(species, species),
                "n_sequences": int(len(sub)),
                "n_reassortant_tips": int(y[-1]),
                "prevalence": float(y[-1] / len(sub)),
                "log_fit_intercept": float(beta[0]),
                "log_fit_slope": float(beta[1]),
                "log_fit_r2": r2,
                "first_positive_date": None if first_positive is None else first_positive["date"],
                "sampled_sequences_until_first_positive": None
                if first_positive is None
                else int(first_positive["cumulative_sampled_sequences"]),
            }
        )

    return curve, pd.DataFrame(fit_rows)


def permutation_envelope(curve: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[pd.DataFrame] = []

    for species in SPECIES_ORDER:
        sub = curve[curve["species"].eq(species)].copy()
        if sub.empty:
            continue

        labels = sub["is_reassortant"].to_numpy(dtype=np.int16)
        n = len(labels)
        if n == 0:
            continue

        permutations = np.empty((N_PERMUTATIONS, n), dtype=np.int16)
        for idx in range(N_PERMUTATIONS):
            permutations[idx] = np.cumsum(rng.permutation(labels))

        rows.append(
            pd.DataFrame(
                {
                    "species": species,
                    "species_label": SPECIES_LABELS.get(species, species),
                    "cumulative_sampled_sequences": np.arange(1, n + 1),
                    "permutation_p025": np.percentile(permutations, 2.5, axis=0),
                    "permutation_mean": np.mean(permutations, axis=0),
                    "permutation_p975": np.percentile(permutations, 97.5, axis=0),
                    "n_permutations": N_PERMUTATIONS,
                    "rng_seed": RNG_SEED,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def plot_discovery_curve(curve: pd.DataFrame) -> None:
    set_plot_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "font.size": 12,
            "axes.labelsize": 16,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(13.6, 11.2), dpi=300, constrained_layout=True)
    axes = fig.subplots(3, 3, sharex=False, sharey=False).ravel()
    letters = ["a)", "b)", "c)", "d)", "e)", "f)", "g)"]

    for ax, species, letter in zip(axes, SPECIES_ORDER, letters):
        sub = curve[curve["species"].eq(species)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        label = SPECIES_LABELS.get(species, species)
        link_color = _darken(PASTEL.get(label, "#9aa8b6"), amount=0.22)
        env_color = greyed_color(link_color, frac_to_grey=0.55)
        x = sub["cumulative_sampled_sequences"].to_numpy(dtype=float)
        y = sub["cumulative_reassortant_tips"].to_numpy(dtype=float)
        y_fit = sub["log_ls_fit"].to_numpy(dtype=float)
        lo = sub["permutation_p025"].to_numpy(dtype=float)
        hi = sub["permutation_p975"].to_numpy(dtype=float)
        years_ordered = sub["collection_year"].to_numpy(dtype=int)

        if len(years_ordered) > 0:
            change_idx = np.where(np.diff(years_ordered) != 0)[0]
            boundaries = np.r_[0, change_idx + 1, len(sub)]
            shade = False
            for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
                if shade:
                    ax.axvspan(b0 + 0.5, b1 + 0.5, facecolor="k", edgecolor="none", alpha=0.04, zorder=0)
                shade = not shade

        ax.plot(x, lo, color=env_color, lw=1.4, zorder=6)
        ax.plot(x, hi, color=env_color, lw=1.4, zorder=6)
        ax.fill_between(
            x,
            lo,
            hi,
            facecolor="none",
            edgecolor=env_color,
            hatch="///",
            linewidth=0.0,
            zorder=5,
        )

        ax.plot(x, y, color=link_color, lw=2.1, alpha=0.45, zorder=18)
        ax.scatter(x, y, s=16, facecolor=link_color, edgecolor="none", zorder=20)

        ax.plot(x, y_fit, color="white", lw=4.2, zorder=21)
        ax.plot(x, y_fit, color=LOG_FIT_COLOR, lw=2.3, zorder=22)

        ax.text(
            0.02,
            0.96,
            f"{letter} {label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
        )

        y_top = np.nanmax(np.r_[y, hi, y_fit, 1.0])
        y_pad = max(1e-6, y_top * 0.03)
        ax.set_ylim(-y_pad, max(1e-6, y_top * 1.06))
        ax.set_xlim(0, float(x.max()) * 1.04)
        ax.grid(axis="y", ls="--", color="lightgrey", zorder=0)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=5))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(length=3.5, color="#666666", width=0.8)

    axes[7].set_axis_off()
    axes[8].set_visible(False)

    fig.supxlabel("Sampled sequences", fontsize=16)
    fig.supylabel("Cumulative reassortant tips", fontsize=16)

    style_handles = [
        Patch(facecolor="none", edgecolor=greyed_color("#9aa8b6"), hatch="///", label="Permutation 95% interval"),
        Line2D(
            [0],
            [0],
            color="#9aa8b6",
            lw=2.1,
            alpha=0.45,
            marker="o",
            markersize=5,
            markerfacecolor="#9aa8b6",
            markeredgecolor="none",
            label="Observed",
        ),
        Line2D([0], [0], color=LOG_FIT_COLOR, lw=2.3, label="Log least-squares fit"),
    ]
    axes[7].legend(
        handles=style_handles,
        frameon=False,
        fontsize=11,
        loc="center",
        handlelength=2.3,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=450, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_direct516()
    curve = build_curve_table(labels)
    curve, fits = log_least_squares_fit(curve)
    envelope = permutation_envelope(curve)
    curve = curve.merge(
        envelope[
            [
                "species",
                "cumulative_sampled_sequences",
                "permutation_p025",
                "permutation_mean",
                "permutation_p975",
            ]
        ],
        on=["species", "cumulative_sampled_sequences"],
        how="left",
    )
    curve.to_csv(CURVE_PATH, sep="\t", index=False)
    fits.to_csv(FIT_PATH, sep="\t", index=False)
    envelope.to_csv(ENVELOPE_PATH, sep="\t", index=False)
    plot_discovery_curve(curve)
    print(PNG_PATH)
    print(SVG_PATH)
    print(CURVE_PATH)
    print(FIT_PATH)
    print(ENVELOPE_PATH)


if __name__ == "__main__":
    main()
