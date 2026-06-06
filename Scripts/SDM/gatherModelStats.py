#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGS_DIR = REPO_ROOT

PAT_JOB_ID = re.compile(r"\.(\d+)_\d+\.out$")
PAT_SPECIES = re.compile(r"INPUT_FILE=.*?/Inputs/(.+?)\.geojson")
PAT_PRES = re.compile(r"Presences not thinned:\s*(\d+)")
PAT_EXTRA_ARGS = re.compile(r"^\[INFO\] EXTRA_ARGS:(.*)$", re.MULTILINE)
PAT_METRIC = re.compile(
    r"\[(xgb|lgbm|ensemble)(?:\s+([^\]]+))?\]\s*ROC-AUC=(\S+)\s*\|\s*PR-AUC=(\S+)\s*\|\s*Brier=(\S+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse SDM SLURM .out logs and write a CSV summary of XGBoost, LightGBM, "
            "and Ensemble training metrics."
        ),
        epilog=(
            "Examples:\n"
            "  python src/gatherModelStats.py 21498643 training_summary_21498643_tuned.csv\n"
            "  python src/gatherModelStats.py --logs-dir /path/to/logs --glob 'SDM_Array*.out'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "keyword",
        nargs="?",
        default="",
        help="Only include log files whose name contains this substring, e.g. a SLURM job ID.",
    )
    parser.add_argument(
        "outfile",
        nargs="?",
        default="training_summary.csv",
        help="CSV file to write. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(DEFAULT_LOGS_DIR),
        help=f"Directory containing SLURM .out files. Default: {DEFAULT_LOGS_DIR}",
    )
    parser.add_argument(
        "--glob",
        default="SDM_Array_HigherResolution.*.out",
        help="Glob used to select log files inside --logs-dir.",
    )
    return parser.parse_args()


def read_rows(logs_dir: Path, glob_pattern: str, keyword: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(logs_dir.glob(glob_pattern)):
        if not path.is_file():
            continue
        if keyword and keyword not in path.name:
            continue

        text = path.read_text(errors="replace")
        species = PAT_SPECIES.search(text)
        pres = PAT_PRES.search(text)
        extra_args = PAT_EXTRA_ARGS.search(text)
        metrics = {
            model: {
                "label": (label or "").strip(),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier": brier,
            }
            for model, label, roc_auc, pr_auc, brier in PAT_METRIC.findall(text)
        }

        if not (species and pres and metrics.get("xgb") and metrics.get("lgbm")):
            continue

        job_id_match = PAT_JOB_ID.search(path.name)
        tuning_requested = bool(extra_args and "--tune" in extra_args.group(1))

        rows.append(
            {
                "job_id": job_id_match.group(1) if job_id_match else "",
                "species": species.group(1),
                "presences_not_thinned": pres.group(1),
                "tuning_requested": tuning_requested,
                "xgb_label": metrics["xgb"]["label"],
                "xgb_roc_auc": metrics["xgb"]["roc_auc"],
                "xgb_pr_auc": metrics["xgb"]["pr_auc"],
                "xgb_brier": metrics["xgb"]["brier"],
                "lgbm_label": metrics["lgbm"]["label"],
                "lgbm_roc_auc": metrics["lgbm"]["roc_auc"],
                "lgbm_pr_auc": metrics["lgbm"]["pr_auc"],
                "lgbm_brier": metrics["lgbm"]["brier"],
                "ensemble_label": metrics.get("ensemble", {}).get("label", ""),
                "ensemble_roc_auc": metrics.get("ensemble", {}).get("roc_auc", ""),
                "ensemble_pr_auc": metrics.get("ensemble", {}).get("pr_auc", ""),
                "ensemble_brier": metrics.get("ensemble", {}).get("brier", ""),
                "source_file": path.name,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir).expanduser().resolve()
    outfile = Path(args.outfile).expanduser()

    if not logs_dir.is_dir():
        print(f"[ERROR] Logs directory does not exist: {logs_dir}", file=sys.stderr)
        return 1

    rows = read_rows(logs_dir, args.glob, args.keyword)
    if not rows:
        print(
            "[ERROR] No matching logs with both XGBoost and LightGBM metrics were found.\n"
            f"         logs_dir={logs_dir}\n"
            f"         glob={args.glob}\n"
            f"         keyword={args.keyword or '<none>'}",
            file=sys.stderr,
        )
        return 1

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {outfile.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
