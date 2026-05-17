from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "01_origin_cluster_robustness.py",
    "02_label_uncertainty_propagation.py",
    "03_within_lineage_contrasts.py",
    "04_phylogenetic_nulls.py",
    "05_mechanism_ablation.py",
    "06_sampling_sensitivity.py",
    "07_manuscript_robustness_summary.py",
]


def main() -> None:
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(BASE_DIR / script)], check=True)


if __name__ == "__main__":
    main()
