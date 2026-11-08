#!/usr/bin/env python3
"""
Retry Augur tree building with deterministic DNA-safe fallbacks.

Workflow:
1) Run `augur tree` with `--tree-builder-args "-st DNA"`.
2) If it fails with unknown-sequence-type signal, rewrite alignment by
   normalizing non-ACGTN/- characters to N and retry once.
3) Bubble up non-recoverable failures as non-zero exit.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


AMBIGUOUS_TO_N = set("URYKMSWBDHVX")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry augur tree with DNA-safe fallback.")
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output-tree", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--augur-bin", default="augur")
    parser.add_argument("--substitution-model", default="GTR+G")
    return parser.parse_args()


def run_augur_tree(
    augur_bin: str,
    alignment: Path,
    output_tree: Path,
    threads: int,
    substitution_model: str,
) -> subprocess.CompletedProcess:
    cmd = [
        augur_bin,
        "tree",
        "--alignment",
        str(alignment),
        "--output",
        str(output_tree),
        "--method",
        "iqtree",
        "--nthreads",
        str(threads),
        "--substitution-model",
        substitution_model,
        "--tree-builder-args",
        "-st DNA",
    ]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def is_unknown_sequence_type(stderr_text: str) -> bool:
    low = stderr_text.lower()
    return "unknown sequence type" in low or "sequence type" in low and "unknown" in low


def normalize_alignment_to_dna(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as in_handle, dest.open("w") as out_handle:
        for raw in in_handle:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                out_handle.write(line + "\n")
                continue
            seq = line.strip().upper()
            if not seq:
                out_handle.write("\n")
                continue
            normalized = []
            for ch in seq:
                if ch in {"A", "C", "G", "T", "N", "-"}:
                    normalized.append(ch)
                elif ch == "?":
                    normalized.append("N")
                elif ch in AMBIGUOUS_TO_N:
                    normalized.append("N")
                else:
                    normalized.append("N")
            out_handle.write("".join(normalized) + "\n")


def main() -> None:
    args = parse_args()
    alignment = args.alignment.resolve()
    output_tree = args.output_tree.resolve()

    if not alignment.exists():
        raise FileNotFoundError(f"Alignment not found: {alignment}")

    try:
        run_augur_tree(
            augur_bin=args.augur_bin,
            alignment=alignment,
            output_tree=output_tree,
            threads=args.threads,
            substitution_model=args.substitution_model,
        )
        print(f"augur tree succeeded on first attempt: {alignment}")
        return
    except subprocess.CalledProcessError as exc:
        stderr_text = (exc.stderr or "") + "\n" + (exc.stdout or "")
        if not is_unknown_sequence_type(stderr_text):
            raise RuntimeError(
                "augur tree failed and error is not recoverable by DNA-safe retry."
            ) from exc

    safe_alignment = alignment.with_name(alignment.stem + "_dna_safe" + alignment.suffix)
    normalize_alignment_to_dna(alignment, safe_alignment)

    run_augur_tree(
        augur_bin=args.augur_bin,
        alignment=safe_alignment,
        output_tree=output_tree,
        threads=args.threads,
        substitution_model=args.substitution_model,
    )
    print(f"augur tree succeeded after DNA-safe normalization: {safe_alignment}")


if __name__ == "__main__":
    main()
