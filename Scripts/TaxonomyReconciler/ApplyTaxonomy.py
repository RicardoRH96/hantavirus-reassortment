#!/usr/bin/env python3
"""
Assign standardized taxonomy labels to the master AllHanta dataset using the
lookup table produced by ClassifyHantas.py.

Usage:
    python ApplyTaxonomy.py --lookup hantaviridae_ncbi_taxonomy_lookup.tsv \
        --input Data/TreeGrafting/espalier_pipeline/inputs/master_Data/AllHanta.tsv \
        --output Data/TreeGrafting/espalier_pipeline/inputs/master_Data/AllHanta_taxonomy.tsv
"""
from __future__ import annotations

import argparse
import math
import pandas as pd
import unicodedata
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

RANK_PRIORITY = {
    "species": 0,
    "subspecies": 1,
    "varietas": 2,
    "forma": 3,
    "genus": 4,
    "no rank": 5,
}


def norm_name(value: Optional[str]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_lineage(field: str) -> List[str]:
    if not isinstance(field, str):
        return []
    return [chunk.strip() for chunk in field.split(";") if chunk.strip()]


def select_lineage_ancestor(row: Dict[str, str]) -> Optional[Dict[str, object]]:
    names = parse_lineage(row.get("lineage_names", ""))
    taxids = parse_lineage(row.get("lineage_taxids", ""))
    ranks = parse_lineage(row.get("lineage_ranks", ""))
    lineage = list(zip(names, taxids, ranks))
    if not lineage:
        return None
    # remove the terminal node and iterate ancestors from nearest to farthest
    for name, taxid, rank in reversed(lineage[:-1]):
        lname = name.lower()
        if "strain" in lname:
            continue
        if lname == "root":
            continue
        if lname.startswith("unclassified"):
            parts = lname.split(maxsplit=1)
            suffix = parts[1] if len(parts) > 1 else ""
            clean = f"Unclassified {suffix.title()}".strip()
        else:
            clean = name
        try:
            tid = int(taxid)
        except (TypeError, ValueError):
            tid = None
        return {
            "assigned_taxid": tid,
            "assigned_rank": rank if rank else "no rank",
            "assigned_name": clean,
            "assignment_source": "lineage_rollup",
        }
    return None


def build_lookup(lookup_path: Path) -> Tuple[Dict[str, List[Dict[str, object]]], Set[str]]:
    df = pd.read_csv(lookup_path, sep="\t")
    records = df.to_dict("records")
    name_map: Dict[str, List[Dict[str, object]]] = {}
    genus_whitelist: Set[str] = set()
    for rec in records:
        sci_name = rec.get("sci_name")
        rank = (rec.get("rank") or "").lower()
        if isinstance(sci_name, str) and rank == "species":
            genus = sci_name.split()[0].lower()
            genus_whitelist.add(genus)
        key = norm_name(rec.get("name_txt") or sci_name)
        if not key:
            continue
        name_map.setdefault(key, []).append(rec)
        alt_key = norm_name(rec.get("name_norm"))
        if alt_key and alt_key != key:
            name_map.setdefault(alt_key, []).append(rec)
    return name_map, genus_whitelist


def pick_assignment(candidates: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    sorted_candidates = sorted(
        candidates,
        key=lambda r: RANK_PRIORITY.get((r.get("rank") or "no rank").lower(), 99),
    )
    for rec in sorted_candidates:
        sci_name = rec.get("sci_name") or rec.get("name_txt")
        rank = (rec.get("rank") or "no rank").lower()
        taxid = rec.get("taxid")
        if isinstance(taxid, float) and math.isnan(taxid):
            taxid = None
        if rank == "species" and sci_name:
            if _is_generic_species(sci_name):
                continue
            return {
                "assigned_taxid": int(taxid) if taxid is not None else None,
                "assigned_rank": "species",
                "assigned_name": sci_name,
                "assignment_source": "species_exact",
            }
        lineage_pick = select_lineage_ancestor(rec)
        if lineage_pick:
            if lineage_pick["assigned_taxid"] is None and taxid is not None:
                lineage_pick["assigned_taxid"] = int(taxid)
            return lineage_pick
        if sci_name:
            return {
                "assigned_taxid": int(taxid) if taxid is not None else None,
                "assigned_rank": rec.get("rank") or "no rank",
                "assigned_name": sci_name,
                "assignment_source": "fallback",
            }
    return None


def heuristic_assignment(raw_name: str, genus_whitelist: Set[str]) -> Optional[Dict[str, object]]:
    if not raw_name:
        return None
    stripped = raw_name.strip()
    if not stripped:
        return None
    if _is_placeholder_species(stripped):
        return {
            "assigned_taxid": None,
            "assigned_rank": "no rank",
            "assigned_name": "Unclassified Orthohantavirus",
            "assignment_source": "heuristic_placeholder",
        }
    parts = stripped.split()
    first = parts[0].lower()
    if first == "hantavirus" or stripped.lower().startswith("unclassified hantavirus"):
        return {
            "assigned_taxid": None,
            "assigned_rank": "no rank",
            "assigned_name": "Unclassified Orthohantavirus",
            "assignment_source": "heuristic_unclassified",
        }
    if first in genus_whitelist and len(parts) >= 2:
        canonical = f"{parts[0]} {parts[1]}"
        return {
            "assigned_taxid": None,
            "assigned_rank": "species",
            "assigned_name": canonical,
            "assignment_source": "heuristic_genus",
        }
    if first.endswith("virus") and len(parts) >= 2:
        canonical = f"{parts[0]} {parts[1]}"
        return {
            "assigned_taxid": None,
            "assigned_rank": "species",
            "assigned_name": canonical,
            "assignment_source": "heuristic_virus",
        }
    return None


def _is_generic_species(name: str) -> bool:
    lname = name.lower()
    return lname.startswith("hantavirus ") or lname.startswith("unclassified ")


def _is_placeholder_species(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.strip().lower()
    if not lowered:
        return False
    if lowered.startswith("species:") or lowered.startswith("species "):
        return True
    if lowered.startswith("species"):
        head = lowered.split(maxsplit=1)[0]
        if any(sep in lowered[:10] for sep in (":", "-", ";")):
            return True
    return False


def resolve_metadata_species(row: pd.Series) -> Tuple[Optional[str], str]:
    raw = row.get("Species")
    if not isinstance(raw, str) or not raw.strip():
        return None, ""
    genus_raw = row.get("Genus")
    genus = genus_raw.strip().lower() if isinstance(genus_raw, str) else ""
    species = raw.strip()
    if _is_placeholder_species(species):
        return "Unclassified Orthohantavirus", "metadata_placeholder"
    normalized = species
    if genus == "orthohantavirus" and "orthohantavirus" not in species.lower():
        normalized = "Unclassified Orthohantavirus"
        return normalized, "metadata_unclassified"
    if _is_generic_species(species):
        return None, ""
    return normalized, "metadata_species"


def annotate_master(master_path: Path, lookup_path: Path, output_path: Path) -> None:
    master_df = pd.read_csv(master_path, sep="\t")
    name_map, genus_whitelist = build_lookup(lookup_path)
    assignments: List[Dict[str, object]] = []
    for _, row in master_df.iterrows():
        organism = row.get("Organism_Name") or row.get("Species") or row.get("Organism") or ""
        norm = norm_name(organism)
        assignment = None
        if norm:
            candidates = name_map.get(norm)
            if candidates:
                assignment = pick_assignment(candidates)
        meta_species, meta_source = resolve_metadata_species(row)
        species_valid = bool(meta_species and meta_source)
        if assignment and assignment.get("assignment_source") == "fallback" and species_valid:
            assignment = {
                "assigned_taxid": None,
                "assigned_rank": "species",
                "assigned_name": meta_species,
                "assignment_source": meta_source,
            }
        if assignment is None:
            if species_valid:
                assignment = {
                    "assigned_taxid": None,
                    "assigned_rank": "species",
                    "assigned_name": meta_species,
                    "assignment_source": meta_source,
                }
        if assignment is None:
            heuristic = heuristic_assignment(str(organism or "").strip(), genus_whitelist)
            if heuristic:
                assignment = heuristic
        if assignment is None:
            assignment = {
                "assigned_taxid": None,
                "assigned_rank": "unassigned",
                "assigned_name": organism,
                "assignment_source": "unmatched",
            }
        assignments.append(assignment)
    assign_df = pd.DataFrame(assignments)
    annotated = pd.concat([master_df, assign_df], axis=1)
    annotated.to_csv(output_path, sep="\t", index=False)
    kept = (assign_df["assignment_source"] != "unmatched").sum()
    print(f"Annotated {kept}/{len(assign_df)} rows. Output written to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply taxonomy lookup to the AllHanta master dataset.")
    parser.add_argument("--lookup", type=Path, required=True, help="Path to hantaviridae taxonomy lookup TSV.")
    parser.add_argument("--input", type=Path, required=True, help="Path to AllHanta master TSV.")
    parser.add_argument("--output", type=Path, required=True, help="Destination TSV with annotations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    annotate_master(args.input, args.lookup, args.output)
