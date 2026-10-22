"""Read measured binding changes from a SKEMPI 2.0 CSV for use with the mutant scan.

SKEMPI (Jankauskaite et al., Bioinformatics 2019) lists measured affinities of wild-type and
mutant protein-protein complexes. The file is **not shipped here**: download it yourself and check
its terms of use. This module only parses it:

* keeps single substitutions on one chain of one entry (the binder),
* converts affinities to ``ddG = R T ln(Kd_mut / Kd_wt)`` in kcal/mol (**positive = weaker
  binding**, the same sign as a computed energy change),
* averages repeated measurements of the same substitution (in ddG, i.e. log affinity),
* maps PDB residue numbers (with insertion codes) to the 0-based index along the chain in the
  prepared structure, which is what :mod:`cinder.validation.mutant_scan` expects.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

GAS_CONSTANT = 0.0019872041  # kcal / (mol K)
DEFAULT_TEMPERATURE = 298.0

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip

_MUTATION = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Z])$")


@dataclass(frozen=True)
class Measurement:
    """One single substitution with its measured change in binding free energy."""

    pdb: str  # SKEMPI entry, e.g. ``3SGB_E_I``
    chain: str
    wild_type: str
    residue: str  # PDB number with insertion code, e.g. ``45`` or ``52A``
    mutant: str
    ddg: float  # kcal/mol, positive = weaker binding
    n_measurements: int = 1


def ddg_kcal(kd_mut: float, kd_wt: float, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """``R T ln(Kd_mut / Kd_wt)``; positive when the mutant binds more weakly."""
    if kd_mut <= 0 or kd_wt <= 0:
        raise ValueError("affinities must be positive")
    return GAS_CONSTANT * temperature * math.log(kd_mut / kd_wt)


def _temperature(text: str) -> float:
    m = re.search(r"\d+(\.\d+)?", text or "")
    value = float(m.group()) if m else DEFAULT_TEMPERATURE
    return value if 200.0 < value < 400.0 else DEFAULT_TEMPERATURE


def read_single_mutants(path: str | Path, entry: str, chain: str) -> list[Measurement]:
    """Single substitutions on ``chain`` of SKEMPI entry ``entry`` (e.g. ``3SGB_E_I``), with
    repeated measurements averaged. Rows without both affinities are skipped."""
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row["#Pdb"].upper() != entry.upper() or "," in row["Mutation(s)_PDB"]:
                continue
            m = _MUTATION.match(row["Mutation(s)_PDB"].strip())
            if not m or m.group(2) != chain:
                continue
            try:
                kd_mut = float(row["Affinity_mut_parsed"])
                kd_wt = float(row["Affinity_wt_parsed"])
            except ValueError:
                continue
            temperature = _temperature(row["Temperature"])
            grouped[(m.group(1), m.group(3), m.group(4))].append(
                ddg_kcal(kd_mut, kd_wt, temperature)
            )
    return [
        Measurement(entry.upper(), chain, wt, res, mut, sum(v) / len(v), len(v))
        for (wt, res, mut), v in sorted(grouped.items(), key=lambda kv: _sort_key(kv[0][1]))
    ]


def _sort_key(residue: str) -> tuple[int, str]:
    m = re.match(r"(-?\d+)(.*)", residue)
    return (int(m.group(1)), m.group(2)) if m else (0, residue)


def residue_index_map(pdb_path: str | Path, chain: str) -> dict[str, tuple[int, str]]:
    """``{pdb residue label: (0-based index, one-letter code)}`` along ``chain`` of a PDB file,
    in file order, counting each residue (with CA) once. Labels are number plus insertion code."""
    out: dict[str, tuple[int, str]] = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  ") or line[21] != chain or line[12:16].strip() != "CA":
                continue
            label = f"{int(line[22:26])}{line[26].strip()}"
            if label not in out:
                out[label] = (len(out), THREE_TO_ONE.get(line[17:20], "X"))
    return out


def to_scan_rows(
    measurements: list[Measurement], index_map: dict[str, tuple[int, str]]
) -> tuple[list[dict[str, str | float]], list[str]]:
    """CSV rows (``name``, ``mutation``, ``measured``) for :mod:`mutant_scan`, plus a list of
    skipped entries (residue absent from the structure, or wild-type letter mismatch)."""
    rows: list[dict[str, str | float]] = []
    skipped: list[str] = []
    for m in measurements:
        name = f"{m.wild_type}{m.chain}{m.residue}{m.mutant}"
        hit = index_map.get(m.residue)
        if hit is None:
            skipped.append(f"{name}: residue {m.residue} not in the structure")
        elif hit[1] != m.wild_type:
            skipped.append(f"{name}: structure has {hit[1]} at {m.residue}")
        else:
            rows.append(
                {"name": name, "mutation": f"{m.wild_type}{hit[0]}{m.mutant}", "measured": m.ddg}
            )
    return rows, skipped
