"""Fetch and prepare benchmark complexes from the RCSB PDB.

Entries are downloaded at run time into a cache directory (``data/`` by default, git-ignored)
rather than committed. Prepared files keep the first model, ATOM records only, no hydrogens,
and the first alternate location, for the chains requested.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


@dataclass(frozen=True)
class ComplexSpec:
    """One benchmark complex: PDB id plus which chains are target and binder."""

    pdb_id: str
    target_chains: list[str]
    binder_chains: list[str]
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not _PDB_ID.match(self.pdb_id):
            raise ValueError(f"invalid PDB id {self.pdb_id!r}")
        if not self.target_chains or not self.binder_chains:
            raise ValueError("target_chains and binder_chains must be non-empty")
        if set(self.target_chains) & set(self.binder_chains):
            raise ValueError("a chain cannot be both target and binder")

    @property
    def chains(self) -> list[str]:
        return [*self.target_chains, *self.binder_chains]


def load_manifest(path: str | Path) -> list[ComplexSpec]:
    """Read ``complexes:`` entries (``id``, ``target_chains``, ``binder_chains``, ...)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    specs = []
    for entry in raw.get("complexes", []):
        unknown = set(entry) - {"id", "target_chains", "binder_chains", "description", "tags"}
        if unknown:
            raise ValueError(f"unknown manifest keys {sorted(unknown)} in {entry.get('id')}")
        specs.append(
            ComplexSpec(
                pdb_id=entry["id"].upper(),
                target_chains=list(entry["target_chains"]),
                binder_chains=list(entry["binder_chains"]),
                description=entry.get("description", ""),
                tags=list(entry.get("tags", [])),
            )
        )
    return specs


def fetch_pdb(pdb_id: str, cache_dir: str | Path, timeout: float = 60.0) -> Path:
    """Download ``pdb_id`` into ``cache_dir`` (reused if present) and sanity-check the content."""
    if not _PDB_ID.match(pdb_id):
        raise ValueError(f"invalid PDB id {pdb_id!r}")
    dest = Path(cache_dir) / "raw" / f"{pdb_id.upper()}.pdb"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RCSB_URL.format(pdb_id=pdb_id.upper()), timeout=timeout) as r:  # noqa: S310
        text = r.read().decode("ascii", errors="replace")
    if "ATOM  " not in text:
        raise ValueError(f"{pdb_id}: downloaded file contains no ATOM records")
    dest.write_text(text)
    return dest


def extract_chains(source: str | Path, chains: list[str], dest: str | Path) -> Path:
    """Write first-model heavy-atom ATOM records of ``chains`` (first altloc only), in the
    order the chains are requested. Raises if any requested chain is absent."""
    by_chain: dict[str, list[str]] = {c: [] for c in chains}
    with open(source) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  "):
                continue
            chain, altloc = line[21], line[16]
            element = line[76:78].strip() or line[12:16].strip()[:1]
            if chain not in by_chain or altloc not in (" ", "A") or element in ("H", "D"):
                continue
            by_chain[chain].append(line[:16] + " " + line[17:].rstrip("\n"))
    missing = [c for c, lines in by_chain.items() if not lines]
    if missing:
        raise ValueError(f"chains {missing} not found in {source}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = [ln for c in chains for ln in by_chain[c]]
    dest.write_text("\n".join(out) + "\nEND\n")
    return dest


def prepare_complex(spec: ComplexSpec, cache_dir: str | Path = "data") -> Path:
    """Fetch (if needed) and extract ``spec``'s chains; returns the prepared PDB path."""
    raw = fetch_pdb(spec.pdb_id, cache_dir)
    return extract_chains(raw, spec.chains, Path(cache_dir) / "prepared" / f"{spec.pdb_id}.pdb")


def helical_fraction(path: str | Path, chains: list[str] | None = None) -> dict[str, float]:
    """Fraction of residues annotated alpha-helix per chain (biotite P-SEA). Needs biotite,
    which ships with TMol's dependencies."""
    import biotite.structure as struc  # noqa: PLC0415
    import biotite.structure.io.pdb as pdb  # noqa: PLC0415

    atoms = pdb.PDBFile.read(str(path)).get_structure(model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    out: dict[str, float] = {}
    for chain in sorted(set(atoms.chain_id)):
        if chains is not None and chain not in chains:
            continue
        sse = struc.annotate_sse(atoms[atoms.chain_id == chain])
        out[str(chain)] = float((sse == "a").sum() / len(sse))
    return out


def chain_breaks(
    path: str | Path, max_ca_distance: float = 5.0
) -> dict[str, list[tuple[int, int, float]]]:
    """Per-chain gaps: consecutive CA atoms farther than ``max_ca_distance`` Å apart.

    Returns ``{chain: [(res_before, res_after, distance), ...]}`` for chains with breaks. A
    structure with missing residues makes TMol bridge the gap with a stretched bond, so such
    chains are unsuitable for energy-based benchmarks unless handled explicitly. The 5.0 Å default
    avoids flagging numbering gaps (a single missing residue gives at least ~5.3 Å).
    """
    last: dict[str, tuple[int, tuple[float, float, float]]] = {}
    out: dict[str, list[tuple[int, int, float]]] = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            chain, resi = line[21], int(line[22:26])
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            if chain in last:
                prev_resi, prev = last[chain]
                dist = sum((a - b) ** 2 for a, b in zip(xyz, prev, strict=True)) ** 0.5
                if dist > max_ca_distance:
                    out.setdefault(chain, []).append((prev_resi, resi, round(dist, 1)))
            last[chain] = (resi, xyz)
    return out


def resolve_complex(
    complex_id: str | None,
    manifest: str | Path,
    data_dir: str | Path,
    pdb: str | Path,
    target: Sequence[str],
    binder: Sequence[str],
    allow_breaks: bool = False,
) -> tuple[str, list[str], list[str]]:
    """``(pdb path, target chains, binder chains)`` from a manifest id (downloaded and prepared on
    first use) or, if ``complex_id`` is ``None``, from an explicit ``pdb`` path and chains.

    Raises :class:`ValueError` for an unknown id or, unless ``allow_breaks``, a structure with
    chain breaks (TMol would bridge the gap with a stretched bond).
    """
    if complex_id:
        specs = {s.pdb_id: s for s in load_manifest(manifest)}
        if complex_id.upper() not in specs:
            raise ValueError(f"{complex_id} not in {manifest}: {sorted(specs)}")
        spec = specs[complex_id.upper()]
        path, targets, binders = (
            str(prepare_complex(spec, data_dir)),
            spec.target_chains,
            spec.binder_chains,
        )
    else:
        path, targets, binders = str(pdb), list(target), list(binder)
    breaks = chain_breaks(path)
    if breaks and not allow_breaks:
        raise ValueError(f"chain breaks in {path}: {breaks} (pass allow_breaks to override)")
    return path, list(targets), list(binders)
