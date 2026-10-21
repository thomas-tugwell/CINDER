"""Hydrogen-bond roles from a TMol pose, for the backend-independent H-bond graph.

Donors, acceptors and polar hydrogens come from TMol's own hydrogen-bond database
(``ParameterDatabase.scoring.hbond``: donor and acceptor *atom types*), so there are no residue
name tables to maintain. Only ``energy/tmol*.py`` modules import TMol.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cinder.energy.tmol import TMolContext, _import_tmol
from cinder.objectives.hbond_graph import PolarRoles

POLAR_HYDROGEN_PREFIXES = ("Hpol", "HNbb")


def _hbond_atom_types() -> tuple[set[str], set[str]]:
    """``(donor heavy-atom types, acceptor atom types)`` from TMol's hbond database."""
    _import_tmol()
    from tmol import ParameterDatabase  # noqa: PLC0415

    db = ParameterDatabase.get_default().scoring.hbond
    return {g.d for g in db.donor_atom_types}, {g.a for g in db.acceptor_atom_types}


def _atoms_of_pose(pose: Any) -> list[tuple[int, str, str, int]]:
    """``(global atom index, atom name, atom type, residue index)`` for every atom of pose 0."""
    bts = pose.packed_block_types.active_block_types
    off = pose.block_coord_offset[0].long()
    out = []
    for res in range(int(pose.n_res_per_pose[0])):
        bt = bts[int(pose.block_type_ind[0, res])]
        for i, atom in enumerate(bt.atoms):
            out.append((int(off[res]) + i, atom.name, atom.atom_type, res))
    return out


def polar_roles(context: TMolContext, region_cutoff: float | None = 8.0) -> PolarRoles:
    """Donor/acceptor roles of the packed complex held by ``context``.

    ``region_cutoff`` (Å, default 8) restricts the graph to residues with a heavy atom within that
    distance of a heavy atom on the *other* chain, measured on the pose's own coordinates, i.e. an
    interface region that includes second-shell residues. ``None`` keeps every polar atom.
    """
    pose = context.poses["complex"]
    donor_types, acceptor_types = _hbond_atom_types()
    n_atoms = pose.coords.shape[1]
    atoms = _atoms_of_pose(pose)
    if len(atoms) != n_atoms:
        raise ValueError("atom bookkeeping does not match the pose")

    residue = torch.zeros(n_atoms, dtype=torch.long)
    is_h = torch.zeros(n_atoms, dtype=torch.bool)
    types = [""] * n_atoms
    for idx, _name, atype, res in atoms:
        residue[idx] = res
        types[idx] = atype
        is_h[idx] = atype.startswith("H")
    chain = torch.zeros(n_atoms, dtype=torch.long)
    binder_atoms = context.atoms["binder"]
    assert binder_atoms is not None, "context must define binder atoms"
    chain[binder_atoms.cpu()] = 1  # target = 0, binder = 1

    # polar hydrogens and the heavy atom each is bonded to (via TMol's per-residue bond lists)
    bts = pose.packed_block_types.active_block_types
    off = pose.block_coord_offset[0].long()
    donor_heavy: list[int] = []
    donor_h: list[int] = []
    for res in range(int(pose.n_res_per_pose[0])):
        bt = bts[int(pose.block_type_ind[0, res])]
        base = int(off[res])
        for i, j in bt.bond_indices.tolist():
            if (
                bt.atoms[j].atom_type.startswith(POLAR_HYDROGEN_PREFIXES)
                and bt.atoms[i].atom_type in donor_types
            ):
                donor_heavy.append(base + i)
                donor_h.append(base + j)
    acceptors = [i for i, t in enumerate(types) if t in acceptor_types]

    heavy = (~is_h).nonzero().flatten()
    node_atoms = _region_atoms(pose.coords[0].detach().cpu(), residue, chain, heavy, region_cutoff)
    return PolarRoles(
        donor_heavy=torch.tensor(donor_heavy, dtype=torch.long),
        donor_h=torch.tensor(donor_h, dtype=torch.long),
        acceptor=torch.tensor(acceptors, dtype=torch.long),
        residue_of_atom=residue,
        chain_of_atom=chain,
        heavy_atoms=heavy,
        node_atoms=node_atoms,
    )


def _region_atoms(
    coords: Tensor, residue: Tensor, chain: Tensor, heavy: Tensor, cutoff: float | None
) -> Tensor | None:
    """Atoms of residues that have a heavy atom within ``cutoff`` Å of the other chain."""
    if cutoff is None:
        return None
    pos = coords[heavy]
    ch = chain[heavy]
    d = torch.cdist(pos, pos)
    near = ((d < cutoff) & (ch[:, None] != ch[None, :])).any(dim=1)
    region_res = torch.unique(residue[heavy[near]])
    return torch.isin(residue, region_res).nonzero().flatten()
