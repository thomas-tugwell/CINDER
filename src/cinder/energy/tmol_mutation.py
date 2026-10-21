"""Sequence-specific TMol poses: mutate binder residues and repack side chains.

Mutating residues changes the atom count, so a pose for sequence ``S`` cannot share a coordinate
tensor with the native pose. :class:`TMolSequenceBuilder` builds one full-atom pose per sequence
with TMol's packer (backbone fixed, side chains of a *fixed repack region* optimised), caches it,
and scores it. The repack region depends only on the design positions, never on the sequence, so
every sequence -- including the native one, which also goes through the packer -- is treated
identically and energies are comparable.

Only ``energy/tmol*.py`` modules may import TMol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cinder.energy.tmol import (
    TMolContext,
    _ca_atom_indices,
    _import_tmol,
    _residue_atom_indices,
    _sync_subpose,
    group_energies,
)
from cinder.systems.specification import MolecularSystem
from cinder.types import AA_TO_INDEX, SequenceBatch, encode_sequence

log = logging.getLogger("cinder.tmol.mutation")

_ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS",
    "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN",
    "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}  # fmt: skip

REPACK_MODES = ("design", "neighbors", "all")


class MutationError(RuntimeError):
    """The packer did not produce the requested residue (never fail silently)."""


@dataclass
class MutantPoses:
    """Packed complex plus the chain sub-poses sharing its coordinates."""

    sequence: str
    complex: Any
    target: Any
    binder: Any
    context: TMolContext

    @property
    def coordinates(self) -> Tensor:
        """Complex coordinates ``[n_atoms, 3]`` of the packed pose."""
        return self.complex.coords[0].detach()


def _residue_range(system: MolecularSystem, key: str) -> tuple[int, int]:
    value: Any = system.extras[key]
    return int(value[0]), int(value[1])


class TMolSequenceBuilder:
    """Build, cache and score full-atom poses for binder sequences.

    ``repack``: ``design`` repacks only the design positions; ``neighbors`` also repacks every
    residue with a heavy atom within ``cutoff`` Å of a design residue (measured on the native
    pose, so the region is the same for all sequences); ``all`` repacks every residue.
    """

    def __init__(
        self,
        system: MolecularSystem,
        repack: str = "neighbors",
        cutoff: float = 6.0,
        seed: int = 0,
    ) -> None:
        if repack not in REPACK_MODES:
            raise ValueError(f"repack must be one of {REPACK_MODES}, got {repack!r}")
        self.system = system
        self.ctx: TMolContext = system.extras["tmol"]  # type: ignore[assignment]
        self.b_lo, self.b_hi = _residue_range(system, "binder_residue_range")
        self.t_lo, self.t_hi = _residue_range(system, "target_residue_range")
        self.repack, self.cutoff, self.seed = repack, cutoff, seed
        self._cache: dict[str, MutantPoses] = {}
        self._systems: dict[str, MolecularSystem] = {}
        self._repack_mask = self._compute_repack_mask()
        self._design_blocks = [self.b_lo + p for p in system.design_space.positions]

    # ------------------------------------------------------------------ region selection

    def _compute_repack_mask(self) -> Tensor:
        pose = self.ctx.poses["complex"]
        n_blocks = pose.max_n_blocks
        design = torch.zeros(n_blocks, dtype=torch.bool)
        for p in self.system.design_space.positions:
            design[self.b_lo + p] = True
        if self.repack == "design":
            return design
        if self.repack == "all":
            return torch.arange(n_blocks) < int(pose.n_res_per_pose[0])
        off = pose.block_coord_offset[0].long()
        n_at = pose.n_ats_per_block[0].long()
        coords = pose.coords[0].detach()
        n_res = int(pose.n_res_per_pose[0])
        owner = torch.repeat_interleave(torch.arange(n_res), n_at[:n_res])
        # distance from every atom to the nearest design-residue atom
        design_atoms = design[owner]
        d = torch.cdist(coords, coords[design_atoms]).min(dim=1).values
        near = torch.zeros(n_blocks, dtype=torch.bool)
        near[:n_res] = torch.zeros(n_res, dtype=torch.bool).index_put(
            (owner[d < self.cutoff],), torch.tensor(True)
        )
        _ = off  # offsets are implied by ``owner``
        return near | design

    # ------------------------------------------------------------------------- building

    def _validate(self, sequence: str) -> Tensor:
        space = self.system.design_space
        if len(sequence) != space.length:
            raise ValueError(f"sequence length {len(sequence)} != binder length {space.length}")
        tokens = encode_sequence(sequence)
        fixed = [i for i in range(space.length) if i not in space.positions]
        bad = [i for i in fixed if sequence[i] != space.native[i]]
        if bad:
            raise ValueError(f"non-design positions {bad} differ from the native sequence")
        mask = space.mask()
        for row, p in enumerate(space.positions):
            if not mask[row, AA_TO_INDEX[sequence[p]]]:
                raise ValueError(f"{sequence[p]} is not allowed at design position {p}")
        return tokens

    def build(self, sequence: str) -> MutantPoses:
        """Poses for ``sequence`` on the native backbone (cached). Raises :class:`MutationError`
        if a requested residue is not what the packer returned."""
        if sequence not in self._cache:
            self._cache[sequence] = self.build_from(sequence, None)
        return self._cache[sequence]

    def build_from(self, sequence: str, coordinates: Tensor | None) -> MutantPoses:
        """Pack ``sequence`` on a given backbone (not cached).

        ``coordinates`` ``[n_atoms, 3]`` are in the *native topology* (the atom layout of the
        loaded complex), e.g. a perturbed copy of ``system.reference_coordinates``; the packer then
        builds side chains for the requested residues on that backbone. ``None`` uses the crystal
        pose.
        """
        self._validate(sequence)
        tmol = _import_tmol()
        import attr  # noqa: PLC0415  (a TMol dependency)
        from tmol import ParameterDatabase  # noqa: PLC0415
        from tmol.pack import PackerPalette, PackerTask, pack_rotamers  # noqa: PLC0415
        from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler  # noqa: PLC0415
        from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database  # noqa: PLC0415

        pose = self.ctx.poses["complex"]
        if coordinates is not None:
            if tuple(coordinates.shape) != tuple(pose.coords.shape[1:]):
                want, got = tuple(pose.coords.shape[1:]), tuple(coordinates.shape)
                raise ValueError(f"coordinates must be {want}, got {got}")
            pose = attr.evolve(pose, coords=coordinates.detach().to(pose.device)[None].clone())
        dev = self.ctx.device
        n_blocks = pose.max_n_blocks
        task = PackerTask(pose, PackerPalette())
        for aa in sorted({sequence[p] for p in self.system.design_space.positions}):
            mask = torch.zeros(1, n_blocks, dtype=torch.bool, device=dev)
            for p in self.system.design_space.positions:
                if sequence[p] == aa:
                    mask[0, self.b_lo + p] = True
            task.restrict_absent_name3s([_ONE_TO_THREE[aa]], block_mask=mask)
        design = torch.zeros(1, n_blocks, dtype=torch.bool, device=dev)
        design[0, self._design_blocks] = True
        repack = self._repack_mask.to(dev)[None]
        task.restrict_to_repacking(block_mask=~design)  # non-design residues keep identity
        task.disable_packing_by_block_mask(~repack)  # and those outside the region stay fixed
        db = ParameterDatabase.get_default()
        task.add_conformer_sampler(create_dunbrack_sampler_from_database(db, dev))
        task.add_conformer_sampler(IncludeCurrentSampler())
        task.add_conformer_sampler(FixedAAChiSampler())  # Ala / Gly have no chi angles
        packed = pack_rotamers(pose, self.ctx.sfxn, task, seed=self.seed)

        self._check_identities(packed, sequence)
        target, binder, t_atoms, b_atoms = self._subposes(tmol, packed, dev)
        ctx = TMolContext(packed, target, binder, t_atoms, b_atoms, dev)
        return MutantPoses(sequence, packed, target, binder, ctx)

    def _check_identities(self, packed: Any, sequence: str) -> None:
        bts = packed.packed_block_types.active_block_types
        for pos, block in zip(self.system.design_space.positions, self._design_blocks, strict=True):
            got = bts[int(packed.block_type_ind[0, block])].name3
            want = _ONE_TO_THREE[sequence[pos]]
            if got != want:
                raise MutationError(
                    f"binder position {pos}: requested {sequence[pos]} ({want}) but the packer "
                    f"returned {got}"
                )

    def _subposes(
        self, tmol: Any, packed: Any, dev: torch.device
    ) -> tuple[Any, Any, Tensor, Tensor]:
        """Chain sub-poses of the packed complex (re-read from PDB text, coordinates synced)."""
        from tmol.io import pose_stack_to_pdb_string  # noqa: PLC0415

        text = pose_stack_to_pdb_string(packed)
        out = []
        for lo, hi, name in ((self.t_lo, self.t_hi, "target"), (self.b_lo, self.b_hi, "binder")):
            atoms = _residue_atom_indices(packed, lo, hi)
            sub = tmol.pose_stack_from_pdb(text, dev, residue_start=lo, residue_end=hi)
            out.append((_sync_subpose(packed, sub, atoms, lo, hi, name), atoms))
        (target, t_atoms), (binder, b_atoms) = out
        return target, binder, t_atoms, b_atoms

    def system_for(self, sequence: str) -> MolecularSystem:
        """A :class:`MolecularSystem` for ``sequence``'s packed pose: its coordinates, TMol
        context and atom index sets (binder/target/CA), cached. Atom indices differ between
        sequences because mutations change atom counts, so always use this system, never the
        native one, with a mutant's coordinates."""
        if sequence not in self._systems:
            self._systems[sequence] = self.system_from(self.build(sequence))
        return self._systems[sequence]

    def system_from(self, mutant: MutantPoses) -> MolecularSystem:
        """A :class:`MolecularSystem` for any built :class:`MutantPoses` (not cached), e.g. one
        packed on a perturbed backbone by :meth:`build_from`."""
        ctx = mutant.context
        extras = dict(self.system.extras)
        extras.update(
            tmol=ctx,
            binder_atoms=ctx.atoms["binder"],
            target_atoms=ctx.atoms["target"],
            binder_ca_atoms=_ca_atom_indices(mutant.complex, self.b_lo, self.b_hi),
            target_ca_atoms=_ca_atom_indices(mutant.complex, self.t_lo, self.t_hi),
        )
        binder_atoms: Any = extras["binder_atoms"]
        binder_mask = torch.zeros(mutant.coordinates.shape[0], 1)
        binder_mask[binder_atoms.cpu()] = 1.0
        return replace(
            self.system,
            reference_coordinates=mutant.coordinates,
            move_mask=binder_mask,
            extras=extras,
        )

    def write_pdb(self, sequence: str | MutantPoses, path: str | Path) -> Path:
        """Write the packed complex of ``sequence`` (a sequence string, built on the native
        backbone, or an already-built :class:`MutantPoses`) as a PDB with hydrogens."""
        from cinder.energy.tmol import write_pdb  # noqa: PLC0415

        if isinstance(sequence, MutantPoses):
            return write_pdb(self.system_from(sequence), sequence.coordinates, path)
        return write_pdb(self.system_for(sequence), self.build(sequence).coordinates, path)

    # ------------------------------------------------------------------------- scoring

    def score(self, sequence: str) -> dict[str, float]:
        """TMol-weighted term values of the packed single structure, one per entry of
        :data:`cinder.energy.tmol.TERM_GROUPS` (``interface``, ``repulsion``, ``strain``,
        ``hbond``, ``elec``, ``internal``), plus ``n_atoms``."""
        mutant = self.build(sequence)
        tokens = encode_sequence(sequence)[None]
        seq = SequenceBatch(tokens)
        x = mutant.coordinates[None]
        out: dict[str, float] = {}
        for name, model in group_energies(mutant.context, tokens[0]).items():
            out[name] = float(model.energy(self.system, x, seq)[0])
        out["n_atoms"] = float(x.shape[1])
        return out


__all__ = ["REPACK_MODES", "MutantPoses", "MutationError", "TMolSequenceBuilder"]
