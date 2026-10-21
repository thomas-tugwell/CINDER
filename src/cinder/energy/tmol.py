"""TMol adapter -- the *only* module in cinder that imports TMol.

Written against TMol 0.1.x (inspected, not assumed): ``tmol.pose_stack_from_pdb``,
``tmol.beta2016_score_function`` and ``ScoreFunction.render_whole_pose_scoring_module``, whose
call ``module(coords, sum_terms=False)`` returns *weighted* per-term scores ``[n_terms, n_poses]``
in the order of ``ScoreFunction.all_score_types()`` (verified in ``tests/integration``).

Coordinate convention: ``[N, n_atoms, 3]`` -- the atoms of the loaded complex's single pose in
TMol order (hydrogens included). Topology is therefore fixed to the **native sequence**; asking
for a different sequence raises :class:`SequenceMismatchError`. Evaluating mutated sequences
requires rebuilding the pose (see ``docs/tmol.md``, "Variable atom count").

Design-energy terms (assembled in :mod:`cinder.energy.tmol_design`):

* ``interface``  -- ``E_group(complex) - E_group(target) - E_group(binder)`` over the pairwise
  physics terms (LJ attraction, solvation, electrostatics, H-bonds, LK-ball). Intra-chain
  contributions cancel exactly because all three poses share coordinates.
* ``repulsion``  -- the same decomposition for ``fa_ljrep`` alone.
* ``strain``     -- covalent-geometry / torsion terms of the complex (not decomposed).
* ``internal``   -- every pairwise and geometry term of the binder alone: keeps a flexible binder
  folded (the interface decomposition removes intra-chain energy by construction).
* ``restraint``  -- harmonic tether to the reference pose (backend independent).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from cinder.energy.base import autograd_gradient
from cinder.systems.design_space import DesignSpace
from cinder.systems.specification import MolecularSystem
from cinder.types import SequenceBatch, decode_sequence
from cinder.utils.io import sha256_file

if TYPE_CHECKING:
    from cinder.config import SystemConfig

log = logging.getLogger("cinder.tmol")

INTERFACE_TERMS = (
    "fa_ljatr", "fa_lk", "fa_elec", "hbond",
    "lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl",
)  # fmt: skip
REPULSION_TERMS = ("fa_ljrep",)
HBOND_TERMS = ("hbond",)
ELEC_TERMS = ("fa_elec",)
STRAIN_TERMS = (
    "cart_lengths", "cart_angles", "cart_torsions", "cart_impropers", "cart_hxltorsions",
    "rama", "omega", "dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot",
)  # fmt: skip


SCOPES = ("interface", "complex", "binder")
INTERNAL_TERMS = INTERFACE_TERMS + REPULSION_TERMS + STRAIN_TERMS


@dataclass(frozen=True)
class TermGroup:
    """A named group of TMol score terms and the part of the system it is evaluated on.

    ``scope``: ``interface`` = ``E(complex) - E(target) - E(binder)`` (binder-target interactions
    only), ``complex`` = the whole complex, ``binder`` = the binder alone (its internal energy,
    which the interface decomposition removes by construction).
    """

    terms: tuple[str, ...]
    scope: str

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {self.scope!r}")


# The single definition of the named energy terms. ``interface`` already contains ``hbond`` and
# ``fa_elec``; the separate ``hbond`` / ``elec`` groups let them be weighted *in addition*
# (default weight 0, so the default design energy is unchanged).
TERM_GROUPS: dict[str, TermGroup] = {
    "interface": TermGroup(INTERFACE_TERMS, "interface"),
    "repulsion": TermGroup(REPULSION_TERMS, "interface"),
    "strain": TermGroup(STRAIN_TERMS, "complex"),
    "hbond": TermGroup(HBOND_TERMS, "interface"),
    "elec": TermGroup(ELEC_TERMS, "interface"),
    # the binder's own energy: keeps a flexible binder folded (needed for torsion/chi DOFs)
    "internal": TermGroup(INTERNAL_TERMS, "binder"),
}


def group_energies(context: TMolContext, native_tokens: Tensor) -> dict[str, TMolGroupEnergy]:
    """One :class:`TMolGroupEnergy` per entry of :data:`TERM_GROUPS` for ``context``."""
    return {
        name: TMolGroupEnergy(context, g.terms, native_tokens, g.scope)
        for name, g in TERM_GROUPS.items()
    }


_THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
    "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
    "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "HIS_D": "H", "CYD": "C",
}  # fmt: skip


class TMolUnavailableError(RuntimeError):
    """TMol is not installed."""


class SequenceMismatchError(ValueError):
    """The requested sequence differs from the pose the TMol adapter was built on."""


def tmol_available() -> bool:
    return util.find_spec("tmol") is not None


def _import_tmol() -> Any:
    if not tmol_available():
        raise TMolUnavailableError("TMol is not installed: `pip install 'cinder[tmol]'`")
    import tmol  # noqa: PLC0415

    return tmol


def tmol_device(requested: str) -> torch.device:
    """TMol runs on CPU or CUDA. ``auto`` picks CUDA when available, else CPU; an explicit
    ``cuda`` name is honoured when CUDA exists; anything else (including MPS) falls back to CPU."""
    wants_cuda = requested == "auto" or requested.startswith("cuda")
    if wants_cuda and torch.cuda.is_available():
        return torch.device("cuda" if requested == "auto" else requested)
    return torch.device("cpu")


class TMolContext:
    """Poses, score function and rendered-module cache for one complex.

    ``scores(which, coords)`` returns weighted per-term scores ``[n_terms, N]`` for the
    ``complex`` / ``target`` / ``binder`` pose; modules are cached per ``(which, N)``.
    """

    def __init__(
        self,
        complex_pose: Any,
        target_pose: Any,
        binder_pose: Any,
        target_atoms: Tensor,
        binder_atoms: Tensor,
        device: torch.device,
    ) -> None:
        tmol = _import_tmol()
        from tmol import ParameterDatabase  # noqa: PLC0415

        self.device = device
        self.poses = {"complex": complex_pose, "target": target_pose, "binder": binder_pose}
        self.atoms = {
            "complex": None,
            "target": target_atoms.to(device),
            "binder": binder_atoms.to(device),
        }
        # A fresh (non-memoised) score function: we never mutate TMol's shared default.
        self.sfxn = tmol.beta2016_score_function(device, param_db=ParameterDatabase.get_default())
        self.score_types = [st.name for st in self.sfxn.all_score_types()]
        self._modules: dict[tuple[str, int], Any] = {}

    def term_rows(self, names: Iterable[str]) -> list[int]:
        missing = [n for n in names if n not in self.score_types]
        if missing:
            raise KeyError(f"TMol score terms not available: {missing}")
        return [self.score_types.index(n) for n in names]

    def _module(self, which: str, n: int) -> Any:
        key = (which, n)
        if key not in self._modules:
            from tmol.pose import PoseStackBuilder  # noqa: PLC0415

            single = self.poses[which]
            stack = single if n == 1 else PoseStackBuilder.from_poses([single] * n, self.device)
            self._modules[key] = self.sfxn.render_whole_pose_scoring_module(stack)
        return self._modules[key]

    def scores(self, which: str, coords: Tensor) -> Tensor:
        """Weighted per-term scores ``[n_terms, N]``; ``coords`` is ``[N, n_atoms(which), 3]``."""
        if coords.device != self.device:
            raise ValueError(f"coordinates on {coords.device}, TMol context on {self.device}")
        return self._module(which, coords.shape[0])(coords.float(), sum_terms=False)

    def split(self, coords: Tensor, which: str) -> Tensor:
        idx = self.atoms[which]
        return coords if idx is None else coords[:, idx]


class TMolGroupEnergy:
    """Energy of a named group of TMol score terms on a given ``scope`` (see :class:`TermGroup`)."""

    def __init__(
        self,
        context: TMolContext,
        terms: Iterable[str],
        native_binder_tokens: Tensor,
        scope: str,
    ) -> None:
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        self.ctx = context
        self.terms = tuple(terms)
        self.rows = context.term_rows(self.terms)
        self.native = native_binder_tokens
        self.scope = scope

    def _check_sequence(self, sequence: SequenceBatch) -> None:
        native = self.native.to(sequence.tokens.device)
        bad = (sequence.tokens != native[None]).any(dim=1)
        if bad.any():
            first = decode_sequence(sequence.tokens[int(bad.nonzero()[0])])
            raise SequenceMismatchError(
                f"TMol adapter is built on the native sequence {decode_sequence(native)}; "
                f"got {first}. Mutated sequences need a pose rebuild (docs/tmol.md)."
            )

    def energy(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        self._check_sequence(sequence)
        if self.scope == "binder":
            sub = self.ctx.split(coordinates, "binder")
            return self.ctx.scores("binder", sub)[self.rows].sum(0)
        total = self.ctx.scores("complex", coordinates)[self.rows].sum(0)
        if self.scope == "interface":
            for which in ("target", "binder"):
                sub = self.ctx.split(coordinates, which)
                total = total - self.ctx.scores(which, sub)[self.rows].sum(0)
        return total

    def gradient(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        return autograd_gradient(lambda x: self.energy(system, x, sequence), coordinates)


# ------------------------------------------------------------------------------ loading


def _chain_residue_range(chain_labels: list[str], chains: list[str]) -> tuple[int, int]:
    idx = [i for i, c in enumerate(chain_labels) if c in chains]
    if not idx:
        raise ValueError(f"chains {chains} not found; available: {sorted(set(chain_labels))}")
    lo, hi = min(idx), max(idx) + 1
    if len(idx) != hi - lo:
        raise ValueError(
            f"chains {chains} are not a contiguous block of residues in the file; reorder the PDB"
        )
    return lo, hi


def _residue_atom_indices(pose: Any, lo: int, hi: int) -> Tensor:
    off = pose.block_coord_offset[0].long()
    n_at = pose.n_ats_per_block[0].long()
    start = int(off[lo])
    stop = int(off[hi - 1] + n_at[hi - 1])
    return torch.arange(start, stop)


def _ca_atom_indices(pose: Any, lo: int, hi: int) -> Tensor:
    """Indices (into the complex's atom axis) of the CA atom of residues ``lo..hi-1``."""
    bts = pose.packed_block_types.active_block_types
    off = pose.block_coord_offset[0].long()
    idx = []
    for i in range(lo, hi):
        bt = bts[int(pose.block_type_ind[0, i])]
        if "CA" not in bt.atom_to_idx:
            raise ValueError(f"residue {i} ({bt.name}) has no CA atom")
        idx.append(int(off[i]) + int(bt.atom_to_idx["CA"]))
    return torch.tensor(idx, dtype=torch.long)


def _one_letter(pose: Any, lo: int, hi: int) -> str:
    bts = pose.packed_block_types.active_block_types
    out = []
    for i in range(lo, hi):
        name = bts[int(pose.block_type_ind[0, i])].name
        out.append(_THREE_TO_ONE.get(name.split(":")[0], "X"))
    return "".join(out)


def _sync_subpose(pose: Any, sub: Any, atoms: Tensor, lo: int, hi: int, name: str) -> Any:
    """Make ``sub`` (a chain read on its own) share the complex's coordinates exactly.

    Heavy atoms missing from the file are rebuilt by TMol and can be placed differently when a
    chain is read alone vs. inside the complex, which would break the interface decomposition
    ``E(complex) - E(target) - E(binder)``. Topology (atom counts per residue) must agree; the
    coordinates are then overwritten with the complex's.
    """
    import attr  # noqa: PLC0415  (a TMol dependency)

    same_counts = torch.equal(sub.n_ats_per_block[0].long(), pose.n_ats_per_block[0, lo:hi].long())
    if sub.coords.shape[1] != len(atoms) or not same_counts:
        raise ValueError(f"{name} sub-pose topology does not match the complex")
    shared = pose.coords[0, atoms.to(pose.device)].detach().clone()[None]
    n_diff = int((sub.coords[0] - shared[0]).norm(dim=-1).gt(1e-3).sum())
    if n_diff:
        log.warning(
            "%s: %d rebuilt atoms differed between sub-pose and complex; synced", name, n_diff
        )
    return attr.evolve(sub, coords=shared)


def load_structure_system(sc: SystemConfig, device: str = "cpu") -> MolecularSystem:
    """Load a complex PDB into a :class:`MolecularSystem` with a TMol context attached."""
    tmol = _import_tmol()
    dev = tmol_device(device)
    path = Path(sc.structure_path or "")
    if not path.exists():
        raise FileNotFoundError(f"structure not found: {path}")
    pose = tmol.pose_stack_from_pdb(str(path), dev)
    if pose.n_poses != 1:
        raise ValueError("expected a single model")
    labels = [str(c) for c in pose.pdb_info.chain_labels[0][: pose.n_res_per_pose[0]]]
    t_lo, t_hi = _chain_residue_range(labels, sc.target_chains)
    b_lo, b_hi = _chain_residue_range(labels, sc.binder_chains)
    if not (t_hi <= b_lo or b_hi <= t_lo):
        raise ValueError("target and binder chains overlap")

    target = tmol.pose_stack_from_pdb(str(path), dev, residue_start=t_lo, residue_end=t_hi)
    binder = tmol.pose_stack_from_pdb(str(path), dev, residue_start=b_lo, residue_end=b_hi)
    t_atoms = _residue_atom_indices(pose, t_lo, t_hi)
    b_atoms = _residue_atom_indices(pose, b_lo, b_hi)
    target = _sync_subpose(pose, target, t_atoms, t_lo, t_hi, "target")
    binder = _sync_subpose(pose, binder, b_atoms, b_lo, b_hi, "binder")

    native = _one_letter(pose, b_lo, b_hi)
    if sc.native_sequence and sc.native_sequence != native:
        raise ValueError(f"configured native_sequence {sc.native_sequence} != structure {native}")
    positions = sc.design_positions if sc.design_positions is not None else list(range(len(native)))
    space = DesignSpace(len(native), native, positions, dict(sc.allowed_aa), sc.forbidden_aa)

    ref = pose.coords[0].detach().clone()
    move_mask = torch.ones(ref.shape[0], 1, device=dev)
    if sc.movable == "binder":
        move_mask.zero_()
        move_mask[b_atoms] = 1.0
    elif sc.movable != "all":
        raise ValueError(f"unknown system.movable {sc.movable!r}")

    ctx = TMolContext(pose, target, binder, t_atoms, b_atoms, dev)
    return MolecularSystem(
        name=sc.name,
        design_space=space,
        reference_coordinates=ref,
        target_chains=list(sc.target_chains),
        binder_chains=list(sc.binder_chains),
        structure_path=path,
        structure_sha256=sha256_file(path),
        move_mask=move_mask,
        extras={
            "tmol": ctx,
            "binder_atoms": b_atoms,
            "target_atoms": t_atoms,
            "binder_residue_range": (b_lo, b_hi),
            "target_residue_range": (t_lo, t_hi),
            "binder_ca_atoms": _ca_atom_indices(pose, b_lo, b_hi),
            "target_ca_atoms": _ca_atom_indices(pose, t_lo, t_hi),
        },
    )


def write_pdb(system: MolecularSystem, coordinates: Tensor, path: str | Path) -> Path:
    """Write ``coordinates`` (``[n_atoms, 3]``, TMol atom order) as a PDB of the complex.

    Atom order, residues and chains are those of the loaded complex, so files written from the
    reference, a perturbed start and a final pose superimpose directly. Hydrogens are included.
    """
    import attr  # noqa: PLC0415  (a TMol dependency)

    tmol = _import_tmol()
    ctx: TMolContext = system.extras["tmol"]  # type: ignore[assignment]
    if tuple(coordinates.shape) != system.coord_shape:
        raise ValueError(
            f"expected coordinates {system.coord_shape}, got {tuple(coordinates.shape)}"
        )
    pose = attr.evolve(ctx.poses["complex"], coords=coordinates.detach().to(ctx.device)[None])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmol.write_pose_stack_pdb(pose, str(out))
    return out


__all__ = [
    "SCOPES",
    "TERM_GROUPS",
    "TermGroup",
    "group_energies",
    "SequenceMismatchError",
    "TMolContext",
    "TMolGroupEnergy",
    "TMolUnavailableError",
    "load_structure_system",
    "tmol_available",
    "tmol_device",
    "write_pdb",
]
