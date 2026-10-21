"""Differentiable hydrogen-bond graph with first- and second-shell interactions.

Atoms that can donate or accept a hydrogen bond are the *nodes*. A donor hydrogen ``H`` (on heavy
atom ``D``) and an acceptor ``A`` form an *edge* whose strength is a smooth function of geometry:

    a(D-H...A) = sigmoid((r_c - |H-A|) / w_r) * sigmoid((-cos(D-H...A) - cos_c) / w_a)

(close, and close to linear), so every quantity below is differentiable in the coordinates.

Shells (soft, via the adjacency, no hard cutoffs). With ``X`` the cross-chain mask:

* first shell:   ``n1_v = 1 - exp(-sum_u S_vu X_vu)``       (v bonds across the interface)
* second shell:  ``n2_v = (1 - n1_v)(1 - exp(-sum_u S_vu (1-X_vu) n1_u))``
                 (v bonds, within its own chain, to a first-shell node)

Objectives (higher is better; ``HBondGraphEnergy`` returns the negative as an energy):

* first-shell strength    ``sum a_da X_da``                    (bonds across the interface)
* second-shell support    ``sum a_da (1-X_da) s_da``,  ``s = 1-(1-n1_d)(1-n1_a)``  (buttressing
  bonds inside a chain that touch a first-shell node; invisible to interface decompositions
  because intra-chain terms cancel in ``E(complex) - E(target) - E(binder)``)
* cooperativity           ``sum_v c_v i_v / (c_v + i_v)``  (smooth min of the cross-chain and
  intra-chain bonding at the same node: chains target-interface-support)
* buried unsatisfied      ``sum_v burial_v exp(-deg_v / tau)``  (penalty)

Nothing here imports a molecular backend: roles come from :class:`PolarRoles`, built elsewhere
(``cinder.energy.tmol_roles`` for TMol). :func:`hard_hbond_report` is an *independent*
hard-threshold counter (heavy-atom D...A distance based) for validation, so an optimised graph
score is never its own check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from cinder.energy.base import autograd_gradient
from cinder.systems.specification import MolecularSystem
from cinder.types import SequenceBatch


@dataclass
class PolarRoles:
    """Indices (into the atom axis of the coordinate tensor) of hydrogen-bonding atoms.

    ``donor_heavy[i]`` / ``donor_h[i]`` are the heavy atom and one of its polar hydrogens (one
    entry per hydrogen). ``acceptor`` lists acceptor atoms. ``residue_of_atom`` / ``chain_of_atom``
    ``[n_atoms]`` label every atom; ``heavy_atoms`` are all non-hydrogen atoms (for burial).
    ``node_atoms`` optionally restricts which atoms are graph nodes (e.g. an interface region).
    """

    donor_heavy: Tensor
    donor_h: Tensor
    acceptor: Tensor
    residue_of_atom: Tensor
    chain_of_atom: Tensor
    heavy_atoms: Tensor
    node_atoms: Tensor | None = None

    def __post_init__(self) -> None:
        if self.donor_heavy.shape != self.donor_h.shape:
            raise ValueError("donor_heavy and donor_h must have the same length")
        if self.residue_of_atom.shape != self.chain_of_atom.shape:
            raise ValueError("residue_of_atom and chain_of_atom must have the same length")
        self._restrict()

    def _restrict(self) -> None:
        """Drop donors/acceptors outside ``node_atoms`` (if given)."""
        if self.node_atoms is None:
            return
        allowed = torch.zeros(self.residue_of_atom.numel(), dtype=torch.bool)
        allowed[self.node_atoms] = True
        keep_d = allowed[self.donor_heavy]
        self.donor_heavy, self.donor_h = self.donor_heavy[keep_d], self.donor_h[keep_d]
        self.acceptor = self.acceptor[allowed[self.acceptor]]

    @property
    def n_donors(self) -> int:
        return int(self.donor_heavy.numel())

    @property
    def n_acceptors(self) -> int:
        return int(self.acceptor.numel())

    def pair_mask(self) -> Tensor:
        """``[D, A]`` bool: pairs allowed to bond (not the same atom, residue or a neighbour)."""
        res_d = self.residue_of_atom[self.donor_heavy][:, None]
        res_a = self.residue_of_atom[self.acceptor][None, :]
        chain_d = self.chain_of_atom[self.donor_heavy][:, None]
        chain_a = self.chain_of_atom[self.acceptor][None, :]
        same_atom = self.donor_heavy[:, None] == self.acceptor[None, :]
        near = (chain_d == chain_a) & ((res_d - res_a).abs() <= 1)
        return ~(same_atom | near)

    def nodes(self) -> tuple[Tensor, Tensor, Tensor]:
        """``(node_atoms, node_of_donor, node_of_acceptor)``: unique atoms acting as donor
        heavy atom or acceptor, and the node index of every donor / acceptor entry."""
        atoms = torch.unique(torch.cat([self.donor_heavy, self.acceptor]))
        lookup = torch.full((self.residue_of_atom.numel(),), -1, dtype=torch.long)
        lookup[atoms] = torch.arange(atoms.numel())
        return atoms, lookup[self.donor_heavy], lookup[self.acceptor]


@dataclass(frozen=True)
class HBondGraphConfig:
    """Smoothing parameters (Å, dimensionless) and objective weights."""

    r_cut: float = 2.5  # H...A distance at which an edge is half formed
    r_width: float = 0.2
    cos_cut: float = 0.5  # -cos(D-H...A) > 0.5  <=>  angle > 120 degrees
    cos_width: float = 0.1
    # Burial = heavy-atom neighbours within ``burial_radius`` of a polar atom. In a folded protein
    # (barnase/barstar, 6 A) polar atoms have ~25-51 neighbours (10th-90th percentile, median 42),
    # so the midpoint sits near the 75th percentile.
    burial_radius: float = 6.0
    burial_width: float = 0.5
    burial_midpoint: float = 45.0  # neighbour count at which a node is half buried
    burial_count_width: float = 4.0
    unsat_tau: float = 0.5  # bonding strength that "satisfies" a buried polar atom
    w_first: float = 1.0
    w_second: float = 0.5
    w_coop: float = 0.25
    w_unsat: float = 0.5


DEFAULT_GRAPH_CONFIG = HBondGraphConfig()


@dataclass
class GraphComponents:
    """Per-sample ``[N]`` graph quantities (all differentiable)."""

    first_shell: Tensor
    second_shell: Tensor
    cooperativity: Tensor
    unsatisfied: Tensor
    n_first_nodes: Tensor
    n_second_nodes: Tensor

    def score(self, cfg: HBondGraphConfig) -> Tensor:
        """Higher is better."""
        return (
            cfg.w_first * self.first_shell
            + cfg.w_second * self.second_shell
            + cfg.w_coop * self.cooperativity
            - cfg.w_unsat * self.unsatisfied
        )


def edge_strengths(coords: Tensor, roles: PolarRoles, cfg: HBondGraphConfig) -> Tensor:
    """Soft bond strength of every allowed donor-hydrogen/acceptor pair: ``[N, D, A]``."""
    xd, xh = coords[:, roles.donor_heavy], coords[:, roles.donor_h]
    xa = coords[:, roles.acceptor]
    v_dh = xd - xh  # [N, D, 3]
    v_ha = xa[:, None, :, :] - xh[:, :, None, :]  # [N, D, A, 3]
    dist = v_ha.norm(dim=-1)  # H...A
    cos = (v_dh[:, :, None, :] * v_ha).sum(-1) / (
        v_dh.norm(dim=-1)[:, :, None] * dist.clamp_min(1e-8)
    )
    s_dist = torch.sigmoid((cfg.r_cut - dist) / cfg.r_width)
    s_ang = torch.sigmoid((-cos - cfg.cos_cut) / cfg.cos_width)
    return s_dist * s_ang * roles.pair_mask().to(coords.device)


def graph_components(
    coords: Tensor, roles: PolarRoles, cfg: HBondGraphConfig = DEFAULT_GRAPH_CONFIG
) -> GraphComponents:
    """All shell / cooperativity / unsatisfied quantities for ``coords`` ``[N, n_atoms, 3]``."""
    n = coords.shape[0]
    dev = coords.device
    a = edge_strengths(coords, roles, cfg)  # [N, D, A]
    node_atoms, nd, na = roles.nodes()
    n_nodes = node_atoms.numel()
    cross = (
        roles.chain_of_atom[roles.donor_heavy][:, None]
        != roles.chain_of_atom[roles.acceptor][None, :]
    ).to(dev, a.dtype)
    intra = 1.0 - cross
    nd, na = nd.to(dev), na.to(dev)

    def to_nodes(per_donor: Tensor, per_acceptor: Tensor) -> Tensor:
        out = torch.zeros(n, n_nodes, dtype=a.dtype, device=dev)
        out = out.index_add(1, nd, per_donor)
        return out.index_add(1, na, per_acceptor)

    cross_node = to_nodes((a * cross).sum(2), (a * cross).sum(1))  # cross-chain strength at node
    n1 = 1 - torch.exp(-cross_node)  # [N, V]
    n1_d, n1_a = n1[:, nd], n1[:, na]  # first-shell weight of each donor / acceptor entry
    # strength of intra-chain bonds to first-shell partners, seen from each node
    to_first_d = (a * intra * n1_a[:, None, :]).sum(2)  # donor side sees acceptor's n1
    to_first_a = (a * intra * n1_d[:, :, None]).sum(1)
    n2 = (1 - n1) * (1 - torch.exp(-to_nodes(to_first_d, to_first_a)))
    intra_node = to_nodes((a * intra).sum(2), (a * intra).sum(1))

    support = 1 - (1 - n1_d[:, :, None]) * (1 - n1_a[:, None, :])
    first = (a * cross).sum((1, 2))
    second = (a * intra * support).sum((1, 2))
    coop = (cross_node * intra_node / (cross_node + intra_node + 1e-6)).sum(1)

    heavy = coords[:, roles.heavy_atoms]
    pos = coords[:, node_atoms.to(dev)]
    d = torch.cdist(pos, heavy)  # [N, V, H]
    is_self = node_atoms.to(dev)[:, None] == roles.heavy_atoms.to(dev)[None, :]
    counts = (torch.sigmoid((cfg.burial_radius - d) / cfg.burial_width) * ~is_self).sum(-1)
    burial = torch.sigmoid((counts - cfg.burial_midpoint) / cfg.burial_count_width)
    unsat = (burial * torch.exp(-(cross_node + intra_node) / cfg.unsat_tau)).sum(1)
    return GraphComponents(first, second, coop, unsat, n1.sum(1), n2.sum(1))


class HBondGraphEnergy:
    """:class:`~cinder.types.EnergyModel` returning ``-score`` (lower is better)."""

    def __init__(self, roles: PolarRoles, cfg: HBondGraphConfig = DEFAULT_GRAPH_CONFIG) -> None:
        self.roles = roles
        self.cfg = cfg

    def components(self, coordinates: Tensor) -> GraphComponents:
        return graph_components(coordinates, self.roles, self.cfg)

    def energy(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        return -self.components(coordinates).score(self.cfg)

    def gradient(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        return autograd_gradient(lambda x: self.energy(system, x, sequence), coordinates)


@dataclass
class HardReport:
    """Hard-threshold hydrogen-bond census for one structure (independent of the soft graph)."""

    n_cross: int
    n_intra_support: int  # intra-chain bonds touching a first-shell node
    n_first_nodes: int
    n_second_nodes: int
    n_buried_unsatisfied: int
    bonds: list[tuple[int, int, int]] = field(default_factory=list)  # (donor, H, acceptor)
    first_atoms: list[int] = field(default_factory=list)  # nodes bonding across the interface
    second_atoms: list[int] = field(default_factory=list)  # nodes bonding to a first-shell node


def hard_hbond_report(
    coords: Tensor,
    roles: PolarRoles,
    max_da: float = 3.5,
    min_angle_deg: float = 110.0,
    burial_radius: float = 6.0,
    burial_neighbors: int = 45,
) -> HardReport:
    """Count hydrogen bonds with hard thresholds on one structure ``coords`` ``[n_atoms, 3]``.

    A bond needs heavy-atom ``D...A <= max_da`` and ``D-H...A >= min_angle_deg``. Uses the heavy
    atom distance (not the hydrogen position) as the primary criterion so it does not inherit
    the soft graph's reliance on built hydrogens.
    """
    xd, xh, xa = coords[roles.donor_heavy], coords[roles.donor_h], coords[roles.acceptor]
    d_da = torch.cdist(xd, xa)
    v1 = xd - xh
    v2 = xa[None, :, :] - xh[:, None, :]
    cos = (v1[:, None, :] * v2).sum(-1) / (
        v1.norm(dim=-1)[:, None] * v2.norm(dim=-1).clamp_min(1e-8)
    )
    ang = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))  # angle D-H...A (180 = linear)
    ok = (d_da <= max_da) & (ang >= min_angle_deg) & roles.pair_mask().to(coords.device)
    d_idx, a_idx = ok.nonzero(as_tuple=True)
    chain_d = roles.chain_of_atom[roles.donor_heavy][d_idx]
    chain_a = roles.chain_of_atom[roles.acceptor][a_idx]
    cross = chain_d != chain_a
    atom_d, atom_a = roles.donor_heavy[d_idx], roles.acceptor[a_idx]
    first_atoms = set(atom_d[cross].tolist()) | set(atom_a[cross].tolist())
    intra_pairs = [
        (int(x), int(y))
        for x, y, c in zip(atom_d.tolist(), atom_a.tolist(), cross.tolist(), strict=True)
        if not c
    ]
    support = [(x, y) for x, y in intra_pairs if x in first_atoms or y in first_atoms]
    second_atoms = ({x for x, y in support} | {y for x, y in support}) - first_atoms
    bonded = set(atom_d.tolist()) | set(atom_a.tolist())
    node_atoms, _, _ = roles.nodes()
    heavy = coords[roles.heavy_atoms]
    unsat = 0
    for atom in node_atoms.tolist():
        if atom in bonded:
            continue
        d = (heavy - coords[atom]).norm(dim=-1)
        if int((d < burial_radius).sum()) - 1 >= burial_neighbors:
            unsat += 1
    return HardReport(
        n_cross=int(cross.sum()),
        n_intra_support=len(support),
        n_first_nodes=len(first_atoms),
        n_second_nodes=len(second_atoms),
        n_buried_unsatisfied=unsat,
        first_atoms=sorted(first_atoms),
        second_atoms=sorted(second_atoms),
        bonds=[
            (int(roles.donor_heavy[i]), int(roles.donor_h[i]), int(roles.acceptor[j]))
            for i, j in zip(d_idx.tolist(), a_idx.tolist(), strict=True)
        ],
    )


def shell_residues(
    report: HardReport, roles: PolarRoles, chain: int
) -> tuple[list[int], list[int]]:
    """Residue indices on ``chain`` that contribute first-shell and second-shell atoms (a residue
    appears in the second list only if it is not already first-shell)."""

    def residues(atoms: list[int]) -> set[int]:
        return {
            int(roles.residue_of_atom[a]) for a in atoms if int(roles.chain_of_atom[a]) == chain
        }

    first = residues(report.first_atoms)
    second = residues(report.second_atoms) - first
    return sorted(first), sorted(second)
