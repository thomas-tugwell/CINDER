import math

import pytest
import torch

from cinder.energy.base import finite_difference_gradient
from cinder.objectives.hbond_graph import (
    HBondGraphConfig,
    HBondGraphEnergy,
    PolarRoles,
    edge_strengths,
    graph_components,
    hard_hbond_report,
    shell_residues,
)
from cinder.systems.rigid_body import rotation_from_vector

from ..conftest import make_toy_system

CFG = HBondGraphConfig()
DT = torch.float64


def make_roles(donors, acceptors, residue, chain, n_atoms, node_atoms=None):
    """donors: [(heavy, H)], acceptors: [atom]; residue/chain: per-atom lists."""
    heavy = [i for i in range(n_atoms)]
    return PolarRoles(
        donor_heavy=torch.tensor([d for d, _ in donors]),
        donor_h=torch.tensor([h for _, h in donors]),
        acceptor=torch.tensor(acceptors),
        residue_of_atom=torch.tensor(residue),
        chain_of_atom=torch.tensor(chain),
        heavy_atoms=torch.tensor(heavy),
        node_atoms=None if node_atoms is None else torch.tensor(node_atoms),
    )


def pair_geometry(acceptor_xyz):
    """Donor D at origin, hydrogen 1.01 Å along +x, acceptor A at ``acceptor_xyz``; D and A on
    different chains."""
    coords = torch.tensor([[0.0, 0, 0], [1.01, 0, 0], list(acceptor_xyz)], dtype=DT)[None]
    roles = make_roles([(0, 1)], [2], residue=[0, 0, 5], chain=[0, 0, 1], n_atoms=3)
    return coords, roles


def test_linear_bond_is_strong_bent_and_far_are_weak():
    linear, roles = pair_geometry((2.9, 0, 0))  # H...A = 1.89 Å, angle 180
    bent, _ = pair_geometry((1.01, 2.0, 0))  # H...A = 2.0 Å but D-H...A = 90 degrees
    far, _ = pair_geometry((6.0, 0, 0))
    s = [float(edge_strengths(c, roles, CFG).sum()) for c in (linear, bent, far)]
    assert s[0] > 0.9
    assert s[1] < 0.05
    assert s[2] < 0.01
    expected = torch.sigmoid(torch.tensor((2.5 - 1.89) / 0.2)) * torch.sigmoid(
        torch.tensor((1.0 - 0.5) / 0.1)
    )
    assert s[0] == pytest.approx(float(expected), abs=1e-3)


def test_same_residue_and_adjacent_residue_pairs_are_masked():
    coords = torch.tensor([[0.0, 0, 0], [1.01, 0, 0], [2.9, 0, 0]], dtype=DT)[None]
    same = make_roles([(0, 1)], [2], residue=[3, 3, 3], chain=[0, 0, 0], n_atoms=3)
    adjacent = make_roles([(0, 1)], [2], residue=[3, 3, 4], chain=[0, 0, 0], n_atoms=3)
    distant = make_roles([(0, 1)], [2], residue=[3, 3, 7], chain=[0, 0, 0], n_atoms=3)
    s = [float(edge_strengths(coords, r, CFG).sum()) for r in (same, adjacent, distant)]
    assert s[0] == 0.0 and s[1] == 0.0 and s[2] > 0.9


def shell_system():
    """D1-H1 (chain 0) bonds across to A1 (chain 1): first shell. D2-H2 (chain 1) bonds, within
    chain 1, to A1: second shell. D3-H3...A3 is an isolated intra-chain pair far away: it must
    NOT count as second-shell support."""
    c = [
        [0.0, 0, 0], [1.01, 0, 0],  # 0 D1, 1 H1 (chain 0)
        [2.9, 0, 0],  # 2 A1 (chain 1, res 5)
        [2.9, 2.9, 0], [2.9, 1.89, 0],  # 3 D2, 4 H2 (chain 1, res 7); H2 points at A1
        [40.0, 0, 0], [40.0, 1.01, 0],  # 5 D3, 6 H3 (chain 1, res 20)
        [40.0, 2.9, 0],  # 7 A3 (chain 1, res 25)
    ]  # fmt: skip
    coords = torch.tensor(c, dtype=DT)[None]
    roles = make_roles(
        donors=[(0, 1), (3, 4), (5, 6)],
        acceptors=[2, 7],
        residue=[0, 0, 5, 7, 7, 20, 20, 25],
        chain=[0, 0, 1, 1, 1, 1, 1, 1],
        n_atoms=8,
    )
    return coords, roles


def test_first_and_second_shell_quantities():
    coords, roles = shell_system()
    a = edge_strengths(coords, roles, CFG)[0]  # [D=3, A=2]
    a_cross, a_support, a_isolated = float(a[0, 0]), float(a[1, 0]), float(a[2, 1])
    assert min(a_cross, a_support, a_isolated) > 0.9  # all three bonds are well formed
    comp = graph_components(coords, roles, CFG)
    assert float(comp.first_shell) == pytest.approx(a_cross, abs=1e-6)
    n1_a1 = 1 - math.exp(-a_cross)  # A1 bonds across the interface
    # the support bond touches first-shell A1; the isolated pair touches no first-shell node
    assert float(comp.second_shell) == pytest.approx(a_support * n1_a1, rel=1e-4)
    assert float(comp.n_first_nodes) == pytest.approx(2 * n1_a1, rel=1e-4)  # D1 and A1
    assert float(comp.n_second_nodes) > 0.0


def test_isolated_intra_pair_contributes_nothing_to_second_shell():
    coords, roles = shell_system()
    full = graph_components(coords, roles, CFG)
    # remove the supporting donor: only the isolated pair remains inside chain 1
    keep = make_roles(
        donors=[(0, 1), (5, 6)],
        acceptors=[2, 7],
        residue=[0, 0, 5, 7, 7, 20, 20, 25],
        chain=[0, 0, 1, 1, 1, 1, 1, 1],
        n_atoms=8,
    )
    without = graph_components(coords, keep, CFG)
    assert float(without.second_shell) == pytest.approx(0.0, abs=1e-9)
    assert float(full.second_shell) > 0.4
    assert float(without.first_shell) == pytest.approx(float(full.first_shell))


def test_cooperativity_needs_both_cross_and_intra_bonds_at_a_node():
    coords, roles = shell_system()
    comp = graph_components(coords, roles, CFG)
    assert float(comp.cooperativity) > 0.3  # A1 has a cross bond and an intra bond
    only_cross, roles_cross = pair_geometry((2.9, 0, 0))
    assert float(graph_components(only_cross, roles_cross, CFG).cooperativity) < 1e-3


def test_gradient_matches_finite_differences_and_is_rigid_invariant():
    coords, roles = shell_system()
    system = make_toy_system(dim=3)
    energy = HBondGraphEnergy(roles, CFG)
    seq = None
    x = coords + 0.05 * torch.randn(
        coords.shape, dtype=DT, generator=torch.Generator().manual_seed(0)
    )
    g = energy.gradient(system, x, seq)
    fd = finite_difference_gradient(lambda c: energy.energy(system, c, seq)[0], x, eps=1e-6)
    assert torch.allclose(g, fd, atol=1e-5)
    # rigid motion of the whole system leaves the score unchanged
    rot = rotation_from_vector(torch.tensor([[0.3, -0.5, 0.2]], dtype=DT))[0]
    moved = x @ rot.T + torch.tensor([3.0, -1.0, 2.0], dtype=DT)
    assert float(energy.energy(system, moved, seq)) == pytest.approx(
        float(energy.energy(system, x, seq))
    )


def test_unsatisfied_buried_polar_is_penalised_and_bonding_relieves_it():
    # a polar acceptor (atom 0) enclosed by 70 heavy atoms within 5 Å, no donor nearby
    g = torch.Generator().manual_seed(1)
    shell = torch.randn(70, 3, generator=g, dtype=DT)
    shell = shell / shell.norm(dim=1, keepdim=True) * 5.0
    coords = torch.cat([torch.zeros(1, 3, dtype=DT), shell])[None]
    n = coords.shape[1]
    # the graph needs a donor to exist; place one far away so it forms no bond
    far = torch.tensor([[200.0, 0, 0], [201.01, 0, 0]], dtype=DT)
    coords2 = torch.cat([coords, far[None]], dim=1)
    n2 = coords2.shape[1]
    roles = make_roles(
        [(n, n + 1)], [0], residue=list(range(n2)), chain=[1] * n + [0, 0], n_atoms=n2
    )
    comp = graph_components(coords2, roles, CFG)
    assert float(comp.unsatisfied) > 0.5  # buried, nothing bonded to it
    # now place the donor so it bonds linearly to the acceptor; the penalty must drop
    bonded = coords2.clone()
    bonded[0, n] = torch.tensor([2.9, 0, 0], dtype=DT)
    bonded[0, n + 1] = torch.tensor([1.89, 0, 0], dtype=DT)
    assert float(graph_components(bonded, roles, CFG).unsatisfied) < float(comp.unsatisfied) * 0.6


def test_hard_report_counts_bonds_and_shells_independently():
    coords, roles = shell_system()
    r = hard_hbond_report(coords[0], roles)
    assert r.n_cross == 1
    assert r.n_intra_support == 1  # D2...A1, which touches first-shell A1
    assert r.n_first_nodes == 2 and r.n_second_nodes == 1
    # the isolated D3-H3...A3 pair is a real bond but not a support bond
    assert len(r.bonds) == 3
    bent_coords, bent_roles = pair_geometry((1.01, 2.0, 0))
    assert hard_hbond_report(bent_coords[0], bent_roles).n_cross == 0  # angle too small


def test_roles_validation_and_node_restriction():
    with pytest.raises(ValueError):
        PolarRoles(
            donor_heavy=torch.tensor([0]),
            donor_h=torch.tensor([1, 2]),
            acceptor=torch.tensor([3]),
            residue_of_atom=torch.zeros(4, dtype=torch.long),
            chain_of_atom=torch.zeros(4, dtype=torch.long),
            heavy_atoms=torch.arange(4),
        )
    roles = make_roles(
        [(0, 1), (3, 4)], [2, 5], residue=[0, 0, 5, 7, 7, 9], chain=[0] * 6, n_atoms=6,
        node_atoms=[0, 1, 2],
    )  # fmt: skip
    assert roles.n_donors == 1 and roles.n_acceptors == 1  # outside atoms dropped


def test_shell_residues_split_first_and_second_shell_by_chain():
    coords, roles = shell_system()
    report = hard_hbond_report(coords[0], roles)
    # chain 1: A1 (residue 5) bonds across the interface; D2 (residue 7) only supports it
    first, second = shell_residues(report, roles, chain=1)
    assert first == [5] and second == [7]
    first0, second0 = shell_residues(report, roles, chain=0)
    assert first0 == [0] and second0 == []  # chain 0 has only the cross-bonding donor
    assert report.first_atoms == sorted(report.first_atoms) and len(report.second_atoms) == 1
