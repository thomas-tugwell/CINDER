"""H-bond graph on a real complex (barnase chain A / barstar chain D, 1BRS). Needs TMol."""

from pathlib import Path

import pytest
import torch

from cinder.config import config_from_dict
from cinder.energy.base import finite_difference_gradient
from cinder.energy.parametrized import ParametrizedEnergy
from cinder.energy.tmol import load_structure_system, tmol_available
from cinder.energy.tmol_design import (
    build_hbond_graph_energy,
    build_tmol_design_energy,
    propose_design_positions,
)
from cinder.energy.tmol_roles import polar_roles
from cinder.objectives.hbond_graph import graph_components, hard_hbond_report
from cinder.systems.rigid_body import RigidBodyParametrization
from cinder.types import SequenceBatch

pytestmark = [
    pytest.mark.tmol,
    pytest.mark.skipif(not tmol_available(), reason="tmol not installed"),
]

PDB = Path(__file__).resolve().parents[1] / "fixtures" / "barnase_barstar_AD.pdb"


@pytest.fixture(scope="module")
def cfg():
    return config_from_dict(
        {
            "system": {
                "kind": "structure",
                "structure_path": str(PDB),
                "target_chains": ["A"],
                "binder_chains": ["D"],
                "movable": "binder",
            },
            "energy": {"backend": "tmol", "hbond_graph_weight": 1.0},
        }
    )


@pytest.fixture(scope="module")
def system(cfg):
    return load_structure_system(cfg.system)


@pytest.fixture(scope="module")
def roles(system):
    return polar_roles(system.extras["tmol"], 8.0)


def _seq(system):
    return SequenceBatch(system.design_space.native_tokens()[None])


def test_roles_are_chemically_sensible(system, roles):
    x = system.reference_coordinates
    assert roles.n_donors > 50 and roles.n_acceptors > 50
    # every polar hydrogen sits ~1 Å from its heavy atom
    dh = (x[roles.donor_heavy] - x[roles.donor_h]).norm(dim=-1)
    assert float(dh.min()) > 0.9 and float(dh.max()) < 1.1
    # chain labels: target atoms 0, binder atoms 1
    assert set(roles.chain_of_atom.tolist()) == {0, 1}
    assert int(roles.chain_of_atom.sum()) == len(system.extras["binder_atoms"])
    # restricting to the interface region keeps fewer atoms than using everything
    everything = polar_roles(system.extras["tmol"], None)
    assert roles.n_donors < everything.n_donors


def test_soft_graph_agrees_with_independent_hard_counter(system, roles):
    x = system.reference_coordinates
    comp = graph_components(x[None], roles)
    hard = hard_hbond_report(x, roles)
    # barnase-barstar has on the order of 10-20 cross-chain hydrogen bonds
    assert 8 <= hard.n_cross <= 25
    assert 0.6 * hard.n_cross <= float(comp.first_shell) <= 1.4 * hard.n_cross
    assert hard.n_first_nodes > hard.n_second_nodes >= 0
    assert float(comp.second_shell) >= 0 and float(comp.unsatisfied) >= 0


def test_graph_term_enters_design_energy_with_its_weight(cfg, system):
    energy = build_tmol_design_energy(cfg.energy, system)
    out = energy.evaluate(system, system.reference_coordinates[None], _seq(system))
    assert "hbond_graph" in out.terms
    graph = build_hbond_graph_energy(system.extras["tmol"], cfg.energy.options)
    expected = graph.energy(system, system.reference_coordinates[None], _seq(system))
    assert float(out.terms["hbond_graph"]) == pytest.approx(float(expected), rel=1e-4)
    off = config_from_dict(
        {**cfg.to_dict(), "energy": {**cfg.energy.__dict__, "hbond_graph_weight": 0.0}}
    )
    assert (
        "hbond_graph"
        not in build_tmol_design_energy(off.energy, system)
        .evaluate(system, system.reference_coordinates[None], _seq(system))
        .terms
    )


def test_graph_gradient_matches_finite_differences(system, roles):
    from cinder.objectives.hbond_graph import HBondGraphEnergy

    energy = HBondGraphEnergy(roles)
    x = system.reference_coordinates[None].clone()
    g = energy.gradient(system, x, _seq(system))
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0
    top = g.abs().reshape(-1).topk(6).indices.tolist()
    fd = finite_difference_gradient(
        lambda c: float(energy.energy(system, c, _seq(system))[0]), x, eps=1e-2, indices=top
    )
    for i in top:
        assert float(g.reshape(-1)[i]) == pytest.approx(
            float(fd.reshape(-1)[i]), rel=0.05, abs=0.02
        )


def test_graph_gradient_chains_through_rigid_body_dofs(system, roles):
    from cinder.objectives.hbond_graph import HBondGraphEnergy

    param = RigidBodyParametrization(system.reference_coordinates, system.extras["binder_atoms"])
    wrapped = ParametrizedEnergy(HBondGraphEnergy(roles), param, system)
    dof_sys = param.dof_system(system)
    q = torch.tensor([[0.1, -0.05, 0.08, 0.1, -0.1, 0.05]])
    g = wrapped.gradient(dof_sys, q, _seq(system))
    fd = finite_difference_gradient(
        lambda v: float(wrapped.energy(dof_sys, v, _seq(system))[0]), q, eps=1e-2
    )
    assert torch.allclose(g, fd, rtol=0.1, atol=0.05), (g, fd)


def test_proposed_positions_recover_known_barstar_hotspots(system):
    """First-shell proposals include barstar's classic interface residues (binder indices of
    Y29, N33, D35, D39 and E76), found from the native H-bond geometry alone."""
    hotspots = {28, 32, 34, 38, 73}
    first = set(propose_design_positions(system, "first"))
    assert len(hotspots & first) >= 4
    both = propose_design_positions(system, "both")
    assert set(first) <= set(both) and both == sorted(both)
    assert all(0 <= p < system.design_space.length for p in both)
    with pytest.raises(ValueError, match="shells"):
        propose_design_positions(system, "third")


def test_contact_positions_include_the_hbonded_first_shell(system):
    """A 4.5 Å heavy-atom contact list is a superset of the cross-interface H-bond residues and
    shrinks as the cutoff does."""
    first = set(propose_design_positions(system, "first"))
    contact = propose_design_positions(system, "contact")
    assert first <= set(contact) and contact == sorted(contact)
    tight = propose_design_positions(system, "contact", contact_cutoff=3.5)
    assert set(tight) <= set(contact) and len(tight) < len(contact)
