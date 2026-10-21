"""Torsion-space (phi/psi) DOFs on a real complex (barnase A / barstar D, 1BRS). Needs TMol."""

from pathlib import Path

import pytest
import torch

from cinder.config import config_from_dict
from cinder.energy.base import finite_difference_gradient
from cinder.energy.parametrized import CoordinateParametrization, ParametrizedEnergy
from cinder.energy.tmol import load_structure_system, tmol_available
from cinder.energy.tmol_design import build_tmol_design_energy
from cinder.energy.tmol_torsion import CHI_TORSIONS, TMolTorsionParametrization
from cinder.systems.rigid_body import ComposedParametrization, RigidBodyParametrization
from cinder.types import SequenceBatch
from cinder.validation import Perturbation, rmsd, run_dof_recovery

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
            "energy": {"backend": "tmol", "restraint_weight": 0.0},
        }
    )


@pytest.fixture(scope="module")
def system(cfg):
    return load_structure_system(cfg.system)


@pytest.fixture(scope="module")
def param(system):
    return TMolTorsionParametrization(system.extras["tmol"], system.extras["binder_residue_range"])


def _seq(system):
    return SequenceBatch(system.design_space.native_tokens()[None])


def test_protocol_and_dof_count(param):
    assert isinstance(param, CoordinateParametrization)
    n_res = 87
    assert 2 * (n_res - 1) - 2 <= param.n_dofs <= 2 * n_res  # phi/psi minus the termini
    assert {name for _, name in param.selected} == {"phi", "psi"}


def test_round_trip_reproduces_the_reference_and_leaves_the_target_alone(system, param):
    x0 = param.to_cartesian(torch.zeros(1, param.n_dofs))[0]
    ref = system.reference_coordinates
    assert float((x0 - ref).norm(dim=-1).max()) < 1e-3  # kinematics reproduce the pose exactly
    q = param.dofs(1.0, torch.Generator().manual_seed(0))[None]
    moved = (param.to_cartesian(q)[0] - ref).norm(dim=-1)
    assert float(moved[system.extras["target_atoms"]].max()) < 1e-3  # target never moves
    assert float(moved[system.extras["binder_atoms"]].max()) > 0.05  # binder does


def test_torsion_moves_preserve_bond_lengths(system, param):
    """Only torsions change: covalent N-CA and CA-C lengths stay put (unlike Cartesian noise)."""
    pose = system.extras["tmol"].poses["complex"]
    bts = pose.packed_block_types.active_block_types
    off = pose.block_coord_offset[0].long()
    lo, hi = system.extras["binder_residue_range"]
    pairs = []
    for res in range(lo, hi):
        idx = bts[int(pose.block_type_ind[0, res])].atom_to_idx
        base = int(off[res])
        pairs += [(base + idx["N"], base + idx["CA"]), (base + idx["CA"], base + idx["C"])]
    pairs_t = torch.tensor(pairs)

    def lengths(x):
        return (x[pairs_t[:, 0]] - x[pairs_t[:, 1]]).norm(dim=-1)

    q = param.dofs(3.0, torch.Generator().manual_seed(1))[None]
    after = param.to_cartesian(q)[0]
    assert torch.allclose(lengths(after), lengths(system.reference_coordinates), atol=1e-3)


def test_independent_torsion_noise_compounds_along_the_chain(system, param):
    """Lever-arm effect: the same per-angle noise displaces far more atoms than its size."""
    g = torch.Generator().manual_seed(2)
    ref = system.reference_coordinates
    binder = system.extras["binder_atoms"]
    small = rmsd(param.to_cartesian(param.dofs(0.5, g)[None])[0], ref, binder)
    large = rmsd(param.to_cartesian(param.dofs(5.0, g)[None])[0], ref, binder)
    assert large > 3 * small


def test_gradient_wrt_torsions_matches_finite_differences(cfg, system, param):
    energy = build_tmol_design_energy(cfg.energy, system)
    wrapped = ParametrizedEnergy(energy, param, system)
    dof_system = param.dof_system(system)
    seq = _seq(system)
    q = param.dofs(0.5, torch.Generator().manual_seed(3))[None]
    g = wrapped.gradient(dof_system, q, seq)
    assert g.shape == (1, param.n_dofs) and torch.isfinite(g).all()
    top = g.abs().reshape(-1).topk(5).indices.tolist()
    fd = finite_difference_gradient(
        lambda v: float(wrapped.energy(dof_system, v, seq)[0]), q, eps=2e-2, indices=top
    )
    for i in top:
        assert float(g[0, i]) == pytest.approx(float(fd[0, i]), rel=0.1, abs=0.5), (i,)


def test_composed_torsion_and_rigid_body_dofs(cfg, system, param):
    rigid = RigidBodyParametrization(system.reference_coordinates, system.extras["binder_atoms"])
    comp = ComposedParametrization(param, rigid)
    assert comp.n_dofs == param.n_dofs + 6
    q = torch.zeros(1, comp.n_dofs)
    assert float((comp.to_cartesian(q)[0] - system.reference_coordinates).norm(dim=-1).max()) < 1e-3
    q[0, param.n_dofs : param.n_dofs + 3] = torch.tensor([1.0, 0.0, 0.0])  # a 1 Å jump
    moved = comp.to_cartesian(q)[0] - system.reference_coordinates
    assert float(moved[system.extras["binder_atoms"]].norm(dim=-1).mean()) == pytest.approx(
        1.0, abs=1e-2
    )


def _torsion_recovery(system, param, internal_weight):
    from cinder.conformations import LangevinSampler

    cfg = config_from_dict(
        {
            "system": {
                "kind": "structure",
                "structure_path": str(PDB),
                "target_chains": ["A"],
                "binder_chains": ["D"],
                "movable": "binder",
            },
            "energy": {
                "backend": "tmol",
                "restraint_weight": 0.0,
                "strain_weight": 0.1,
                "internal_weight": internal_weight,
            },
        }
    )
    energy = build_tmol_design_energy(cfg.energy, system)
    sampler = LangevinSampler.from_energy(
        ParametrizedEnergy(energy, param, system),
        strength=10.0,
        step_size=2e-5,
        n_steps=300,
        max_grad_norm=2000.0,
    )
    g = torch.Generator().manual_seed(4)
    q0 = param.dofs(1.0, g)
    return run_dof_recovery(
        system, energy, param, sampler, _seq(system), Perturbation("torsion", sigma_deg=1.0), q0,
        system.extras["binder_atoms"], g, metric_atoms=system.extras["binder_ca_atoms"],
    )  # fmt: skip


def test_internal_energy_is_what_keeps_a_torsion_flexible_binder_folded(system, param):
    """Regression for a real finding: the interface energy removes every intra-chain term, so
    guided torsion sampling alone unwinds the binder (here 2.0 -> 4.2 A CA RMSD, fold distorted
    1.0 -> 2.9 A). Adding the binder's own ('internal') energy repairs it (2.0 -> 0.9 A, fold
    1.0 -> 0.5 A)."""
    without = _torsion_recovery(system, param, internal_weight=0.0)
    with_internal = _torsion_recovery(system, param, internal_weight=1.0)
    assert without.final_rmsd > without.start_rmsd  # the helix unwinds
    assert with_internal.final_rmsd < 0.7 * with_internal.start_rmsd
    assert with_internal.metrics["ca_aligned"][1] < with_internal.metrics["ca_aligned"][0]
    assert with_internal.metrics["ca_aligned"][1] < 0.5 * without.metrics["ca_aligned"][1]


def test_chi_torsions_are_selectable(system):
    chi = TMolTorsionParametrization(
        system.extras["tmol"], system.extras["binder_residue_range"], torsion_names=CHI_TORSIONS
    )
    assert chi.n_dofs > 20 and {n for _, n in chi.selected} <= set(CHI_TORSIONS)
    ref = system.reference_coordinates
    moved = chi.to_cartesian(chi.dofs(10.0, torch.Generator().manual_seed(5))[None])[0] - ref
    # side-chain rotations move side-chain atoms but leave every CA exactly where it was
    ca = system.extras["binder_ca_atoms"]
    assert float(moved[ca].norm(dim=-1).max()) < 1e-3
    assert float(moved.norm(dim=-1).max()) > 0.1
    with pytest.raises(ValueError, match="no torsions"):
        TMolTorsionParametrization(
            system.extras["tmol"], system.extras["binder_residue_range"], torsion_names=("nope",)
        )
