import math

import pytest
import torch

from cinder.conformations import LangevinSampler
from cinder.energy.base import finite_difference_gradient
from cinder.energy.composite import CompositeEnergy, EnergyTerm
from cinder.energy.restraints import HarmonicRestraint
from cinder.energy.rigid_body import RigidBodyEnergy
from cinder.systems.rigid_body import (
    N_RIGID_DOF,
    RigidBodyParametrization,
    rotation_from_vector,
)
from cinder.types import SequenceBatch
from cinder.validation import Perturbation, rigid_body_perturb, rmsd, run_rigid_recovery
from cinder.validation.perturb import rotation_matrix

from ..conftest import make_toy_system


def _system(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    system = make_toy_system(dim=3)
    system.reference_coordinates = torch.randn(n, 3, generator=g, dtype=torch.float64) * 4
    return system


def _seq(n=1):
    return SequenceBatch(torch.zeros(n, 2, dtype=torch.long))


def test_rotation_from_vector_matches_axis_angle_reference_and_identity():
    omega = torch.tensor([[0.0, 0.0, 0.0], [0.3, -0.2, 0.5], [1e-5, 0.0, 0.0]], dtype=torch.float64)
    r = rotation_from_vector(omega)
    assert torch.allclose(r[0], torch.eye(3, dtype=torch.float64))
    axis, angle = omega[1] / omega[1].norm(), float(omega[1].norm())
    ref = rotation_matrix(axis, angle).double()
    assert torch.allclose(r[1], ref, atol=1e-10)
    assert torch.allclose(
        r @ r.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(3, 3, 3), atol=1e-12
    )
    assert torch.allclose(torch.linalg.det(r), torch.ones(3, dtype=torch.float64))


def test_rotation_gradient_finite_at_zero():
    omega = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    rotation_from_vector(omega).sum().backward()
    assert omega.grad is not None and torch.isfinite(omega.grad).all()


def test_to_cartesian_identity_translation_and_rigidity():
    system = _system()
    atoms = torch.arange(5)
    p = RigidBodyParametrization(system.reference_coordinates, atoms)
    x0 = p.to_cartesian(torch.zeros(1, N_RIGID_DOF, dtype=torch.float64))
    assert torch.allclose(x0[0], system.reference_coordinates)
    q = torch.zeros(1, N_RIGID_DOF, dtype=torch.float64)
    q[0, :3] = torch.tensor([1.0, -2.0, 0.5])
    shifted = p.to_cartesian(q)[0]
    assert torch.allclose(
        shifted[atoms] - system.reference_coordinates[atoms], q[0, :3].expand(5, 3)
    )
    assert torch.equal(shifted[5:], system.reference_coordinates[5:])  # others untouched
    q[0, 3:] = torch.tensor([3.0, -1.0, 2.0])
    moved = p.to_cartesian(q)[0]
    d0 = torch.cdist(system.reference_coordinates[atoms], system.reference_coordinates[atoms])
    assert torch.allclose(torch.cdist(moved[atoms], moved[atoms]), d0, atol=1e-10)  # rigid


def test_matches_cartesian_rigid_body_perturb_with_same_draws():
    """The jump transform equals the Cartesian ``rigid_body_perturb`` for the same random jump."""
    system = _system()
    atoms = torch.arange(8)
    p = RigidBodyParametrization(system.reference_coordinates, atoms)
    q = p.dofs(20.0, 1.5, torch.Generator().manual_seed(7), dtype=torch.float64)
    via_q = p.to_cartesian(q[None])[0]
    via_cart = rigid_body_perturb(
        system.reference_coordinates, atoms, 20.0, 1.5, torch.Generator().manual_seed(7)
    )
    assert torch.allclose(via_q, via_cart, atol=1e-9)


def test_rotation_scale_default_is_radius_of_gyration_and_floor():
    ref = torch.randn(30, 3, dtype=torch.float64) * 5
    p = RigidBodyParametrization(ref, torch.arange(30))
    rg = float((ref - ref.mean(0)).pow(2).sum(-1).mean().sqrt())
    assert p.rotation_scale == pytest.approx(rg)
    tiny = RigidBodyParametrization(ref * 0.01, torch.arange(30))
    assert tiny.rotation_scale == 1.0


def test_input_validation():
    ref = torch.zeros(4, 3)
    with pytest.raises(ValueError):
        RigidBodyParametrization(ref[:, :2], torch.arange(2))
    with pytest.raises(ValueError):
        RigidBodyParametrization(ref, torch.tensor([], dtype=torch.long))
    with pytest.raises(ValueError):
        RigidBodyParametrization(ref, torch.arange(2)).to_cartesian(torch.zeros(1, 5))


def test_energy_gradient_wrt_dofs_matches_finite_differences():
    system = _system()
    atoms = torch.arange(6)
    p = RigidBodyParametrization(system.reference_coordinates, atoms)
    # a restraint to a *different* target makes the gradient non-trivial
    target = system.reference_coordinates + 0.7
    ref_sys = _system()
    ref_sys.reference_coordinates = target
    base = HarmonicRestraint(k=2.0)
    wrapped = RigidBodyEnergy(
        _Fixed(base, ref_sys), p, system
    )  # energy measured against ``target``
    dof_system = p.dof_system(system)
    q = torch.randn(1, N_RIGID_DOF, dtype=torch.float64) * 0.3
    g = wrapped.gradient(dof_system, q, _seq())
    fd = finite_difference_gradient(lambda v: wrapped.energy(dof_system, v, _seq())[0], q, eps=1e-6)
    assert torch.allclose(g, fd, atol=1e-5)


class _Fixed:
    """Evaluate a base energy against a fixed alternative system (test helper)."""

    def __init__(self, base, system):
        self.base, self.system = base, system

    def energy(self, system, coordinates, sequence):
        return self.base.energy(self.system, coordinates, sequence)

    def gradient(self, system, coordinates, sequence):
        return self.base.gradient(self.system, coordinates, sequence)


def test_langevin_in_jump_space_recovers_reference_and_stays_rigid():
    system = _system(n=14)
    atoms = torch.arange(8)
    p = RigidBodyParametrization(system.reference_coordinates, atoms)
    energy = CompositeEnergy([EnergyTerm("well", HarmonicRestraint(k=1.0))])
    sampler = LangevinSampler.from_energy(
        RigidBodyEnergy(energy, p, system), strength=40.0, step_size=0.002, n_steps=800
    )
    res = run_rigid_recovery(
        system, energy, p, sampler, _seq(), Perturbation("rigid", 15.0, 2.0),
        torch.Generator().manual_seed(1), metric_atoms=atoms,
    )  # fmt: skip
    assert res.start_rmsd > 2.0
    assert res.final_rmsd < 0.25 * res.start_rmsd
    # internal geometry of the moving group is untouched by construction
    d_ref = torch.cdist(system.reference_coordinates[atoms], system.reference_coordinates[atoms])
    d_fin = torch.cdist(res.final_coordinates[atoms], res.final_coordinates[atoms]).double()
    assert torch.allclose(d_ref, d_fin, atol=1e-3)  # float32 sampler
    assert res.metrics["ca_aligned"][1] < 1e-4  # fold exactly preserved
    assert rmsd(res.final_coordinates, system.reference_coordinates, atoms) == pytest.approx(
        res.final_rmsd
    )


def test_run_rigid_recovery_rejects_non_rigid_perturbation():
    system = _system()
    p = RigidBodyParametrization(system.reference_coordinates, torch.arange(4))
    energy = CompositeEnergy([EnergyTerm("w", HarmonicRestraint())])
    sampler = LangevinSampler.from_energy(RigidBodyEnergy(energy, p, system))
    with pytest.raises(ValueError, match="rigid"):
        run_rigid_recovery(system, energy, p, sampler, _seq(), Perturbation("cartesian", sigma=1.0))


def test_dof_system_is_six_dimensional_and_keeps_original():
    system = _system()
    p = RigidBodyParametrization(system.reference_coordinates, torch.arange(4))
    ds = p.dof_system(system)
    assert ds.coord_shape == (N_RIGID_DOF,) and ds.move_mask is None
    assert ds.extras["cartesian_system"] is system
    assert math.isclose(float(ds.reference_coordinates.abs().sum()), 0.0)
