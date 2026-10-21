import pytest
import torch

from cinder.conformations import LangevinSampler
from cinder.energy.composite import CompositeEnergy, EnergyTerm
from cinder.energy.restraints import HarmonicRestraint
from cinder.types import SequenceBatch
from cinder.validation import (
    Perturbation,
    cartesian_perturb,
    kabsch_rmsd,
    result_row,
    rigid_body_perturb,
    rmsd,
    run_recovery,
)

from ..conftest import make_toy_system


def test_rigid_perturb_preserves_internal_distances_and_leaves_others():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(10, 3)
    atoms = torch.arange(5)
    y = rigid_body_perturb(x, atoms, 20.0, 2.0, g)
    d0, d1 = torch.cdist(x[atoms], x[atoms]), torch.cdist(y[atoms], y[atoms])
    assert torch.allclose(d0, d1, atol=1e-5)
    assert torch.equal(y[5:], x[5:])
    assert rmsd(y, x, atoms) > 0.5


def test_rigid_perturb_pure_translation_has_exact_rmsd():
    x = torch.randn(6, 3)
    y = rigid_body_perturb(x, torch.arange(6), 0.0, 1.5, torch.Generator().manual_seed(1))
    assert rmsd(y, x) == pytest.approx(1.5, abs=1e-5)


def test_cartesian_perturb_scale_and_mask():
    x = torch.zeros(4000, 3)
    y = cartesian_perturb(x, torch.arange(2000), 0.3, torch.Generator().manual_seed(2))
    assert y[:2000].std().item() == pytest.approx(0.3, rel=0.05)
    assert torch.equal(y[2000:], x[2000:])


def test_recovery_toy_harmonic_returns_to_reference():
    """Under a harmonic well centred on the reference, guided dynamics must reduce RMSD."""
    system = make_toy_system(dim=3)
    system.reference_coordinates = torch.zeros(5, 3)
    energy = CompositeEnergy([EnergyTerm("well", HarmonicRestraint(k=1.0))])
    sampler = LangevinSampler.from_energy(energy, strength=50.0, step_size=0.005, n_steps=400)
    seq = SequenceBatch(torch.zeros(1, 2, dtype=torch.long))
    res = run_recovery(
        system, energy, sampler, seq, Perturbation("cartesian", sigma=2.0), torch.arange(5),
        torch.Generator().manual_seed(3),
    )  # fmt: skip
    assert res.start_rmsd > 2.0
    assert res.final_rmsd < 0.5 * res.start_rmsd
    assert res.improvement > 0 and set(res.start_terms) == {"total", "well"}


def test_unknown_perturbation_kind():
    with pytest.raises(ValueError):
        Perturbation("bogus").apply(torch.zeros(2, 3), torch.arange(2))


def test_result_carries_coordinates_and_flattens_to_row():
    system = make_toy_system(dim=3)
    system.reference_coordinates = torch.zeros(4, 3)
    energy = CompositeEnergy([EnergyTerm("well", HarmonicRestraint(k=1.0))])
    sampler = LangevinSampler.from_energy(energy, strength=10.0, step_size=0.01, n_steps=5)
    seq = SequenceBatch(torch.zeros(1, 2, dtype=torch.long))
    p = Perturbation("rigid", rotation_deg=2.0, translation=1.0)
    res = run_recovery(
        system, energy, sampler, seq, p, torch.arange(4), torch.Generator().manual_seed(0)
    )
    assert res.start_coordinates.shape == res.final_coordinates.shape == (4, 3)
    assert rmsd(res.start_coordinates, system.reference_coordinates) == pytest.approx(
        res.start_rmsd
    )
    row = result_row("toy", 1, res)
    assert row["perturbation"] == "rigid_r2_t1" and row["complex"] == "toy"
    assert {"start_well", "final_well", "start_total", "final_total"} <= set(row)


def test_perturbation_slugs_are_filesystem_safe():
    assert Perturbation("rigid", 2.0, 1.5).slug() == "rigid_r2_t1.5"
    assert Perturbation("cartesian", sigma=0.5).slug() == "cart_s0.5"


def test_kabsch_rmsd_is_zero_for_rigid_motion_and_detects_deformation():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 3, generator=g)
    moved = rigid_body_perturb(x, torch.arange(20), 40.0, 5.0, g)
    assert rmsd(moved, x) > 1.0  # frame-dependent RMSD sees the motion
    assert kabsch_rmsd(moved, x) == pytest.approx(0.0, abs=1e-5)  # fold unchanged
    bent = x.clone()
    bent[:10, 0] += 1.0
    assert kabsch_rmsd(bent, x) > 0.3


def test_kabsch_rejects_reflection_as_a_match():
    x = torch.randn(30, 3, generator=torch.Generator().manual_seed(1))
    mirrored = x * torch.tensor([-1.0, 1.0, 1.0])
    assert kabsch_rmsd(mirrored, x) > 0.5  # a mirror image is not the same fold


def test_metric_atoms_select_primary_rmsd():
    system = make_toy_system(dim=3)
    system.reference_coordinates = torch.zeros(6, 3)
    energy = CompositeEnergy([EnergyTerm("well", HarmonicRestraint(k=1.0))])
    sampler = LangevinSampler.from_energy(energy, strength=1.0, step_size=0.01, n_steps=0)
    seq = SequenceBatch(torch.zeros(1, 2, dtype=torch.long))
    res = run_recovery(
        system, energy, sampler, seq, Perturbation("rigid", 0.0, 1.0), torch.arange(6),
        torch.Generator().manual_seed(0), metric_atoms=torch.tensor([0, 2]),
    )  # fmt: skip
    assert res.start_rmsd == pytest.approx(1.0, abs=1e-5)  # pure shift of 1 Å
    assert res.metrics["ca_aligned"][0] == pytest.approx(0.0, abs=1e-5)
    assert result_row("toy", 0, res)["start_rmsd_ca_aligned"] == pytest.approx(0.0, abs=1e-5)
