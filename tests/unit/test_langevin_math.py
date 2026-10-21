"""Mathematical validation of the sampler against closed-form distributions."""

import pytest
import torch

from cinder.conformations import EnergyGuidance, LangevinSampler
from cinder.energy.base import finite_difference_gradient
from cinder.energy.toy import DoubleWellEnergy, GaussianMixtureEnergy, HarmonicEnergy
from cinder.types import SequenceBatch

from ..conftest import make_toy_system


def _seq(n: int, length: int = 2) -> SequenceBatch:
    return SequenceBatch(torch.zeros(n, length, dtype=torch.long))


@pytest.mark.parametrize("lam,k", [(1.0, 1.0), (2.0, 1.0), (1.0, 4.0)])
def test_harmonic_moments(lam, k):
    # p ∝ exp(-lam k x^2 / 2)  =>  N(0, 1/(lam k))
    system = make_toy_system(dim=1)
    sampler = LangevinSampler.from_energy(
        HarmonicEnergy(k=k), strength=lam, step_size=0.01 / (lam * k), n_steps=1500
    )
    g = torch.Generator().manual_seed(0)
    x = sampler.sample(system, _seq(1), 4000, g).coordinates.reshape(-1)
    var = 1.0 / (lam * k)
    assert abs(x.mean()) < 4 * (var / 4000) ** 0.5
    assert abs(x.var() / var - 1) < 0.12  # ULA bias O(eta) plus MC error


def test_harmonic_shifted_mean():
    system = make_toy_system(dim=2)
    sampler = LangevinSampler.from_energy(
        HarmonicEnergy(1.0, center=torch.tensor([3.0, -2.0])), step_size=0.02, n_steps=800
    )
    x = sampler.sample(system, _seq(1), 2000, torch.Generator().manual_seed(1)).coordinates[0]
    assert torch.allclose(x.mean(0), torch.tensor([3.0, -2.0]), atol=0.15)


def test_double_well_populates_both_wells():
    system = make_toy_system(dim=2)
    sampler = LangevinSampler.from_energy(
        DoubleWellEnergy(a=1.0, b=1.0), strength=1.0, step_size=0.01, n_steps=3000, init_std=1.0
    )
    x = sampler.sample(system, _seq(1), 1500, torch.Generator().manual_seed(2)).coordinates[0]
    frac_right = (x[:, 0] > 0).float().mean()
    assert 0.42 < frac_right < 0.58  # symmetric wells
    assert abs(x[:, 0].abs().mean() - 1.0) < 0.25  # concentrated near the minima
    assert abs(x[:, 1].var() - 1.0) < 0.15  # independent harmonic direction


def test_gaussian_mixture_mode_weights():
    means = torch.tensor([[-3.0, 0.0], [3.0, 0.0]])
    energy = GaussianMixtureEnergy(means, sigma=1.0, weights=torch.tensor([0.5, 0.5]))
    system = make_toy_system(dim=2)
    sampler = LangevinSampler.from_energy(energy, step_size=0.05, n_steps=2000, init_std=3.0)
    x = sampler.sample(system, _seq(1), 1500, torch.Generator().manual_seed(3)).coordinates[0]
    assert abs((x[:, 0] > 0).float().mean() - 0.5) < 0.08
    assert abs(x[:, 1].var() - 1.0) < 0.15


def test_energy_guided_product_of_gaussians():
    """p_base = N(0, 1), U = (x-a)^2/2, lam: guided precision 1+lam, mean lam*a/(1+lam)."""

    class StdNormalScore:
        def score(self, system, coordinates, sequence, t=None):
            return -coordinates

    a, lam = 2.0, 3.0
    system = make_toy_system(dim=1)
    guidance = EnergyGuidance(
        HarmonicEnergy(1.0, center=a), strength=lam, base_model=StdNormalScore()
    )
    sampler = LangevinSampler(guidance, step_size=0.005, n_steps=1500)
    x = sampler.sample(system, _seq(1), 4000, torch.Generator().manual_seed(4)).coordinates.reshape(
        -1
    )
    assert abs(x.mean() - lam * a / (1 + lam)) < 0.05
    assert abs(x.var() * (1 + lam) - 1) < 0.1


def test_autograd_gradient_matches_finite_difference():
    system = make_toy_system(dim=3)
    for energy in (DoubleWellEnergy(), GaussianMixtureEnergy(torch.randn(3, 3))):
        x = torch.randn(1, 3, dtype=torch.float64)
        seq = _seq(1)
        g = energy.gradient(system, x, seq)
        fd = finite_difference_gradient(
            lambda c, e=energy, s=seq: e.energy(system, c, s)[0], x, eps=1e-5
        )
        assert torch.allclose(g, fd, atol=1e-4)


def test_log_prob_unsupported():
    from cinder.types import UnsupportedOperation

    sampler = LangevinSampler.from_energy(HarmonicEnergy())
    with pytest.raises(UnsupportedOperation):
        sampler.log_prob(make_toy_system(), _seq(1), torch.zeros(1, 2))


def test_initial_coordinates_override():
    system = make_toy_system(dim=2)
    sampler = LangevinSampler.from_energy(HarmonicEnergy(1.0), step_size=0.01, n_steps=0)
    x0 = torch.tensor([5.0, -5.0])
    out = sampler.sample(system, _seq(2), 3, initial_coordinates=x0)
    assert out.coordinates.shape == (2, 3, 2) and torch.equal(out.coordinates[0, 0], x0)


def test_noise_scale_zero_is_deterministic_gradient_descent_and_default_is_unchanged():
    system = make_toy_system(dim=2)
    energy = HarmonicEnergy(1.0, center=torch.tensor([3.0, -2.0]))
    quench = LangevinSampler.from_energy(
        energy, strength=1.0, step_size=0.05, n_steps=400, noise_scale=0.0
    )
    start = torch.tensor([[-4.0, 5.0]])
    a = quench.sample(
        system, _seq(1), 1, torch.Generator().manual_seed(0), initial_coordinates=start
    )
    b = quench.sample(
        system, _seq(1), 1, torch.Generator().manual_seed(9), initial_coordinates=start
    )
    assert torch.equal(a.coordinates, b.coordinates)  # no randomness left
    assert torch.allclose(a.coordinates[0, 0], torch.tensor([3.0, -2.0]), atol=1e-3)  # the minimum
    default = LangevinSampler.from_energy(energy, step_size=0.05, n_steps=5)
    assert default.noise_scale == 1.0
    with pytest.raises(ValueError, match="noise_scale"):
        LangevinSampler.from_energy(energy, noise_scale=-0.1)
