import pytest
import torch

from cinder.conformations.diffusion import DiffusionConformationModel
from cinder.conformations.flow_matching import FlowMatchingConformationModel
from cinder.conformations.guidance import EnergyGuidance
from cinder.conformations.networks import ConditionalMLP
from cinder.conformations.normalizing_flow import sequence_context
from cinder.conformations.training import fit
from cinder.energy.toy import HarmonicEnergy
from cinder.types import N_AA, ConformationModel, SequenceBatch, UnsupportedOperation

from ..conftest import make_toy_system

SIGMA = 0.5
MEANS = {0: torch.tensor([2.0, 0.0]), 1: torch.tensor([-2.0, 1.0])}  # sequence token -> data mean


def _data(n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randint(0, 2, (n,), generator=g)
    mu = torch.stack([MEANS[int(t)] for t in tok])
    x = mu + SIGMA * torch.randn(n, 2, generator=g)
    return x, SequenceBatch(tok[:, None])


def _seq(token):
    return SequenceBatch(torch.tensor([[token]]))


def _trained(kind, guidance=None, steps=5000):
    torch.manual_seed(0)
    net = ConditionalMLP(dim=2, context_dim=N_AA, hidden=64, depth=3)
    model = (
        DiffusionConformationModel(net, n_steps=100, guidance=guidance)
        if kind == "diffusion"
        else FlowMatchingConformationModel(net, n_steps=50, guidance=guidance)
    )
    x, seqs = _data()
    history = fit(model, x, seqs, steps=steps, batch_size=512, lr=2e-3)
    return model, history


@pytest.fixture(scope="module", params=["diffusion", "flow_matching"])
def trained(request):
    model, history = _trained(request.param)
    return request.param, model, history


def test_satisfies_protocol_and_has_no_density(trained):
    _, model, _ = trained
    sys_ = make_toy_system(n_positions=1)
    assert isinstance(model, ConformationModel)
    with pytest.raises(UnsupportedOperation):
        model.log_prob(sys_, _seq(0), torch.zeros(1, 2))


def test_training_reduces_the_loss(trained):
    _, _, history = trained
    assert sum(history[-100:]) / 100 < 0.8 * sum(history[:50]) / 50


def test_samples_match_the_sequence_conditional_data_distribution(trained):
    """Per-sequence mean and spread of the samples match the Gaussian each sequence generated."""
    _, model, _ = trained
    sys_ = make_toy_system(n_positions=1)
    g = torch.Generator().manual_seed(1)
    for token, mu in MEANS.items():
        out = model.sample(sys_, _seq(token), 3000, g).coordinates[0]
        assert out.shape == (3000, 2)
        assert torch.allclose(out.mean(0), mu, atol=0.15), (token, out.mean(0))
        assert torch.allclose(out.std(0), torch.tensor(SIGMA), atol=0.12), (token, out.std(0))


@pytest.mark.parametrize("kind", ["diffusion", "flow_matching"])
def test_energy_guidance_pulls_samples_toward_the_energy_minimum(kind):
    """Base data ~ N(mu, sigma^2); a harmonic pull to ``c`` of strength lambda moves the mean
    toward the exact tilted mean ``(mu/s^2 + lam c)/(1/s^2 + lam)``. Guidance is heuristic: the
    diffusion sampler lands close to it; flow matching moves the right way but overshoots, so only
    the direction is asserted for it."""
    sys_ = make_toy_system(n_positions=1)
    center, lam = torch.tensor([0.0, 3.0]), 2.0
    energy = HarmonicEnergy(k=1.0, center=center)
    model, _ = _trained(kind, guidance=EnergyGuidance(energy, lam))
    g = torch.Generator().manual_seed(2)
    mu = MEANS[0]
    guided = model.sample(sys_, _seq(0), 3000, g).coordinates[0].mean(0)
    exact = (mu / SIGMA**2 + lam * center) / (1 / SIGMA**2 + lam)
    assert torch.dist(guided, center) < torch.dist(mu, center) - 0.5
    if kind == "diffusion":
        assert torch.dist(guided, exact) < torch.dist(mu, exact)


def test_fit_validates_inputs_and_networks_validate_shapes():
    net = ConditionalMLP(dim=2, context_dim=N_AA, hidden=8, depth=1)
    model = DiffusionConformationModel(net)
    with pytest.raises(ValueError):
        fit(model, torch.zeros(3, 2), SequenceBatch(torch.zeros(2, 1, dtype=torch.long)), steps=1)
    with pytest.raises(ValueError):
        ConditionalMLP(dim=0, context_dim=1)
    with pytest.raises(ValueError):
        DiffusionConformationModel(net, n_steps=1)
    ctx = sequence_context(SequenceBatch(torch.zeros(4, 1, dtype=torch.long)))
    assert net(torch.zeros(4, 2), torch.zeros(4), ctx).shape == (4, 2)
