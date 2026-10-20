import math

import pytest
import torch

from cinder.conformations.normalizing_flow import (
    ConditionalFlow,
    NormalizingFlowSampler,
    sequence_context,
    train_flow,
)
from cinder.energy.toy import GaussianMixtureEnergy, HarmonicEnergy, ToySequenceEnergy
from cinder.types import SequenceBatch

from ..conftest import make_toy_system


def _seqs(n=1, length=2):
    return SequenceBatch(torch.zeros(n, length, dtype=torch.long))


def test_flow_invertible_and_density_consistent():
    torch.manual_seed(0)
    flow = ConditionalFlow(dim=3, context_dim=4, n_layers=4, hidden=16)
    for p in flow.parameters():  # randomise so the flow is not the identity
        torch.nn.init.normal_(p, std=0.3)
    ctx = torch.randn(50, 4)
    x, log_q = flow.sample_with_log_prob(ctx)
    # density from sampling path equals density from the forward (data->latent) path
    assert torch.allclose(log_q, flow.log_prob(x, ctx), atol=1e-4)


def test_flow_density_normalizes_2d():
    torch.manual_seed(0)
    flow = ConditionalFlow(dim=2, context_dim=1, n_layers=4, hidden=16)
    for p in flow.parameters():
        torch.nn.init.normal_(p, std=0.2)
    g = torch.linspace(-12, 12, 400)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)
    with torch.no_grad():
        lp = flow.log_prob(pts, torch.zeros(len(pts), 1))
    mass = lp.exp().sum() * (g[1] - g[0]) ** 2
    assert mass.item() == pytest.approx(1.0, abs=0.02)


def test_variational_loss_recovers_gaussian():
    """L >= -log Z, tight at q = p_U; Gaussian log Z = (d/2) log(2 pi / (lam k))."""
    torch.manual_seed(0)
    system = make_toy_system(dim=2)
    lam, k = 1.0, 2.0
    center = torch.tensor([1.0, -1.0])
    energy = HarmonicEnergy(k, center)
    flow = ConditionalFlow(dim=2, context_dim=40, n_layers=4, hidden=32)
    hist = train_flow(flow, energy, system, _seqs(), lam, steps=600)
    neg_log_z = -(math.log(2 * math.pi / (lam * k)))  # d/2 * log(.) with d=2
    final = sum(hist[-50:]) / 50
    assert final >= neg_log_z - 0.05  # variational bound (up to MC noise)
    assert final - neg_log_z < 0.05  # and tight
    x = NormalizingFlowSampler(flow).sample(system, _seqs(), 4000).coordinates[0]
    assert torch.allclose(x.mean(0), center, atol=0.08)
    assert x.var(0).mean().item() == pytest.approx(1 / (lam * k), rel=0.15)


def test_variational_loss_recovers_bimodal():
    torch.manual_seed(0)
    system = make_toy_system(dim=2)
    means = torch.tensor([[-1.5, 0.0], [1.5, 0.0]])
    energy = GaussianMixtureEnergy(means, sigma=1.0)  # normalised density => log Z = 0
    flow = ConditionalFlow(dim=2, context_dim=40, n_layers=6, hidden=64)
    hist = train_flow(flow, energy, system, _seqs(), 1.0, steps=1500, lr=2e-3)
    final = sum(hist[-100:]) / 100
    assert final >= -0.05
    assert final < 0.2  # a single-mode collapse would cost ~log 2 = 0.69
    x = NormalizingFlowSampler(flow).sample(system, _seqs(), 4000).coordinates[0]
    assert 0.3 < (x[:, 0] > 0).float().mean() < 0.7


def test_conditional_flow_tracks_sequence_dependent_mean():
    torch.manual_seed(0)
    toy = ToySequenceEnergy(n_positions=2, dim=2, seed=3)
    system = make_toy_system(2, 2)
    pool = SequenceBatch(torch.randint(0, 20, (6, 2)))
    flow = ConditionalFlow(dim=2, context_dim=40, n_layers=4, hidden=64)
    train_flow(flow, toy, system, pool, lam=1.0, steps=1500, lr=3e-3)
    sampler = NormalizingFlowSampler(flow)
    x = sampler.sample(system, pool, 3000).coordinates  # [6, 3000, 2]
    assert torch.allclose(x.mean(1), toy.equilibrium_mean(pool.tokens), atol=0.15)
    assert sequence_context(pool).shape == (6, 40)


def test_log_prob_supported_on_sampler():
    flow = ConditionalFlow(dim=2, context_dim=40)
    s = NormalizingFlowSampler(flow)
    out = s.sample(make_toy_system(), _seqs(2), 3)
    assert out.log_prob is not None and out.log_prob.shape == (2, 3)
