import math

import pytest
import torch

from cinder.energy.composite import CompositeEnergy, EnergyTerm
from cinder.energy.toy import HarmonicEnergy
from cinder.objectives.ensemble import (
    EnergyEnsembleObjective,
    free_energy_aggregate,
    make_aggregator,
    mean_aggregate,
    median_aggregate,
    quantile_aggregate,
    softmin_aggregate,
    variational_free_energy,
)
from cinder.objectives.interface import DesignEnergyWeights, build_design_energy
from cinder.optimization.reinforce import reinforce_loss
from cinder.sequence import CategoricalPolicy
from cinder.types import N_AA, ConformationBatch, SequenceBatch

from ..conftest import make_toy_system


def test_reinforce_gradient_matches_score_function():
    # 1 position, 3 AAs, reward r(a); analytic grad of J = sum_a p_a r_a  (no baseline)
    mask = torch.zeros(1, N_AA, dtype=torch.bool)
    mask[0, :3] = True
    pol = CategoricalPolicy(mask)
    r_table = torch.tensor([1.0, 3.0, -2.0])
    tokens = torch.arange(3)[:, None]
    p = pol.probs()[0, :3]
    # exact expectation of (R-b) grad log q  == grad J  when summed over all outcomes weighted by p
    lp = pol.log_prob(tokens)
    loss = -(p * r_table * lp).sum()  # b = 0, exact expectation
    loss.backward()
    g_exact = pol.logits.grad.clone()
    pol.zero_grad()
    j = (pol.log_probs().exp()[0, :3] * r_table).sum()
    j.backward()
    assert torch.allclose(g_exact, -pol.logits.grad, atol=1e-6)


def test_reinforce_loss_baseline_and_entropy():
    lp = torch.tensor([-1.0, -2.0], requires_grad=True)
    r = torch.tensor([1.0, 3.0])
    ent = torch.tensor(2.0)
    loss = reinforce_loss(lp, r, ent, 0.1)
    # baseline = 2 -> adv = [-1, 1]; loss = -mean([-1*-1, 1*-2]) - 0.1*2 = 0.5 - 0.2
    assert loss.item() == pytest.approx(0.3)
    loss.backward()
    assert torch.allclose(lp.grad, torch.tensor([0.5, -0.5]))
    # reward carries no gradient
    r2 = torch.tensor([1.0, 3.0], requires_grad=True)
    reinforce_loss(lp.detach().requires_grad_(), r2).backward()
    assert r2.grad is None


def test_aggregators():
    u = torch.tensor([[1.0, 2.0, 3.0, 10.0]])
    assert mean_aggregate(u).item() == pytest.approx(4.0)
    assert median_aggregate(u).item() == pytest.approx(2.0)
    assert quantile_aggregate(u, 0.0).item() == pytest.approx(1.0)
    # softmin: T large -> mean, T small -> min
    assert softmin_aggregate(u.double(), 1e5).item() == pytest.approx(4.0, abs=1e-2)
    assert softmin_aggregate(u, 1e-3).item() == pytest.approx(1.0 + 1e-3 * math.log(4), abs=1e-4)
    assert free_energy_aggregate(u, 1.0).item() == pytest.approx(
        -math.log(torch.exp(-u).mean().item())
    )
    assert make_aggregator("softmin", temperature=2.0)(u).shape == (1,)
    with pytest.raises(ValueError):
        make_aggregator("nope")


def test_variational_free_energy():
    e = torch.tensor([[1.0, 3.0]])
    lq = torch.tensor([[-1.0, -3.0]])
    assert variational_free_energy(e, lq, 2.0).item() == pytest.approx(((1 - 0.5) + (3 - 1.5)) / 2)


def test_composite_energy_terms_weights_and_gradient():
    system = make_toy_system(dim=2)
    a, b = HarmonicEnergy(1.0), HarmonicEnergy(2.0, center=1.0)
    comp = CompositeEnergy([EnergyTerm("a", a, 1.0), EnergyTerm("b", b, 0.5)])
    x = torch.randn(4, 2)
    seq = SequenceBatch(torch.zeros(4, 2, dtype=torch.long))
    out = comp.evaluate(system, x, seq)
    assert torch.allclose(out.terms["b"], 0.5 * b.energy(system, x, seq))
    assert torch.allclose(out.total, out.terms["a"] + out.terms["b"])
    assert torch.allclose(comp.gradient(system, x, seq), x + 0.5 * 2.0 * (x - 1.0))
    with pytest.raises(ValueError):
        CompositeEnergy([EnergyTerm("a", a), EnergyTerm("a", b)])
    with pytest.raises(ValueError):
        CompositeEnergy([])


def test_build_design_energy_drops_zero_weights_and_rejects_unknown():
    models = {"interface": HarmonicEnergy(), "strain": HarmonicEnergy()}
    comp = build_design_energy(models, DesignEnergyWeights(interface=2.0, strain=0.0))
    assert [t.name for t in comp.terms] == ["interface"]
    with pytest.raises(ValueError):
        build_design_energy({"mystery": HarmonicEnergy()}, DesignEnergyWeights())


def test_ensemble_objective_shapes_and_reward_sign():
    system = make_toy_system(dim=2)
    comp = CompositeEnergy([EnergyTerm("h", HarmonicEnergy(1.0))])
    obj = EnergyEnsembleObjective(comp)
    x = torch.randn(3, 5, 2)
    seq = SequenceBatch(torch.zeros(3, 2, dtype=torch.long))
    res = obj(system, seq, ConformationBatch(x))
    assert res.energies.shape == (3, 5) and res.reward.shape == (3,)
    assert torch.allclose(res.reward, -(0.5 * x.pow(2).sum(-1)).mean(1))
    assert res.terms["h"].shape == (3, 5)


def test_design_weights_include_hbond_and_elec_terms_only_when_nonzero():
    models = {n: HarmonicEnergy() for n in ("interface", "hbond", "elec")}
    default = build_design_energy(models, DesignEnergyWeights())
    assert [t.name for t in default.terms] == ["interface"]  # extra terms default to weight 0
    on = build_design_energy(models, DesignEnergyWeights(hbond=2.0, elec=0.5))
    assert {t.name: t.weight for t in on.terms} == {"interface": 1.0, "hbond": 2.0, "elec": 0.5}
