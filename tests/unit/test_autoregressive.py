import itertools

import pytest
import torch

from cinder.config import ConfigError, config_from_dict
from cinder.experiments.runner import run_design
from cinder.optimization.reinforce import reinforce_loss
from cinder.sequence import AutoregressivePolicy, CategoricalPolicy, TrainablePolicy
from cinder.systems import DesignSpace
from cinder.types import N_AA


def _mask(p=3, allowed=(0, 1, 2, 3)):
    m = torch.zeros(p, N_AA, dtype=torch.bool)
    m[:, list(allowed)] = True
    return m


def test_distribution_is_normalised_and_matches_sampling():
    torch.manual_seed(0)
    pol = AutoregressivePolicy(_mask(3))
    with torch.no_grad():
        pol.coupling.normal_(0, 0.7)  # make it genuinely non-factorised
        pol.base.normal_(0, 0.5)
    seqs = torch.tensor(list(itertools.product(range(4), repeat=3)))
    probs = pol.log_prob(seqs).exp()
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)  # exact normalisation
    sample = pol.sample(40000, torch.Generator().manual_seed(1)).tokens
    idx = sample[:, 0] * 16 + sample[:, 1] * 4 + sample[:, 2]
    emp = torch.bincount(idx, minlength=64).float() / len(idx)
    assert (emp - probs).abs().max() < 0.012


def test_masks_are_respected_and_zero_coupling_equals_independent_policy():
    mask = _mask(3, allowed=(0, 5, 9))
    ar = AutoregressivePolicy(mask)
    cat = CategoricalPolicy(mask)
    tokens = ar.sample(500, torch.Generator().manual_seed(0)).tokens
    assert set(tokens.flatten().tolist()) <= {0, 5, 9}
    toks = torch.tensor([[0, 5, 9], [9, 9, 0]])
    assert torch.allclose(ar.log_prob(toks), cat.log_prob(toks), atol=1e-6)
    assert ar.entropy().item() == pytest.approx(cat.entropy().item(), rel=1e-5)


def test_entropy_estimate_is_unbiased_for_a_coupled_policy():
    torch.manual_seed(2)
    pol = AutoregressivePolicy(_mask(2), entropy_samples=4000)
    with torch.no_grad():
        pol.coupling.normal_(0, 1.0)
    seqs = torch.tensor(list(itertools.product(range(4), repeat=2)))
    lp = pol.log_prob(seqs)
    exact = -(lp.exp() * lp).sum().item()
    assert pol.entropy().item() == pytest.approx(exact, abs=0.03)


def test_satisfies_trainable_policy_protocol_and_greedy_decode():
    pol = AutoregressivePolicy(_mask(3))
    assert isinstance(pol, TrainablePolicy)
    assert pol.probs().shape == (3, N_AA) and pol.argmax_tokens().shape == (3,)
    assert pol.probs().sum(-1).tolist() == pytest.approx([1.0] * 3)


def _reward(tokens):
    """1 only for the coupled pairs (0, 1) and (2, 3)."""
    a, b = tokens[:, 0], tokens[:, 1]
    return (((a == 0) & (b == 1)) | ((a == 2) & (b == 3))).float()


def _train_entropy_regularised(policy, alpha, steps=600, batch=128, lr=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        s = policy.sample(batch, g)
        loss = reinforce_loss(s.log_prob, _reward(s.tokens), policy.entropy(), alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()


def _objective(policy, alpha, seqs):
    with torch.no_grad():
        lp = policy.log_prob(seqs)
    p = lp.exp()
    return float((p * _reward(seqs)).sum() - alpha * (p * lp).sum())  # E[R] + alpha * H


def test_autoregressive_represents_the_coupled_optimum_the_independent_policy_cannot():
    """Maximise E[R] + alpha*H. The optimum is q ∝ exp(R/alpha), value alpha*log(sum exp(R/alpha)),
    a *coupled* distribution (reward only for the pairs (0,1) and (2,3)). Position-independent
    policies cannot represent it, so they plateau below it; an autoregressive policy gets there.
    (Without an entropy term both collapse onto one valid pair and are equally good: the benefit
    of coupling is representing several good solutions at once.)"""
    alpha = 0.5
    seqs = torch.tensor(list(itertools.product(range(4), repeat=2)))
    optimum = alpha * torch.logsumexp(_reward(seqs) / alpha, dim=0).item()
    torch.manual_seed(0)
    ar = AutoregressivePolicy(_mask(2))
    _train_entropy_regularised(ar, alpha)
    torch.manual_seed(0)
    cat = CategoricalPolicy(_mask(2))
    _train_entropy_regularised(cat, alpha)
    assert _objective(ar, alpha, seqs) > optimum - 0.03  # reaches the optimum
    assert _objective(cat, alpha, seqs) < optimum - 0.08  # independent policy cannot


def test_runner_trains_the_autoregressive_backend_end_to_end(tmp_path):
    cfg = config_from_dict(
        {
            "seed": 1,
            "output_dir": str(tmp_path),
            "system": {"binder_length": 2},
            "conformation": {"n_samples": 4, "n_steps": 20, "step_size": 0.05},
            "sequence": {"backend": "autoregressive", "entropy_weight": 0.005},
            "optimization": {"iterations": 6, "batch_size": 8, "learning_rate": 0.1},
        }
    )
    result, run_dir = run_design(cfg)
    assert len(result.history) == 6 and (run_dir / "checkpoints" / "final.pt").exists()
    with pytest.raises(ConfigError):
        config_from_dict({"sequence": {"backend": "transformer"}})


def test_design_space_validation_unchanged_for_the_new_backend():
    from cinder.sequence import policy_from_design_space

    space = DesignSpace(3, "AAA", [0, 2])
    assert isinstance(
        policy_from_design_space(space, backend="autoregressive"), AutoregressivePolicy
    )
    with pytest.raises(ValueError, match="backend"):
        policy_from_design_space(space, backend="nope")
