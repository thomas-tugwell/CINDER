import itertools

import pytest
import torch

from cinder.energy.toy import ToySequenceEnergy
from cinder.objectives.pairwise_table import (
    PairwiseEnergyTable,
    approximation_error,
    contact_pairs,
    tables_from_scorer,
)
from cinder.optimization import ExpectedEnergySettings, optimize_expected_energy
from cinder.sequence import CategoricalPolicy
from cinder.types import AMINO_ACIDS, N_AA, decode_sequence, encode_sequence

ALPHABET = "ACDE"  # a 4-letter alphabet keeps enumeration tiny
IDX = [AMINO_ACIDS.index(a) for a in ALPHABET]


def _mask(p):
    m = torch.zeros(p, N_AA, dtype=torch.bool)
    m[:, IDX] = True
    return m


def _random_table(p=3, pairs=((0, 1), (0, 2), (1, 2)), seed=0):
    g = torch.Generator().manual_seed(seed)
    e1 = torch.randn(p, N_AA, generator=g)
    e2 = torch.randn(len(pairs), N_AA, N_AA, generator=g)
    return PairwiseEnergyTable(0.7, e1, torch.tensor(pairs), e2)


def _all_sequences(p):
    return torch.tensor(
        [[IDX[i] for i in combo] for combo in itertools.product(range(4), repeat=p)]
    )


def test_expected_energy_equals_brute_force_average_over_the_factorised_distribution():
    table = _random_table()
    probs = torch.softmax(torch.randn(3, N_AA) * 0.8, dim=-1)
    probs = probs * _mask(3)
    probs = probs / probs.sum(-1, keepdim=True)
    seqs = _all_sequences(3)
    weights = torch.stack([probs[i, seqs[:, i]] for i in range(3)]).prod(0)  # q(S)
    brute = float((weights * table.energy(seqs)).sum())
    assert float(table.expected_energy(probs)) == pytest.approx(brute, rel=1e-5)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)


def test_expected_energy_gradient_matches_finite_differences():
    table = _random_table(seed=1)
    pol = CategoricalPolicy(_mask(3))
    with torch.no_grad():
        pol.logits.normal_(0, 0.5)
    table.expected_energy(pol.marginals()).backward()
    grad = pol.logits.grad.clone()
    eps = 1e-3
    for i, a in itertools.product(range(3), IDX):
        for sign in (+1, -1):
            with torch.no_grad():
                pol.logits[i, a] += sign * eps
            with torch.no_grad():
                val = float(table.expected_energy(pol.marginals()))
            with torch.no_grad():
                pol.logits[i, a] -= sign * eps
            if sign == +1:
                plus = val
            else:
                minus = val
        assert grad[i, a].item() == pytest.approx((plus - minus) / (2 * eps), abs=2e-3)


def test_one_body_optimum_is_the_boltzmann_distribution():
    """With no pair terms, minimising E_q[U] - alpha*H gives p_i(a) ∝ exp(-e1_i(a)/alpha)."""
    g = torch.Generator().manual_seed(2)
    e1 = torch.randn(2, N_AA, generator=g) * 0.5
    table = PairwiseEnergyTable(
        0.0, e1, torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, N_AA, N_AA)
    )
    pol = CategoricalPolicy(_mask(2))
    optimize_expected_energy(
        pol,
        table,
        ExpectedEnergySettings(iterations=600, learning_rate=0.2, alpha_start=0.5, alpha_end=0.5),
    )
    expected = torch.softmax((-e1 / 0.5).masked_fill(~_mask(2), -1e9), dim=-1)
    assert torch.allclose(pol.marginals().detach(), expected, atol=2e-3)


def test_tables_from_scorer_recovers_a_pairwise_function_exactly():
    truth = _random_table(p=3, seed=3)
    native = "AAA"

    def score(seq: str) -> float:
        tokens = encode_sequence(seq)[None]
        # the "true" energy: a table defined relative to native A at each position
        return float(truth.energy(tokens)[0])

    # shift so the reference (all-A) has the e0 we expect from inclusion-exclusion
    built = tables_from_scorer(score, native, [0, 1, 2], [(0, 1), (0, 2), (1, 2)], ALPHABET)
    for seq in ("ACD", "EDC", "AAE", "CAA", "DEA"):
        tokens = encode_sequence(seq)[None]
        assert float(built.energy(tokens)[0]) == pytest.approx(score(seq), abs=1e-4)
    assert float(built.e0) == pytest.approx(score(native), abs=1e-5)
    with pytest.raises(ValueError, match="unknown amino acid"):
        tables_from_scorer(score, native, [0], [], "AZ")


def test_approximation_error_exposes_three_body_effects_that_a_pairwise_table_misses():
    base = _random_table(p=3, seed=4)

    def score(seq: str) -> float:
        t = encode_sequence(seq)
        three_body = 2.0 * float(seq[0] == "C" and seq[1] == "D" and seq[2] == "E")
        return float(base.energy(t[None])[0]) + three_body

    table = tables_from_scorer(score, "AAA", [0, 1, 2], [(0, 1), (0, 2), (1, 2)], ALPHABET)
    seqs = ["".join(c) for c in itertools.product(ALPHABET, repeat=3)]
    err = approximation_error(table, score, "AAA", [0, 1, 2], seqs)
    assert 0.0 < err["max_abs_error"] <= 2.5  # only the three-body combination is missed
    assert err["mean_abs_error"] < err["one_body_mean_abs_error"]  # pairs help over one-body
    assert err["pearson"] > 0.99 and err["n"] == 64


def test_contact_pairs_uses_minimum_heavy_atom_distance():
    # residue 0: atoms 0,1 near origin; residue 1: atom 2 at 4 Å; residue 2: atom 3 at 20 Å
    coords = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0], [20.0, 0, 0]])
    residue_of_atom = torch.tensor([0, 0, 1, 2])
    heavy = torch.arange(4)
    assert contact_pairs(coords, residue_of_atom, heavy, [0, 1, 2], cutoff=6.0) == [(0, 1)]
    assert contact_pairs(coords, residue_of_atom, heavy, [0, 1, 2], cutoff=30.0) == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    assert contact_pairs(coords, residue_of_atom, heavy, [0, 2], cutoff=6.0) == []


def test_table_validation():
    with pytest.raises(ValueError):
        PairwiseEnergyTable(
            0.0, torch.zeros(2, 5), torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, 20, 20)
        )
    with pytest.raises(ValueError, match="i < j"):
        PairwiseEnergyTable(
            0.0, torch.zeros(2, N_AA), torch.tensor([[1, 0]]), torch.zeros(1, N_AA, N_AA)
        )
    with pytest.raises(ValueError):
        PairwiseEnergyTable(
            0.0, torch.zeros(2, N_AA), torch.tensor([[0, 1]]), torch.zeros(2, N_AA, N_AA)
        )
    with pytest.raises(ValueError, match="probs must be"):
        _random_table().expected_energy(torch.zeros(2, N_AA))
    with pytest.raises(ValueError, match="positions"):
        optimize_expected_energy(CategoricalPolicy(_mask(2)), _random_table())
    with pytest.raises(ValueError):
        ExpectedEnergySettings(alpha_start=0.1, alpha_end=1.0)


def test_exact_gradient_trainer_finds_the_brute_force_optimum_of_the_toy_sequence_problem():
    """The toy ensemble energy is exactly pairwise, so the table is exact and the exact-gradient
    trainer must land on the true optimum found by enumerating all 8000 sequences."""
    toy = ToySequenceEnergy(n_positions=3, dim=2, seed=0)
    lam = 1.0

    def score(seq: str) -> float:
        return float(toy.expected_energy(encode_sequence(seq)[None], lam)[0])

    native = "AAA"
    pairs = [(0, 1), (0, 2), (1, 2)]
    table = tables_from_scorer(score, native, [0, 1, 2], pairs, AMINO_ACIDS)
    all_toks = torch.tensor(list(itertools.product(range(N_AA), repeat=3)))
    true_energy = toy.expected_energy(all_toks, lam)
    assert float((table.energy(all_toks) - true_energy).abs().max()) < 1e-3  # exactly pairwise
    best = decode_sequence(all_toks[int(true_energy.argmin())])

    pol = CategoricalPolicy(torch.ones(3, N_AA, dtype=torch.bool))
    result = optimize_expected_energy(pol, table, ExpectedEnergySettings(iterations=500))
    assert decode_sequence(result.tokens) == best
    assert result.table_energy == pytest.approx(float(true_energy.min()), abs=1e-3)
    assert result.expected_energy == pytest.approx(result.table_energy, abs=0.05)  # concentrated
    assert result.history[-1]["entropy"] < result.history[0]["entropy"]


def _double_cancellation_scorer():
    """A clash at position 1 (+1000) that is relieved by *either* neighbour (position 0 or 2) but
    can only be relieved once: the true energy of the triple is the single-relief value, while a
    pairwise table subtracts the relief twice (a real failure seen on barnase/barstar)."""

    def score(seq: str) -> float:
        clash = seq[1] == "E"  # E at position 1 clashes ...
        relieved = (
            seq[0] == "D" or seq[2] == "D"
        )  # ... unless a neighbour is D (relief is once only)
        return 1000.0 if clash and not relieved else (50.0 if clash else 0.0)

    return score


def test_inclusion_exclusion_double_counts_a_relief_that_can_only_happen_once():
    score = _double_cancellation_scorer()
    table = tables_from_scorer(score, "AAA", [0, 1, 2], [(0, 1), (1, 2)], "ADE")
    triple = encode_sequence("DEE")[None]  # D at 0, E at 1, E at 2: still just one relief
    true = score("DEE")
    assert true == 50.0
    assert float(table.energy(triple)[0]) == pytest.approx(true, abs=1e-3)  # one relief: fine
    both = encode_sequence("DED")[None]  # D at 0 and 2: two neighbours relieve the same clash
    assert score("DED") == 50.0
    # the table subtracts the relief twice: 1000 - 950 - 950 = -900 instead of the true +50
    assert float(table.energy(both)[0]) == pytest.approx(-900.0, abs=1e-3)


def test_clipping_removes_the_spurious_cancellation():
    score = _double_cancellation_scorer()
    raw = tables_from_scorer(score, "AAA", [0, 1, 2], [(0, 1), (1, 2)], "ADE")
    clipped = raw.clipped(cap=60.0)
    both = encode_sequence("DED")[None]
    assert float(raw.energy(both)[0]) == pytest.approx(-900.0, abs=1e-3)  # error of 950
    # clipped at 60 the table says 60 - 10 - 10 = 40 vs the true 50: error 950 shrinks to 10
    assert float(clipped.energy(both)[0]) == pytest.approx(40.0, abs=1e-3)
    assert abs(float(clipped.energy(both)[0]) - score("DED")) < 0.02 * abs(
        float(raw.energy(both)[0]) - score("DED")
    )
    # clipping a table equals building it from a clipped scorer
    capped = tables_from_scorer(
        lambda s: min(score(s), 60.0), "AAA", [0, 1, 2], [(0, 1), (1, 2)], "ADE"
    )
    assert torch.allclose(clipped.e1, capped.e1, atol=1e-4)
    assert torch.allclose(clipped.e2, capped.e2, atol=1e-4)
    assert clipped.e0 == raw.e0 and torch.equal(clipped.pairs, raw.pairs)
    with pytest.raises(ValueError, match="cap"):
        raw.clipped(0.0)


def test_mutation_cost_pulls_the_policy_toward_the_reference():
    g = torch.Generator().manual_seed(5)
    e1 = torch.randn(3, N_AA, generator=g) * 0.1
    native = torch.tensor([0, 0, 0])  # all 'A'
    e1[torch.arange(3), native] = 0.0
    table = PairwiseEnergyTable(
        0.0, e1, torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, N_AA, N_AA), native
    )
    free = CategoricalPolicy(_mask(3))
    costly = CategoricalPolicy(_mask(3))
    base = ExpectedEnergySettings(iterations=300, learning_rate=0.2, alpha_start=0.3, alpha_end=0.3)
    optimize_expected_energy(free, table, base)
    optimize_expected_energy(
        costly, table, ExpectedEnergySettings(**{**base.__dict__, "mutation_cost": 2.0})
    )
    assert float(table.expected_mutations(costly.marginals())) < float(
        table.expected_mutations(free.marginals())
    )
    assert float(table.expected_mutations(costly.marginals())) < 0.5
    with pytest.raises(ValueError, match="native"):
        PairwiseEnergyTable(
            0.0, e1, torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, N_AA, N_AA)
        ).expected_mutations(free.marginals())
    with pytest.raises(ValueError, match="mutation_cost"):
        ExpectedEnergySettings(mutation_cost=-1.0)
