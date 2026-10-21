import statistics

import pytest
import torch

from cinder.types import ObjectiveResult, decode_sequence
from cinder.validation import (
    MutationSampler,
    apply_mutations,
    position_scan,
    summarize_effects,
)

from ..conftest import make_toy_system

NATIVE = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 residues
INTERFACE = {10, 11, 12, 40, 41, 70, 71, 72}


def test_expected_mutation_count_follows_the_rate():
    s = MutationSampler(NATIVE, range(100), seed=0)
    muts = [s.sample(0.03, min_mutations=0) for _ in range(2000)]
    mean_n = statistics.mean(len(m.positions) for m in muts)
    assert mean_n == pytest.approx(3.0, abs=0.2)  # rate * L
    assert s.probabilities(0.0) == {p: 0.0 for p in range(100)}
    assert sum(s.probabilities(0.03).values()) == pytest.approx(3.0)


def test_bias_concentrates_mutations_on_interface_positions():
    uniform = MutationSampler(NATIVE, range(100), INTERFACE, bias=0.0, seed=1)
    biased = MutationSampler(NATIVE, range(100), INTERFACE, bias=9.0, seed=1)

    def interface_fraction(sampler):
        hits = total = 0
        for _ in range(1500):
            m = sampler.sample(0.03, min_mutations=1)
            hits += sum(p in INTERFACE for p in m.positions)
            total += len(m.positions)
        return hits / total

    assert interface_fraction(uniform) == pytest.approx(len(INTERFACE) / 100, abs=0.03)
    # weight 10 on 8 positions vs 1 on 92: expected share 80 / 172
    assert interface_fraction(biased) == pytest.approx(80 / 172, abs=0.05)
    assert interface_fraction(biased) > 4 * interface_fraction(uniform)


def test_mutants_are_valid_native_checked_and_respect_constraints():
    s = MutationSampler(NATIVE, [3, 4, 5, 90], forbidden="CPW", seed=2)
    for _ in range(300):
        m = s.sample(0.2)
        assert apply_mutations(NATIVE, m.spec) == m.sequence  # spec round-trips
        assert set(m.positions) <= {3, 4, 5, 90}  # never outside the allowed positions
        for p in m.positions:
            assert m.sequence[p] != NATIVE[p] and m.sequence[p] not in "CPW"
        assert [i for i in range(100) if m.sequence[i] != NATIVE[i]] == list(m.positions)


def test_min_mutations_and_reproducibility_and_distinctness():
    s = MutationSampler(NATIVE, range(100), seed=3)
    assert all(len(s.sample(0.0, min_mutations=2).positions) == 2 for _ in range(50))
    a = MutationSampler(NATIVE, range(100), seed=7).sample_many(20, 0.03)
    b = MutationSampler(NATIVE, range(100), seed=7).sample_many(20, 0.03)
    assert [m.spec for m in a] == [m.spec for m in b]
    assert len({m.spec for m in a}) == 20
    with pytest.raises(RuntimeError):
        MutationSampler(NATIVE, [0], seed=0).sample_many(500, 0.01)  # only 18 distinct mutants


def test_sampler_validation():
    with pytest.raises(ValueError):
        MutationSampler(NATIVE, [], seed=0)
    with pytest.raises(ValueError):
        MutationSampler(NATIVE, [500], seed=0)
    with pytest.raises(ValueError):
        MutationSampler(NATIVE, [1], bias=-1.0)
    with pytest.raises(ValueError):
        MutationSampler(NATIVE, [1]).probabilities(1.5)


def test_summarize_effects_recovers_known_per_mutation_costs():
    g = torch.Generator().manual_seed(0)
    n_int = torch.randint(0, 4, (200,), generator=g).tolist()
    n_oth = torch.randint(0, 4, (200,), generator=g).tolist()
    noise = (torch.randn(200, generator=g) * 0.2).tolist()
    deltas = [2.0 * a + 0.5 * b + e for a, b, e in zip(n_int, n_oth, noise, strict=True)]
    s = summarize_effects(deltas, n_int, n_oth)
    assert s.cost_interface == pytest.approx(2.0, abs=0.1)
    assert s.cost_other == pytest.approx(0.5, abs=0.1)
    assert 0.0 < s.fraction_destabilizing <= 1.0 and s.n == 200
    with pytest.raises(ValueError):
        summarize_effects([1.0, 2.0], [1, 0], [0, 1])


class _FakeReward:
    """Energy = hydrophobic-count penalty, so substitutions have a known ordering."""

    penalty = {"A": 0.0, "G": 0.5, "S": 0.7, "W": 5.0, "L": 1.0}

    def __call__(self, system, sequence):
        seq = decode_sequence(sequence.tokens[0])
        e = sum(self.penalty.get(c, 2.0) for c in seq)
        return ObjectiveResult(reward=torch.tensor([-e]), energies=torch.tensor([[e]]))


def test_position_scan_reports_native_rank_among_all_residues():
    system = make_toy_system(n_positions=3)  # native AAA: every substitution is worse or equal
    prof = position_scan(system, _FakeReward(), [0, 1], allowed={0: "AGSWL", 1: "AGSWL"})
    assert [p.position for p in prof] == [0, 1] and prof[0].native == "A"
    assert prof[0].native_rank == 1  # A (penalty 0) beats every alternative
    assert set(prof[0].deltas) == {"G", "S", "W", "L"} and all(
        d > 0 for d in prof[0].deltas.values()
    )
    assert prof[0].mean_delta == pytest.approx((0.5 + 0.7 + 5.0 + 1.0) / 4)


def test_position_scan_rank_when_a_substitution_is_better():
    system = make_toy_system(n_positions=3)
    system.design_space.native = "WWW"  # type: ignore[misc]
    prof = position_scan(system, _FakeReward(), [1], allowed={1: "AGSWL"})
    assert prof[0].native == "W" and prof[0].native_rank == 5  # all four alternatives beat W
