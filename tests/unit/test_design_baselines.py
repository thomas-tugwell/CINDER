import random

import pytest
import torch

from cinder.objectives.pairwise_table import PairwiseEnergyTable
from cinder.optimization import best_single_tokens, one_body_tokens, random_search


def _table() -> tuple[PairwiseEnergyTable, torch.Tensor]:
    e1 = torch.zeros(3, 20)
    e1[0, 2] = -1.0
    e1[1, 5] = -3.0
    e1[1, 6] = -2.0
    e1[2, 7] = 4.0
    e1[2, 9] = -9.0  # disallowed below
    allowed = torch.zeros(3, 20, dtype=torch.bool)
    allowed[:, [0, 2, 5, 6, 7]] = True
    native = torch.zeros(3, dtype=torch.long)
    pairs = torch.zeros(0, 2, dtype=torch.long)
    return PairwiseEnergyTable(0.0, e1, pairs, torch.zeros(0, 20, 20), native), allowed


def test_best_single_picks_the_lowest_allowed_substitution():
    table, allowed = _table()
    assert best_single_tokens(table, allowed).tolist() == [0, 5, 0]


def test_one_body_optimum_ignores_disallowed_and_harmful_letters():
    table, allowed = _table()
    assert one_body_tokens(table, allowed).tolist() == [2, 5, 0]


def test_baselines_keep_the_reference_when_nothing_helps():
    table, allowed = _table()
    table.e1.clamp_(min=0)
    assert best_single_tokens(table, allowed).tolist() == [0, 0, 0]


def test_baselines_need_native_residues():
    table, allowed = _table()
    table = PairwiseEnergyTable(table.e0, table.e1, table.pairs, table.e2)
    with pytest.raises(ValueError):
        one_body_tokens(table, allowed)


def test_random_search_is_sorted_unique_and_bounded():
    calls: list[str] = []

    def score(seq: str) -> float:
        calls.append(seq)
        return float(sum(c != "A" for c in seq))

    out = random_search(score, "AAAA", [0, 1, 2, 3], "ADE", 10, 2, random.Random(0))
    assert len(out) == len(set(s for s, _ in out)) == len(calls) == 10
    assert [e for _, e in out] == sorted(e for _, e in out)
    assert all(1 <= e <= 2 for _, e in out)


def _landscape(seq: str) -> float:
    """Native 'AAAA'. D at 0 and E at 1 each help a little (-0.5) but together help a lot (-5,
    a pair effect); G at 2 helps by 2."""
    d, e = seq[0] == "D", seq[1] == "E"
    pair = -5.0 if d and e else -0.5 * d - 0.5 * e
    return pair - (2.0 if seq[2] == "G" else 0.0)


def test_hill_climb_is_monotone_and_finds_pair_synergy():
    from cinder.optimization import hill_climb

    res = hill_climb(_landscape, "AAAA", [0, 1, 2, 3], "ADEG", max_steps=5, pair_top=8)
    energies = [e for _, e in res.trajectory]
    assert energies == sorted(energies, reverse=True)  # never accepts a worse design
    assert res.sequence == "DEGA" and res.energy == -7.0
    assert res.n_evaluations > 0


def test_hill_climb_returns_native_when_nothing_improves():
    from cinder.optimization import hill_climb

    res = hill_climb(lambda s: float(sum(c != "A" for c in s)), "AAA", [0, 1, 2], "ADE")
    assert res.sequence == "AAA" and len(res.trajectory) == 1
