import pytest
import torch

from cinder.types import ObjectiveResult
from cinder.validation import ScanRow, apply_mutations, correlate, scan_mutants, spearman

from ..conftest import make_toy_system


def test_apply_mutations_single_multi_and_validation():
    assert apply_mutations("ACDEF", "C1A") == "AADEF"
    assert apply_mutations("ACDEF", "A0G+F4W") == "GCDEW"
    with pytest.raises(ValueError, match="native residue"):
        apply_mutations("ACDEF", "D1A")  # index 1 is C, not D
    with pytest.raises(ValueError, match="out of range"):
        apply_mutations("ACDEF", "A9G")
    with pytest.raises(ValueError, match="bad mutation spec"):
        apply_mutations("ACDEF", "nonsense")


def test_spearman_matches_known_values_and_handles_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)
    assert spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.8944, abs=1e-3)  # ties
    assert torch.isnan(torch.tensor(spearman([1, 1, 1], [1, 2, 3])))  # constant -> nan
    with pytest.raises(ValueError):
        spearman([1, 2], [1, 2])


class _Reward:
    """Energy = number of 'W' residues (so a mutant to W is worse); ensemble adds noise spread."""

    def __init__(self, k):
        self.k = k

    def __call__(self, system, sequence):
        from cinder.types import decode_sequence

        base = float(sum(c == "W" for c in decode_sequence(sequence.tokens[0])))
        e = torch.full((1, self.k), base) + torch.arange(self.k)[None] * 0.0
        if self.k > 1:
            e = e + torch.linspace(-0.5, 0.5, self.k)[None]
        return ObjectiveResult(reward=-e.mean(1), energies=e)


def test_scan_mutants_reports_energy_change_and_spread():
    system = make_toy_system(n_positions=4)  # native "AAAA"
    rows = scan_mutants(
        system,
        {"single": _Reward(1), "ensemble": _Reward(5)},
        {"toW": "A0W", "double": "A0W+A1W", "same": "A2A"},
    )
    by = {r.name: r for r in rows}
    assert by["toW"].delta["single"] == pytest.approx(1.0)  # energy rose by one W
    assert by["double"].delta["ensemble"] == pytest.approx(2.0)
    assert by["same"].delta["single"] == pytest.approx(0.0)
    assert by["toW"].spread["single"] == 0.0 and by["toW"].spread["ensemble"] > 0.3
    assert isinstance(rows[0], ScanRow) and "ensemble_delta" in rows[0].as_dict()


def test_correlate_ranks_models_against_measurements():
    system = make_toy_system(n_positions=4)
    muts = {"m1": "A0W", "m2": "A0W+A1W", "m3": "A0W+A1W+A2W", "m4": "A3A"}
    rows = scan_mutants(system, {"e": _Reward(3)}, muts)
    measured = {"m1": 1.0, "m2": 2.0, "m3": 3.0, "m4": 0.0}  # more W is worse, as predicted
    assert correlate(rows, measured)["e"] == pytest.approx(1.0)
    anti = {k: -v for k, v in measured.items()}
    assert correlate(rows, anti)["e"] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match=">= 3"):
        correlate(rows, {"m1": 1.0})
