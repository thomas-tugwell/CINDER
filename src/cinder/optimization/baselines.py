"""Baselines the table optimiser has to beat: best single mutant, one-body optimum, random search.

The first two read the table only (its one-body entries are exact single-mutant energies, so
they cost no extra scorer calls); random search spends real scorer calls.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from cinder.objectives.pairwise_table import PairwiseEnergyTable


def best_single_tokens(table: PairwiseEnergyTable, allowed: Tensor) -> Tensor:
    """Tokens ``[P]`` of the best single substitution; the reference if none lowers the energy.

    ``allowed`` is a ``[P, 20]`` bool mask; the reference residue at each position needs no entry.
    """
    _require_native(table)
    tokens = table.native.clone()  # type: ignore[union-attr]
    e1 = _masked(table, allowed)
    best = e1.min()
    if best < 0:
        i, a = divmod(int(e1.argmin()), e1.shape[1])
        tokens[i] = a
    return tokens


def one_body_tokens(table: PairwiseEnergyTable, allowed: Tensor) -> Tensor:
    """Tokens ``[P]`` minimising the one-body energy at every position independently (the
    optimum if all pair terms were zero)."""
    _require_native(table)
    tokens = table.native.clone()  # type: ignore[union-attr]
    e1 = _masked(table, allowed)
    for i in range(e1.shape[0]):
        if e1[i].min() < 0:
            tokens[i] = int(e1[i].argmin())
    return tokens


def random_search(
    score: Callable[[str], float],
    native: str,
    positions: Sequence[int],
    alphabet: str,
    n: int,
    max_mutations: int,
    rng: random.Random,
) -> list[tuple[str, float]]:
    """Score ``n`` random designs with 1..``max_mutations`` substitutions; returns (sequence,
    energy) sorted best first."""
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < 20 * n:
        attempts += 1
        seq = list(native)
        for pos in rng.sample(list(positions), rng.randint(1, min(max_mutations, len(positions)))):
            seq[pos] = rng.choice([a for a in alphabet if a != native[pos]])
        s = "".join(seq)
        if s in seen:
            continue
        seen.add(s)
        out.append((s, float(score(s))))
    return sorted(out, key=lambda r: r[1])


@dataclass
class HillClimbResult:
    sequence: str
    energy: float
    n_evaluations: int
    trajectory: list[tuple[str, float]]  # accepted sequences with their energies, native first


def hill_climb(
    score: Callable[[str], float],
    native: str,
    positions: Sequence[int],
    alphabet: str,
    max_steps: int = 10,
    pair_top: int = 6,
) -> HillClimbResult:
    """Greedy search with the real scorer: accept only improvements, never worse than ``native``.

    Each step scores every single substitution at ``positions`` around the current sequence plus
    the double substitutions among the ``pair_top`` best singles at distinct positions (cheap
    two-body moves), then moves to the best of them if it lowers the energy. The table optimiser's
    failure mode (selecting designs outside the region where its table is valid) cannot occur
    because every accepted design was scored for real.
    """
    cache: dict[str, float] = {}

    def value(seq: str) -> float:
        if seq not in cache:
            cache[seq] = float(score(seq))
        return cache[seq]

    current = native
    trajectory = [(current, value(current))]
    for _ in range(max_steps):
        base = value(current)
        singles: list[tuple[float, int, str]] = []
        for pos in positions:
            for aa in alphabet:
                if aa != current[pos]:
                    singles.append((value(_with(current, {pos: aa})), pos, aa))
        singles.sort()
        best_energy, best_seq = base, current
        if singles and singles[0][0] < best_energy:
            best_energy, best_seq = singles[0][0], _with(current, {singles[0][1]: singles[0][2]})
        for (_, p1, a1), (_, p2, a2) in itertools.combinations(singles[:pair_top], 2):
            if p1 != p2:
                seq = _with(current, {p1: a1, p2: a2})
                if value(seq) < best_energy:
                    best_energy, best_seq = value(seq), seq
        if best_seq == current:
            break
        current = best_seq
        trajectory.append((current, best_energy))
    return HillClimbResult(current, value(current), len(cache), trajectory)


def _with(sequence: str, changes: dict[int, str]) -> str:
    seq = list(sequence)
    for pos, aa in changes.items():
        seq[pos] = aa
    return "".join(seq)


def _require_native(table: PairwiseEnergyTable) -> None:
    if table.native is None:
        raise ValueError("this table has no reference residues (native) recorded")


def _masked(table: PairwiseEnergyTable, allowed: Tensor) -> Tensor:
    """``e1`` with the reference residue and disallowed letters set to +inf."""
    e1 = table.e1.clone()
    e1[~allowed] = float("inf")
    e1[torch.arange(e1.shape[0]), table.native] = float("inf")
    return e1
