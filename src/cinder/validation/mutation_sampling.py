"""Low-rate random mutations (optionally biased toward interface positions) and their statistics.

``MutationSampler`` draws mutants of a native binder sequence. The expected number of mutations is
``rate * L``; they are spread over positions in proportion to a weight (``1 + bias`` on interface
positions, ``1`` elsewhere), so ``bias = 0`` is uniform. Substitutions are uniform over the allowed
amino acids other than the native one.

``summarize_effects`` regresses each mutant's energy change on how many interface and
non-interface positions it mutates, giving the average cost of one interface mutation versus one
non-interface mutation. ``position_scan`` substitutes every alternative at chosen positions and
reports the rank of the native residue among all 20 (lower energy = better).
"""

from __future__ import annotations

import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import torch

from cinder.systems.specification import MolecularSystem
from cinder.types import AMINO_ACIDS, RewardModel
from cinder.validation.mutant_scan import ScanRow, scan_mutants


@dataclass(frozen=True)
class SampledMutant:
    """A mutant of the native sequence: ``spec`` (e.g. ``Y28A+D34N``), the new sequence, and the
    mutated 0-based binder positions."""

    spec: str
    sequence: str
    positions: tuple[int, ...]


class MutationSampler:
    def __init__(
        self,
        native: str,
        positions: Sequence[int],
        interface: Collection[int] = (),
        bias: float = 0.0,
        forbidden: str = "CP",
        seed: int = 0,
    ) -> None:
        if bias < 0:
            raise ValueError("bias must be >= 0")
        bad = [p for p in positions if not 0 <= p < len(native)]
        if bad or not positions:
            raise ValueError(f"positions must be non-empty and within the sequence; bad: {bad}")
        self.native = native
        self.positions = sorted(set(positions))
        self.interface = set(interface) & set(self.positions)
        self.bias = float(bias)
        self.alphabet = [a for a in AMINO_ACIDS if a not in forbidden]
        self.rng = random.Random(seed)
        self.weights = {
            p: 1.0 + (self.bias if p in self.interface else 0.0) for p in self.positions
        }

    def probabilities(self, rate: float) -> dict[int, float]:
        """Per-position mutation probability: expected total ``rate * L``, proportional to weight
        (capped at 1)."""
        if not 0 <= rate <= 1:
            raise ValueError("rate must be in [0, 1]")
        total_w = sum(self.weights.values())
        expected = rate * len(self.native)
        return {p: min(1.0, expected * w / total_w) for p, w in self.weights.items()}

    def sample(self, rate: float, min_mutations: int = 1) -> SampledMutant:
        """One mutant; if the draw produced fewer than ``min_mutations`` it is topped up by
        weighted sampling without replacement."""
        probs = self.probabilities(rate)
        chosen = {p for p, pr in probs.items() if self.rng.random() < pr}
        remaining = [p for p in self.positions if p not in chosen]
        while len(chosen) < min(min_mutations, len(self.positions)):
            pick = self.rng.choices(remaining, weights=[self.weights[p] for p in remaining])[0]
            chosen.add(pick)
            remaining.remove(pick)
        seq = list(self.native)
        parts = []
        for p in sorted(chosen):
            new = self.rng.choice([a for a in self.alphabet if a != self.native[p]])
            parts.append(f"{self.native[p]}{p}{new}")
            seq[p] = new
        return SampledMutant("+".join(parts), "".join(seq), tuple(sorted(chosen)))

    def sample_many(self, n: int, rate: float, min_mutations: int = 1) -> list[SampledMutant]:
        """``n`` distinct mutants (duplicates are redrawn)."""
        seen: dict[str, SampledMutant] = {}
        guard = 0
        while len(seen) < n:
            m = self.sample(rate, min_mutations)
            seen.setdefault(m.spec, m)
            guard += 1
            if guard > 50 * n + 100:
                raise RuntimeError("could not draw enough distinct mutants; raise the rate or n")
        return list(seen.values())


@dataclass
class EffectSummary:
    """Averages over a set of scored mutants (energy change; positive = worse)."""

    n: int
    mean_delta: float
    median_delta: float
    fraction_destabilizing: float
    cost_interface: float  # mean energy change per interface mutation (regression)
    cost_other: float  # mean energy change per non-interface mutation (regression)


def summarize_effects(
    deltas: Sequence[float], n_interface: Sequence[int], n_other: Sequence[int]
) -> EffectSummary:
    """Least-squares fit ``delta ≈ a * n_interface + b * n_other`` (no intercept: a mutant with no
    mutations has zero change) plus simple averages. Needs mutants that vary in both counts."""
    if not (len(deltas) == len(n_interface) == len(n_other)) or len(deltas) < 3:
        raise ValueError("need three equal-length sequences with at least 3 mutants")
    y = torch.tensor(deltas, dtype=torch.float64)
    x = torch.stack(
        [
            torch.tensor(n_interface, dtype=torch.float64),
            torch.tensor(n_other, dtype=torch.float64),
        ],
        dim=1,
    )
    coef = torch.linalg.lstsq(x, y[:, None]).solution[:, 0]
    return EffectSummary(
        n=len(deltas),
        mean_delta=float(y.mean()),
        median_delta=float(y.median()),
        fraction_destabilizing=float((y > 0).double().mean()),
        cost_interface=float(coef[0]),
        cost_other=float(coef[1]),
    )


@dataclass
class PositionProfile:
    position: int
    native: str
    deltas: dict[str, float]  # substituted amino acid -> energy change vs native
    native_rank: int  # 1 = native has the lowest energy among all 20

    @property
    def mean_delta(self) -> float:
        return sum(self.deltas.values()) / len(self.deltas)


def position_scan(
    system: MolecularSystem,
    model: RewardModel,
    positions: Sequence[int],
    forbidden: str = "",
    allowed: Mapping[int, str] | None = None,
) -> list[PositionProfile]:
    """Substitute every other amino acid at each position (others stay native) and score with
    ``model`` (the single packed structure is the usual choice). ``native_rank`` is the native
    residue's rank among all tried residues by energy (1 = lowest = best)."""
    native = system.design_space.native
    out = []
    for pos in positions:
        letters = (allowed or {}).get(pos, AMINO_ACIDS)
        subs = [a for a in letters if a != native[pos] and a not in forbidden]
        mutants = {a: f"{native[pos]}{pos}{a}" for a in subs}
        rows: list[ScanRow] = scan_mutants(system, {"m": model}, mutants)
        deltas = {r.name: r.delta["m"] for r in rows}
        rank = 1 + sum(1 for d in deltas.values() if d < 0)  # substitutions better than native
        out.append(PositionProfile(pos, native[pos], deltas, rank))
    return out
