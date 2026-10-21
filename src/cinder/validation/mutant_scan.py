"""Score point/multi-mutants with several reward models and compare them to measured data.

The scientific question this supports: does ranking mutants by an *ensemble* energy track
measured binding better than ranking by a single packed structure? ``scan_mutants`` evaluates
each mutant with any number of named :class:`~cinder.types.RewardModel` objects and reports, per
mutant, the energy change relative to the wild type (positive = worse binding by that score).
``correlate`` computes Spearman rank correlations against measured values (supplied by the user;
nothing here ships experimental numbers).

Mutation specs are ``<wt><index><mut>`` with a **0-based binder-local index**, e.g. ``Y28A``; a
multi-mutant joins specs with ``+`` (``Y28A+D34N``). The wild-type letter is checked against the
native sequence so a wrong index fails loudly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from cinder.systems.specification import MolecularSystem
from cinder.types import RewardModel, SequenceBatch, encode_sequence

_SPEC = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def apply_mutations(native: str, spec: str) -> str:
    """Apply ``spec`` (see module docstring) to ``native`` and return the mutant sequence."""
    seq = list(native)
    for part in spec.split("+"):
        m = _SPEC.match(part.strip())
        if not m:
            raise ValueError(f"bad mutation spec {part!r}; expected e.g. 'Y28A' (0-based index)")
        wt, idx, new = m.group(1), int(m.group(2)), m.group(3)
        if not 0 <= idx < len(native):
            raise ValueError(f"index {idx} out of range for a binder of length {len(native)}")
        if native[idx] != wt:
            raise ValueError(
                f"{part}: native residue at index {idx} is {native[idx]!r}, not {wt!r}"
            )
        seq[idx] = new
    return "".join(seq)


@dataclass
class ScanRow:
    name: str
    sequence: str
    reward: dict[str, float] = field(default_factory=dict)  # model -> -aggregate energy
    delta: dict[str, float] = field(default_factory=dict)  # model -> E(mut) - E(wt)
    spread: dict[str, float] = field(default_factory=dict)  # model -> std over the ensemble

    def as_dict(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {"name": self.name, "sequence": self.sequence}
        for model in self.reward:
            out[f"{model}_delta"] = self.delta[model]
            out[f"{model}_spread"] = self.spread[model]
        return out


def scan_mutants(
    system: MolecularSystem,
    models: Mapping[str, RewardModel],
    mutants: Mapping[str, str],
) -> list[ScanRow]:
    """Evaluate the wild type and every mutant (``name -> spec``) with every model in ``models``."""
    native = system.design_space.native
    wild = _evaluate(system, models, native)
    rows = []
    for name, spec in mutants.items():
        seq = apply_mutations(native, spec)
        row = ScanRow(name=name, sequence=seq)
        got = _evaluate(system, models, seq)
        for model, (reward, spread) in got.items():
            row.reward[model] = reward
            row.delta[model] = wild[model][0] - reward  # energy change = -(reward change)
            row.spread[model] = spread
        rows.append(row)
    return rows


def _evaluate(
    system: MolecularSystem, models: Mapping[str, RewardModel], sequence: str
) -> dict[str, tuple[float, float]]:
    batch = SequenceBatch(encode_sequence(sequence)[None])
    out = {}
    for name, model in models.items():
        res = model(system, batch)
        e = res.energies[0]
        out[name] = (float(res.reward[0]), float(e.std()) if e.numel() > 1 else 0.0)
    return out


def _ranks(x: Tensor) -> Tensor:
    """Average ranks (ties share the mean rank)."""
    order = x.argsort()
    ranks = torch.empty_like(x, dtype=torch.float64)
    ranks[order] = torch.arange(len(x), dtype=torch.float64)
    for value in x.unique():
        tied = x == value
        if int(tied.sum()) > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation of two equal-length samples (``nan`` if either is constant)."""
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("need two equal-length samples with at least 3 points")
    rx, ry = _ranks(torch.tensor(x)), _ranks(torch.tensor(y))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(rx.norm() * ry.norm())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def correlate(rows: list[ScanRow], measured: Mapping[str, float]) -> dict[str, float]:
    """Spearman correlation of each model's predicted energy change with ``measured`` values
    (same sign convention: larger = worse). Mutants without a measurement are skipped."""
    used = [r for r in rows if r.name in measured]
    if len(used) < 3:
        raise ValueError(f"need >= 3 mutants with measurements, got {len(used)}")
    truth = [measured[r.name] for r in used]
    return {model: spearman([r.delta[model] for r in used], truth) for model in used[0].delta}
