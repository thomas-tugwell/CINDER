"""Ensemble aggregation and the sequence reward.

All aggregators map energies ``[B, K]`` to one *energy-like* value per sequence ``[B]``
(lower is better); the reward is its negative. Aggregators operate on tensors only, so they
are agnostic to where the ensemble came from.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from cinder.energy.composite import CompositeEnergy
from cinder.systems.specification import MolecularSystem
from cinder.types import (
    ConformationBatch,
    ConformationModel,
    ObjectiveResult,
    SequenceBatch,
)

Aggregator = Callable[[Tensor], Tensor]


def mean_aggregate(u: Tensor) -> Tensor:
    return u.mean(dim=1)


def median_aggregate(u: Tensor) -> Tensor:
    return u.median(dim=1).values


def softmin_aggregate(u: Tensor, temperature: float = 1.0) -> Tensor:
    """``-T log mean exp(-U/T)``; ``T -> inf`` is the mean, ``T -> 0`` the minimum."""
    k = u.shape[1]
    return -temperature * (torch.logsumexp(-u / temperature, dim=1) - math.log(k))


def quantile_aggregate(u: Tensor, q: float = 0.25) -> Tensor:
    return torch.quantile(u, q, dim=1)


def free_energy_aggregate(u: Tensor, lam: float = 1.0) -> Tensor:
    """Free-energy surrogate ``-(1/lam) log mean_k exp(-lam U_k)`` (ensemble partition-function
    estimate under samples from the *base* distribution)."""
    return softmin_aggregate(u, temperature=1.0 / lam)


def make_aggregator(name: str, **kwargs: float) -> Aggregator:
    table: dict[str, Callable[..., Tensor]] = {
        "mean": mean_aggregate,
        "median": median_aggregate,
        "softmin": softmin_aggregate,
        "quantile": quantile_aggregate,
        "free_energy": free_energy_aggregate,
    }
    if name not in table:
        raise ValueError(f"unknown aggregator {name!r}; choose from {sorted(table)}")
    fn = table[name]
    return lambda u: fn(u, **kwargs)


def variational_free_energy(energy: Tensor, log_q: Tensor, lam: float = 1.0) -> Tensor:
    """``F_q = E_q[U + (1/lam) log q]`` for samples from a model with tractable density.

    ``energy`` and ``log_q`` are ``[B, K]``; returns ``[B]``.
    """
    return (energy + log_q / lam).mean(dim=1)


class EnergyEnsembleObjective:
    """``R(S) = -aggregate_k U_design(X_k, S, T)``."""

    def __init__(self, energy: CompositeEnergy, aggregator: Aggregator = mean_aggregate) -> None:
        self.energy = energy
        self.aggregator = aggregator

    @torch.no_grad()
    def __call__(
        self, system: MolecularSystem, sequence: SequenceBatch, conformations: ConformationBatch
    ) -> ObjectiveResult:
        b, k = conformations.batch_size, conformations.n_samples
        out = self.energy.evaluate(system, conformations.flatten(), sequence.repeat_interleave(k))
        energies = out.total.reshape(b, k)
        terms = {name: v.reshape(b, k) for name, v in out.terms.items()}
        return ObjectiveResult(reward=-self.aggregator(energies), energies=energies, terms=terms)


class SampledEnsembleReward:
    """:class:`~cinder.types.RewardModel`: sample ``n_samples`` conformations per sequence from one
    conformation model, then reduce their energies with an :class:`EnergyEnsembleObjective`."""

    def __init__(
        self,
        conformations: ConformationModel,
        objective: EnergyEnsembleObjective,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        self.conformations = conformations
        self.objective = objective
        self.n_samples = n_samples
        self.generator = generator

    def __call__(self, system: MolecularSystem, sequence: SequenceBatch) -> ObjectiveResult:
        confs = self.conformations.sample(system, sequence, self.n_samples, self.generator)
        return self.objective(system, sequence, confs)
