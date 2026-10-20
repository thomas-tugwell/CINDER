"""Reusable energy guidance: ``score_guided = score_base - lambda * grad_X U``.

This is the single place where learned scores (or velocities) are combined with energy
gradients, so diffusion, flow-matching and Langevin samplers share it rather than each
re-implementing it. With ``base=None`` the base score is zero in the *energy-only* sense, i.e.
the target is ``p_U ∝ exp(-lambda U)``.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from cinder.systems.specification import MolecularSystem
from cinder.types import EnergyModel, ScoreModel, SequenceBatch

StrengthSchedule = Callable[[float | None], float]


def constant_schedule(value: float) -> StrengthSchedule:
    return lambda t: value


def linear_schedule(start: float, end: float) -> StrengthSchedule:
    """Ramp guidance strength linearly in sampler progress ``t ∈ [0, 1]``."""

    def f(t: float | None) -> float:
        tt = 0.0 if t is None else min(max(t, 0.0), 1.0)
        return start + (end - start) * tt

    return f


class EnergyGuidance:
    """Implements :class:`~cinder.types.ScoreModel`: ``s_base(X,t) - lambda(t) grad U(X)``."""

    def __init__(
        self,
        energy_model: EnergyModel,
        strength: float | StrengthSchedule = 1.0,
        base_model: ScoreModel | None = None,
    ) -> None:
        self.energy_model = energy_model
        self.strength: StrengthSchedule = (
            constant_schedule(strength) if isinstance(strength, (int, float)) else strength
        )
        self.base_model = base_model

    def score(
        self,
        system: MolecularSystem,
        coordinates: Tensor,
        sequence: SequenceBatch,
        t: float | None = None,
    ) -> Tensor:
        s = -self.strength(t) * self.energy_model.gradient(system, coordinates, sequence)
        if self.base_model is not None:
            s = s + self.base_model.score(system, coordinates, sequence, t)
        return s


def energy_guided_velocity(
    base_velocity: Tensor, energy_gradient: Tensor, strength: float
) -> Tensor:
    """Flow-matching guidance ``v_guided = v_theta - lambda(t) grad_X U`` (heuristic: it
    matches the score-guidance form but is not exact for an arbitrary learned velocity field)."""
    return base_velocity - strength * energy_gradient


def energy_guided_score(base_score: Tensor, energy_gradient: Tensor, strength: float) -> Tensor:
    """Diffusion guidance ``s_guided = s_theta - lambda grad_X U``."""
    return base_score - strength * energy_gradient
