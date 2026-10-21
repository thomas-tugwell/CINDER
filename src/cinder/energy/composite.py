"""Weighted sum of independently configurable energy terms (the ``U_design`` composition)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from cinder.systems.specification import MolecularSystem
from cinder.types import EnergyModel, EnergyOutput, SequenceBatch


@dataclass
class EnergyTerm:
    name: str
    model: EnergyModel
    weight: float = 1.0


class CompositeEnergy:
    """``U_design = sum_k w_k U_k``. Each term is logged separately via :meth:`evaluate`."""

    def __init__(self, terms: list[EnergyTerm]) -> None:
        if not terms:
            raise ValueError("CompositeEnergy needs at least one term")
        names = [t.name for t in terms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate energy term names: {names}")
        self.terms = terms

    def evaluate(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> EnergyOutput:
        parts = {
            t.name: t.weight * t.model.energy(system, coordinates, sequence) for t in self.terms
        }
        total = sum(parts.values())
        return EnergyOutput(total=total, terms=parts)  # type: ignore[arg-type]

    def energy(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        return self.evaluate(system, coordinates, sequence).total

    def gradient(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        # Sum of per-term gradients keeps backend-native gradients where available.
        grads = [t.weight * t.model.gradient(system, coordinates, sequence) for t in self.terms]
        return sum(grads)  # type: ignore[return-value]


__all__ = ["CompositeEnergy", "EnergyTerm"]
