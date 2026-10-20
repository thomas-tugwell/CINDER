"""Backend-independent harmonic positional restraint ``U = k/2 * sum |m * (X - X_ref)|^2``."""

from __future__ import annotations

from torch import Tensor

from cinder.energy.base import autograd_gradient
from cinder.systems.specification import MolecularSystem
from cinder.types import SequenceBatch


class HarmonicRestraint:
    """Tether coordinates to ``system.reference_coordinates`` (optionally only where ``mask``).

    ``mask`` broadcasts against the coordinates (e.g. ``[n_atoms, 1]`` per-atom booleans).
    """

    def __init__(self, k: float = 1.0, mask: Tensor | None = None) -> None:
        self.k = k
        self.mask = mask

    def energy(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        d = coordinates - system.reference_coordinates.to(coordinates)
        if self.mask is not None:
            d = d * self.mask.to(coordinates)
        return 0.5 * self.k * d.reshape(d.shape[0], -1).pow(2).sum(-1)

    def gradient(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        return autograd_gradient(lambda x: self.energy(system, x, sequence), coordinates)
