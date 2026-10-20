"""Molecular system specification: which chains are target/binder and what is designable."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from cinder.systems.design_space import DesignSpace


@dataclass
class MolecularSystem:
    """A fixed target ``T`` plus a binder whose sequence ``S`` is being designed.

    ``reference_coordinates`` is the starting pose with shape ``coord_shape``; samplers use it
    as the initial state / centre of local perturbations. For toy systems it is a plain vector.
    """

    name: str
    design_space: DesignSpace
    reference_coordinates: Tensor
    target_chains: list[str] = field(default_factory=list)
    binder_chains: list[str] = field(default_factory=list)
    structure_path: Path | None = None
    structure_sha256: str | None = None
    move_mask: Tensor | None = None  # broadcastable to coord_shape; 0 freezes coordinates
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def coord_shape(self) -> tuple[int, ...]:
        return tuple(self.reference_coordinates.shape)

    @property
    def binder_length(self) -> int:
        return self.design_space.length

    def to(self, device: torch.device | str) -> MolecularSystem:
        self.reference_coordinates = self.reference_coordinates.to(device)
        if self.move_mask is not None:
            self.move_mask = self.move_mask.to(device)
        return self
