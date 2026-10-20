"""Diffusion conformation model -- EXPERIMENTAL STUB (Milestone 7).

Forward process ``X_t = alpha_t X_0 + sigma_t eps``; a learned score ``s_theta(X_t, t, S, T)``
is combined with TMol gradients through :class:`~cinder.conformations.guidance.EnergyGuidance`
(``s_guided = s_theta - lambda grad_X U``). Ordinary diffusion has no tractable normalised
density, so ``log_prob`` raises :class:`~cinder.types.UnsupportedOperation`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cinder.systems.specification import MolecularSystem
from cinder.types import ConformationBatch, SequenceBatch, UnsupportedOperation


class DiffusionConformationModel:
    """Placeholder satisfying the ``ConformationModel`` protocol. Not implemented."""

    experimental = True

    def sample(
        self,
        system: MolecularSystem,
        sequence: SequenceBatch,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> ConformationBatch:
        raise NotImplementedError("diffusion sampling is a planned milestone (see docs/models.md)")

    def log_prob(
        self, system: MolecularSystem, sequence: SequenceBatch, coordinates: Tensor
    ) -> Tensor | None:
        raise UnsupportedOperation("diffusion models do not provide a normalised density")
