"""Flow-matching conformation model -- EXPERIMENTAL STUB (Milestone 7).

``dX/dt = v_theta(X, t, S, T)``; energy guidance ``v_guided = v_theta - lambda(t) grad_X U`` is
provided by :func:`cinder.conformations.guidance.energy_guided_velocity`. An exact density is
available via the instantaneous change-of-variables formula (at ODE-solve cost), so ``log_prob``
is a candidate for later implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cinder.systems.specification import MolecularSystem
from cinder.types import ConformationBatch, SequenceBatch


class FlowMatchingConformationModel:
    """Placeholder satisfying the ``ConformationModel`` protocol. Not implemented."""

    experimental = True

    def sample(
        self,
        system: MolecularSystem,
        sequence: SequenceBatch,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> ConformationBatch:
        raise NotImplementedError("flow-matching is a planned milestone (see docs/models.md)")

    def log_prob(
        self, system: MolecularSystem, sequence: SequenceBatch, coordinates: Tensor
    ) -> Tensor | None:
        raise NotImplementedError("flow-matching log_prob is not implemented")
