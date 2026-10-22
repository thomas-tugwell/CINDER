"""Flow-matching conformation model (conditional optimal-transport path).

Training pairs a noise draw ``x0 ~ N(0, I)`` with data ``x1`` and regresses a velocity network onto
``x1 - x0`` along ``x_t = (1 - t) x0 + t x1``. Sampling integrates ``dX/dt = v_theta(X, t, S)``
from ``t = 0`` to ``1`` with Euler steps. Energy guidance uses
:func:`~cinder.conformations.guidance.energy_guided_velocity`, ``v - lambda grad_X U``; this is a
heuristic (it is not exact for an arbitrary learned velocity field). No normalised density is
provided (it would need an ODE solve with a divergence term), so ``log_prob`` raises
:class:`~cinder.types.UnsupportedOperation`. Toy / low-dimensional coordinates only.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cinder.conformations.guidance import EnergyGuidance, energy_guided_velocity
from cinder.conformations.networks import ConditionalMLP
from cinder.conformations.normalizing_flow.flow import sequence_context
from cinder.systems.specification import MolecularSystem
from cinder.types import ConformationBatch, SequenceBatch, UnsupportedOperation


class FlowMatchingConformationModel:
    """:class:`~cinder.types.ConformationModel` backed by a learned velocity field."""

    def __init__(
        self, network: ConditionalMLP, n_steps: int = 50, guidance: EnergyGuidance | None = None
    ) -> None:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.network = network
        self.n_steps = n_steps
        self.guidance = guidance

    def loss(self, x1: Tensor, context: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Flow-matching loss ``E |v_theta(x_t, t, S) - (x1 - x0)|^2`` (data ``x1`` ``[N, D]``)."""
        t = torch.rand(x1.shape[0], generator=generator)
        x0 = torch.randn(x1.shape, generator=generator)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        return (self.network(xt, t, context) - (x1 - x0)).pow(2).sum(-1).mean()

    @torch.no_grad()
    def sample(
        self,
        system: MolecularSystem,
        sequence: SequenceBatch,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> ConformationBatch:
        context = sequence_context(sequence.repeat_interleave(n_samples))
        seqs = sequence.repeat_interleave(n_samples)
        n = context.shape[0]
        x = torch.randn(n, self.network.dim, generator=generator)
        dt = 1.0 / self.n_steps
        for k in range(self.n_steps):
            t = k * dt
            v = self.network(x, torch.full((n,), t), context)
            if self.guidance is not None:
                shaped = x.reshape(n, *system.coord_shape)
                with torch.enable_grad():
                    grad = -self.guidance.score(system, shaped, seqs, t)  # = lambda grad U
                v = energy_guided_velocity(v, grad.reshape(n, -1), 1.0)
            x = x + dt * v
        b = len(sequence)
        return ConformationBatch(x.reshape(b, n_samples, *system.coord_shape))

    def log_prob(
        self, system: MolecularSystem, sequence: SequenceBatch, coordinates: Tensor
    ) -> Tensor | None:
        raise UnsupportedOperation("flow-matching models do not provide a normalised density here")
