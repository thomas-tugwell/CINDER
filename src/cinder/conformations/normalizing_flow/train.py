"""Variational training: minimise ``L(theta) = E_{X~q_theta}[lambda U(X,S) + log q_theta(X|S)]``.

``L = KL(q || p_U) - log Z`` with ``p_U ∝ exp(-lambda U)``, so ``L >= -log Z`` with equality iff
``q = p_U``. This is mode-seeking (reverse KL): see ``docs/models.md`` for the consequences.
"""

from __future__ import annotations

import torch

from cinder.conformations.normalizing_flow.flow import ConditionalFlow, sequence_context
from cinder.systems.specification import MolecularSystem
from cinder.types import EnergyModel, SequenceBatch


def variational_loss(
    flow: ConditionalFlow,
    energy: EnergyModel,
    system: MolecularSystem,
    sequence: SequenceBatch,
    lam: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    x, log_q = flow.sample_with_log_prob(sequence_context(sequence), generator)
    u = energy.energy(system, x.reshape(x.shape[0], *system.coord_shape), sequence)
    return (lam * u + log_q).mean()


def train_flow(
    flow: ConditionalFlow,
    energy: EnergyModel,
    system: MolecularSystem,
    sequences: SequenceBatch,
    lam: float = 1.0,
    steps: int = 1000,
    samples_per_step: int = 256,
    lr: float = 2e-3,
    generator: torch.Generator | None = None,
) -> list[float]:
    """Train on a fixed pool of conditioning sequences (rows of ``sequences``)."""
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    history: list[float] = []
    n_seq = len(sequences)
    for _ in range(steps):
        idx = torch.randint(0, n_seq, (samples_per_step,), generator=generator)
        batch = SequenceBatch(sequences.tokens[idx])
        loss = variational_loss(flow, energy, system, batch, lam, generator)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        history.append(float(loss.detach()))
    return history
