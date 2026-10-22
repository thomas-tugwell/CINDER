"""Generic maximum-likelihood-style training loop for the data-driven conformation models."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn

from cinder.conformations.normalizing_flow.flow import sequence_context
from cinder.types import SequenceBatch


class _HasLoss(Protocol):
    network: nn.Module

    def loss(
        self, x: Tensor, context: Tensor, generator: torch.Generator | None = ...
    ) -> Tensor: ...


def fit(
    model: _HasLoss,
    coordinates: Tensor,
    sequences: SequenceBatch,
    steps: int = 2000,
    batch_size: int = 256,
    lr: float = 2e-3,
    generator: torch.Generator | None = None,
) -> list[float]:
    """Fit ``model`` to paired data: ``coordinates`` ``[N, D]`` observed for ``sequences`` (``N``
    rows). Returns the loss history."""
    if len(coordinates) != len(sequences):
        raise ValueError("coordinates and sequences must have the same number of rows")
    context = sequence_context(sequences)
    opt = torch.optim.Adam(model.network.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        idx = torch.randint(0, len(coordinates), (batch_size,), generator=generator)
        loss = model.loss(coordinates[idx], context[idx], generator)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.network.parameters(), 10.0)
        opt.step()
        history.append(float(loss.detach()))
    return history
