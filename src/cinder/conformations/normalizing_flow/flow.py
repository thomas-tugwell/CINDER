"""Small conditional RealNVP flow ``q_theta(X | S, T)`` for low-dimensional toy coordinates.

The sequence enters through a context vector (one-hot of the binder sequence, flattened).
This is a *prototype for validating the variational objective* on analytic landscapes -- it is
deliberately not an equivariant protein model (see ``docs/models.md``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from cinder.systems.specification import MolecularSystem
from cinder.types import N_AA, ConformationBatch, SequenceBatch

_LOG_2PI = math.log(2 * math.pi)


def sequence_context(sequence: SequenceBatch) -> Tensor:
    """One-hot flatten: ``[N, L] -> [N, L*20]``."""
    oh = torch.nn.functional.one_hot(sequence.tokens, N_AA).float()
    return oh.reshape(oh.shape[0], -1)


class _Coupling(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden: int, parity: int, scale_cap: float):
        super().__init__()
        mask = torch.zeros(dim)
        mask[parity::2] = 1.0
        self.register_buffer("mask", mask)
        self.scale_cap = scale_cap
        self.net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2 * dim),
        )  # fmt: skip
        last: nn.Linear = self.net[-1]  # type: ignore[assignment]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _st(self, x: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
        m: Tensor = self.mask  # type: ignore[assignment]
        h = self.net(torch.cat([x * m, ctx], -1))
        s, t = h.chunk(2, -1)
        s = self.scale_cap * torch.tanh(s / self.scale_cap) * (1 - m)
        return s, t * (1 - m)

    def forward(self, x: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
        """Data -> latent. Returns (z, log|det dz/dx|)."""
        s, t = self._st(x, ctx)
        return x * torch.exp(s) + t, s.sum(-1)

    def inverse(self, z: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
        """Latent -> data. Returns (x, log|det dx/dz|)."""
        m: Tensor = self.mask  # type: ignore[assignment]
        s, t = self._st(z * m, ctx)  # conditioner only reads the unchanged (masked) half
        return (z - t) * torch.exp(-s), -s.sum(-1)


class ConditionalFlow(nn.Module):
    """Stack of affine couplings with a standard-normal base. ``dim >= 2``."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        n_layers: int = 6,
        hidden: int = 64,
        scale_cap: float = 3.0,
    ) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("ConditionalFlow requires dim >= 2")
        self.dim = dim
        self.layers = nn.ModuleList(
            _Coupling(dim, context_dim, hidden, i % 2, scale_cap) for i in range(n_layers)
        )

    def _base_log_prob(self, z: Tensor) -> Tensor:
        return -0.5 * (z.pow(2).sum(-1) + self.dim * _LOG_2PI)

    def log_prob(self, x: Tensor, ctx: Tensor) -> Tensor:
        z, total = x, torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            z, ld = layer(z, ctx)
            total = total + ld
        return self._base_log_prob(z) + total

    def sample_with_log_prob(
        self, ctx: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Reparameterised samples and their exact log density; differentiable in theta."""
        z = torch.randn(ctx.shape[0], self.dim, generator=generator, device=ctx.device)
        log_q = self._base_log_prob(z)
        x = z
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x, ctx)  # type: ignore[operator]
            log_q = log_q - ld
        return x, log_q


class NormalizingFlowSampler:
    """:class:`~cinder.types.ConformationModel` backed by a :class:`ConditionalFlow`."""

    def __init__(self, flow: ConditionalFlow) -> None:
        self.flow = flow

    @torch.no_grad()
    def sample(
        self,
        system: MolecularSystem,
        sequence: SequenceBatch,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> ConformationBatch:
        ctx = sequence_context(sequence.repeat_interleave(n_samples))
        x, log_q = self.flow.sample_with_log_prob(ctx, generator)
        b = len(sequence)
        return ConformationBatch(x.reshape(b, n_samples, -1), log_q.reshape(b, n_samples))

    def log_prob(
        self, system: MolecularSystem, sequence: SequenceBatch, coordinates: Tensor
    ) -> Tensor:
        return self.flow.log_prob(
            coordinates.reshape(len(sequence), -1), sequence_context(sequence)
        )
