"""Analytic energies used to validate samplers, flows and the sequence optimiser on CPU.

These never touch a protein. They exist so that every mathematical claim of the framework can
be checked against a closed form before a real energy function is involved.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from cinder.energy.base import autograd_gradient
from cinder.systems.specification import MolecularSystem
from cinder.types import N_AA, SequenceBatch


class _ToyBase:
    """Provides ``gradient`` via autograd from a ``_u(x, tokens)`` implementation."""

    def _u(self, x: Tensor, tokens: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def energy(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        return self._u(coordinates, sequence.tokens)

    def gradient(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        return autograd_gradient(lambda x: self._u(x, sequence.tokens), coordinates)


class HarmonicEnergy(_ToyBase):
    """``U(x) = k/2 * |x - center|^2`` (sequence independent)."""

    def __init__(self, k: float = 1.0, center: Tensor | float = 0.0) -> None:
        self.k = k
        self.center = torch.as_tensor(center, dtype=torch.float32)

    def _u(self, x: Tensor, tokens: Tensor) -> Tensor:
        d = x - self.center.to(x)
        return 0.5 * self.k * d.reshape(d.shape[0], -1).pow(2).sum(-1)


class DoubleWellEnergy(_ToyBase):
    """``U(x) = a (x0^2 - b^2)^2 + 1/2 sum_{i>0} x_i^2`` -- wells at ``x0 = +-b``."""

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self.a, self.b = a, b

    def _u(self, x: Tensor, tokens: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1)
        u = self.a * (x[:, 0] ** 2 - self.b**2) ** 2
        if x.shape[1] > 1:
            u = u + 0.5 * x[:, 1:].pow(2).sum(-1)
        return u


class GaussianMixtureEnergy(_ToyBase):
    """``U(x) = -log sum_m w_m N(x; mu_m, sigma^2 I)`` (isotropic components)."""

    def __init__(self, means: Tensor, sigma: float = 1.0, weights: Tensor | None = None) -> None:
        self.means = means.float()
        self.sigma = sigma
        w = torch.ones(len(means)) if weights is None else weights.float()
        self.log_w = (w / w.sum()).log()

    def log_density(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1)
        d = x.shape[1]
        sq = (x[:, None, :] - self.means.to(x)[None]).pow(2).sum(-1)
        lp = -0.5 * sq / self.sigma**2 - d * (math.log(self.sigma) + 0.5 * math.log(2 * math.pi))
        return torch.logsumexp(lp + self.log_w.to(x), dim=-1)

    def _u(self, x: Tensor, tokens: Tensor) -> Tensor:
        return -self.log_density(x)


class ToySequenceEnergy(_ToyBase):
    """A sequence-dependent energy with a closed-form ensemble mean.

    ``U(x, S) = sum_i c[i, S_i] + 1/2 |x - m(S)|^2 + 1/2 |x - t|^2`` with
    ``m(S) = sum_i w[S_i]``. Completing the square,
    ``U = sum_i c + |x - (m+t)/2|^2 + |m - t|^2 / 4``, so under ``p ∝ exp(-lam U)``
    ``x ~ N((m+t)/2, I / (2 lam))`` and ``E[U] = sum_i c + d/(2 lam) + |m-t|^2 / 4``.
    The optimum over sequences can therefore be found by brute force and checked against the
    policy-gradient result.
    """

    def __init__(self, n_positions: int, dim: int = 2, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.n_positions, self.dim = n_positions, dim
        self.c = torch.randn(n_positions, N_AA, generator=g)
        self.w = 0.5 * torch.randn(N_AA, dim, generator=g)
        self.t = torch.randn(dim, generator=g)

    def _m(self, tokens: Tensor) -> Tensor:
        return self.w.to(tokens.device)[tokens].sum(1)

    def _u(self, x: Tensor, tokens: Tensor) -> Tensor:
        c = self.c.to(x)
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        seq_term = c[pos[None, :], tokens].sum(1)
        m = self._m(tokens).to(x)
        t = self.t.to(x)
        return seq_term + 0.5 * (x - m).pow(2).sum(-1) + 0.5 * (x - t).pow(2).sum(-1)

    def expected_energy(self, tokens: Tensor, lam: float = 1.0) -> Tensor:
        """Exact ``E_{x ~ p_lam}[U(x, S)]`` for each row of ``tokens`` ``[B, L]``."""
        pos = torch.arange(tokens.shape[1])
        seq_term = self.c[pos[None, :], tokens].sum(1)
        m = self._m(tokens)
        return seq_term + self.dim / (2 * lam) + 0.25 * (m - self.t).pow(2).sum(-1)

    def equilibrium_mean(self, tokens: Tensor) -> Tensor:
        return 0.5 * (self._m(tokens) + self.t)
