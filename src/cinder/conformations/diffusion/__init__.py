"""Denoising-diffusion conformation model (variance-preserving, epsilon prediction).

Forward process ``X_t = sqrt(abar_t) X_0 + sqrt(1 - abar_t) eps``; a network predicts ``eps`` and
the score is ``s_theta = -eps_theta / sqrt(1 - abar_t)``. Sampling runs the ancestral (DDPM)
reverse chain. Energy guidance adds ``-lambda grad_X U`` to the score at every step through
:class:`~cinder.conformations.guidance.EnergyGuidance`.

Caveats: the guidance gradient is evaluated at the *noisy* ``X_t``, so the guided distribution is
only approximately ``p_base exp(-lambda U)`` (exact in the limit of small noise at the final
steps); ordinary diffusion has no tractable normalised density, so ``log_prob`` raises
:class:`~cinder.types.UnsupportedOperation`. Toy / low-dimensional coordinates only.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cinder.conformations.guidance import EnergyGuidance
from cinder.conformations.networks import ConditionalMLP
from cinder.conformations.normalizing_flow.flow import sequence_context
from cinder.systems.specification import MolecularSystem
from cinder.types import ConformationBatch, SequenceBatch, UnsupportedOperation


class DiffusionConformationModel:
    """:class:`~cinder.types.ConformationModel` backed by a learned denoiser."""

    def __init__(
        self,
        network: ConditionalMLP,
        n_steps: int = 100,
        beta_start: float = 1e-3,
        beta_end: float = 0.1,
        guidance: EnergyGuidance | None = None,
    ) -> None:
        if n_steps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("need n_steps >= 2 and 0 < beta_start < beta_end < 1")
        self.network = network
        self.n_steps = n_steps
        self.guidance = guidance
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def loss(self, x0: Tensor, context: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Denoising loss ``E |eps - eps_theta(X_t, t, S)|^2`` for clean data ``x0`` ``[N, D]``."""
        t = torch.randint(0, self.n_steps, (x0.shape[0],), generator=generator)
        eps = torch.randn(x0.shape, generator=generator)
        ab = self.alpha_bar[t][:, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        pred = self.network(xt, t.float() / self.n_steps, context)
        return (pred - eps).pow(2).sum(-1).mean()

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
        for t in reversed(range(self.n_steps)):
            tt = torch.full((n,), t / self.n_steps)
            eps = self.network(x, tt, context)
            score = -eps / (1 - self.alpha_bar[t]).sqrt()
            if self.guidance is not None:
                shaped = x.reshape(n, *system.coord_shape)
                with torch.enable_grad():
                    g = self.guidance.score(system, shaped, seqs, t / self.n_steps)
                score = score + g.reshape(n, -1)
            x = (x + self.betas[t] * score) / self.alphas[t].sqrt()
            if t > 0:
                x = x + self.betas[t].sqrt() * torch.randn(x.shape, generator=generator)
        b = len(sequence)
        return ConformationBatch(x.reshape(b, n_samples, *system.coord_shape))

    def log_prob(
        self, system: MolecularSystem, sequence: SequenceBatch, coordinates: Tensor
    ) -> Tensor | None:
        raise UnsupportedOperation("diffusion models do not provide a normalised density")
