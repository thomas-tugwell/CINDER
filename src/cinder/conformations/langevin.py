"""Unadjusted Langevin sampler (the analytic / energy-only baseline conformation model).

For a score ``s(X) = grad log p(X)`` the discretisation is
``X' = X + eta * s(X) + sqrt(2 eta) * eps``; with ``s = -lambda grad U`` this targets
``p_U ∝ exp(-lambda U)``. Being unadjusted (no Metropolis step) it carries an O(eta)
discretisation bias, and it provides no normalised density, so ``log_prob`` is unsupported.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cinder.conformations.guidance import EnergyGuidance, StrengthSchedule
from cinder.systems.specification import MolecularSystem
from cinder.types import (
    ConformationBatch,
    EnergyModel,
    ScoreModel,
    SequenceBatch,
    UnsupportedOperation,
)


class LangevinSampler:
    """Runs ``B*K`` independent chains and returns ``K`` end-states per sequence."""

    def __init__(
        self,
        score_model: ScoreModel,
        step_size: float = 1e-2,
        n_steps: int = 500,
        init_std: float = 1.0,
        max_grad_norm: float | None = None,
        noise_scale: float = 1.0,
    ) -> None:
        if step_size <= 0 or n_steps < 0:
            raise ValueError("step_size must be > 0 and n_steps >= 0")
        if noise_scale < 0:
            raise ValueError("noise_scale must be >= 0")
        self.score_model = score_model
        self.step_size = step_size
        self.n_steps = n_steps
        self.init_std = init_std
        self.max_grad_norm = max_grad_norm
        # 1 = Langevin; 0 = noise-free (clipped) gradient descent, a quench to a nearby minimum
        self.noise_scale = noise_scale

    @classmethod
    def from_energy(
        cls,
        energy: EnergyModel,
        strength: float | StrengthSchedule = 1.0,
        base_model: ScoreModel | None = None,
        **kwargs: float | int | None,
    ) -> LangevinSampler:
        return cls(EnergyGuidance(energy, strength, base_model), **kwargs)  # type: ignore[arg-type]

    def _clip(self, s: Tensor) -> Tensor:
        if self.max_grad_norm is None:
            return s
        norm = s.reshape(s.shape[0], -1).norm(dim=-1).clamp_min(1e-12)
        scale = (self.max_grad_norm / norm).clamp(max=1.0)
        return s * scale.reshape(-1, *([1] * (s.ndim - 1)))

    @torch.no_grad()
    def sample(
        self,
        system: MolecularSystem,
        sequence: SequenceBatch,
        n_samples: int,
        generator: torch.Generator | None = None,
        initial_coordinates: Tensor | None = None,
    ) -> ConformationBatch:
        """``initial_coordinates`` (``coord_shape`` or ``[B, K, *coord_shape]``) overrides the
        default start of ``reference_coordinates + init_std * noise``."""
        b = len(sequence)
        ref = system.reference_coordinates
        seq = sequence.repeat_interleave(n_samples)
        if initial_coordinates is None:
            x = ref.expand(b * n_samples, *ref.shape).clone()
        else:
            x0 = initial_coordinates.to(ref)
            if x0.shape == ref.shape:
                x0 = x0.expand(b, n_samples, *ref.shape)
            x = x0.reshape(b * n_samples, *ref.shape).clone()
        mask = None if system.move_mask is None else system.move_mask.to(x)
        if initial_coordinates is None:
            init_noise = torch.randn(x.shape, generator=generator, device=x.device)
            x = x + self.init_std * (init_noise if mask is None else init_noise * mask)
        eta = self.step_size
        for step in range(self.n_steps):
            t = step / max(self.n_steps - 1, 1)
            s = self._clip(self.score_model.score(system, x, seq, t))
            noise = torch.randn(x.shape, generator=generator, device=x.device)
            if mask is not None:
                s, noise = s * mask, noise * mask
            x = x + eta * s + self.noise_scale * (2 * eta) ** 0.5 * noise
        return ConformationBatch(x.reshape(b, n_samples, *ref.shape))

    def log_prob(
        self, system: MolecularSystem, sequence: SequenceBatch, coordinates: Tensor
    ) -> Tensor | None:
        raise UnsupportedOperation("LangevinSampler has no normalised density")
