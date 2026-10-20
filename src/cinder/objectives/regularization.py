"""Sequence-level regularisers."""

from __future__ import annotations

from torch import Tensor


def entropy_bonus(entropy: Tensor, weight: float) -> Tensor:
    """Term to *add to the loss*: ``-alpha * H[q]`` (discourages premature collapse)."""
    return -weight * entropy


def ensemble_spread_penalty(energies: Tensor, weight: float) -> Tensor:
    """Penalise large ensemble spread: ``weight * std_k(U)``, shape ``[B]``."""
    return weight * energies.std(dim=1, unbiased=False)
