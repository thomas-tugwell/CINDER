"""Sequence-level regularisers."""

from __future__ import annotations

from torch import Tensor


def entropy_bonus(entropy: Tensor, weight: float) -> Tensor:
    """Term to *add to the loss*: ``-alpha * H[q]`` (discourages premature collapse)."""
    return -weight * entropy
