"""Energy helpers shared by implementations."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def autograd_gradient(fn: Callable[[Tensor], Tensor], coordinates: Tensor) -> Tensor:
    """``d sum(fn(X)) / dX``; per-sample energies are independent so this is per-sample dU/dX."""
    with torch.enable_grad():
        x = coordinates.detach().clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(fn(x).sum(), x)
    return grad.detach()


def finite_difference_gradient(
    fn: Callable[[Tensor], Tensor], coordinates: Tensor, eps: float = 1e-4, indices=None
) -> Tensor:
    """Central finite-difference gradient of a *single-sample* ``fn`` (for gradient checks).

    ``indices`` optionally restricts to a list of flat coordinate indices (others are zero).
    """
    flat = coordinates.detach().clone().reshape(-1)
    grad = torch.zeros_like(flat)
    idx = range(flat.numel()) if indices is None else indices
    for i in idx:
        plus, minus = flat.clone(), flat.clone()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (fn(plus.reshape(coordinates.shape)) - fn(minus.reshape(coordinates.shape))) / (
            2 * eps
        )
    return grad.reshape(coordinates.shape)
