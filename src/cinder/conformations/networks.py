"""Small networks shared by the learned conformation models (toy / low-dimensional coordinates).

These are deliberately *not* equivariant protein architectures: they let the diffusion and
flow-matching samplers be validated against closed forms on CPU (see ``docs/models.md``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConditionalMLP(nn.Module):
    """``f(x, t, context) -> R^dim``: the denoiser (diffusion) or velocity field (flow matching).

    ``t`` is a ``[N]`` tensor in ``[0, 1]``; ``context`` is a ``[N, context_dim]`` encoding of the
    sequence (see :func:`cinder.conformations.normalizing_flow.sequence_context`).
    """

    def __init__(self, dim: int, context_dim: int, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        if dim < 1 or depth < 1:
            raise ValueError("dim and depth must be >= 1")
        self.dim = dim
        layers: list[nn.Module] = [nn.Linear(dim + 1 + context_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor, context: Tensor) -> Tensor:
        return self.net(torch.cat([x, t[:, None], context], dim=-1))
