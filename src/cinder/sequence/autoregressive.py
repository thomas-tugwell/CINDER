"""Autoregressive sequence policy: ``q(S) = prod_i q(S_i | S_<i)``.

Position ``i``'s logits are ``b_i + sum_{j<i} J[i, j][:, S_j]``, so earlier choices shift later
ones: the policy can represent *coupled* choices (a donor here only if an acceptor there), which a
position-independent policy cannot. With ``J = 0`` it is exactly the independent categorical policy.

``log_prob`` is exact and differentiable. The entropy of the full joint is intractable, so
``entropy`` returns a Monte-Carlo estimate: the sum of per-position conditional entropies at
prefixes sampled from the policy (an unbiased, differentiable estimator of ``H[q]``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cinder.types import N_AA, SequenceSample

_NEG = -1e9


class AutoregressivePolicy(nn.Module):
    """Learnable ``base`` ``[P, 20]`` logits and ``coupling`` ``[P, P, 20, 20]`` (only ``j < i`` is
    used), with a hard boolean ``mask`` ``[P, 20]``."""

    def __init__(
        self,
        mask: Tensor,
        init_logits: Tensor | None = None,
        entropy_samples: int = 16,
        probs_samples: int = 512,
    ) -> None:
        super().__init__()
        if mask.ndim != 2 or mask.shape[1] != N_AA:
            raise ValueError(f"mask must be [P, {N_AA}]")
        if not mask.any(dim=1).all():
            raise ValueError("every position needs at least one allowed amino acid")
        self.register_buffer("mask", mask.bool())
        p = int(mask.shape[0])
        base = torch.zeros(mask.shape) if init_logits is None else init_logits.clone().float()
        if base.shape != mask.shape:
            raise ValueError("init_logits must match mask shape")
        self.base = nn.Parameter(base)
        self.coupling = nn.Parameter(torch.zeros(p, p, N_AA, N_AA))
        self.entropy_samples = entropy_samples
        self.probs_samples = probs_samples

    @property
    def n_positions(self) -> int:
        return int(self.base.shape[0])

    def _conditional_logits(self, i: int, prefix: Tensor) -> Tensor:
        """Masked logits ``[B, 20]`` of position ``i`` given ``prefix`` ``[B, i]``."""
        logits = self.base[i].expand(prefix.shape[0], N_AA)
        if i > 0:
            j = torch.arange(i, device=prefix.device)[None, :]
            logits = logits + self.coupling[i, j, :, prefix].sum(1)  # [B, i, 20] -> [B, 20]
        return logits.masked_fill(~self.mask[i], _NEG)  # type: ignore[index]

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> SequenceSample:
        tokens = torch.zeros(batch_size, 0, dtype=torch.long)
        with torch.no_grad():
            for i in range(self.n_positions):
                p = torch.softmax(self._conditional_logits(i, tokens), dim=-1)
                nxt = torch.multinomial(p, 1, generator=generator)
                tokens = torch.cat([tokens, nxt], dim=1)
        return SequenceSample(tokens=tokens, log_prob=self.log_prob(tokens))

    def log_prob(self, tokens: Tensor) -> Tensor:
        """Exact ``[B]`` log q(S) for ``tokens`` ``[B, P]`` (differentiable)."""
        total = torch.zeros(tokens.shape[0], device=tokens.device)
        for i in range(self.n_positions):
            logp = torch.log_softmax(self._conditional_logits(i, tokens[:, :i]), dim=-1)
            total = total + logp.gather(1, tokens[:, i : i + 1]).squeeze(1)
        return total

    def entropy(self, generator: torch.Generator | None = None) -> Tensor:
        """Monte-Carlo estimate of ``H[q]`` (see module docstring); differentiable."""
        with torch.no_grad():
            prefixes = self.sample(self.entropy_samples, generator).tokens
        total = torch.zeros(())
        for i in range(self.n_positions):
            logp = torch.log_softmax(self._conditional_logits(i, prefixes[:, :i]), dim=-1)
            h = -(logp.exp() * logp.masked_fill(~self.mask[i], 0.0)).sum(-1)  # type: ignore[index]
            total = total + h.mean()
        return total

    def probs(self) -> Tensor:
        """Estimated marginal amino-acid probabilities ``[P, 20]`` (detached), from samples."""
        tokens = self.sample(self.probs_samples).tokens
        counts = torch.zeros(self.n_positions, N_AA)
        for i in range(self.n_positions):
            counts[i] = torch.bincount(tokens[:, i], minlength=N_AA).float()
        return counts / self.probs_samples

    def argmax_tokens(self) -> Tensor:
        """Greedy decoding: the most likely amino acid at each position given the earlier ones."""
        tokens = torch.zeros(1, 0, dtype=torch.long)
        with torch.no_grad():
            for i in range(self.n_positions):
                nxt = self._conditional_logits(i, tokens).argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, nxt], dim=1)
        return tokens[0]
