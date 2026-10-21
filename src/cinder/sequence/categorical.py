"""Position-factorised categorical policy ``q_phi(S|T) = prod_i q_phi(S_i|T)``."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cinder.types import N_AA, SequenceSample

_NEG = -1e9


class CategoricalPolicy(nn.Module):
    """Trainable logits ``[P, 20]`` with a hard boolean ``mask`` ``[P, 20]``.

    Masked amino acids have exactly zero probability, are never sampled and contribute zero
    entropy. ``init_logits`` allows native-AA or uniform initialisation.
    """

    def __init__(self, mask: Tensor, init_logits: Tensor | None = None) -> None:
        super().__init__()
        if mask.ndim != 2 or mask.shape[1] != N_AA:
            raise ValueError(f"mask must be [P, {N_AA}]")
        if not mask.any(dim=1).all():
            raise ValueError("every position needs at least one allowed amino acid")
        self.register_buffer("mask", mask.bool())
        logits = torch.zeros(mask.shape) if init_logits is None else init_logits.clone().float()
        if logits.shape != mask.shape:
            raise ValueError("init_logits must match mask shape")
        self.logits = nn.Parameter(logits)

    @property
    def n_positions(self) -> int:
        return int(self.mask.shape[0])  # type: ignore[index]

    def _masked_logits(self) -> Tensor:
        return self.logits.masked_fill(~self.mask, _NEG)  # type: ignore[operator]

    def log_probs(self) -> Tensor:
        """``[P, 20]`` log-probabilities (``-inf``-like where masked)."""
        return torch.log_softmax(self._masked_logits(), dim=-1)

    def probs(self) -> Tensor:
        return self.log_probs().exp().detach()

    def marginals(self) -> Tensor:
        """Per-position probabilities ``[P, 20]`` *with* gradients (masked entries are 0); the
        ``q`` in exact expected-energy objectives."""
        return self.log_probs().exp()

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> SequenceSample:
        with torch.no_grad():
            p = self.log_probs().exp()
            flat = torch.multinomial(
                p.repeat(batch_size, 1), 1, replacement=True, generator=generator
            )
            tokens = flat.reshape(batch_size, self.n_positions)
        return SequenceSample(tokens=tokens, log_prob=self.log_prob(tokens))

    def log_prob(self, tokens: Tensor) -> Tensor:
        """``[B]`` log q(S) for ``tokens`` ``[B, P]`` (differentiable)."""
        lp = self.log_probs()
        pos = torch.arange(self.n_positions, device=tokens.device)
        return lp[pos[None, :], tokens].sum(-1)

    def entropy(self) -> Tensor:
        """Entropy of the factorised distribution (sum over positions), differentiable."""
        lp = self.log_probs()
        p = lp.exp()
        return -(p * lp.masked_fill(~self.mask, 0.0)).sum()  # type: ignore[operator]

    def argmax_tokens(self) -> Tensor:
        return self._masked_logits().argmax(dim=-1)
