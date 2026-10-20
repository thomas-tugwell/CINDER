"""REINFORCE (policy gradient) with a batch baseline and entropy regularisation.

``grad J = E[(R - b) grad log q(S)]``; the loss minimised is
``-mean((R - b) log q(S)) - alpha * H[q]``.
"""

from __future__ import annotations

from torch import Tensor

from cinder.objectives.regularization import entropy_bonus


def batch_baseline(reward: Tensor) -> Tensor:
    return reward.mean()


def reinforce_loss(
    log_prob: Tensor,
    reward: Tensor,
    entropy: Tensor | None = None,
    entropy_weight: float = 0.0,
    baseline: Tensor | None = None,
    normalize_advantage: bool = False,
) -> Tensor:
    """``log_prob`` ``[B]`` (carries grad), ``reward`` ``[B]`` (treated as constant)."""
    r = reward.detach()
    b = batch_baseline(r) if baseline is None else baseline.detach()
    adv = r - b
    if normalize_advantage and adv.numel() > 1:
        adv = adv / (adv.std() + 1e-8)
    loss = -(adv * log_prob).mean()
    if entropy is not None and entropy_weight != 0.0:
        loss = loss + entropy_bonus(entropy, entropy_weight)
    return loss
