"""Gradient-based sequence optimisation on a pairwise energy table (no REINFORCE).

Minimises ``F(q) = E_q[U] - alpha * H[q]`` over the categorical policy's logits, where ``E_q[U]`` is
the exact expectation of a :class:`~cinder.objectives.pairwise_table.PairwiseEnergyTable` (smooth in
the logits) and ``H`` the policy entropy. At temperature ``alpha`` the optimum of the *full* joint
objective is ``q ∝ exp(-U/alpha)``; for a one-body table the factorised optimum equals it exactly,
and with pair terms it is the mean-field approximation. Annealing ``alpha`` towards zero
concentrates the policy on a low-energy sequence.

Gradients are exact (autograd through the closed-form expectation), so unlike REINFORCE there is no
sampling variance. The price is the table: it approximates the energy by one- and two-body terms
on a fixed backbone (see :func:`~cinder.objectives.pairwise_table.approximation_error`), so use it
as a fast proposal generator and verify the winners with the real scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from cinder.objectives.pairwise_table import PairwiseEnergyTable
from cinder.sequence.categorical import CategoricalPolicy


@dataclass(frozen=True)
class ExpectedEnergySettings:
    iterations: int = 400
    learning_rate: float = 0.1
    alpha_start: float = 1.0  # entropy weight (temperature) at the start
    alpha_end: float = 0.01  # ... and at the end (geometric anneal)
    mutation_cost: float = (
        0.0  # energy charged per expected mutation (a prior toward the reference)
    )

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.learning_rate <= 0:
            raise ValueError("iterations >= 1 and learning_rate > 0 are required")
        if not 0 <= self.alpha_end <= self.alpha_start:
            raise ValueError("need 0 <= alpha_end <= alpha_start")
        if self.mutation_cost < 0:
            raise ValueError("mutation_cost must be >= 0")


@dataclass
class ExpectedEnergyResult:
    tokens: Tensor  # [P] greedy (argmax) residues of the final policy
    table_energy: float  # table energy of ``tokens`` (exact for the table)
    expected_energy: float  # E_q[U] at the end
    history: list[dict[str, float]] = field(default_factory=list)


def optimize_expected_energy(
    policy: CategoricalPolicy,
    table: PairwiseEnergyTable,
    settings: ExpectedEnergySettings = ExpectedEnergySettings(),  # noqa: B008
) -> ExpectedEnergyResult:
    """Train ``policy`` in place on ``table`` and return its greedy sequence."""
    if policy.n_positions != table.n_positions:
        raise ValueError(f"policy covers {policy.n_positions} positions, table {table.n_positions}")
    opt = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)
    history = []
    n = settings.iterations
    ratio = (settings.alpha_end / settings.alpha_start) if settings.alpha_start > 0 else 1.0
    for it in range(n):
        alpha = settings.alpha_start * ratio ** (it / max(n - 1, 1))
        marginals = policy.marginals()
        expected = table.expected_energy(marginals)
        entropy = policy.entropy()
        loss = expected - alpha * entropy
        if settings.mutation_cost:
            loss = loss + settings.mutation_cost * table.expected_mutations(marginals)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(
            {
                "iteration": it,
                "alpha": alpha,
                "expected_energy": float(expected.detach()),
                "entropy": float(entropy.detach()),
            }
        )
    tokens = policy.argmax_tokens()
    with torch.no_grad():
        final_expected = float(table.expected_energy(policy.marginals()))
        table_energy = float(table.energy(tokens[None])[0])
    return ExpectedEnergyResult(tokens, table_energy, final_expected, history)
