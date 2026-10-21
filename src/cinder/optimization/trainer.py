"""Sequence-policy training loop: sample S -> sample X -> U -> R -> REINFORCE update.

The trainer depends only on the Protocols in :mod:`cinder.types`; it never imports a concrete
conformation or energy model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cinder.experiments.checkpoint import save_checkpoint
from cinder.experiments.logging import CsvLogger
from cinder.optimization.reinforce import reinforce_loss
from cinder.sequence.base import TrainablePolicy
from cinder.systems.specification import MolecularSystem
from cinder.types import (
    AMINO_ACIDS,
    RewardModel,
    SequenceBatch,
)

log = logging.getLogger("cinder.trainer")


@dataclass
class TrainerSettings:
    batch_size: int = 32
    learning_rate: float = 0.05
    iterations: int = 200
    entropy_weight: float = 0.01
    normalize_advantage: bool = False
    checkpoint_every: int = 50
    log_probs_every: int = 10


@dataclass
class TrainResult:
    best_sequence: str
    best_reward: float
    history: list[dict[str, float]] = field(default_factory=list)


class SequenceTrainer:
    def __init__(
        self,
        system: MolecularSystem,
        policy: TrainablePolicy,
        reward: RewardModel,
        settings: TrainerSettings,
        run_dir: Path | None = None,
        generator: torch.Generator | None = None,
        on_iteration: Callable[[int, dict[str, float]], None] | None = None,
    ) -> None:
        self.system, self.policy = system, policy
        self.reward = reward
        self.s = settings
        self.run_dir = run_dir
        self.generator = generator
        self.on_iteration = on_iteration
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)
        self._metrics = CsvLogger(run_dir / "metrics.csv") if run_dir else None
        self._seqs = CsvLogger(run_dir / "sequences.csv") if run_dir else None
        self._probs = CsvLogger(run_dir / "probabilities.csv") if run_dir else None

    def step(self, iteration: int) -> tuple[dict[str, float], list[str], torch.Tensor]:
        sample = self.policy.sample(self.s.batch_size, self.generator)
        full = SequenceBatch(self.system.design_space.assemble(sample.tokens))
        result = self.reward(self.system, full)
        entropy = self.policy.entropy()
        loss = reinforce_loss(
            sample.log_prob,
            result.reward,
            entropy,
            self.s.entropy_weight,
            normalize_advantage=self.s.normalize_advantage,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        reward = result.reward.detach()
        strings = full.strings()
        metrics: dict[str, float] = {
            "iteration": iteration,
            "loss": float(loss.detach()),
            "reward_mean": float(reward.mean()),
            "reward_max": float(reward.max()),
            "reward_var": float(reward.var(unbiased=False)),
            "policy_entropy": float(entropy.detach()),
            "n_unique_sequences": len(set(strings)),
        }
        for name, v in result.terms.items():
            metrics[f"term/{name}"] = float(v.mean())
        return metrics, strings, reward

    def train(self) -> TrainResult:
        best_seq, best_reward = "", float("-inf")
        history: list[dict[str, float]] = []
        for it in range(self.s.iterations):
            metrics, strings, reward = self.step(it)
            i = int(reward.argmax())
            if float(reward[i]) > best_reward:
                best_reward, best_seq = float(reward[i]), strings[i]
            metrics["best_reward"] = best_reward
            history.append(metrics)
            if self._metrics:
                self._metrics.write({**metrics, "best_sequence": best_seq})
            if self._seqs:
                for seq, r in zip(strings, reward.tolist(), strict=True):
                    self._seqs.write({"iteration": it, "sequence": seq, "reward": r})
            if self._probs and it % self.s.log_probs_every == 0:
                self._log_probs(it)
            if self.run_dir and self.s.checkpoint_every and (it + 1) % self.s.checkpoint_every == 0:
                self._checkpoint(it + 1, best_seq, best_reward)
            if self.on_iteration:
                self.on_iteration(it, metrics)
            log.info(
                "it=%d R=%.4f best=%.4f H=%.3f uniq=%d",
                it, metrics["reward_mean"], best_reward, metrics["policy_entropy"],
                metrics["n_unique_sequences"],
            )  # fmt: skip
        if self.run_dir:
            self._checkpoint(self.s.iterations, best_seq, best_reward, name="final")
        return TrainResult(best_seq, best_reward, history)

    def _checkpoint(self, it: int, seq: str, reward: float, name: str | None = None) -> None:
        assert self.run_dir is not None
        path = self.run_dir / "checkpoints" / f"{name or f'iter_{it:06d}'}.pt"
        save_checkpoint(
            path, self.policy, self.optimizer, it, {"best_sequence": seq, "best_reward": reward}
        )

    def _log_probs(self, it: int) -> None:
        assert self._probs is not None
        probs = self.policy.probs()
        for pos_i, pos in enumerate(self.system.design_space.positions):
            row: dict[str, float | int] = {"iteration": it, "position": pos}
            row.update({aa: float(probs[pos_i, j]) for j, aa in enumerate(AMINO_ACIDS)})
            self._probs.write(row)


__all__ = ["SequenceTrainer", "TrainResult", "TrainerSettings"]
