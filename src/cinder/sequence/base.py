"""Sequence-policy interfaces.

:class:`~cinder.types.SequencePolicy` is the minimal contract (sample, log-probability, entropy).
:class:`TrainablePolicy` adds what the trainer and loggers need: learnable parameters, per-position
amino-acid probabilities for logging, and a greedy "consensus" sequence.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn

from cinder.types import SequencePolicy, SequenceSample


@runtime_checkable
class TrainablePolicy(SequencePolicy, Protocol):
    """A :class:`SequencePolicy` the sequence trainer can train."""

    def parameters(self) -> Iterator[nn.Parameter]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> Any: ...

    @property
    def n_positions(self) -> int: ...

    def probs(self) -> Tensor:
        """Per-position amino-acid probabilities ``[P, 20]`` (detached) for logging."""
        ...

    def argmax_tokens(self) -> Tensor:
        """Greedy consensus tokens ``[P]``."""
        ...


__all__ = ["SequencePolicy", "SequenceSample", "TrainablePolicy"]
