"""Core data containers and typed interfaces shared by all components.

Components communicate *only* through the containers and Protocols defined here, so that a
conformation model never imports a concrete energy model and a sequence policy knows nothing
about energies at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from cinder.systems.specification import MolecularSystem

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_AA = len(AMINO_ACIDS)
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class UnsupportedOperation(NotImplementedError):  # noqa: N818
    """Raised when a component cannot provide an optional capability (e.g. ``log_prob``)."""


def encode_sequence(seq: str) -> Tensor:
    """Encode a one-letter amino-acid string as a ``[L]`` long tensor."""
    try:
        return torch.tensor([AA_TO_INDEX[c] for c in seq], dtype=torch.long)
    except KeyError as exc:
        raise ValueError(f"unknown amino-acid letter {exc} in {seq!r}") from exc


def decode_sequence(tokens: Tensor) -> str:
    """Decode a ``[L]`` token tensor into a one-letter string."""
    return "".join(AMINO_ACIDS[int(i)] for i in tokens)


@dataclass
class SequenceBatch:
    """A batch of full binder sequences, ``tokens`` has shape ``[B, L]``."""

    tokens: Tensor

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2:
            raise ValueError(f"tokens must be [B, L], got {tuple(self.tokens.shape)}")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def strings(self) -> list[str]:
        return [decode_sequence(t) for t in self.tokens]

    def repeat_interleave(self, k: int) -> SequenceBatch:
        """``[B, L] -> [B*k, L]`` with each sequence repeated ``k`` times consecutively."""
        return SequenceBatch(self.tokens.repeat_interleave(k, dim=0))


@dataclass
class SequenceSample:
    """Sequences drawn from a policy, over the *designable* positions only.

    ``tokens``: ``[B, P]`` amino-acid indices at the P designable positions.
    ``log_prob``: ``[B]`` log q(S) (differentiable w.r.t. policy parameters).
    """

    tokens: Tensor
    log_prob: Tensor


@dataclass
class ConformationBatch:
    """``K`` conformations for each of ``B`` sequences.

    ``coordinates``: ``[B, K, *coord_shape]``.
    ``log_prob``: optional ``[B, K]`` density of each sample under the sampler.
    """

    coordinates: Tensor
    log_prob: Tensor | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.coordinates.shape[1])

    def flatten(self) -> Tensor:
        """``[B, K, ...] -> [B*K, ...]`` (sequence-major)."""
        b, k = self.coordinates.shape[:2]
        return self.coordinates.reshape(b * k, *self.coordinates.shape[2:])


@dataclass
class EnergyOutput:
    """Total energy ``[N]`` plus its named (already weighted) components, each ``[N]``."""

    total: Tensor
    terms: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class ObjectiveResult:
    """Result of reducing an ensemble to one scalar reward per sequence."""

    reward: Tensor  # [B]
    energies: Tensor  # [B, K] design energy of each ensemble member
    terms: dict[str, Tensor] = field(default_factory=dict)  # name -> [B, K]


# --------------------------------------------------------------------------- interfaces


@runtime_checkable
class EnergyModel(Protocol):
    """Deterministic energy ``U(X, S, T)``; one value per sample."""

    def energy(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        """``coordinates`` is ``[N, *coord_shape]``; ``sequence`` has N rows. Returns ``[N]``."""
        ...

    def gradient(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        """``dU/dX`` with the shape of ``coordinates``."""
        ...


@runtime_checkable
class ScoreModel(Protocol):
    """A coordinate score ``grad_X log p(X | S, T)`` (optionally time dependent)."""

    def score(
        self,
        system: MolecularSystem,
        coordinates: Tensor,
        sequence: SequenceBatch,
        t: float | None = None,
    ) -> Tensor: ...


@runtime_checkable
class ConformationModel(Protocol):
    """Access to a conformational distribution ``p(X | S, T)``."""

    def sample(
        self,
        system: MolecularSystem,
        sequence: SequenceBatch,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> ConformationBatch: ...

    def log_prob(
        self, system: MolecularSystem, sequence: SequenceBatch, coordinates: Tensor
    ) -> Tensor | None:
        """Normalized log density ``[N]``, or raise/return ``None`` if unavailable."""
        ...


@runtime_checkable
class SequencePolicy(Protocol):
    """Sequence distribution ``q_phi(S | T)`` over designable positions."""

    def sample(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> SequenceSample: ...

    def log_prob(self, tokens: Tensor) -> Tensor: ...

    def entropy(self) -> Tensor: ...


@runtime_checkable
class RewardModel(Protocol):
    """Maps a batch of full binder sequences to one reward each, via whatever ensemble the
    implementation builds for each sequence.

    This is the only thing a sequence optimiser needs from the structure/energy side. Sampling
    one shared conformation model and one energy
    (:class:`~cinder.objectives.ensemble.SampledEnsembleReward`) is one implementation; building
    a different pose per sequence (the TMol mutant ensemble) is another.
    """

    def __call__(self, system: MolecularSystem, sequence: SequenceBatch) -> ObjectiveResult: ...
