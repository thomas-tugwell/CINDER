"""Optional BioEmu2 adapter boundary -- EXPERIMENTAL STUB (Milestone 8).

Intended role: ``p_base(X | S, T)``, optionally combined with TMol guidance
``p_guided ∝ p_base exp(-lambda U)``. No BioEmu calls are made here: its API must be inspected
from the *installed* version before wiring, and this module must fail gracefully if absent.
"""

from __future__ import annotations

from importlib import util

import torch
from torch import Tensor

from cinder.systems.specification import MolecularSystem
from cinder.types import ConformationBatch, SequenceBatch, UnsupportedOperation


class BackendUnavailableError(RuntimeError):
    """An optional backend is not installed."""


def bioemu_available() -> bool:
    return util.find_spec("bioemu") is not None


class BioEmu2Adapter:
    experimental = True

    def __init__(self) -> None:
        if not bioemu_available():
            raise BackendUnavailableError(
                "BioEmu is not installed. Install it separately (see its documentation) to use "
                "this adapter; the rest of cinder works without it."
            )

    def sample(
        self,
        system: MolecularSystem,
        sequence: SequenceBatch,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> ConformationBatch:
        raise NotImplementedError(
            "BioEmu2 sampling is not wired yet; inspect the installed API first"
        )

    def log_prob(
        self, system: MolecularSystem, sequence: SequenceBatch, coordinates: Tensor
    ) -> Tensor | None:
        raise UnsupportedOperation("BioEmu2 does not expose a normalised density")
