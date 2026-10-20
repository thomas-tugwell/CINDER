"""Fixed-size structural representations, kept separate from energy evaluation.

Sequence mutations change the atom count (Ala -> Trp adds atoms), so a raw ``[n_atoms, 3]``
tensor cannot be the common currency of a *sequence-conditioned* generative model: its shape
would change with ``S``. A fixed-size representation avoids this. Candidates (see
``docs/models.md``): atom14 coordinates + mask, residue rigid frames, backbone/side-chain
torsions. For the MVP, toy systems use plain coordinate vectors and the TMol adapter keeps
TMol's own pose representation; ``Atom14Layout`` below is the first fixed-size container.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

ATOM14_SLOTS = 14


@dataclass
class Atom14Layout:
    """Fixed-size per-residue coordinates: ``coords`` ``[L, 14, 3]``, ``mask`` ``[L, 14]``.

    Slots beyond a residue type's real atoms are masked out, so the tensor shape is
    independent of the sequence.
    """

    coords: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if self.coords.shape[-2:] != (ATOM14_SLOTS, 3):
            raise ValueError(f"coords must be [..., {ATOM14_SLOTS}, 3]")
        if self.mask.shape != self.coords.shape[:-1]:
            raise ValueError("mask must match coords[..., :3] leading shape")

    def flat_atoms(self) -> Tensor:
        """Real (unmasked) atoms as ``[n_atoms, 3]`` -- the variable-size view."""
        return self.coords[self.mask.bool()]

    @classmethod
    def empty(cls, n_residues: int) -> Atom14Layout:
        return cls(torch.zeros(n_residues, ATOM14_SLOTS, 3), torch.zeros(n_residues, ATOM14_SLOTS))
