"""Which binder positions are designable, and which amino acids each may take."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from cinder.types import AA_TO_INDEX, AMINO_ACIDS, N_AA, encode_sequence


@dataclass
class DesignSpace:
    """Design constraints over a binder of ``length`` residues.

    ``positions`` are 0-based indices into the binder sequence. Every other residue is held at
    its ``native`` identity. ``allowed`` maps a position to its permitted amino-acid letters
    (default: all 20); ``forbidden`` removes letters at every designable position.
    """

    length: int
    native: str
    positions: list[int]
    allowed: dict[int, str] = field(default_factory=dict)
    forbidden: str = ""

    def __post_init__(self) -> None:
        if len(self.native) != self.length:
            raise ValueError(f"native has length {len(self.native)} but length={self.length}")
        encode_sequence(self.native)
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("duplicate design positions")
        bad = [p for p in self.positions if not 0 <= p < self.length]
        if bad:
            raise ValueError(f"design positions out of range: {bad}")
        for p, letters in self.allowed.items():
            if p not in self.positions:
                raise ValueError(f"allowed-AA entry for non-design position {p}")
            for c in letters:
                if c not in AA_TO_INDEX:
                    raise ValueError(f"unknown amino acid {c!r} at position {p}")
        for c in self.forbidden:
            if c not in AA_TO_INDEX:
                raise ValueError(f"unknown forbidden amino acid {c!r}")
        for i, p in enumerate(self.positions):
            if not self.mask()[i].any():
                raise ValueError(f"no amino acids allowed at position {p}")

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    def mask(self) -> Tensor:
        """Boolean ``[P, 20]`` mask of permitted amino acids at each designable position."""
        m = torch.zeros(self.n_positions, N_AA, dtype=torch.bool)
        for i, p in enumerate(self.positions):
            letters = self.allowed.get(p, AMINO_ACIDS)
            for c in letters:
                if c not in self.forbidden:
                    m[i, AA_TO_INDEX[c]] = True
        return m

    def native_tokens(self) -> Tensor:
        """``[L]`` native binder tokens."""
        return encode_sequence(self.native)

    def native_design_tokens(self) -> Tensor:
        """``[P]`` native tokens at designable positions."""
        return self.native_tokens()[torch.tensor(self.positions, dtype=torch.long)]

    def assemble(self, design_tokens: Tensor) -> Tensor:
        """Insert ``[B, P]`` designed tokens into the native scaffold -> ``[B, L]``."""
        if design_tokens.ndim != 2 or design_tokens.shape[1] != self.n_positions:
            raise ValueError(f"expected [B, {self.n_positions}], got {tuple(design_tokens.shape)}")
        full = self.native_tokens().to(design_tokens.device).repeat(design_tokens.shape[0], 1)
        full[:, torch.tensor(self.positions, dtype=torch.long, device=design_tokens.device)] = (
            design_tokens
        )
        return full
