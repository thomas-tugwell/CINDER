"""Pairwise (one-body + two-body) sequence-energy tables and their exact expectation.

For designable positions ``i`` with amino acid ``s_i`` the table models

    U(S) ~= e0 + sum_i e1[i, s_i] + sum_{(i,j) in pairs} e2[(i,j)][s_i, s_j]

If ``q`` is a factorised distribution with marginals ``p_i(a)`` (our categorical policy), the
expected energy is **exact and smooth in ``p``**:

    E_q[U] = e0 + sum_i p_i . e1[i] + sum_pairs p_i^T e2[(i,j)] p_j

so gradients with respect to the policy logits need no REINFORCE sampling and carry no variance.
Whether ``U`` is well approximated by a pairwise table is an empirical question about the energy;
:func:`approximation_error` measures it on held-out sequences.

Tables are built from any scorer ``score(sequence) -> energy`` by inclusion-exclusion around a
reference (native) sequence (:func:`tables_from_scorer`), so this module is backend independent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from cinder.types import AA_TO_INDEX, AMINO_ACIDS, N_AA


@dataclass
class PairwiseEnergyTable:
    """``e0`` (energy of the reference), ``e1`` ``[P, 20]`` one-body changes relative to the
    reference, and for each contact pair ``m`` the indices ``pairs[m] = (i, j)`` (``i < j``, into
    the list of designable positions) and interaction corrections ``e2[m]`` ``[20, 20]``.

    Entries for the reference residue are zero, and for amino acids that were not scanned are
    zero too: keep those masked in the policy (probability 0) so they never contribute.
    """

    e0: float
    e1: Tensor
    pairs: Tensor
    e2: Tensor
    native: Tensor | None = None  # [P] residue index of the reference at each position

    def __post_init__(self) -> None:
        if self.e1.ndim != 2 or self.e1.shape[1] != N_AA:
            raise ValueError(f"e1 must be [P, {N_AA}]")
        if self.pairs.ndim != 2 or (self.pairs.numel() and self.pairs.shape[1] != 2):
            raise ValueError("pairs must be [M, 2]")
        if self.e2.shape != (self.pairs.shape[0], N_AA, N_AA):
            raise ValueError(f"e2 must be [M, {N_AA}, {N_AA}] matching pairs")
        if self.pairs.numel() and not bool((self.pairs[:, 0] < self.pairs[:, 1]).all()):
            raise ValueError("every pair needs i < j")

    @property
    def n_positions(self) -> int:
        return int(self.e1.shape[0])

    def expected_mutations(self, probs: Tensor) -> Tensor:
        """Expected number of mutated positions: ``sum_i (1 - p_i[native_i])``."""
        if self.native is None:
            raise ValueError("this table has no reference residues (native) recorded")
        pos = torch.arange(self.n_positions)
        return (1.0 - probs[pos, self.native]).sum()

    def clipped(self, cap: float) -> PairwiseEnergyTable:
        """The table that results from clipping every substitution energy at ``cap`` above the
        reference, i.e. scoring with ``f(U) = e0 + min(U - e0, cap)``.

        Raw energies are recovered from the table (single ``e1`` and joint
        ``e1_i + e1_j + e2``), clipped, and the inclusion-exclusion redone. This removes the
        failure mode where one huge clash is "cancelled" by two different neighbours at once, which
        a pairwise table then double counts and an optimiser exploits.
        """
        if cap <= 0:
            raise ValueError("cap must be > 0 (energy above the reference at which to clip)")
        e1 = self.e1.clamp(max=cap)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        joint = self.e1[i][:, :, None] + self.e1[j][:, None, :] + self.e2
        e2 = joint.clamp(max=cap) - e1[i][:, :, None] - e1[j][:, None, :]
        # entries for the reference (or unscanned) residue stay exactly 0:
        # clamp(e1_j) - 0 - clamp(e1_j)
        return PairwiseEnergyTable(self.e0, e1, self.pairs, e2, self.native)

    def energy(self, tokens: Tensor) -> Tensor:
        """Table energy of sequences ``tokens`` ``[B, P]`` (designable-position residues)."""
        b = tokens.shape[0]
        pos = torch.arange(self.n_positions, device=tokens.device)
        total = self.e0 + self.e1[pos[None, :], tokens].sum(1)
        if self.pairs.numel():
            a = tokens[:, self.pairs[:, 0]]  # [B, M]
            c = tokens[:, self.pairs[:, 1]]
            m = torch.arange(self.pairs.shape[0], device=tokens.device)
            total = total + self.e2[m[None, :], a, c].sum(1)
        return total.expand(b) if total.ndim == 0 else total

    def expected_energy(self, probs: Tensor) -> Tensor:
        """Exact ``E_q[U]`` for factorised marginals ``probs`` ``[P, 20]`` (differentiable)."""
        if probs.shape != self.e1.shape:
            raise ValueError(f"probs must be {tuple(self.e1.shape)}, got {tuple(probs.shape)}")
        total = self.e0 + (probs * self.e1).sum()
        if self.pairs.numel():
            pi, pj = probs[self.pairs[:, 0]], probs[self.pairs[:, 1]]  # [M, 20]
            total = total + torch.einsum("ma,mab,mb->", pi, self.e2, pj)
        return total


def contact_pairs(
    coords: Tensor,
    residue_of_atom: Tensor,
    heavy_atoms: Tensor,
    residues: Sequence[int],
    cutoff: float = 6.0,
) -> list[tuple[int, int]]:
    """Pairs ``(i, j)`` (``i < j``, indices into ``residues``) whose residues have any heavy
    atoms within ``cutoff`` Å; only these can interact strongly enough to need a two-body term."""
    res_idx = {r: k for k, r in enumerate(residues)}
    heavy = heavy_atoms[torch.isin(residue_of_atom[heavy_atoms], torch.tensor(list(residues)))]
    owner = residue_of_atom[heavy]
    pos = coords[heavy]
    d = torch.cdist(pos, pos)
    out: set[tuple[int, int]] = set()
    close = (d < cutoff).nonzero()
    for a, b in close.tolist():
        ra, rb = int(owner[a]), int(owner[b])
        if ra != rb:
            i, j = sorted((res_idx[ra], res_idx[rb]))
            out.add((i, j))
    return sorted(out)


def tables_from_scorer(
    score: Callable[[str], float],
    native: str,
    positions: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    alphabet: str = AMINO_ACIDS,
    on_progress: Callable[[str], None] | None = None,
) -> PairwiseEnergyTable:
    """Build a table by inclusion-exclusion around ``native``.

    ``positions`` are 0-based sequence indices of the designable positions; ``pairs`` index into
    that list. ``e1[i][a] = U(a at i) - U(native)`` and, with ``a != native`` and ``b != native``,
    ``e2[i,j][a,b] = U(a at i, b at j) - U(native) - e1[i][a] - e1[j][b]``. ``score`` is called
    ``1 + P(|A|-1) + M(|A|-1)^2`` times at most.
    """
    for c in alphabet:
        if c not in AA_TO_INDEX:
            raise ValueError(f"unknown amino acid {c!r} in alphabet")
    p = len(positions)
    e0 = float(score(native))
    e1 = torch.zeros(p, N_AA, dtype=torch.float64)
    cache: dict[tuple[tuple[int, str], ...], float] = {(): e0}

    def mutate(changes: tuple[tuple[int, str], ...]) -> str:
        seq = list(native)
        for pos, aa in changes:
            seq[positions[pos]] = aa
        return "".join(seq)

    def value(changes: tuple[tuple[int, str], ...]) -> float:
        if changes not in cache:
            cache[changes] = float(score(mutate(changes)))
        return cache[changes]

    for i in range(p):
        for a in alphabet:
            if a == native[positions[i]]:
                continue
            e1[i, AA_TO_INDEX[a]] = value(((i, a),)) - e0
        if on_progress:
            on_progress(f"one-body position {positions[i]} done")
    pair_list = [tuple(sorted(pr)) for pr in pairs]
    e2 = torch.zeros(len(pair_list), N_AA, N_AA, dtype=torch.float64)
    for m, (i, j) in enumerate(pair_list):
        for a in alphabet:
            if a == native[positions[i]]:
                continue
            for b in alphabet:
                if b == native[positions[j]]:
                    continue
                joint = value(((i, a), (j, b)))
                e2[m, AA_TO_INDEX[a], AA_TO_INDEX[b]] = (
                    joint - e0 - e1[i, AA_TO_INDEX[a]].item() - e1[j, AA_TO_INDEX[b]].item()
                )
        if on_progress:
            on_progress(f"pair {positions[i]}-{positions[j]} done ({m + 1}/{len(pair_list)})")
    return PairwiseEnergyTable(
        e0=e0,
        e1=e1.float(),
        pairs=torch.tensor(pair_list, dtype=torch.long).reshape(-1, 2),
        e2=e2.float(),
        native=torch.tensor([AA_TO_INDEX[native[p]] for p in positions], dtype=torch.long),
    )


def approximation_error(
    table: PairwiseEnergyTable,
    score: Callable[[str], float],
    native: str,
    positions: Sequence[int],
    sequences: Sequence[str],
) -> dict[str, float]:
    """How well the table predicts the true ``score`` on held-out ``sequences`` (full sequences
    differing from native only at ``positions``): mean/max absolute error, Pearson and Spearman
    correlation, and the error of the one-body-only model for comparison."""
    from cinder.types import encode_sequence  # local: avoid widening the module's import surface
    from cinder.validation.mutant_scan import spearman

    truth = torch.tensor([float(score(s)) for s in sequences], dtype=torch.float64)
    toks = torch.stack([encode_sequence(s)[list(positions)] for s in sequences])
    pred = table.energy(toks).double()
    one_body = (
        PairwiseEnergyTable(table.e0, table.e1, table.pairs[:0], table.e2[:0]).energy(toks).double()
    )
    err = (pred - truth).abs()
    pearson = float(torch.corrcoef(torch.stack([pred, truth]))[0, 1])
    return {
        "n": float(len(sequences)),
        "mean_abs_error": float(err.mean()),
        "max_abs_error": float(err.max()),
        "pearson": pearson,
        "spearman": spearman(pred.tolist(), truth.tolist()),
        "one_body_mean_abs_error": float((one_body - truth).abs().mean()),
        "native_energy": float(table.e0),
        "truth_std": float(truth.std()),
    }
