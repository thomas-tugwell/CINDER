# TMol adapter

Written against TMol 0.1.x after inspecting the installed package (`tmol.pose_stack_from_pdb`,
`beta2016_score_function`, `ScoreFunction.render_whole_pose_scoring_module`). All
version-specific code is in `src/cinder/energy/tmol.py`. Install with `pip install -e ".[tmol]"`;
`cinder doctor` reports availability.

## What is implemented (Milestones 2–3)

* Load a complex PDB; target/binder chains must each be a contiguous block of residues.
* Per-term weighted scores, total score, coordinate gradients (autograd through TMol).
* Interface decomposition `E(complex) − E(target) − E(binder)` over pairwise terms
  (`INTERFACE_TERMS`), and for `fa_ljrep` alone; covalent `strain` terms; harmonic restraint.
* Batched evaluation (N conformations → an N-pose stack, module cached per N).
* A fresh score function is created; TMol's memoised default is never mutated.

Validation (`pytest -m tmol`): reproducible energies; per-term row order confirmed against an
isolated score function; decomposition equals the manual computation; batch == single;
autograd gradients agree with central finite differences on the largest-gradient atoms;
restrained binder-only Langevin stays bound and clash-free.

## Variable atom count and sequence-specific poses

Mutating $\mathrm{Ala}\to\mathrm{Trp}$ changes the atom set, so one coordinate tensor cannot be
shared across sequences. `TMolGroupEnergy` therefore stays tied to one topology and raises
`SequenceMismatchError` for any other sequence. Sequence changes are handled one level up by
`energy/tmol_mutation.py::TMolSequenceBuilder`:

* It builds one full-atom pose per sequence with TMol's packer (`PackerTask` +
  `pack_rotamers`), with the backbone fixed and the side chains in a **fixed repack region**
  optimised (`design` positions only, `neighbors` within a cutoff of any design residue, or
  `all`). The region depends only on the design positions, so every sequence, **including the
  native one, goes through the same protocol**; compare mutants to the repacked native, not to
  the raw input pose (repacking alone can change the energy a lot: raw 2ZTA total 392.9 vs
  repacked about -1).
* Rotamers come from the Dunbrack sampler, `IncludeCurrentSampler`, and `FixedAAChiSampler`.
  The last is required for Ala/Gly (no chi angles): without it the packer **silently keeps the
  original residue**. The builder checks every design residue after packing and raises
  `MutationError` if it is not what was requested.
* Chain sub-poses of the packed complex share its coordinates exactly, so the interface
  decomposition still holds. Poses are cached by sequence string (about 0.4 s each for a
  62-residue complex on CPU).
* `builder.score(seq)` returns the single-structure `interface`, `repulsion` and `strain` terms.
  This is the "one structure per sequence" baseline for the ensemble comparison.

Not yet done: sampling an *ensemble* per mutant sequence (the builder gives the starting pose),
and a fixed-size coordinate representation so one batch can span many sequences.

## Caveats

* Cartesian Langevin on all atoms heats stiff covalent terms (equipartition), so `strain`
  rises while interface/repulsion stay near reference. Prefer `movable: binder`, a small step,
  and consider torsion-space moves later.
* TMol runs on CPU/CUDA; MPS requests fall back to CPU.
* Energy units are Rosetta-style score units, not kcal/mol.
