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

## Variable atom count (important limitation)

Mutating $\mathrm{Ala}\to\mathrm{Trp}$ changes the atom set, so one coordinate tensor cannot be
shared across sequences. The adapter therefore fixes topology to the **native sequence**
and raises `SequenceMismatchError` for any other. Consequently the *TMol-driven sequence
optimisation loop is not yet runnable on a real complex*; the toy loop is. Planned route
(not implemented): build a per-sequence pose with `tmol.pack.PackerTask.restrict_absent_name3s`
at design positions + `pack_rotamers`, cache by sequence, and adopt a fixed-size coordinate
representation (atom14 / frames / torsions, `systems/structure.py`) for sequence-conditioned
generative models.

## Caveats

* Cartesian Langevin on all atoms heats stiff covalent terms (equipartition), so `strain`
  rises while interface/repulsion stay near reference. Prefer `movable: binder`, a small step,
  and consider torsion-space moves later.
* TMol runs on CPU/CUDA; MPS requests fall back to CPU.
* Energy units are Rosetta-style score units, not kcal/mol.
