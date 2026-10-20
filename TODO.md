# TODO and next steps

Status as of 2026-10-20. Milestones 0, 1, 2, 3, 4 (toy only) and 5 (toy only) are done; see
`README.md` for the roadmap checklist. This file lists what is not done and the order to do it in.

## 1. Highest priority: TMol-driven sequence design on a real complex

**Blocker:** mutations change the atom count, so one coordinate tensor cannot be shared across
sequences. `energy/tmol.py` evaluates the native sequence only and raises
`SequenceMismatchError` for anything else. The toy loop works; the real-complex loop does not.

**Progress:** steps 1, 2 and 4 below are done in `energy/tmol_mutation.py`
(`TMolSequenceBuilder`: mutation + repack, cache, single-structure `score`; see `docs/tmol.md`).
Remaining: ensemble sampling per mutant sequence, coordinate representation across sequences,
wiring into `run_design`.

Steps:
1. (done) Add a per-sequence pose builder. Use `tmol.pack.PackerTask` with
   `restrict_to_repacking` on non-design residues and `restrict_absent_name3s` at design
   positions, then `pack_rotamers`. Cache poses by sequence string.
2. Build a `TMolContext` per sequence (target pose is shared; binder and complex poses change).
3. Decide how conformations are represented across sequences. Options: process sequences one at
   a time with per-sequence systems, or adopt a fixed-size representation (atom14 + mask, residue
   frames, torsions; `systems/structure.py` has the `Atom14Layout` container only).
4. First deliverable: `score_sequence(system, seq)` for the single-packed-structure baseline.
   Test it on a few barstar interface mutations.
5. Then ensemble scoring per sequence, then wire into `run_design` and the CLI.
6. Run policy optimization on barnase/barstar design positions. A sanity check is that known
   favourable or unfavourable mutations move in the expected direction.

## 1b. Priorities requested by the project owner

1. **Rigid-body (jump) DOFs** (done as of this version: `systems/rigid_body.py`,
   `energy/rigid_body.py`, `--space rigid` in the recovery script). Next: use jump-space ensembles
   as the conformational model inside the sequence loop, and compare against TMol's own native
   fold-forest jumps (`MoveMap`, `run_kin_min`) as a cross-check.
2. **Separate `hbond` and `fa_elec` rewards** (second priority). `fa_elec` stands in for a
   salt-bridge term; no custom term is needed. Add them as separately weighted design-energy
   terms (config `hbond_weight`, `elec_weight`), log them per term, and report hbond/elec in the
   run CSVs. Needs care: decompose into interface contributions like `interface` does.
3. Torsion-space moves (phi/psi/chi via TMol `MoveMap` / `set_named_torsions`) for flexible
   binders without bond heating, and torsion perturbations for benchmarks.

## 2. Benchmarks (the scientific question)

Nothing here exists yet. Needs step 1 first.
- Single minimized structure vs ensemble mean vs ensemble free-energy surrogate.
- Unguided vs TMol-guided conformational model.
- Categorical policy vs random search vs MPNN proposals.
- Store the metrics needed for these comparisons and add an `analyze` mode that produces them.
- If experimental binding data are available, correlate rankings against them.

## 3. Conformation models

- **Restricted/internal-coordinate sampling:** Cartesian Langevin on all atoms heats stiff
  covalent terms. Try torsion-space or local-frame moves, or a per-atom force clip.
- **Normalizing flow:** only toy coordinates, `dim >= 2`, reverse-KL (mode-seeking). Add
  1D support, a forward-KL or data-driven option, and wire it into `runner.build_sampler`.
- **Diffusion and flow matching:** stubs only. Implement behind `ConformationModel`, reuse
  `EnergyGuidance` / `energy_guided_velocity`, and validate against the same analytic targets
  as Langevin. Check whether velocity guidance is valid for learned fields (currently a heuristic).
- **Protein-scale learned model (Milestone 6):** choose a representation (atom14,
  SE(3)-equivariant Cartesian, rigid frames, torsions, internal coordinates) and document the choice.
- **BioEmu2 (Milestone 8):** adapter is a stub. Inspect the installed API first, then wire it
  as `p_base` with optional TMol guidance.

## 4. Sequence optimization

- Only a position-factorised categorical policy exists. Possible extensions: pairwise or
  autoregressive policies, MPNN-style proposals as a prior or baseline.
- Variance reduction beyond the batch baseline (leave-one-out baseline, advantage normalization
  is available but off by default).
- Support the variational free-energy objective (`variational_free_energy` exists) as a selectable
  reward for models with tractable density.

## 5. TMol adapter extras

- Restricted minimization and optional restrained relaxation (`tmol.relax`, `run_cart_min`).
- Selectable per-term subsets via config (term groups are currently hard-coded constants).
- GPU validation; all TMol tests so far ran on CPU.
- Chains must be contiguous residue blocks in the PDB; relax this or add a reorder helper.
- Write sampled ensembles from `cinder design` to `runs/<id>/structures/` (the recovery script now
  writes PDBs there via `write_pdb`; design runs do not yet).

## 6. Repository housekeeping

- Nothing has been committed or pushed. `origin` is set to
  `https://github.com/thomas-tugwell/CINDER.git`.
- CI workflow is untested (not yet run on GitHub). It installs no TMol and no CUDA by design.
- Add a docs build if desired; `docs/` is plain Markdown.
- `examples/notebooks/` is empty.
- `CITATION.cff` lists only the repository owner; update authors and add a DOI/paper if relevant.
- Add `torch` CPU-only note or constraint if CI install time becomes a problem.
- On the external drive, macOS creates `._*` files; they are git-ignored and the config test
  skips them.
