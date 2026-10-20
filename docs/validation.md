# Validation protocols

Two kinds of validation exist: **analytic** (machinery vs closed forms; see the README table and
`scripts/validation_report.py`) and **perturb-and-recover** on real complexes (this page).

## Perturb-and-recover

**Question.** If the *binder* is displaced from the complex fold, does energy-guided sampling
return it toward the reference? Only the binder is perturbed; the target is frozen
(`movable: binder`).

**Protocol** (`cinder.validation`, CLI `scripts/structure_recovery.py`, needs TMol):

1. Load the complex; reference pose = the file's coordinates.
2. Perturb binder atoms: rigid-body (rotation about the binder centroid + shift; internal
   geometry preserved) or Cartesian noise (distorts bonds).
3. Run TMol-guided Langevin from the perturbed start with an annealed guidance strength
   (`λ` ramped linearly), **with no restraint to the reference** (`restraint_weight: 0`), so only
   the physics can pull the structure back. A restraint would make the test circular.
4. Report binder RMSD to the reference before/after (no superposition: the target is fixed) and
   the interface energy before/after.

**Sampling spaces** (`--space`). `cartesian` moves every binder atom independently (heats
stiff bonds, distorts the fold). `rigid` moves only the six binder jump DOFs (translation +
rotation vector about the binder centroid; `cinder.systems.rigid_body`), so the internal geometry
is preserved exactly and the sampler explores docking poses. Energy gradients dU/dq are the
chain rule through the differentiable jump transform and TMol's coordinate autograd
(`cinder.energy.rigid_body.RigidBodyEnergy`; checked against finite differences on barnase/barstar).
Rotation is expressed in Å-equivalent units (scaled by the binder's radius of gyration) so one
step size suits all six dimensions.

**Outputs.** Each invocation creates a new run directory under `runs/` (never overwritten):
`config.yaml`, `metadata.json` (git commit, versions, seed, input checksum), `results.csv` (one
row per perturbed run, with *every* energy term before and after: total, interface, repulsion,
strain) and `structures/` holding `<ID>_reference.pdb` plus `<ID>_<perturbation>_<k>_start.pdb`
and `_final.pdb`. All files share the complex's atom order, so they superimpose directly (for
example in PyMOL). Always read the repulsion and strain columns, not just the interface energy:
a displaced pose can look better on interface energy while clashing badly.

**Always run the control** (no perturbation). The reference is a crystal/NMR pose, not a TMol
minimum; the control's final RMSD is the floor of what "recovered" can mean, and its interface
energy shows how far below the reference TMol's own optimum lies.

**Interpretation.** Recovery to the *control floor* means the TMol landscape funnels the binder
back to the reference basin. Final energy as low as the reference but RMSD far from it means a
different, equally good minimum, i.e. the energy does not discriminate the reference pose at that
resolution. Neither says anything about binding affinity.

**Not yet done:** perturbing the binder *sequence* as well. That needs the per-sequence pose
builder (see `TODO.md`).

## Benchmark complexes

`configs/benchmarks/helical_complexes.yaml` lists helix-rich complexes from the RCSB PDB
(public domain). Files are downloaded at run time into `data/` (git-ignored) by
`cinder.validation.datasets`, keeping model 1, ATOM records, no hydrogens, first altloc.
Screen new candidates with `chain_breaks()` (a gap makes TMol bridge residues with a stretched
bond; 1PQ1 was excluded for this) and `helical_fraction()`.

### Teddymer

[Teddymer](https://teddymer.steineggerlab.workers.dev/) (AlphaFold-predicted domain–domain
dimers clustered from TED; CC BY-NC 4.0; archives 20.5–352 GB) was considered and **not used**:
its structures are predictions, it has no binding labels, and its license is non-commercial, so
derived files must not be committed to this Apache-2.0 repository. If used later, download at run
time only and keep it out of version control.
