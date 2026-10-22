# TODO and next steps

Status as of 2026-10-20. See `README.md` for the roadmap checklist and `docs/` for details.

## Done since the first version (for orientation)

Rigid-body and torsion (phi/psi/chi) DOFs with gradients; mutation + repacking with per-sequence
ensembles and a `RewardModel` interface; a TMol design loop (`cinder design`); separately weighted
`hbond`, `elec` (stands in for salt bridges), `internal` and `hbond_graph` (first/second shell)
terms; design positions proposed from the shells; an autoregressive (coupled) sequence policy;
mutant-scan tooling; PDB output; recovery benchmarks on three helical complexes.

## 1. Highest priority: the ensemble hypothesis

Done: `scripts/skempi_validation.py` compares scores with measured SKEMPI ddG on three protease
inhibitor complexes (see `docs/validation.md`). The energy has signal (Spearman 0.35 to 0.50, above
a BLOSUM baseline); the ensemble is not distinguishable from the single packed structure.

Next:

1. Done in part (jump-space sweep on 3SGB: all intervals contain zero, see `docs/validation.md`). Still open:
   torsion or backbone flexibility, not only jump-space; report the paired bootstrap interval.
2. Ablations: `hbond`/`elec`/`hbond_graph` terms on and off; fixed-coordinates interface energy
   against a relaxed unbound state (ddG protocols include this; we do not).
3. More complexes that do not share an inhibitor (SKEMPI has many; check mutations on both
   partners, which needs target mutation support), and a check for SKEMPI leakage if a learned
   model is added.
4. SKEMPI license/terms were not located; the file is not shipped.

## 2. Validate the new terms

* H-bond graph / second shell: compare with an independent tool and with experiment; check that
  second-shell support rewards are not gamed by strained geometry; tune burial and switching
  parameters (currently heuristics calibrated on barnase/barstar only).
* `hbond`/`elec` emphasis: ablations in the recovery and design benchmarks.

## 2b. Noise, mutate, refine: close the gap on large binders

On barstar (87 residues) the refined noisy mutants end a median +121 energy units above the same
sequence packed on the native backbone (+25 without noise); on 2ZTA (31 residues) the gap was about
5. About one third is docking (final CA 1.2 to 2.5 Å from native), two thirds the binder's own fold.
A one-mutant diagnostic (see `docs/validation.md`) shows the gap scales mainly with the noise size
(+154 at the default noise, +30 at a 0.67 Å start); a 5x longer quench helps only the internal part
(+154 to +112) and more annealing does not help. Next: length-scaled noise levels, a refinement
stage that can move the docking pose, a better reference (the baseline is only Cartesian-polished,
so it is too weak where a mutation needs the binder to adapt), and more mutants, complexes, seeds.

## 2c. Pairwise-table design (exact gradients)

Implemented (`scripts/table_design.py`) but **it does not beat baselines**: on four complexes the
table optimiser lost to the best single mutant, the one-body optimum or random search, and a greedy
real-scorer hill climb beat it clearly on 1CDL (-14.7 vs +15.2; see `docs/sequence_optimization.md`).
Next, in order: iterative re-fitting of the table around the
current design (trust region), a three-body check on selected designs, a real mutational dataset.
Open: three-body effects (the table is exact only for one- and two-body substitutions), tables on
an ensemble or refined backbone instead of the fixed native one, scaling the pair scan (cost grows
with contact pairs times alphabet squared), and using the table inside the `cinder design` loop.

## 3. Design loop

* Torsion DOFs inside `rigid_langevin` (currently jump-space ensembles only; torsion ensembles are
  validated separately). Needs the `internal` weight > 0.
* Pair/triple mutation proposals; richer policies (the autoregressive policy is a first step).
* Mutations in the target or in both partners (the design space, sampler and builder are tied to
  the binder chain today); most measured mutational datasets include both.
* Cost: each sequence needs a packed pose plus a short ensemble; GPU is untested (TMol runs on
  CUDA; `device: auto` is wired but has never run on a GPU here).
* Variance reduction beyond the batch baseline.

## 4. Conformation models

* Torsion-space recovery is weaker than rigid-body recovery (see `docs/validation.md`); investigate
  correlated torsion moves, a colder final stage, or per-DOF step sizes (independent per-angle
  noise compounds like a lever).
* Normalizing flow, diffusion and flow matching: toy coordinates only (tested on CPU, not wired into
  the runner); BioEmu2 is a stub (needs the installed API, weights and probably a GPU).
* Protein-scale learned model: choose a representation (atom14, frames, torsions) and document it.

## 5. Benchmarks and evidence

* Recovery benchmark: the 1G5J failures (2 Å shifts) have not been tested with longer runs or
  restarts.
* Guided vs unguided, vs random search, vs MPNN proposals at equal compute.
* Literature: two prior-art searches failed (differentiable-physics design through ensembles,
  BioEmu-based design); a full review is needed before any publication claim.

## 6. TMol adapter

* Uses TMol internals (`MoveMap`, `MinimizerMap`, `inverse_kin`, packer samplers); re-check on
  every TMol upgrade. Chains must be contiguous residue blocks; chains with gaps are rejected by
  `chain_breaks()` rather than handled.
* Restricted minimization / relaxation (`run_cart_min`, `run_kin_min`) as a baseline to compare
  against our guided sampler.
* The packer repack region is fixed from the native pose; test sensitivity to `cutoff`.

## 7. Repository housekeeping

* Commit identity: `tht13@pitt.edu` is set for this clone only; earlier commits carry the machine
  hostname address and are not linked to the GitHub account.
* CI has not been confirmed green on GitHub since the TMol-free matrix was introduced.
* `examples/notebooks/` is empty. `CITATION.cff` lists only the repository owner.
