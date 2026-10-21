# TODO and next steps

Status as of 2026-10-20. See `README.md` for the roadmap checklist and `docs/` for details.

## Done since the first version (for orientation)

Rigid-body and torsion (phi/psi/chi) DOFs with gradients; mutation + repacking with per-sequence
ensembles and a `RewardModel` interface; a TMol design loop (`cinder design`); separately weighted
`hbond`, `elec` (stands in for salt bridges), `internal` and `hbond_graph` (first/second shell)
terms; design positions proposed from the shells; an autoregressive (coupled) sequence policy;
mutant-scan tooling; PDB output; recovery benchmarks on three helical complexes.

## 1. Highest priority: test the central claim on real data

Nothing has been compared with experiment. The tooling exists (`scripts/mutant_scan.py`,
`cinder.validation.mutant_scan`), so the blocker is data.

1. Find and vet a mutational binding dataset for complexes we can load (check its license, the
   structures used, and the sign convention; SKEMPI is the obvious candidate but its contents and
   license have not been verified here).
2. Score the mutants with the single packed structure and with the jump-space ensemble; report
   Spearman correlation with measured changes in binding, plus enrichment of the top hits (most
   mutations are neutral or destabilising, so overall correlation is not enough).
3. Add baselines: random ordering, a physics-free substitution score, and the same protocol with
   `hbond`/`elec`/`hbond_graph` terms on and off.
4. Known caveat to address: the interface energy at fixed coordinates ignores unbound-state
   relaxation and entropy, which standard ddG protocols handle.

## 2. Validate the new terms

* H-bond graph / second shell: compare with an independent tool and with experiment; check that
  second-shell support rewards are not gamed by strained geometry; tune burial and switching
  parameters (currently heuristics calibrated on barnase/barstar only).
* `hbond`/`elec` emphasis: ablations in the recovery and design benchmarks.

## 3. Design loop

* Torsion DOFs inside `rigid_langevin` (currently jump-space ensembles only; torsion ensembles are
  validated separately). Needs the `internal` weight > 0.
* Pair/triple mutation proposals; richer policies (the autoregressive policy is a first step).
* Cost: each sequence needs a packed pose plus a short ensemble; GPU is untested (TMol runs on
  CUDA; `device: auto` is wired but has never run on a GPU here).
* Variance reduction beyond the batch baseline.

## 4. Conformation models

* Torsion-space recovery is weaker than rigid-body recovery (see `docs/validation.md`); investigate
  correlated torsion moves, a colder final stage, or per-DOF step sizes (independent per-angle
  noise compounds like a lever).
* Normalizing flow: toy coordinates only; diffusion, flow matching and BioEmu2 are stubs.
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
