# Results and validation status

This file collects what has been measured and what has not. Details, protocols and caveats are in
[validation.md](validation.md) and [sequence_optimization.md](sequence_optimization.md). The
numbers marked below come from `python scripts/validation_report.py` (seeded, CPU, about a minute;
the TMol rows need TMol installed).

## Summary

* The machinery works on analytic toy problems, and the TMol evaluator and its gradients are
  verified on real complexes.
* Binder structure recovery is validated on three helical complexes in Cartesian, rigid-body and
  torsion spaces.
* Mutated sequences are built and scored (single packed structure or jump-space ensemble), and
  `cinder design` runs a sequence-optimisation loop on a real complex with TMol.
* **Against experiment (SKEMPI 2.0, three protease-inhibitor complexes):** the TMol interface
  energy tracks measured binding (Spearman 0.35 to 0.50, above a negative-BLOSUM62 baseline). The
  jump-space ensemble was not distinguishable from the single packed structure (paired bootstrap
  intervals include zero). The central ensemble hypothesis is **not supported yet, and not refuted**.
* **Sequence optimisation:** the pairwise-table optimiser did not beat a best-single-mutant scan,
  and a greedy hill climb on the real scorer beat it on 1CDL. See
  [sequence_optimization.md](sequence_optimization.md).
* Diffusion and flow matching are implemented for toy coordinates only; BioEmu2 is a stub.

## Measured binding changes (SKEMPI 2.0)

Single-structure vs jump-space ensemble vs BLOSUM, 190 to 191 single mutants of the binder (OMTKY3)
per entry. Ensemble minus single is the Spearman difference with a 95% paired bootstrap interval.

| entry | single Spearman / AUC | ensemble Spearman / AUC | BLOSUM Spearman / AUC | ensemble - single |
|---|---|---|---|---|
| 3SGB_E_I | +0.460 / 0.778 | +0.498 / 0.779 | +0.087 / 0.559 | +0.038 (-0.018, +0.098) |
| 1R0R_E_I | +0.345 / 0.691 | +0.366 / 0.690 | +0.148 / 0.578 | +0.020 (-0.043, +0.086) |
| 1PPF_E_I | +0.369 / 0.671 | +0.370 / 0.663 | +0.159 / 0.590 | +0.000 (-0.039, +0.037) |

Caveats: the three entries share one inhibitor (they are not independent), about ten contact
positions, binder-only mutations, no unbound-state relaxation, untuned ensemble settings. A sweep
of ensemble settings on 3SGB (more samples, more noise, stronger guidance) also gave intervals that
contain zero. SKEMPI is not shipped; see [validation.md](validation.md).

## Verification details

**Verified on analytic toy problems with known answers.** These test the machinery (samplers,
guidance, flow objective, policy gradient), not protein biology.

| Check | Measured | Expected |
|---|---|---|
| Langevin, harmonic: variance | 0.1670 | 0.1667 |
| Guided product of Gaussians: mean | 1.4914 | 1.5000 |
| Guided product of Gaussians: variance | 0.2686 | 0.2500 |
| Double well: fraction in right well | 0.508 | 0.500 |
| Flow, Gaussian: final loss vs bound −log Z | −1.1391 | −1.1447 (bound) |
| Flow, bimodal: final loss vs −log Z | 0.0751 | 0 (bound) |
| Flow, bimodal: fraction in right mode | 0.505 | 0.500 |
| Policy: consensus sequence | IR | IR (brute force over 400) |
| Policy: mean reward, last 10 iterations | 2.195 | 2.214 (analytic optimum) |
| Policy: entropy first → last iteration | 5.99 → 0.11 | decreasing |

The guided variance is about 7% high, consistent with the O(η) bias of unadjusted Langevin plus
Monte Carlo error. The bimodal flow loss sits 0.075 above its bound, i.e. both modes are
covered but the fit is not exact.

**Verified: the TMol evaluator on barnase/barstar (PDB 1BRS, native pose).** Autograd
coordinate gradients match central finite differences on the largest-gradient atoms:

| Term | Weighted value |
|---|---|
| interface | −62.93 |
| repulsion | 6.99 |
| strain | 231.08 |
| restraint | 0.00 |

| Coordinate | Autograd dU/dx | Finite difference |
|---|---|---|
| atom 2733, axis y | −72.044 | −72.043 |
| atom 2745, axis y | 66.310 | 66.315 |
| atom 2745, axis x | −14.158 | −14.046 |
| atom 2733, axis x | 11.868 | 11.874 |
| atom 864, axis z | −8.192 | −8.200 |
| atom 1677, axis z | 7.394 | 7.396 |

**Verified: structure recovery on three helical complexes (binder displaced, TMol-guided
sampling, no restraint to the reference).** Moving only the six rigid-body binder DOFs returned
the binder to its control floor in 21 of 24 perturbed runs (2ZTA 8/8, 1CDL 8/8, 1G5J 5/8), with
the fold preserved exactly. Moving every atom independently (Cartesian) returned none of 21
completed runs to the floor. Details, caveats and the three failures are in
[validation.md](validation.md#results-perturb-and-recover-cartesian-vs-rigid-body-sampling).
This tests the energy landscape and the sampler, not sequence design or ensemble scoring.

**Verified: mutation and repacking (TMol).** `TMolSequenceBuilder` builds one full-atom pose per
sequence by mutating residues and repacking side chains with the backbone fixed. Tests check that
each design residue has the requested identity after packing, that every CA stays in place, that
atom counts order as expected (Gly < Ala < Trp), that sub-poses share the complex's coordinates,
and that results are cached and deterministic. A silent failure mode was found and guarded: without
`FixedAAChiSampler` the packer quietly keeps the original residue for Ala and Gly, so the builder
raises `MutationError` whenever a requested residue is not what the packer returned. Only
single-structure scores are produced; no ensemble around each mutant yet.

**Verified: the TMol design loop and new terms.** `cinder design` on barnase/barstar (5 designable
positions, per-sequence packed poses, jump-space ensembles) runs end to end; `hbond`, `elec`,
`internal` and `hbond_graph` are separately weighted and logged per term; the soft H-bond graph
agrees with an independent hard-threshold counter (13.2 vs 15 cross-interface bonds) and its first
shell recovers barstar's classic interface residues from geometry alone. The autoregressive
sequence policy represents coupled optima that the independent policy cannot. The loop has not
been shown to improve anything real: three iterations only demonstrate that it runs.

**Verified: mutations and the noise, mutate, refine protocol (barnase/barstar).** Low-rate random
mutants are penalised far more at the interface than elsewhere (median energy change +21 for mutants
with an interface mutation vs +0.5 for surface-only ones), and across the 9 first-shell positions the
native residue is the lowest-energy choice at 7 and in the top 3 at 8 (mean rank 2.4 of 20; random
10.5). The noise, mutate, refine pipeline (torsion, rigid-body and Cartesian noise on the binder,
mutant packed on the noisy backbone, then gradient refinement in torsion/rigid-body and Cartesian
space) removes the clashes and strain it starts with (about +3000 down to about 0) but, for the
87-residue binder, ends a median of +121 energy units above the same sequence on the native backbone
(+25 without noise); on the 31-residue 2ZTA it came within about 5. All use TMol as both guide and
judge; none compares with experiment. Details and caveats: [validation.md](validation.md).

**Tests.** 155 CPU tests run by default; more (plus one skip) need TMol (`pytest -m tmol`).

## Not verified

- That any of the new terms (H-bond graph, second shell, `hbond`/`elec` emphasis) improves a real
  design. They are implemented and tested for correctness; ablations against measured data are open.
- That ensemble scoring beats single-structure scoring (SKEMPI so far: no detectable difference).
- Anything about absolute binding affinity, or any experimentally validated design.
- Protein-scale learned conformation models and BioEmu2.
- The GPU path (no GPU was available).
