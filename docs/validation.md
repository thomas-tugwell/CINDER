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
(`cinder.energy.parametrized.ParametrizedEnergy`; checked against finite differences on barnase/barstar).
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

## Results: perturb-and-recover, Cartesian vs rigid-body sampling

Protocol as above, binder-only perturbation (2° or 5° rotation × 1 or 2 Å shift, two repeats
each, plus an unperturbed control), 4000 steps, `λ` ramped 5 → 50, no restraint to the reference.
"Recovered" means final CA RMSD (target frame) within 0.15 Å of that complex's own control
(unperturbed) final RMSD, the floor reachable under the same protocol. Reproduce with
`scripts/structure_recovery.py --complex <ID> [--space rigid]`.

| Complex | Sampling space | Control floor (Å) | Perturbed finals (Å) | Recovered |
|---|---|---|---|---|
| 2ZTA (coiled-coil homodimer) | Cartesian | 0.25 | 0.96 – 2.32 | 0 / 8 |
| 2ZTA | rigid-body | 0.54 | 0.49 – 0.56 | 8 / 8 |
| 1CDL (calmodulin + peptide) | Cartesian | 0.34 | 0.90 – 1.84 | 0 / 8 |
| 1CDL | rigid-body | 0.41 | 0.39 – 0.41 | 8 / 8 |
| 1G5J (Bcl-xL + Bad peptide) | Cartesian | 0.27 | 0.91 – 2.04 | 0 / 5 (run stopped early) |
| 1G5J | rigid-body | 0.52 | 0.52 – 2.43 | 5 / 8 |

What this does and does not show:

* Moving every atom independently (Cartesian) never returned the binder to the control floor.
  Moving only the six rigid-body DOFs did in 21 of 24 perturbed runs. The earlier conclusion that
  the energy "does not discriminate the reference pose" was an artifact of the sampling space.
* All rigid-body runs preserve the binder's internal geometry exactly (aligned CA RMSD 0.00).
* On 2ZTA and 1CDL all finals agree to within about 1 REU in interface energy, i.e. a single
  basin. The floor is 0.4 – 0.5 Å, not 0, because the crystal pose is not TMol's own minimum.
* The three 1G5J failures (all with the 2 Å shift) end 1.3 – 2.4 Å away in a different basin:
  weaker interface energy (−38 to −45 vs −63 for the control) but lower repulsion (15 – 20 vs 38),
  with essentially the same total design energy (159.6 – 161.9 vs 159.5). So under TMol those poses
  are equally good as the control, not worse: the landscape is nearly degenerate there, or the
  run was too short or too cold to cross a barrier. Not yet tested: longer runs, restarts.
* Caveats: three small helical complexes, two repeats per setting, perturbations at most 5° and
  2 Å, target frozen, and TMol is both guide and judge. The 1G5J Cartesian row used only 5 runs.
  This validates the energy landscape and the sampler, not ensemble scoring or sequence design.

## Results: torsion-space and combined recovery

Same protocol, but the sampler moves binder torsions (phi/psi) with the binder's own `internal`
energy (and, for `torsion_rigid`, a rigid-body jump as well). Perturbations: 0.5, 1 and 2 degrees of
torsion noise (two repeats each), and 1 degree plus rigid 2/5 degrees and 1/2 Å (one each). "Within
floor" means final CA RMSD within 0.15 Å of the unperturbed control's final RMSD.

| Complex | Space | Control floor (Å) | n | start (Å) | final (Å) | within floor | fold distortion at end (Å) |
|---|---|---|---|---|---|---|---|
| 1CDL | torsion | 0.44 (fold 0.36) | 6 | 0.27–1.97 | 0.39–0.71 | 3/6 | 0.32–0.51 |
| 1CDL | torsion + rigid | 0.53 (fold 0.39) | 4 | 1.12–2.12 | 0.53–0.64 | 4/4 | 0.31–0.42 |
| 2ZTA | torsion | 0.56 (fold 0.36) | 6 | 0.76–3.32 | 0.47–1.10 | 2/6 | 0.35–0.49 |
| 2ZTA | torsion + rigid | 0.51 (fold 0.38) | 4 | 0.99–2.46 | 0.69–0.97 | 0/4 | 0.37–0.54 |
| 1G5J | torsion | 2.12 (fold 1.03) | 6 | 0.44–3.24 | 1.92–2.30 | 5/6 | 0.88–1.13 |
| 1G5J | torsion + rigid | 1.67 (fold 1.18) | 4 | 1.37–2.43 | 1.55–2.14 | 1/4 | 0.95–1.19 |

* Torsion-space recovery is less clean than rigid-body recovery (21 of 24 within the floor). 1CDL
  recovers well; 2ZTA ends 0.5 to 1 Å from the structure but keeps its fold (distortion 0.35 to 0.54
  Å, the same as the control's); results are mixed rather than uniformly good.
* **1G5J is not a clean test in torsion space:** the *unperturbed* control drifts 2.1 Å (fold
  distortion about 1 Å), so "within the floor" there only means "ended where the control ended".
  The Bad peptide in 1G5J is a flexible helix from an NMR-derived entry; torsion-space sampling moves
  it substantially even from the starting pose. Do not read the 5/6 as recovery.
* Without the `internal` term torsion sampling unwinds the helix (regression test in
  `tests/integration/test_tmol_torsion.py`).

## Results: low-rate random mutations (barnase/barstar, single packed structure)

`scripts/random_mutants.py`: 40 mutants per set at a 3 percent rate (about 2.6 mutations of 87
residues); one set is biased 10 to 1 toward the 9 first/second-shell interface positions, the other
uniform. Each mutant is packed on the native backbone and scored; the energy change is relative to
the repacked wild type (positive = worse). No measured data are used.

| Set | Median change | Destabilising | Tail |
|---|---|---|---|
| interface-biased (1.5 interface mutations per mutant) | +13.1 | 95% | p75 +165, p90 +261, max +1030 |
| uniform (0.2 interface mutations per mutant) | +0.7 | 78% | p90 +47, max +169 |
| mutants touching no interface position (n = 40) | +0.5 | – | p90 +2.9, 10% above +5 |
| mutants with at least one interface mutation (n = 40) | +20.8 | – | 70% above +5, 40% above +50 |

* The pipeline penalises interface mutations far more than surface ones, as it should.
* The largest changes are mostly steric clashes (Gly31 to Phe is +1030): the backbone is fixed and
  nothing relaxes, so magnitudes are exaggerated. Use medians; a pooled regression
  (+57.8 per interface mutation vs +1.4 per other) is dominated by a few outliers.
* Random substitutions often go to large residues, so clashes dominate; subtler losses (a broken
  hydrogen bond or salt bridge) are under-represented. 40 mutants per set; no experimental data.
* The rerun with the same seed reproduced these numbers exactly.

## Results: position scan (barnase/barstar, native sequence recovery)

`scripts/position_scan.py`: all 19 substitutions at each of the 9 first-shell positions (backbone
fixed, same repack protocol), ranking the native residue among 20 by energy.

| Binder index | Native | Native rank (of 20) | Mean change of substitutions |
|---|---|---|---|
| 28 | Tyr | 1 | +5.1 |
| 30 | Gly | 1 | +466 |
| 32 | Asn | 2 | +5.8 |
| 33 | Leu | 1 | +2.4 |
| 34 | Asp | 1 | +58.8 |
| 38 | Asp | 1 | +15.0 |
| 41 | Thr | 13 | +0.5 |
| 42 | Gly | 1 | +29.1 |
| 73 | Glu | 1 | +2.7 |

Native is best at 7/9 positions and in the top 3 at 8/9 (mean rank 2.4; random 10.5). Caveats: the
fixed native backbone favours native residues by construction; glycines rank first almost
automatically because any side chain clashes; 9 positions on one complex. This is a sanity check of
the energy function, not a validation against binding data.

## Results: noise the structure, mutate the sequence, refine with gradients

`scripts/mutate_refine.py` (library: `cinder.experiments.mutate_refine`). For each of 6 low-rate,
interface-biased mutants of barstar (barnase/barstar, 87-residue binder): noise the native binder
(torsion 1 degree per phi/psi, rigid 2 degrees / 1 Å, Cartesian 0.05 Å), pack the mutant **on the
noisy backbone**, refine in stage A (annealed gradients in binder torsions phi/psi/chi plus the six
rigid-body DOFs, then a noise-free quench; 2000 + 300 steps) and stage B (Cartesian polish of the
binder, then a quench; 300 + 300 steps), with the binder's `internal` energy included throughout.
**Reference:** the same sequence packed on the native backbone and given the identical Cartesian
polish (`baseline_relaxed`). Every condition is also run noise-free as a control. Gap = final minus
reference total design energy (lower is better; at or below 0 means the refinement found a structure
at least as good as the reference). One noise setting and one seed per mutant.

| Mutant | Noisy: gap | CA vs native (Å) | Fold (Å) | Noise-free: gap | CA vs native (Å) |
|---|---|---|---|---|---|
| A35K+L40T | +86.9 | 1.88 | 0.68 | -4.1 | 0.71 |
| L33R+D34H+T41M+G78S | +142.5 | 2.53 | 0.73 | +37.5 | 1.11 |
| L23E+L36Q+A39E+L40D | +99.3 | 1.25 | 0.56 | +16.7 | 0.83 |
| N32Y+T82G | +160.6 | 1.57 | 0.84 | +33.1 | 0.65 |
| E31H | +153.8 | 1.49 | 0.67 | +48.1 | 0.76 |
| D34Y+Q57K | -11.6 | 2.07 | 0.77 | -123.4 | 1.95 |
| **median (mean)** | **+120.9 (+105.2)** | | | **+24.9 (+1.3)** | |

Median over the six mutants of final minus reference, per term (noisy / noise-free):
interface +53 / +14, internal (binder's own energy) +79 / +15, repulsion -6 / -2.

* **What works:** the noisy mutant starts around +3000 (clashes and strain; repulsion in the
  hundreds) and gradients bring it to about 0, with repulsion at or below the reference in every
  case. The noise-free runs end close to the reference (median +25).
* **What does not:** in five of six noisy runs the final structure stays 87 to 161 energy units above
  the reference, ending 1.2 to 2.5 Å from the native CAs. Roughly one third of the gap is a weaker
  interface (a less well-docked pose), two thirds is the binder's own fold energy (distortion 0.56
  to 0.84 Å against 0.1 Å for the reference). On the 31-residue 2ZTA the same protocol came within
  about 5 units, so larger binders are harder (lever-arm compounding of torsion noise, limited step
  budget).
* **The reference is imperfect.** It is only polished in Cartesian space, so for a mutation that
  needs the binder to adapt (Asp35 to Tyr, `D34Y+Q57K`) the refinement beats it (gap -12 noisy, -123
  noise-free: the binder moved about 2 Å). Even noise-free, the annealed sampler wanders 0.65 to 1.95
  Å from native before the quench, so part of every gap is method noise, not structural damage.
* One complex, six mutants, one seed, one noise level, TMol as both guide and judge; nothing here
  compares with experiment.

## Output naming conventions

Each script invocation creates `runs/<YYYY-MM-DD_HHMMSS>_<name>/` (a numeric suffix is added on a
collision; nothing is overwritten) containing `config.yaml`, `metadata.json` and a results CSV.
`<name>` is `recovery_<space>_<ID>` (recovery benchmark; one directory for the control and one for
the perturbed grid), `random_mutants`, `position_scan`, `mutate_refine`, `mutant_scan`, or the
config's `name` for `cinder design`.

PDB files in `structures/`:

* recovery: `<ID>_reference.pdb`, `<ID>_<perturbation>_<k>_start.pdb` / `_final.pdb`, where `k` is the
  0-based repeat and the perturbation slug is `rigid_r<deg>_t<Å>`, `tors_s<deg>`,
  `tors_s<deg>_r<deg>_t<Å>` or `cart_s<Å>` (zero values are the controls);
* random mutants: `crystal_reference.pdb`, `wild_type_repacked.pdb`, `<biased|uniform>_<mutation>.pdb`;
* position scan: the same two references plus `<native><index><substitute>.pdb` (by default only the
  best and worst substitution per position);
* mutate-refine: `crystal_reference.pdb` and `<mutation>_<noise>_<stage>.pdb`, with noise slug
  `t<deg>_r<deg>_s<Å>_c<Å>` (no noise: `t0_r0_s0_c0`) and stage `baseline`, `baseline_relaxed`,
  `start`, `stage_a` or `final`.

A mutation is `<native><index><new>`, joined by `+` (`L33R+D34H`), with a **0-based index along the
binder chain** (not the PDB residue number: barstar Y29, N33, D35, D39, E76 are indices 28, 32, 34,
38, 73). CSV columns are `<prefix>_<quantity>` with prefixes `start`, `final`, `baseline`,
`baseline_relaxed`, `stage_a`, and quantities `total`, `interface`, `repulsion`, `strain`,
`internal`, `ca_rmsd` (target frame), `ca_aligned` (after superposition, i.e. fold distortion).
Mutant PDBs have different atom counts from the crystal (mutations add or remove atoms); superimpose
them by chain and residue, not by atom index. Within one mutate-refine mutant all stage files share
one atom layout.

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
