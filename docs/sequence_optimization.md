# Sequence optimisation

`CategoricalPolicy` holds logits `[P, 20]` and a hard boolean mask from `DesignSpace`
(fixed residues, allowed subsets, forbidden amino acids). Initialise `uniform` or `native`.

Loop (`optimization/trainer.py`): sample N sequences → K conformations each → design energy →
aggregate → reward → REINFORCE with batch baseline and entropy bonus.

Logged: `metrics.csv` (iteration, loss, reward mean/max/variance, entropy, unique sequences,
best reward/sequence, mean of each energy component), `sequences.csv`, `probabilities.csv`
(per-position AA probabilities every `log_probs_every` iterations), checkpoints.

Toy validation: `ToySequenceEnergy` has a closed-form ensemble mean, so the optimum is found
by brute force; `tests/integration/test_toy_end_to_end.py` asserts the policy's consensus
sequence equals it.

Variance control: the batch baseline is the only variance reduction so far; use
`normalize_advantage` and larger batches/K if training is noisy.

## Policies

* `categorical` (default): independent positions, `q(S) = prod_i q(S_i)`.
* `autoregressive`: `q(S) = prod_i q(S_i | S_<i)` with logits `b_i + sum_{j<i} J[i,j][:, S_j]`, so
  choices at one position can depend on earlier ones (e.g. a donor only if an acceptor is present).
  `log_prob` is exact; the entropy bonus uses an unbiased Monte-Carlo estimate. With a zero
  coupling it equals the categorical policy. Test: maximising `E[R] + alpha H` on a reward that
  needs coupled pairs, the autoregressive policy reaches the optimum `alpha log sum exp(R/alpha)`
  (1.667 vs 1.68) while the independent policy plateaus at 1.551. Without an entropy term both
  collapse onto one good pair and are equally good: coupling buys the ability to represent several
  good solutions at once.

Select with `sequence.backend`.

## Reward models

The trainer takes any `RewardModel`. `conformation.backend: langevin` (toy systems) samples one
shared conformation model; `rigid_langevin` (TMol structures) builds a packed pose per sequence and
a jump-space ensemble around it, cached per sequence (`conformation.options`: `repack`,
`repack_cutoff`). `n_steps: 0, init_std: 0, n_samples: 1` degenerates to the single packed
structure, the baseline for the ensemble comparison. Example: `configs/experiment/design_tmol.yaml`.

## Exact-gradient optimisation on pairwise tables (no REINFORCE)

Choosing an amino acid is discrete, but if the energy is (approximately) a sum of one-body and
two-body terms, `U(S) ~ e0 + sum_i e1[i, s_i] + sum_{(i,j)} e2[ij][s_i, s_j]`, then the expected
energy under the factorised policy is **exact and smooth in the logits**:
`E_q[U] = e0 + sum_i p_i . e1[i] + sum_pairs p_i^T e2[ij] p_j`
(`objectives/pairwise_table.py`). `optimization/expected_energy.py` minimises
`E_q[U] - alpha * H[q]` by gradient descent, annealing `alpha` toward 0, so there is no sampling
variance. Properties that are tested: the expectation equals brute-force enumeration; its gradient
matches finite differences; for a one-body table the optimum is exactly the Boltzmann distribution
`p ∝ exp(-e1/alpha)`; on the toy sequence problem (exactly pairwise) the trainer lands on the
brute-force optimum.

Tables are built from any scorer by inclusion-exclusion around the native sequence
(`tables_from_scorer`): `e1[i][a] = U(a at i) - U(native)`, and for residues in contact
(`contact_pairs`) `e2[ij][a,b] = U(a at i, b at j) - U(native) - e1[i][a] - e1[j][b]`. One- and
two-body substitutions are reproduced exactly; three-body effects are not, and
`approximation_error` measures that on held-out multi-mutants. With the single packed TMol
structure as the scorer (`scripts/table_design.py`) the table is a fast proposal generator on a
**fixed native backbone**: it ignores ensembles and any backbone adaptation, so verify winners with
the real scorer.

### Failure found on barnase/barstar, and the guards

With 9 interface positions, 11 contact pairs and a 11-letter alphabet, the plain optimiser
selected an 8-mutation design that the table scored at -10 against native but the real scorer
scored at +143 (the other 19 policy samples collapsed onto it; median table-vs-true error 285).
The held-out check (2 to 4 random mutations) had not flagged it. Cause: inclusion-exclusion
double-counts a clash relief that can only happen once (two mutations each remove the same
clash, so their joint energy is not the sum), and the optimiser finds and exploits that.

Two guards are implemented, both reported by `scripts/table_design.py` (`--cap`,
`--mutation-cost`, `--load-table` to re-optimise a saved table without rescoring):

* `PairwiseEnergyTable.clipped(cap)` clips every single and joint energy at a ceiling before the
  optimisation, so a huge clash cannot be cancelled.
* `mutation_cost` charges the optimiser per expected mutation, keeping it where the table is exact.

Verified results on the saved table (true single-structure energy, native 145.3):

| cap | mutation cost | selected design | table | true |
|---|---|---|---|---|
| 20 | 0 | 8 mutations | 135.3 | 288 to 1322 (still exploited) |
| 50 | 0 | L33A+D38N+T41R+G42A | 144.7 | 147.7 |
| 50 or 100 | 0.5 | T41A | 143.3 | 143.3 |
| 100 | 2 | native | 145.3 | 145.3 |

So with the guards the only verified improvement is a single mutation (T41A, -1.9); barnase/barstar
is a natural high-affinity pair, so little headroom is expected. This is one complex and one
fixed backbone: it shows the guards stop the exploit, not that the method finds better binders.
Always verify winners with the real scorer.

### Benchmark against simple baselines (negative result)

`scripts/table_design.py` also scores three baselines with the same single-structure scorer:
the best single mutant, the one-body optimum (all pair terms ignored) and the best of 300 random
1 to 4 mutation designs. Settings were fixed beforehand (cap 50, mutation cost 0.5, taken from the
barnase run), so only barnase/barstar was used for tuning. True energy change against native
(negative is better):

| complex (positions) | table optimiser | best single | one-body | random (300 calls) |
|---|---|---|---|---|
| barnase/barstar (9) | -1.9 | -1.9 | -1.9 | -1.6 |
| 2ZTA (3) | -1.1 | -1.9 | -1.1 | -3.0 |
| 1CDL (8) | +15.2 | -7.5 | -7.5 | -7.5 |
| 1G5J (17, contact shell) | +185.2 | -18.7 | -26.3 | -14.1 |

The table optimiser never beat the baselines. On 1CDL and 1G5J its designs (8 and 17 mutations)
were far worse than native although the table scored them far better (1CDL -63 vs true -0.5;
1G5J -381 vs true +331): the table is exact only for at most two mutations, and the optimiser
walks straight into the region where it is not. Interface mutations on 1CDL are strongly
non-additive (R15L -7.5 and R0Q -2.7 alone, yet +2.6 together; the scorer is deterministic, so
this is real). The 2ZTA space (3 positions) is too small to say much.

### Greedy hill climbing on the real scorer

`hill_climb` (`optimization/baselines.py`) starts at native, repeatedly applies the best improving
single mutation (also trying double moves among the top six singles), and accepts only moves the
real scorer confirms. It needs no surrogate, so it cannot walk into a region where a table is
invalid. Same scorer, same settings (true energy change against native, negative is better):

| complex | table optimiser | best single | random (300 calls) | hill climb |
|---|---|---|---|---|
| barnase/barstar | -1.9 | -1.9 | -1.6 | -2.0 (L33A+T41A) |
| 1CDL | +15.2 | -7.5 | -7.5 | **-14.7** (Q4Y+K5T+H8Y+R15L) |
| 2ZTA | -1.1 | -1.9 | **-3.0** | -2.1 |

Hill climbing is the best or near-best on all three and is the only method that finds the
multi-mutation gain on 1CDL, where the surrogate fails most. One run per complex, one scorer
(single native-backbone structure), so this shows it is a stronger baseline, not a general ranking.

Conclusion: a fixed-backbone pairwise table with these guards is not a better design method than
scanning single mutants or greedy search on the real scorer. Unless it is made trust-region aware
(restrict to the region where the table was validated, re-fit the table around the current design
and iterate, or add a three-body check), do not use it for design. The machinery (exact expected energy and gradients) is correct;
the surrogate is the limit.

