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
