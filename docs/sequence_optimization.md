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
