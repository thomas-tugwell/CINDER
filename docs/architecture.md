# Architecture

```
q_phi(S | T)            SequencePolicy        sequence/categorical.py
    │  S
    ▼
p(X | S, T)             ConformationModel     conformations/{langevin, normalizing_flow, …}
    │  X_1 … X_K            ▲
    │                       └─ EnergyGuidance (conformations/guidance.py): s_base − λ∇U
    ▼
U(X, S, T)              EnergyModel           energy/{toy, tmol, composite, restraints}
    │  U_k
    ▼
R(S) = −agg_k U_k       EnsembleObjective     objectives/ensemble.py
    │  R
    ▼
REINFORCE + entropy     SequenceTrainer       optimization/{reinforce, trainer}.py
```

```
RewardModel (types.py)  sequences -> reward + ensemble energies, the only thing the trainer needs
  SampledEnsembleReward   one conformation model + one energy (toy)
  TMolMutantReward        per-sequence packed pose + jump/torsion ensemble (experiments/tmol_reward.py)
CoordinateParametrization  q -> Cartesian: RigidBody (6), TMolTorsion (k), Composed
SequencePolicy            CategoricalPolicy (independent), AutoregressivePolicy (coupled positions)
```

## Boundaries

* All interfaces and containers live in `src/cinder/types.py`.
* `sequence/` imports nothing about energies. `conformations/` never imports an energy
  implementation; it receives an `EnergyModel` (via `EnergyGuidance`) from the caller.
* TMol is imported **only** in `energy/tmol*.py` (`tmol.py`, `tmol_mutation.py`) and lazily from
  the runner.
* `experiments/runner.py` is the single place concrete classes are wired together;
  `cli.py` is a thin wrapper around it.

## Equation → module

| Equation | Module |
|---|---|
| $U_{\mathrm{design}}=\sum w_kU_k$ | `energy/composite.py`, `objectives/interface.py` |
| $s_{\mathrm{guided}}=s_{\mathrm{base}}-\lambda\nabla U$ | `conformations/guidance.py` |
| Langevin step | `conformations/langevin.py` |
| $L=\mathbb E_q[\lambda U+\log q]$ | `conformations/normalizing_flow/train.py` |
| $R(S)=-\mathrm{agg}_k U_k$, $F_q$ | `objectives/ensemble.py` |
| $(R-b)\nabla\log q-\alpha H$ | `optimization/reinforce.py` |

## Data shapes

Sequences `[B, L]` (full binder) or `[B, P]` (designable positions); conformations
`[B, K, *coord_shape]`; energies `[B, K]`; rewards `[B]`. `DesignSpace.assemble` inserts
designed residues into the native scaffold.

## Run directory

```
runs/<timestamp>_<name>/  config.yaml  metadata.json  metrics.csv  sequences.csv
                          probabilities.csv  checkpoints/  structures/
```
Never overwritten; collisions get a numeric suffix.
