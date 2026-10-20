# Implementation plan

Order of work (each stage ends with a green test run):

1. Foundation: package skeleton, pyproject, config, typing, CI, docs skeleton.
2. Core types + Protocol interfaces (`EnergyModel`, `ConformationModel`, `SequencePolicy`,
   `EnsembleObjective`, `ScoreModel`).
3. Analytic toy energies; Langevin sampler validated against Gaussian / double-well moments;
   energy guidance validated against the analytic product-of-Gaussians.
4. Categorical sequence policy, masks, REINFORCE with batch baseline + entropy bonus.
5. Toy end-to-end loop S -> X -> U -> R -> S with a brute-force known optimum.
6. Run directories, metadata, checkpoints, CLI.
7. Small conditional normalizing flow trained with the variational objective on toy energies.
8. Diffusion / flow-matching / BioEmu2: interfaces + documented experimental stubs.
9. Documentation and README.
10. TMol adapter, written against the *installed* TMol API (inspected, not assumed).

Non-goals for this iteration: protein-scale generative models, Transformer policies,
absolute binding free energies, distributed training.
