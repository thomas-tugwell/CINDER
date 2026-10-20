# Conformation models

All implement `ConformationModel.sample(system, sequence, n_samples)` and `log_prob(...)`.

| Model | Status | `log_prob` |
|---|---|---|
| Langevin (energy-only or score-guided) | implemented, validated on analytic targets | unsupported |
| Conditional RealNVP flow (toy coordinates) | implemented, validated vs. $-\log Z$ | exact |
| Diffusion | stub | unsupported |
| Flow matching | stub | candidate (ODE change of variables) |
| BioEmu2 adapter | stub, fails gracefully if absent | unsupported |

## Normalising flow

Trained with the reverse-KL objective $\mathbb E_q[\lambda U+\log q]$. Tests: Gaussian target
(bound tight to $-\log Z$), bimodal mixture (both modes covered at the chosen separation) and
sequence-dependent means via a one-hot context. Reverse KL is mode-seeking: for well-separated
modes expect collapse onto a subset; this is documented behaviour, not a bug.

## Protein-scale representations (Milestone 6, undecided)

`atom14` + mask, SE(3)-equivariant Cartesian, residue rigid frames, torsional, internal
coordinates. Each must solve the variable-atom-count problem (see `tmol.md`).

## Guidance

Never re-implement guidance inside a model: wrap its score/velocity with
`EnergyGuidance` / `energy_guided_velocity`.

## BioEmu2

Intended as $p_{\mathrm{base}}$ with optional TMol guidance. Not wired: its API must be
inspected from an installed version first.
