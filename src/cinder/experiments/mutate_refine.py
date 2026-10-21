"""Noise the structure, mutate the sequence, then use gradients to find a good structure.

Protocol (wiring layer; TMol-specific, imported lazily by callers):

1. **Noise** the native binder in every way requested by a :class:`~cinder.validation.NoiseSpec`:
   torsion (phi/psi), rigid-body jump, and Cartesian noise, on the native topology.
2. **Mutate**: the packer builds the mutant's side chains *on that noisy backbone*
   (:meth:`TMolSequenceBuilder.build_from`).
3. **Stage A (gradients in internal space):** annealed Langevin in the mutant's binder torsions
   (phi, psi and chi) plus the six rigid-body jump DOFs, with the binder's own (``internal``)
   energy so the fold is held together, then a noise-free quench. Bond lengths and angles are
   fixed in this space.
4. **Stage B (Cartesian polish):** a short Cartesian run on the binder, then a noise-free quench;
   this can repair the bond geometry that Cartesian noise distorted.

Each stage ends with a quench because Langevin sampling at finite guidance carries thermal excess
energy (about ``N_dof / (2 lambda)`` by equipartition), which would otherwise mask the structure's
true energy.

The reference for judging a new sequence is **that sequence packed on the native backbone**
(``baseline``), not the native structure. Results carry every intermediate structure so they can
be written out and inspected.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch
from torch import Tensor

from cinder.config import EnergyConfig
from cinder.conformations.guidance import linear_schedule
from cinder.conformations.langevin import LangevinSampler
from cinder.energy.parametrized import ParametrizedEnergy
from cinder.energy.tmol_design import build_design_energy_for_context
from cinder.energy.tmol_mutation import MutantPoses, TMolSequenceBuilder
from cinder.energy.tmol_torsion import CHI_TORSIONS, TMolTorsionParametrization
from cinder.systems.rigid_body import ComposedParametrization, RigidBodyParametrization
from cinder.systems.specification import MolecularSystem
from cinder.types import SequenceBatch, encode_sequence
from cinder.validation.noise import NoiseSpec
from cinder.validation.perturb import cartesian_perturb, kabsch_rmsd, rmsd

log = logging.getLogger("cinder.mutate_refine")

PHI_PSI = ("phi", "psi")


@dataclass(frozen=True)
class RefineSettings:
    """Sampler settings for the two refinement stages."""

    torsions: tuple[str, ...] = (*PHI_PSI, *CHI_TORSIONS)
    stage_a_steps: int = 2000
    stage_a_step_size: float = 2e-5
    lam_start: float = 5.0
    lam_end: float = 50.0
    stage_b_steps: int = 300
    stage_b_step_size: float = 1e-5
    quench_steps: int = 300  # noise-free gradient descent ending each stage (0 = none)
    max_grad_norm: float | None = 2000.0
    internal_weight: float = 1.0  # binder's own energy; needed for torsion DOFs


class StructureNoiser:
    """Applies a :class:`NoiseSpec` to the native binder (native topology coordinates)."""

    def __init__(self, system: MolecularSystem, torsions: tuple[str, ...] = PHI_PSI) -> None:
        binder_range: Any = system.extras["binder_residue_range"]
        self.system = system
        self.binder_atoms = cast(Tensor, system.extras["binder_atoms"])
        self.torsion = TMolTorsionParametrization(
            system.extras["tmol"],  # type: ignore[arg-type]
            (int(binder_range[0]), int(binder_range[1])),
            torsions,
        )
        self.rigid = RigidBodyParametrization(system.reference_coordinates, self.binder_atoms)
        self.composed = ComposedParametrization(self.torsion, self.rigid)

    def apply(self, noise: NoiseSpec, generator: torch.Generator | None = None) -> Tensor:
        """Noisy coordinates ``[n_atoms, 3]``; with no noise, the crystal coordinates."""
        ref = self.system.reference_coordinates
        if noise.is_none:
            return ref.clone()
        q = torch.cat(
            [
                self.torsion.dofs(noise.torsion_deg, generator),
                self.rigid.dofs(noise.rigid_rotation_deg, noise.rigid_shift, generator),
            ]
        )
        x = self.composed.to_cartesian(q[None])[0]
        if noise.cartesian_sigma:
            x = cartesian_perturb(x, self.binder_atoms, noise.cartesian_sigma, generator)
        return x


@dataclass
class StageSnapshot:
    """One structure along the protocol, with its design-energy terms and CA RMSDs to the native
    backbone (``ca_rmsd``: target frame; ``ca_aligned``: after superposition = fold distortion)."""

    name: str
    coordinates: Tensor
    terms: dict[str, float]
    ca_rmsd: float
    ca_aligned: float


@dataclass
class MutateRefineResult:
    sequence: str
    noise: NoiseSpec
    system: MolecularSystem  # the mutant system (atom layout of every snapshot except crystal)
    mutant: MutantPoses
    baseline: StageSnapshot  # same sequence packed on the native backbone
    baseline_relaxed: StageSnapshot  # ... and put through the same Cartesian polish
    start: StageSnapshot  # mutant packed on the noisy backbone
    stage_a: StageSnapshot
    final: StageSnapshot
    seconds: float
    snapshots: list[StageSnapshot] = field(default_factory=list)

    @property
    def energy_gap(self) -> float:
        """``E(final) - E(baseline_relaxed)`` of the total design energy: the refined noisy
        structure against the same sequence on the native backbone after the identical polish.
        <= 0 means the refinement found a structure at least as good as that reference."""
        return self.final.terms["total"] - self.baseline_relaxed.terms["total"]


def _terms(energy: Any, system: MolecularSystem, x: Tensor, seq: SequenceBatch) -> dict[str, float]:
    out = energy.evaluate(system, x[None], seq)
    return {"total": float(out.total[0]), **{k: float(v[0]) for k, v in out.terms.items()}}


def _polish(
    energy: Any,
    system: MolecularSystem,
    x: Tensor,
    seq: SequenceBatch,
    settings: RefineSettings,
    generator: torch.Generator | None,
) -> Tensor:
    """Cartesian refinement of the binder: a short Langevin run at ``lam_end`` followed by a
    noise-free quench. ``system.move_mask`` restricts motion to the binder."""
    clip = settings.max_grad_norm
    langevin = LangevinSampler.from_energy(
        energy,
        strength=settings.lam_end,
        step_size=settings.stage_b_step_size,
        n_steps=settings.stage_b_steps,
        max_grad_norm=clip,
    )
    x = langevin.sample(system, seq, 1, generator, initial_coordinates=x).coordinates[0, 0]
    if settings.quench_steps:
        quench = LangevinSampler.from_energy(
            energy,
            strength=settings.lam_end,
            step_size=settings.stage_b_step_size,
            n_steps=settings.quench_steps,
            max_grad_norm=clip,
            noise_scale=0.0,
        )
        x = quench.sample(system, seq, 1, generator, initial_coordinates=x).coordinates[0, 0]
    return x


def run_mutate_refine(
    system: MolecularSystem,
    builder: TMolSequenceBuilder,
    noiser: StructureNoiser,
    energy_config: EnergyConfig,
    settings: RefineSettings,
    noise: NoiseSpec,
    sequence: str,
    generator: torch.Generator | None = None,
) -> MutateRefineResult:
    """Run the full protocol for one ``sequence`` (full binder sequence) and ``noise``."""
    t0 = time.time()
    cfg = replace(energy_config, internal_weight=settings.internal_weight)
    tokens = encode_sequence(sequence)
    seq1 = SequenceBatch(tokens[None])
    native_ca = system.reference_coordinates[cast(Tensor, system.extras["binder_ca_atoms"])]

    def snapshot(name: str, msys: MolecularSystem, energy: Any, x: Tensor) -> StageSnapshot:
        ca = x[cast(Tensor, msys.extras["binder_ca_atoms"])]
        return StageSnapshot(
            name, x.detach().clone(), _terms(energy, msys, x, seq1),
            rmsd(ca, native_ca), kabsch_rmsd(ca, native_ca),
        )  # fmt: skip

    # baseline: this sequence packed on the native backbone (the target to match or beat)
    base_mutant = builder.build(sequence)
    base_sys = builder.system_for(sequence)
    base_ctx: Any = base_sys.extras["tmol"]
    base_energy = build_design_energy_for_context(cfg, base_sys, base_ctx, tokens)
    baseline = snapshot("baseline", base_sys, base_energy, base_mutant.coordinates)

    # 1-2. noise the native structure, then mutate on the noisy backbone
    x_noisy = noiser.apply(noise, generator)
    mutant = builder.build_from(sequence, x_noisy)
    msys = builder.system_from(mutant)
    ctx: Any = msys.extras["tmol"]
    energy = build_design_energy_for_context(cfg, msys, ctx, tokens)
    start = snapshot("start", msys, energy, mutant.coordinates)

    # 3. stage A: annealed Langevin in torsions (phi/psi/chi) + rigid-body jumps
    binder_range: Any = msys.extras["binder_residue_range"]
    torsion = TMolTorsionParametrization(
        ctx, (int(binder_range[0]), int(binder_range[1])), settings.torsions
    )
    rigid = RigidBodyParametrization(
        msys.reference_coordinates, cast(Tensor, msys.extras["binder_atoms"])
    )
    composed = ComposedParametrization(torsion, rigid)
    sampler_a = LangevinSampler.from_energy(
        ParametrizedEnergy(energy, composed, msys),
        strength=linear_schedule(settings.lam_start, settings.lam_end),
        step_size=settings.stage_a_step_size,
        n_steps=settings.stage_a_steps,
        max_grad_norm=settings.max_grad_norm,
    )
    dof_msys = composed.dof_system(msys)
    q = sampler_a.sample(
        dof_msys, seq1, 1, generator, initial_coordinates=torch.zeros(composed.n_dofs)
    ).coordinates[0, 0]
    if settings.quench_steps:
        quench_a = LangevinSampler.from_energy(
            ParametrizedEnergy(energy, composed, msys),
            strength=settings.lam_end,
            step_size=settings.stage_a_step_size,
            n_steps=settings.quench_steps,
            max_grad_norm=settings.max_grad_norm,
            noise_scale=0.0,
        )
        q = quench_a.sample(dof_msys, seq1, 1, generator, initial_coordinates=q).coordinates[0, 0]
    x_a = composed.to_cartesian(q[None])[0].detach()
    stage_a = snapshot("stage_a", msys, energy, x_a)

    # 4. stage B: Cartesian polish of the binder (move_mask limits it to binder atoms)
    final = snapshot("final", msys, energy, _polish(energy, msys, x_a, seq1, settings, generator))

    # fair reference: the same sequence on the native backbone, put through the same polish
    baseline_relaxed = snapshot(
        "baseline_relaxed",
        base_sys,
        base_energy,
        _polish(base_energy, base_sys, base_mutant.coordinates, seq1, settings, generator),
    )

    return MutateRefineResult(
        sequence=sequence,
        noise=noise,
        system=msys,
        mutant=mutant,
        baseline=baseline,
        baseline_relaxed=baseline_relaxed,
        start=start,
        stage_a=stage_a,
        final=final,
        seconds=time.time() - t0,
        snapshots=[baseline, baseline_relaxed, start, stage_a, final],
    )
