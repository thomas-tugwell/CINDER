"""Structure-recovery protocol: perturb a reference pose, guide it back, measure RMSD.

Only the energy (and sampler dynamics) pull the structure back: callers must not include a
restraint to the reference in ``energy`` or the test becomes circular.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from cinder.conformations.langevin import LangevinSampler
from cinder.energy.composite import CompositeEnergy
from cinder.energy.parametrized import CoordinateParametrization
from cinder.systems.rigid_body import RigidBodyParametrization
from cinder.systems.specification import MolecularSystem
from cinder.types import SequenceBatch
from cinder.validation.perturb import cartesian_perturb, kabsch_rmsd, rigid_body_perturb, rmsd


@dataclass(frozen=True)
class Perturbation:
    """``kind``: ``rigid`` (rotation_deg, translation), ``cartesian`` (sigma, Å) or ``torsion``
    (sigma_deg: std of noise added to every selected torsion; needs a torsion parametrization) or
    ``torsion_rigid`` (sigma_deg plus a rigid jump of rotation_deg / translation)."""

    kind: str
    rotation_deg: float = 0.0
    translation: float = 0.0
    sigma: float = 0.0
    sigma_deg: float = 0.0

    def label(self) -> str:
        if self.kind == "rigid":
            return f"rigid {self.rotation_deg:g}° / {self.translation:g} Å"
        if self.kind == "torsion":
            return f"torsion σ={self.sigma_deg:g}°"
        if self.kind == "torsion_rigid":
            return (
                f"torsion σ={self.sigma_deg:g}° + rigid {self.rotation_deg:g}° / "
                f"{self.translation:g} Å"
            )
        return f"cartesian σ={self.sigma:g} Å"

    def slug(self) -> str:
        """Filesystem-safe identifier, e.g. ``rigid_r2_t1`` or ``cart_s0.5``."""
        if self.kind == "rigid":
            return f"rigid_r{self.rotation_deg:g}_t{self.translation:g}"
        if self.kind == "torsion":
            return f"tors_s{self.sigma_deg:g}"
        if self.kind == "torsion_rigid":
            return f"tors_s{self.sigma_deg:g}_r{self.rotation_deg:g}_t{self.translation:g}"
        return f"cart_s{self.sigma:g}"

    def apply(
        self, coords: Tensor, atoms: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        if self.kind == "rigid":
            return rigid_body_perturb(coords, atoms, self.rotation_deg, self.translation, generator)
        if self.kind == "cartesian":
            return cartesian_perturb(coords, atoms, self.sigma, generator)
        raise ValueError(
            f"perturbation kind {self.kind!r} cannot be applied in Cartesian space "
            "(torsion perturbations need a torsion parametrization)"
        )


@dataclass
class RecoveryResult:
    """``start_rmsd`` / ``final_rmsd`` are the primary metric: RMSD over ``metric_atoms``
    (CA atoms when provided) in the fixed target frame. ``metrics`` additionally holds
    ``(start, final)`` pairs: ``ca_aligned`` (same atoms after optimal superposition, i.e. is
    the fold preserved) and ``all_atom`` (every perturbed atom, target frame)."""

    perturbation: Perturbation
    start_rmsd: float
    final_rmsd: float
    start_terms: dict[str, float]
    final_terms: dict[str, float]
    seconds: float
    metrics: dict[str, tuple[float, float]] = field(default_factory=dict)
    start_coordinates: Tensor = field(repr=False, default_factory=lambda: torch.empty(0))
    final_coordinates: Tensor = field(repr=False, default_factory=lambda: torch.empty(0))

    @property
    def improvement(self) -> float:
        """Positive when the guided structure ended closer to the reference."""
        return self.start_rmsd - self.final_rmsd


def _terms(
    energy: CompositeEnergy, system: MolecularSystem, x: Tensor, seq: SequenceBatch
) -> dict[str, float]:
    out = energy.evaluate(system, x[None], seq)
    return {"total": float(out.total[0]), **{k: float(v[0]) for k, v in out.terms.items()}}


def _summarize(
    perturbation: Perturbation,
    system: MolecularSystem,
    energy: CompositeEnergy,
    sequence: SequenceBatch,
    start: Tensor,
    final_x: Tensor,
    atoms: Tensor,
    m_atoms: Tensor,
    seconds: float,
) -> RecoveryResult:
    ref = system.reference_coordinates
    return RecoveryResult(
        perturbation=perturbation,
        start_rmsd=rmsd(start, ref, m_atoms),
        final_rmsd=rmsd(final_x, ref, m_atoms),
        metrics={
            "ca_aligned": (
                kabsch_rmsd(start, ref, m_atoms),
                kabsch_rmsd(final_x, ref, m_atoms),
            ),
            "all_atom": (rmsd(start, ref, atoms), rmsd(final_x, ref, atoms)),
        },
        start_terms=_terms(energy, system, start, sequence),
        final_terms=_terms(energy, system, final_x, sequence),
        seconds=seconds,
        start_coordinates=start,
        final_coordinates=final_x,
    )


def run_recovery(
    system: MolecularSystem,
    energy: CompositeEnergy,
    sampler: LangevinSampler,
    sequence: SequenceBatch,
    perturbation: Perturbation,
    atoms: Tensor,
    generator: torch.Generator | None = None,
    metric_atoms: Tensor | None = None,
) -> RecoveryResult:
    """Perturb ``system.reference_coordinates`` on ``atoms``, run ``sampler`` from there, and
    compare RMSD before and after. ``metric_atoms`` (default: ``atoms``) selects the atoms for
    the primary RMSD, typically the binder CA atoms."""
    m_atoms = atoms if metric_atoms is None else metric_atoms
    ref = system.reference_coordinates
    start = perturbation.apply(ref, atoms, generator)
    t0 = time.time()
    final = sampler.sample(system, sequence, 1, generator, initial_coordinates=start)
    final_x = final.coordinates[0, 0]
    return _summarize(
        perturbation, system, energy, sequence, start, final_x, atoms, m_atoms, time.time() - t0
    )


def run_dof_recovery(
    system: MolecularSystem,
    energy: CompositeEnergy,
    param: CoordinateParametrization,
    sampler: LangevinSampler,
    sequence: SequenceBatch,
    perturbation: Perturbation,
    q0: Tensor,
    atoms: Tensor,
    generator: torch.Generator | None = None,
    metric_atoms: Tensor | None = None,
) -> RecoveryResult:
    """Recovery in the DOF space of ``param`` (rigid-body jumps, torsions, or both).

    The system starts at ``q0``; ``sampler`` moves only those DOFs and must be built on
    ``ParametrizedEnergy(energy, param, system)`` and run on ``param.dof_system(system)``.
    ``atoms`` are the atoms perturbed (for the all-atom RMSD); ``metric_atoms`` (default
    ``atoms``) selects the primary RMSD, typically the binder CA atoms.
    """
    m_atoms = atoms if metric_atoms is None else metric_atoms
    start = param.to_cartesian(q0[None])[0]
    t0 = time.time()
    final = sampler.sample(param.dof_system(system), sequence, 1, generator, initial_coordinates=q0)
    final_x = param.to_cartesian(final.coordinates[0, 0][None])[0]
    return _summarize(
        perturbation, system, energy, sequence, start, final_x, atoms, m_atoms, time.time() - t0
    )


def run_rigid_recovery(
    system: MolecularSystem,
    energy: CompositeEnergy,
    param: RigidBodyParametrization,
    sampler: LangevinSampler,
    sequence: SequenceBatch,
    perturbation: Perturbation,
    generator: torch.Generator | None = None,
    metric_atoms: Tensor | None = None,
) -> RecoveryResult:
    """Recovery in rigid-body jump space: a random 6-DOF jump, then sampling of those DOFs only
    (the binder's internal geometry is preserved throughout)."""
    if perturbation.kind != "rigid":
        raise ValueError("run_rigid_recovery needs a 'rigid' perturbation")
    q0 = param.dofs(perturbation.rotation_deg, perturbation.translation, generator)
    return run_dof_recovery(
        system, energy, param, sampler, sequence, perturbation, q0, param.moving_atoms,
        generator, metric_atoms,
    )  # fmt: skip


def result_row(complex_id: str, repeat: int, result: RecoveryResult) -> dict[str, object]:
    """Flatten a result into one CSV row (all energy terms, before and after)."""
    row: dict[str, object] = {
        "complex": complex_id,
        "perturbation": result.perturbation.slug(),
        "repeat": repeat,
        "start_rmsd": result.start_rmsd,
        "final_rmsd": result.final_rmsd,
        "improvement": result.improvement,
        "seconds": result.seconds,
    }
    for name, (start, final) in result.metrics.items():
        row[f"start_rmsd_{name}"] = start
        row[f"final_rmsd_{name}"] = final
    for name, value in result.start_terms.items():
        row[f"start_{name}"] = value
    for name, value in result.final_terms.items():
        row[f"final_{name}"] = value
    return row
