"""CLI for the structure-recovery benchmark (library code: ``cinder.validation``). Needs TMol.

Perturbs the binder away from the complex fold, then runs TMol-guided Langevin with NO restraint
to the reference, so only the physics pulls the structure back.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from cinder.config import config_from_dict
from cinder.conformations import LangevinSampler
from cinder.conformations.guidance import linear_schedule
from cinder.energy.composite import CompositeEnergy
from cinder.energy.parametrized import CoordinateParametrization, ParametrizedEnergy
from cinder.energy.tmol import load_structure_system, write_pdb
from cinder.energy.tmol_design import build_tmol_design_energy
from cinder.energy.tmol_torsion import TMolTorsionParametrization
from cinder.experiments.logging import (
    CsvLogger,
    collect_metadata,
    create_run_dir,
    write_run_header,
)
from cinder.systems.rigid_body import ComposedParametrization, RigidBodyParametrization
from cinder.systems.specification import MolecularSystem
from cinder.types import EnergyModel, SequenceBatch
from cinder.validation import (
    Perturbation,
    result_row,
    run_dof_recovery,
    run_recovery,
)
from cinder.validation.datasets import resolve_complex


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--complex", help="PDB id from --manifest (downloaded to --data-dir)")
    ap.add_argument("--manifest", default="configs/benchmarks/helical_complexes.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pdb", default="tests/fixtures/barnase_barstar_AD.pdb")
    ap.add_argument("--target", default="A")
    ap.add_argument("--binder", default="D")
    ap.add_argument("--allow-breaks", action="store_true", help="skip the chain-break check")
    ap.add_argument("--rot", type=float, nargs="*", default=[3.0], help="rigid rotations (deg)")
    ap.add_argument("--trans", type=float, nargs="*", default=[1.0], help="rigid shifts (Å)")
    ap.add_argument("--cart-noise", type=float, nargs="*", default=[], help="Cartesian σ (Å)")
    ap.add_argument(
        "--space",
        choices=["cartesian", "rigid", "torsion", "torsion_rigid"],
        default="cartesian",
        help="cartesian: every binder atom moves; rigid: the 6 binder jump DOFs only; torsion: "
        "binder torsions only (use --internal-weight > 0); torsion_rigid: both",
    )
    ap.add_argument("--torsion-sigma", type=float, nargs="*", default=[1.0], help="deg per angle")
    ap.add_argument("--torsions", nargs="*", default=["phi", "psi"], help="phi psi chi1 ...")
    ap.add_argument("--internal-weight", type=float, default=None, help="binder-alone energy")
    ap.add_argument("--n", type=int, default=2, help="repeats per setting")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--step-size", type=float, default=2e-5)
    ap.add_argument("--lam0", type=float, default=5.0)
    ap.add_argument("--lam1", type=float, default=50.0)
    ap.add_argument("--clip", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="runs", help="a new run directory is created here")
    return ap.parse_args()


@dataclass
class Space:
    """What the sampler moves: the DOF parametrization, the energy it differentiates, and the
    perturbations to apply."""

    name: str
    param: CoordinateParametrization | None
    energy: EnergyModel
    perturbations: list[Perturbation]
    rigid: RigidBodyParametrization | None = None
    torsion: TMolTorsionParametrization | None = None

    def start(self, p: Perturbation, gen: torch.Generator) -> Tensor:
        """Starting DOFs ``q0`` for perturbation ``p`` (DOF spaces only)."""
        parts: list[Tensor] = []
        if self.torsion is not None:
            parts.append(self.torsion.dofs(p.sigma_deg, gen))
        if self.rigid is not None:
            parts.append(self.rigid.dofs(p.rotation_deg, p.translation, gen))
        return torch.cat(parts)


def make_space(args: argparse.Namespace, system: MolecularSystem, energy: CompositeEnergy) -> Space:
    atoms = system.extras["binder_atoms"]
    rigid_p = [Perturbation("rigid", r, t) for r in args.rot for t in args.trans]
    if args.space == "cartesian":
        cart = [Perturbation("cartesian", sigma=s) for s in args.cart_noise]
        return Space("cartesian", None, energy, rigid_p + cart)
    rigid = RigidBodyParametrization(system.reference_coordinates, atoms)
    if args.space == "rigid":
        return Space(
            "rigid", rigid, ParametrizedEnergy(energy, rigid, system), rigid_p, rigid=rigid
        )
    torsion = TMolTorsionParametrization(
        system.extras["tmol"], system.extras["binder_residue_range"], tuple(args.torsions)
    )
    if args.space == "torsion":
        perts = [Perturbation("torsion", sigma_deg=s) for s in args.torsion_sigma]
        return Space(
            "torsion", torsion, ParametrizedEnergy(energy, torsion, system), perts, torsion=torsion
        )
    composed = ComposedParametrization(torsion, rigid)
    perts = [
        Perturbation("torsion_rigid", r, t, sigma_deg=s)
        for s in args.torsion_sigma
        for r in args.rot
        for t in args.trans
    ]
    return Space(
        "torsion_rigid", composed, ParametrizedEnergy(energy, composed, system), perts,
        rigid=rigid, torsion=torsion,
    )  # fmt: skip


def main() -> None:
    args = parse_args()
    try:
        pdb_path, target_chains, binder_chains = resolve_complex(
            args.complex, args.manifest, args.data_dir, args.pdb, [args.target], [args.binder],
            args.allow_breaks,
        )  # fmt: skip
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    flexible = args.space.startswith("torsion")
    internal = (
        args.internal_weight if args.internal_weight is not None else (1.0 if flexible else 0.0)
    )
    cfg = config_from_dict(
        {
            "system": {
                "kind": "structure",
                "structure_path": pdb_path,
                "target_chains": target_chains,
                "binder_chains": binder_chains,
                "movable": "binder",
            },
            # restraint_weight=0: no information about the reference enters the guidance;
            # internal_weight keeps a flexible (torsion) binder folded
            "energy": {
                "backend": "tmol",
                "strain_weight": 0.1,
                "restraint_weight": 0.0,
                "internal_weight": internal,
            },
        }
    )
    system = load_structure_system(cfg.system)
    energy = build_tmol_design_energy(cfg.energy, system)
    seq = SequenceBatch(system.design_space.native_tokens()[None])
    atoms = system.extras["binder_atoms"]
    ca_atoms = system.extras["binder_ca_atoms"]
    space = make_space(args, system, energy)
    sampler = LangevinSampler.from_energy(
        space.energy,
        strength=linear_schedule(args.lam0, args.lam1),
        step_size=args.step_size,
        n_steps=args.steps,
        max_grad_norm=args.clip,
    )
    gen = torch.Generator().manual_seed(args.seed)

    complex_id = args.complex.upper() if args.complex else Path(pdb_path).stem
    run_dir = create_run_dir(args.output_dir, f"recovery_{args.space}_{complex_id}")
    config = {
        **vars(args),
        "pdb_path": pdb_path,
        "target": target_chains,
        "binder": binder_chains,
        "internal_weight_used": internal,
    }
    write_run_header(run_dir, config, collect_metadata(args.seed, pdb_path))
    results = CsvLogger(run_dir / "results.csv")
    structures = run_dir / "structures"
    write_pdb(system, system.reference_coordinates, structures / f"{complex_id}_reference.pdb")
    print(f"run directory: {run_dir}")

    ref = energy.evaluate(system, system.reference_coordinates[None], seq).terms
    print("reference terms:", {k: round(float(v), 1) for k, v in ref.items()})
    print(
        "| perturbation | CA RMSD start → final (Å) | CA aligned start → final (Å) "
        "| interface | repulsion | s |"
    )
    print("|---|---|---|---|---|---|")
    for p in space.perturbations:
        for k in range(args.n):
            if space.param is None:
                r = run_recovery(system, energy, sampler, seq, p, atoms, gen, metric_atoms=ca_atoms)
            else:
                q0 = space.start(p, gen)
                r = run_dof_recovery(
                    system, energy, space.param, sampler, seq, p, q0, atoms, gen,
                    metric_atoms=ca_atoms,
                )  # fmt: skip
            stem = f"{complex_id}_{p.slug()}_{k}"
            write_pdb(system, r.start_coordinates, structures / f"{stem}_start.pdb")
            write_pdb(system, r.final_coordinates, structures / f"{stem}_final.pdb")
            results.write(result_row(complex_id, k, r))
            print(
                f"| {p.label()} | {r.start_rmsd:.2f} → {r.final_rmsd:.2f} | "
                f"{r.metrics['ca_aligned'][0]:.2f} → {r.metrics['ca_aligned'][1]:.2f} | "
                f"{r.start_terms['interface']:.1f} → {r.final_terms['interface']:.1f} | "
                f"{r.start_terms['repulsion']:.1f} → {r.final_terms['repulsion']:.1f} | "
                f"{r.seconds:.0f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
