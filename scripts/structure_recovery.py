"""CLI for the structure-recovery benchmark (library code: ``cinder.validation``). Needs TMol.

Perturbs the binder away from the complex fold, then runs TMol-guided Langevin with NO restraint
to the reference, so only the physics pulls the structure back.
"""

import argparse
from pathlib import Path

import torch

from cinder.config import config_from_dict
from cinder.conformations import LangevinSampler
from cinder.conformations.guidance import linear_schedule
from cinder.energy.tmol import build_tmol_design_energy, load_structure_system, write_pdb
from cinder.experiments.logging import (
    CsvLogger,
    collect_metadata,
    create_run_dir,
    write_run_header,
)
from cinder.types import SequenceBatch
from cinder.validation import Perturbation, result_row, run_recovery
from cinder.validation.datasets import chain_breaks, load_manifest, prepare_complex


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
    ap.add_argument("--n", type=int, default=2, help="repeats per setting")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--step-size", type=float, default=2e-5)
    ap.add_argument("--lam0", type=float, default=5.0)
    ap.add_argument("--lam1", type=float, default=50.0)
    ap.add_argument("--clip", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="runs", help="a new run directory is created here")
    return ap.parse_args()


def resolve_complex(args: argparse.Namespace) -> tuple[str, list[str], list[str]]:
    """Return (pdb path, target chains, binder chains) from --complex or --pdb."""
    if args.complex:
        specs = {s.pdb_id: s for s in load_manifest(args.manifest)}
        if args.complex.upper() not in specs:
            raise SystemExit(f"{args.complex} not in {args.manifest}: {sorted(specs)}")
        spec = specs[args.complex.upper()]
        path = str(prepare_complex(spec, args.data_dir))
        target, binder = spec.target_chains, spec.binder_chains
    else:
        path, target, binder = args.pdb, [args.target], [args.binder]
    breaks = chain_breaks(path)
    if breaks and not args.allow_breaks:
        raise SystemExit(f"chain breaks in {path}: {breaks} (use --allow-breaks to override)")
    return path, target, binder


def main() -> None:
    args = parse_args()
    pdb_path, target_chains, binder_chains = resolve_complex(args)
    cfg = config_from_dict(
        {
            "system": {
                "kind": "structure",
                "structure_path": pdb_path,
                "target_chains": target_chains,
                "binder_chains": binder_chains,
                "movable": "binder",
            },
            # restraint_weight=0: no information about the reference enters the guidance
            "energy": {"backend": "tmol", "strain_weight": 0.1, "restraint_weight": 0.0},
        }
    )
    system = load_structure_system(cfg.system)
    energy = build_tmol_design_energy(cfg.energy, system)
    seq = SequenceBatch(system.design_space.native_tokens()[None])
    atoms = system.extras["binder_atoms"]
    ca_atoms = system.extras["binder_ca_atoms"]
    sampler = LangevinSampler.from_energy(
        energy,
        strength=linear_schedule(args.lam0, args.lam1),
        step_size=args.step_size,
        n_steps=args.steps,
        max_grad_norm=args.clip,
    )
    gen = torch.Generator().manual_seed(args.seed)
    perturbations = [Perturbation("rigid", r, t) for r in args.rot for t in args.trans]
    perturbations += [Perturbation("cartesian", sigma=s) for s in args.cart_noise]

    complex_id = args.complex.upper() if args.complex else Path(pdb_path).stem
    run_dir = create_run_dir(args.output_dir, f"recovery_{complex_id}")
    config = {**vars(args), "pdb_path": pdb_path, "target": target_chains, "binder": binder_chains}
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
    for p in perturbations:
        for k in range(args.n):
            r = run_recovery(system, energy, sampler, seq, p, atoms, gen, metric_atoms=ca_atoms)
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
