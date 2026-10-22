"""Noise the structure in every way, mutate the sequence, refine with gradients. Needs TMol.

For each sampled low-rate (interface-biased) mutant: noise the native binder (torsion, rigid-body
and Cartesian), pack the mutant on the noisy backbone, refine in two stages (torsion + rigid-body
gradients, then a Cartesian polish, each ending in a noise-free quench) and compare with the same
sequence packed on the native backbone and given the same polish ("baseline_relaxed"). Writes a
PDB for every stage. Library code: ``cinder.experiments.mutate_refine``.
"""

import argparse
import statistics

import torch

from cinder.energy.tmol import write_pdb
from cinder.experiments.logging import (
    CsvLogger,
    collect_metadata,
    create_run_dir,
    write_run_header,
)
from cinder.experiments.mutate_refine import RefineSettings, StructureNoiser, run_mutate_refine
from cinder.experiments.tmol_study import load_study
from cinder.validation import MutationSampler, NoiseSpec
from cinder.validation.datasets import resolve_complex


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--complex", help="PDB id from --manifest")
    ap.add_argument("--manifest", default="configs/benchmarks/helical_complexes.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pdb", default="tests/fixtures/barnase_barstar_AD.pdb")
    ap.add_argument("--target", default="A")
    ap.add_argument("--binder", default="D")
    ap.add_argument("--n", type=int, default=5, help="number of mutants")
    ap.add_argument("--rate", type=float, default=0.03, help="expected mutated fraction of L")
    ap.add_argument("--bias", type=float, default=9.0, help="extra weight on interface positions")
    ap.add_argument("--shells", choices=["first", "second", "both", "contact"], default="both")
    ap.add_argument("--forbidden", default="CP")
    ap.add_argument("--torsion-noise", type=float, default=1.0, help="deg per phi/psi")
    ap.add_argument("--rigid-rotation", type=float, default=2.0, help="deg")
    ap.add_argument("--rigid-shift", type=float, default=1.0, help="Å")
    ap.add_argument("--cartesian-noise", type=float, default=0.05, help="Å per binder atom")
    ap.add_argument(
        "--no-noise-control", action="store_true", help="also run each mutant noise-free"
    )
    ap.add_argument("--stage-a-steps", type=int, default=2000)
    ap.add_argument("--stage-b-steps", type=int, default=300)
    ap.add_argument("--quench-steps", type=int, default=300)
    ap.add_argument("--structures", choices=["all", "final", "none"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="runs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pdb, target, binder = resolve_complex(
            args.complex, args.manifest, args.data_dir, args.pdb, [args.target], [args.binder]
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    study = load_study(pdb, target, binder, shells=args.shells, seed=args.seed)
    system, builder, interface = study.system, study.builder, study.interface
    length = system.design_space.length
    noiser = StructureNoiser(system)
    settings = RefineSettings(
        stage_a_steps=args.stage_a_steps,
        stage_b_steps=args.stage_b_steps,
        quench_steps=args.quench_steps,
    )
    noise = NoiseSpec(
        args.torsion_noise, args.rigid_rotation, args.rigid_shift, args.cartesian_noise
    )
    conditions = [noise] + ([NoiseSpec()] if args.no_noise_control else [])
    sampler = MutationSampler(
        system.design_space.native, range(length), interface, args.bias, args.forbidden, args.seed
    )
    mutants = sampler.sample_many(args.n, args.rate)

    run_dir = create_run_dir(args.output_dir, "mutate_refine")
    write_run_header(
        run_dir,
        {**vars(args), "interface": interface, "noise": noise.label()},
        collect_metadata(args.seed, pdb),
    )
    log = CsvLogger(run_dir / "mutate_refine.csv")
    structures = run_dir / "structures"
    if args.structures != "none":
        write_pdb(
            study.probe, study.probe.reference_coordinates, structures / "crystal_reference.pdb"
        )
    print(f"run directory: {run_dir}\nnoise: {noise.label()}\ninterface positions: {interface}")
    print(
        "| mutation | noise | E baseline(relaxed) | E start | E final | gap | CA start → final (Å) "
        "| fold (Å) | s |\n|---|---|---|---|---|---|---|---|---|"
    )
    gaps: dict[str, list[float]] = {}
    for m in mutants:
        for cond in conditions:
            gen = torch.Generator().manual_seed(args.seed)
            r = run_mutate_refine(
                system, builder, noiser, study.energy_config, settings, cond, m.sequence, gen
            )
            gaps.setdefault(cond.label(), []).append(r.energy_gap)
            row = {
                "mutation": m.spec,
                "noise": cond.label(),
                "gap": r.energy_gap,
                "seconds": r.seconds,
            }
            for snap in r.snapshots:
                row[f"{snap.name}_total"] = snap.terms["total"]
                row[f"{snap.name}_ca_rmsd"] = snap.ca_rmsd
                row[f"{snap.name}_ca_aligned"] = snap.ca_aligned
                for term in ("interface", "repulsion", "internal"):
                    row[f"{snap.name}_{term}"] = snap.terms[term]
            log.write(row)
            if args.structures != "none":
                stem = f"{m.spec}_{cond.slug()}"
                keep = r.snapshots if args.structures == "all" else [r.final]
                for snap in keep:
                    write_pdb(r.system, snap.coordinates, structures / f"{stem}_{snap.name}.pdb")
            print(
                f"| {m.spec} | {cond.label()} | {r.baseline_relaxed.terms['total']:.1f} | "
                f"{r.start.terms['total']:.1f} | {r.final.terms['total']:.1f} | "
                f"{r.energy_gap:+.1f} | {r.start.ca_rmsd:.2f} → {r.final.ca_rmsd:.2f} | "
                f"{r.final.ca_aligned:.2f} | {r.seconds:.0f} |",
                flush=True,
            )
    for label, vals in gaps.items():
        print(
            f"[{label}] energy gap vs relaxed baseline: median {statistics.median(vals):+.1f}, "
            f"mean {statistics.mean(vals):+.1f}, n={len(vals)}"
        )


if __name__ == "__main__":
    main()
