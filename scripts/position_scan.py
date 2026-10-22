"""Substitute every amino acid at interface positions; report the native residue's rank. TMol.

For each chosen binder position all other amino acids are built (backbone fixed, same repack
protocol) and scored with a single packed structure; ``native rank`` 1 means the native residue
has the lowest energy of all 20. Energy change is positive for worse. Reference-free: it checks
whether the energy function prefers the residues evolution selected, not binding affinity.
"""

import argparse

from cinder.energy.tmol import write_pdb
from cinder.experiments.logging import (
    CsvLogger,
    collect_metadata,
    create_run_dir,
    write_run_header,
)
from cinder.experiments.tmol_study import load_study
from cinder.validation import position_scan
from cinder.validation.datasets import resolve_complex


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--complex", help="PDB id from --manifest")
    ap.add_argument("--manifest", default="configs/benchmarks/helical_complexes.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pdb", default="tests/fixtures/barnase_barstar_AD.pdb")
    ap.add_argument("--target", default="A")
    ap.add_argument("--binder", default="D")
    ap.add_argument(
        "--positions", type=int, nargs="*", help="0-based binder indices (default: shells)"
    )
    ap.add_argument("--shells", choices=["first", "second", "both", "contact"], default="first")
    ap.add_argument("--forbidden", default="", help="amino acids to skip as substitutions")
    ap.add_argument(
        "--structures",
        choices=["all", "extremes", "none"],
        default="extremes",
        help="PDBs to write: every substitution, or the native (repacked) plus the best and "
        "worst substitution at each position",
    )
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
    system, builder = study.system, study.builder
    positions = args.positions or study.interface
    model = study.single_structure_reward()

    run_dir = create_run_dir(args.output_dir, "position_scan")
    write_run_header(
        run_dir, {**vars(args), "positions": positions}, collect_metadata(args.seed, pdb)
    )
    log = CsvLogger(run_dir / "position_scan.csv")
    print(f"run directory: {run_dir}\npositions: {positions}")
    profiles = position_scan(system, model, positions, forbidden=args.forbidden)
    structures = run_dir / "structures"
    if args.structures != "none":
        write_pdb(
            study.probe, study.probe.reference_coordinates, structures / "crystal_reference.pdb"
        )
        builder.write_pdb(system.design_space.native, structures / "wild_type_repacked.pdb")
        native = system.design_space.native
        for p in profiles:
            keep = (
                p.deltas
                if args.structures == "all"
                else {
                    min(p.deltas, key=lambda a: p.deltas[a]): None,
                    max(p.deltas, key=lambda a: p.deltas[a]): None,
                }
            )
            for aa in keep:
                mutant = native[: p.position] + aa + native[p.position + 1 :]
                builder.write_pdb(mutant, structures / f"{p.native}{p.position}{aa}.pdb")
    print(
        "| position | native | native rank (of 20) | mean dE of substitutions |\n|---|---|---|---|"
    )
    for p in profiles:
        print(f"| {p.position} | {p.native} | {p.native_rank} | {p.mean_delta:+.2f} |")
        log.write({"position": p.position, "native": p.native, "native_rank": p.native_rank,
                   "mean_delta": p.mean_delta})  # fmt: skip
    ranks = [p.native_rank for p in profiles]
    print(
        f"native is best at {sum(r == 1 for r in ranks)}/{len(ranks)} positions, "
        f"in the top 3 at {sum(r <= 3 for r in ranks)}/{len(ranks)}; mean rank "
        f"{sum(ranks) / len(ranks):.1f} (random would be 10.5)"
    )


if __name__ == "__main__":
    main()
