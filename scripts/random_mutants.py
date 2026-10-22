"""Low-rate random mutants, biased toward interface positions; scored with TMol. Needs TMol.

Draws mutants of the native binder (expected ``rate * L`` mutations, spread over positions with
weight ``1 + bias`` on first/second-shell interface positions), scores each with a single packed
structure (and optionally a jump-space ensemble), and reports the average energy cost of an
interface vs a non-interface mutation. Energy change is positive for worse. No measured data are
used: this checks that the pipeline penalises interface mutations more than surface ones.
"""

import argparse

from cinder.energy.tmol import write_pdb
from cinder.experiments.logging import (
    CsvLogger,
    collect_metadata,
    create_run_dir,
    write_run_header,
)
from cinder.experiments.tmol_reward import MutantEnsembleSettings
from cinder.experiments.tmol_study import load_study
from cinder.validation import MutationSampler, scan_mutants, summarize_effects
from cinder.validation.datasets import resolve_complex


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--complex", help="PDB id from --manifest")
    ap.add_argument("--manifest", default="configs/benchmarks/helical_complexes.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pdb", default="tests/fixtures/barnase_barstar_AD.pdb")
    ap.add_argument("--target", default="A")
    ap.add_argument("--binder", default="D")
    ap.add_argument("--rate", type=float, default=0.03, help="expected mutated fraction of L")
    ap.add_argument("--bias", type=float, default=9.0, help="extra weight on interface positions")
    ap.add_argument("--n", type=int, default=40, help="mutants per set")
    ap.add_argument("--shells", choices=["first", "second", "both", "contact"], default="both")
    ap.add_argument("--forbidden", default="CP")
    ap.add_argument("--ensemble-samples", type=int, default=0, help="K; 0 = single structure only")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument(
        "--structures",
        choices=["all", "none"],
        default="all",
        help="write each mutant's packed complex as a PDB (plus crystal and repacked wild type)",
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
    system, builder, interface = study.system, study.builder, study.interface
    length = system.design_space.length
    models = {"single": study.single_structure_reward()}
    if args.ensemble_samples > 0:
        models["ensemble"] = study.reward(
            MutantEnsembleSettings(n_samples=args.ensemble_samples, n_steps=args.steps)
        )

    run_dir = create_run_dir(args.output_dir, "random_mutants")
    write_run_header(
        run_dir, {**vars(args), "interface": interface}, collect_metadata(args.seed, pdb)
    )
    log = CsvLogger(run_dir / "random_mutants.csv")
    structures = run_dir / "structures"
    if args.structures == "all":
        write_pdb(
            study.probe, study.probe.reference_coordinates, structures / "crystal_reference.pdb"
        )
        builder.write_pdb(system.design_space.native, structures / "wild_type_repacked.pdb")
    print(f"run directory: {run_dir}")
    print(f"binder length {length}; interface positions ({args.shells}) = {interface}")

    pooled: dict[str, list[tuple[float, int, int]]] = {m: [] for m in models}
    for label, bias in (("biased", args.bias), ("uniform", 0.0)):
        sampler = MutationSampler(
            system.design_space.native, range(length), interface, bias, args.forbidden, args.seed
        )
        mutants = sampler.sample_many(args.n, args.rate)
        rows = scan_mutants(system, models, {m.spec: m.spec for m in mutants})
        if args.structures == "all":
            for m in mutants:
                builder.write_pdb(m.sequence, structures / f"{label}_{m.spec}.pdb")
        for m, row in zip(mutants, rows, strict=True):
            n_int = sum(p in set(interface) for p in m.positions)
            record = {
                "set": label, "mutation": m.spec, "n_mut": len(m.positions),
                "n_interface": n_int, "n_other": len(m.positions) - n_int,
            }  # fmt: skip
            for model in models:
                record[f"{model}_delta"] = row.delta[model]
                pooled[model].append((row.delta[model], n_int, len(m.positions) - n_int))
            log.write(record)
        for model in models:
            d = [r.delta[model] for r in rows]
            n_i = [sum(p in set(interface) for p in m.positions) for m in mutants]
            n_o = [len(m.positions) - i for m, i in zip(mutants, n_i, strict=True)]
            if len(d) < 3:
                print(f"[{label:7s}] {model:8s} too few mutants for a summary")
                continue
            s = summarize_effects(d, n_i, n_o)
            print(
                f"[{label:7s}] {model:8s} n={s.n} mean dE={s.mean_delta:+.2f} "
                f"median={s.median_delta:+.2f} destabilising={s.fraction_destabilizing:.0%} "
                f"mean interface mutations/mutant={sum(n_i) / len(n_i):.2f}"
            )
    for model, items in pooled.items():
        if len(items) < 3:
            continue
        d, n_i, n_o = zip(*items, strict=True)
        s = summarize_effects(list(d), list(n_i), list(n_o))
        print(
            f"[pooled ] {model:8s} cost per interface mutation={s.cost_interface:+.2f}, "
            f"per non-interface mutation={s.cost_other:+.2f} (energy units, positive = worse)"
        )


if __name__ == "__main__":
    main()
