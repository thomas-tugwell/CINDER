"""Score mutants with a single packed structure and with a jump-space ensemble; needs TMol.

Input CSV columns: ``name``, ``mutation`` (e.g. ``Y28A`` or ``Y28A+D34N``, 0-based binder-local
index) and optionally ``measured`` (larger = worse binding). Library code: ``cinder.validation``.
"""

import argparse
import csv
import re

import torch

from cinder.config import EnergyConfig, config_from_dict
from cinder.energy.tmol import load_structure_system
from cinder.energy.tmol_mutation import TMolSequenceBuilder
from cinder.experiments.logging import (
    CsvLogger,
    collect_metadata,
    create_run_dir,
    write_run_header,
)
from cinder.experiments.tmol_reward import MutantEnsembleSettings, TMolMutantReward
from cinder.validation import correlate, scan_mutants
from cinder.validation.datasets import chain_breaks, load_manifest, prepare_complex


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mutants", help="CSV with columns name, mutation[, measured]")
    ap.add_argument("--complex", help="PDB id from --manifest")
    ap.add_argument("--manifest", default="configs/benchmarks/helical_complexes.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pdb", default="tests/fixtures/barnase_barstar_AD.pdb")
    ap.add_argument("--target", default="A")
    ap.add_argument("--binder", default="D")
    ap.add_argument("--samples", type=int, default=8, help="ensemble size K")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--step-size", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=2.0, help="fixed guidance strength")
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="runs")
    return ap.parse_args()


def read_mutants(path: str) -> tuple[dict[str, str], dict[str, float]]:
    mutants, measured = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mutants[row["name"]] = row["mutation"]
            if row.get("measured") not in (None, ""):
                measured[row["name"]] = float(row["measured"])
    if not mutants:
        raise SystemExit(f"no mutants in {path}")
    return mutants, measured


def main() -> None:
    args = parse_args()
    mutants, measured = read_mutants(args.mutants)
    if args.complex:
        spec = {s.pdb_id: s for s in load_manifest(args.manifest)}[args.complex.upper()]
        pdb, target, binder = (
            str(prepare_complex(spec, args.data_dir)),
            spec.target_chains,
            spec.binder_chains,
        )
    else:
        pdb, target, binder = args.pdb, [args.target], [args.binder]
    if chain_breaks(pdb):
        raise SystemExit(f"chain breaks in {pdb}: {chain_breaks(pdb)}")
    positions = sorted(
        {int(i) for m in mutants.values() for i in re.findall(r"[A-Z](\d+)[A-Z]", m)}
    )
    cfg = config_from_dict(
        {
            "seed": args.seed,
            "system": {
                "kind": "structure",
                "structure_path": pdb,
                "target_chains": target,
                "binder_chains": binder,
                "design_positions": positions,
            },
            "energy": {"backend": "tmol", "restraint_weight": 0.0, "strain_weight": 0.1},
        }
    )
    system = load_structure_system(cfg.system)
    builder = TMolSequenceBuilder(system, repack="neighbors", cutoff=6.0, seed=args.seed)
    energy_cfg: EnergyConfig = cfg.energy

    def reward(settings: MutantEnsembleSettings) -> TMolMutantReward:
        gen = torch.Generator().manual_seed(args.seed)
        return TMolMutantReward(builder, energy_cfg, settings, generator=gen)

    models = {
        # K=1, no sampling: the packed pose itself (the single-structure baseline)
        "single": reward(MutantEnsembleSettings(n_samples=1, n_steps=0, init_std=0.0)),
        "ensemble": reward(
            MutantEnsembleSettings(
                n_samples=args.samples,
                guidance_strength=args.lam,
                step_size=args.step_size,
                n_steps=args.steps,
                init_std=args.init_std,
            )
        ),
    }
    rows = scan_mutants(system, models, mutants)

    run_dir = create_run_dir(args.output_dir, "mutant_scan")
    write_run_header(
        run_dir, {**vars(args), "positions": positions}, collect_metadata(args.seed, pdb)
    )
    log = CsvLogger(run_dir / "scan.csv")
    for r in rows:
        log.write({**r.as_dict(), **({"measured": measured[r.name]} if r.name in measured else {})})
    print(f"run directory: {run_dir}")
    print("| mutant | single ΔE | ensemble ΔE | ensemble spread |\n|---|---|---|---|")
    for r in rows:
        d = r.delta
        print(
            f"| {r.name} | {d['single']:+.2f} | {d['ensemble']:+.2f} | {r.spread['ensemble']:.2f} |"
        )
    if len(measured) >= 3:
        rho = correlate(rows, measured)
        print("Spearman vs measured:", {k: round(v, 3) for k, v in rho.items()})
    else:
        print("(no measured values: add a 'measured' column to compute rank correlations)")


if __name__ == "__main__":
    main()
