"""Compare computed mutation effects with measured SKEMPI 2.0 binding changes. Needs TMol.

Scores every single substitution on the binder chain of one SKEMPI entry with (a) the single
packed structure and (b) the jump-space ensemble, plus a physics-free baseline (negative BLOSUM62
score), and reports Spearman correlation, ROC AUC for "destabilising by more than --threshold
kcal/mol" and enrichment of the top 20%. Sign convention: larger = weaker binding, for both the
measured ddG (kcal/mol) and the computed energy change.

SKEMPI is not shipped: pass the CSV you downloaded (check its terms of use).
"""

import argparse
import random
from pathlib import Path

from cinder.experiments.logging import (
    CsvLogger,
    collect_metadata,
    create_run_dir,
    write_run_header,
)
from cinder.experiments.tmol_reward import MutantEnsembleSettings
from cinder.experiments.tmol_study import load_study
from cinder.validation import (
    paired_bootstrap_spearman_difference,
    roc_auc,
    scan_mutants,
    spearman,
    top_fraction_enrichment,
)
from cinder.validation.datasets import extract_chains, fetch_pdb, resolve_complex
from cinder.validation.skempi import read_single_mutants, residue_index_map, to_scan_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skempi", required=True, help="skempi_v2.csv")
    ap.add_argument("--entry", required=True, help="SKEMPI entry, e.g. 3SGB_E_I (target_binder)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--subsample", type=int, default=0, help="random subset size (0 = all)")
    ap.add_argument("--ensemble-samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--step-size", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=1.0, help="kcal/mol for the AUC label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="runs")
    return ap.parse_args()


def blosum_penalty(wild: str, mutant: str) -> float:
    """Physics-free baseline: more dissimilar substitution (lower BLOSUM62) = worse."""
    from biotite.sequence.align import SubstitutionMatrix  # noqa: PLC0415

    return -float(SubstitutionMatrix.std_protein_matrix().get_score(wild, mutant))


def main() -> None:
    args = parse_args()
    pdb_id, target_chains, binder_chains = args.entry.split("_")
    raw = fetch_pdb(pdb_id, args.data_dir)
    prepared = extract_chains(
        raw,
        [*target_chains, *binder_chains],
        Path(args.data_dir) / "prepared" / f"{args.entry}.pdb",
    )
    pdb, targets, binders = resolve_complex(
        None, "", args.data_dir, prepared, list(target_chains), list(binder_chains)
    )
    if len(binders) != 1:
        raise SystemExit("this script handles a single binder chain")
    measurements = read_single_mutants(args.skempi, args.entry, binders[0])
    rows, skipped = to_scan_rows(measurements, residue_index_map(pdb, binders[0]))
    if args.subsample and args.subsample < len(rows):
        rows = sorted(
            random.Random(args.seed).sample(rows, args.subsample), key=lambda r: r["name"]
        )
    for line in skipped:
        print("skipped:", line)
    if len(rows) < 10:
        raise SystemExit(f"only {len(rows)} usable single mutants")
    mutants = {str(r["name"]): str(r["mutation"]) for r in rows}
    measured = {str(r["name"]): float(r["measured"]) for r in rows}  # type: ignore[arg-type]

    positions = sorted({int(m[1:-1]) for m in mutants.values()})
    study = load_study(pdb, targets, binders, design_positions=positions, seed=args.seed)
    models = {
        "single": study.single_structure_reward(),
        "ensemble": study.reward(
            MutantEnsembleSettings(
                n_samples=args.ensemble_samples,
                guidance_strength=args.lam,
                step_size=args.step_size,
                n_steps=args.steps,
                init_std=args.init_std,
            )
        ),
    }
    scan = scan_mutants(study.system, models, mutants)

    run_dir = create_run_dir(args.output_dir, f"skempi_{args.entry}")
    write_run_header(run_dir, {**vars(args), "n": len(rows)}, collect_metadata(args.seed, pdb))
    log = CsvLogger(run_dir / "scan.csv")
    truth = [measured[r.name] for r in scan]
    scores = {m: [r.delta[m] for r in scan] for m in models}
    scores["blosum"] = [blosum_penalty(r["mutation"][0], r["mutation"][-1]) for r in rows]  # type: ignore[index]
    for k, r in enumerate(scan):
        log.write({**r.as_dict(), "measured_ddg": measured[r.name], "blosum": scores["blosum"][k]})
    label = [t > args.threshold for t in truth]
    print(
        f"run directory: {run_dir}\n{args.entry}: {len(scan)} single mutants, "
        f"{sum(label)} with ddG > {args.threshold} kcal/mol"
    )
    print("| score | Spearman | AUC | top-20% enrichment |\n|---|---|---|---|")
    for name, s in scores.items():
        print(
            f"| {name} | {spearman(s, truth):+.3f} | {roc_auc(s, label):.3f} | "
            f"{top_fraction_enrichment(s, truth):.2f} |"
        )

    diff, lo, hi = paired_bootstrap_spearman_difference(
        scores["ensemble"], scores["single"], truth, seed=args.seed
    )
    print(
        f"ensemble minus single Spearman: {diff:+.3f} "
        f"(95% bootstrap interval {lo:+.3f} to {hi:+.3f})"
    )


if __name__ == "__main__":
    main()
