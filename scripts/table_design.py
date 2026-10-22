"""Pairwise-table sequence design with exact gradients (no REINFORCE). Needs TMol.

1. Score single substitutions and, for residue pairs in contact, double substitutions with the
   single packed structure (backbone fixed) to build a one-body + two-body energy table.
2. Check the pairwise approximation against the true energy on held-out multi-mutants.
3. Minimise the exact expected energy minus an entropy term over the categorical policy by
   gradient descent, annealing the temperature, and verify the winners with the real scorer.

The table is a fast proposal generator on a fixed backbone, not a replacement for the real
scorer: always look at the verification columns.
"""

import argparse
import random
import statistics

import torch

from cinder.energy.tmol import write_pdb
from cinder.energy.tmol_roles import polar_roles
from cinder.experiments.logging import (
    CsvLogger,
    collect_metadata,
    create_run_dir,
    write_run_header,
)
from cinder.experiments.tmol_study import load_study
from cinder.objectives.pairwise_table import (
    PairwiseEnergyTable,
    approximation_error,
    contact_pairs,
    tables_from_scorer,
)
from cinder.optimization import (
    ExpectedEnergySettings,
    best_single_tokens,
    hill_climb,
    one_body_tokens,
    optimize_expected_energy,
    random_search,
)
from cinder.sequence import CategoricalPolicy
from cinder.types import AA_TO_INDEX, N_AA, decode_sequence
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
    ap.add_argument("--alphabet", default="ADEFHKLNQRSTWY", help="amino acids to consider")
    ap.add_argument(
        "--pair-cutoff", type=float, default=6.0, help="Å between residues for a pair term"
    )
    ap.add_argument("--validate", type=int, default=30, help="held-out multi-mutants to check")
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument(
        "--random-budget", type=int, default=300, help="true-scorer calls for random search"
    )
    ap.add_argument("--load-table", help="table.pt from an earlier run: skip building the table")
    ap.add_argument(
        "--cap", type=float, default=50.0, help="clip single/joint energies at this ceiling"
    )
    ap.add_argument(
        "--mutation-cost", type=float, default=0.5, help="energy penalty per expected mutation"
    )
    ap.add_argument(
        "--samples", type=int, default=20, help="extra candidates sampled from the policy"
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
    native = study.native
    positions = sorted(args.positions or study.interface)
    lo = int(study.probe.extras["binder_residue_range"][0])  # type: ignore[index]
    roles = polar_roles(study.probe.extras["tmol"], None)  # type: ignore[arg-type]
    saved = torch.load(args.load_table) if args.load_table else None
    if saved is not None:
        positions = list(saved["positions"])
    pairs = contact_pairs(
        study.probe.reference_coordinates,
        roles.residue_of_atom,
        roles.heavy_atoms,
        [lo + p for p in positions],
        args.pair_cutoff,
    )
    alphabet = "".join(sorted(set(args.alphabet) | {native[p] for p in positions}))
    if saved is not None:
        alphabet = saved["alphabet"]
    n_calls = 1 + len(positions) * (len(alphabet) - 1) + len(pairs) * (len(alphabet) - 1) ** 2

    run_dir = create_run_dir(args.output_dir, "table_design")
    write_run_header(
        run_dir,
        {**vars(args), "positions": positions, "pairs": pairs, "alphabet": alphabet},
        collect_metadata(args.seed, pdb),
    )
    print(f"run directory: {run_dir}")
    print(f"positions {positions} (native {[native[p] for p in positions]})")
    print(f"{len(pairs)} contact pairs within {args.pair_cutoff} Å; alphabet {alphabet}")
    print(f"at most {n_calls} packed-structure evaluations to build the table")

    score = study.single_structure_energy
    if saved is not None:
        table = PairwiseEnergyTable(
            saved["e0"], saved["e1"], saved["pairs"], saved["e2"],
            torch.tensor([AA_TO_INDEX[native[p]] for p in positions], dtype=torch.long),
        )  # fmt: skip
    else:
        table = tables_from_scorer(
            score, native, positions, pairs, alphabet,
            on_progress=lambda m: print("  " + m, flush=True),
        )  # fmt: skip
        torch.save(
            {"e0": table.e0, "e1": table.e1, "pairs": table.pairs, "e2": table.e2,
             "positions": positions, "native": native, "alphabet": alphabet},
            run_dir / "table.pt",
        )  # fmt: skip
    raw_table = table
    if args.cap is not None:
        table = table.clipped(args.cap)

    # 2. held-out check of the pairwise approximation
    rng = random.Random(args.seed + 1)

    def random_variant(k: int) -> str:
        seq = list(native)
        for pos in rng.sample(positions, k):
            seq[pos] = rng.choice([a for a in alphabet if a != native[pos]])
        return "".join(seq)

    held = [random_variant(rng.randint(2, min(4, len(positions)))) for _ in range(args.validate)]
    if held:
        err = approximation_error(raw_table, score, native, positions, held)
        print(
            f"pairwise table vs true energy on {int(err['n'])} held-out multi-mutants: "
            f"mean |error| {err['mean_abs_error']:.2f} "
            f"(one-body only {err['one_body_mean_abs_error']:.2f}), "
            f"max {err['max_abs_error']:.2f}, "
            f"Pearson {err['pearson']:.3f}, Spearman {err['spearman']:.3f} "
            f"(true energies' std {err['truth_std']:.2f})"
        )

    # 3. optimise with exact gradients, then verify with the real scorer
    mask = torch.zeros(len(positions), N_AA, dtype=torch.bool)
    for i in range(len(positions)):
        for a in alphabet:
            mask[i, AA_TO_INDEX[a]] = True
    policy = CategoricalPolicy(mask)
    result = optimize_expected_energy(
        policy,
        table,
        ExpectedEnergySettings(iterations=args.iterations, mutation_cost=args.mutation_cost),
    )
    tokens = result.tokens
    best = list(native)
    for i, p in enumerate(positions):
        best[p] = decode_sequence(tokens[i : i + 1])
    candidates = {"greedy": "".join(best), "native": native}
    for name, toks in (
        ("best_single", best_single_tokens(table, mask)),
        ("one_body", one_body_tokens(table, mask)),
    ):
        seq = list(native)
        for i, p in enumerate(positions):
            seq[p] = decode_sequence(toks[i : i + 1])
        candidates[name] = "".join(seq)
    searched = random_search(
        score, native, positions, alphabet, args.random_budget, 4, random.Random(args.seed + 2)
    )
    candidates["random_best"] = searched[0][0]
    climbed = hill_climb(score, native, positions, alphabet)
    candidates["hill_climb"] = climbed.sequence
    gen = torch.Generator().manual_seed(args.seed)
    for k, row in enumerate(policy.sample(args.samples, gen).tokens):
        seq = list(native)
        for i, p in enumerate(positions):
            seq[p] = decode_sequence(row[i : i + 1])
        candidates.setdefault(f"sample{k}", "".join(seq))
    log = CsvLogger(run_dir / "candidates.csv")
    print("| candidate | mutations | table E | true E | true - native |\n|---|---|---|---|---|")
    true_native = score(native)
    rows = []
    for name, seq in candidates.items():
        toks = torch.tensor([[AA_TO_INDEX[seq[p]] for p in positions]])
        pred = float(table.energy(toks)[0])
        true = score(seq)
        muts = "+".join(f"{native[p]}{p}{seq[p]}" for p in positions if seq[p] != native[p]) or "-"
        rows.append((name, muts, pred, true))
        log.write({"candidate": name, "mutations": muts, "table": pred, "true": true})
        print(f"| {name} | {muts} | {pred:.1f} | {true:.1f} | {true - true_native:+.1f} |")
    print(f"random search: {len(searched)} true-scorer calls (table build used at most {n_calls})")
    best_true = min(rows, key=lambda r: r[3])
    unique = {r[1] for r in rows}
    print(
        f"best verified: {best_true[0]} ({best_true[1]}) true energy {best_true[3]:.1f} vs native "
        f"{true_native:.1f}; {len(unique)} distinct candidates; median |table-true| "
        f"{statistics.median(abs(r[2] - r[3]) for r in rows):.2f}"
    )
    structures = run_dir / "structures"
    for name in ("greedy", "native"):
        study.builder.write_pdb(candidates[name], structures / f"{name}.pdb")
    write_pdb(study.probe, study.probe.reference_coordinates, structures / "crystal_reference.pdb")


if __name__ == "__main__":
    main()
