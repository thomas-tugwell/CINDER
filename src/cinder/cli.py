"""Thin command-line wrapper; all logic lives in the library."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from cinder.config import ConfigError, load_config
from cinder.experiments.logging import setup_logging


def _doctor(_: argparse.Namespace) -> int:
    from cinder.doctor import format_report, run_doctor

    print(format_report(run_doctor()))
    return 0


def _cfg(args: argparse.Namespace):
    return load_config(args.config)


def _score(args: argparse.Namespace) -> int:
    from cinder.experiments.runner import score_native

    print(json.dumps(score_native(_cfg(args)), indent=2))
    return 0


def _sample(args: argparse.Namespace) -> int:
    from cinder.experiments.runner import sample_ensemble

    print(json.dumps(sample_ensemble(_cfg(args)), indent=2))
    return 0


def _design(args: argparse.Namespace) -> int:
    from cinder.experiments.runner import run_design

    result, run_dir = run_design(_cfg(args))
    print(f"best sequence: {result.best_sequence}  reward: {result.best_reward:.4f}")
    print(f"run directory: {run_dir}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    run = Path(args.run_dir)
    metrics = run / "metrics.csv"
    if not metrics.exists():
        print(f"no metrics.csv in {run}", file=sys.stderr)
        return 1
    with open(metrics) as f:
        rows = list(csv.DictReader(f))
    first, last = rows[0], rows[-1]
    print(f"iterations: {len(rows)}")
    for key in ("reward_mean", "reward_max", "policy_entropy", "n_unique_sequences", "best_reward"):
        print(f"{key:>20}: {float(first[key]):10.4f} -> {float(last[key]):10.4f}")
    print(f"best sequence: {last['best_sequence']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cinder", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report environment and optional backends").set_defaults(
        fn=_doctor
    )
    for name, fn, help_ in (
        ("score", _score, "score the native sequence at the reference pose"),
        ("sample", _sample, "sample a conformational ensemble and report energies"),
        ("design", _design, "optimise binder sequences"),
    ):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("analyze", help="summarise a finished run")
    sp.add_argument("run_dir")
    sp.set_defaults(fn=_analyze)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    try:
        return int(args.fn(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
