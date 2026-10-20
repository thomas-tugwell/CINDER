"""S -> X -> U -> R -> S on a toy problem with a known (brute-force) optimum."""

import itertools

import pytest
import torch

from cinder.config import config_from_dict
from cinder.energy.toy import ToySequenceEnergy
from cinder.experiments.runner import build_components, run_design
from cinder.types import N_AA, decode_sequence


def _cfg(tmp_path, length=2, iters=120):
    return config_from_dict(
        {
            "name": "t",
            "seed": 1,
            "output_dir": str(tmp_path),
            "system": {"kind": "toy", "binder_length": length, "toy_dim": 2},
            "conformation": {"n_samples": 8, "step_size": 0.05, "n_steps": 40},
            "sequence": {"entropy_weight": 0.005},
            "optimization": {"batch_size": 32, "learning_rate": 0.15, "iterations": iters},
        }
    )


def _brute_force_best(toy: ToySequenceEnergy, length: int, lam: float) -> tuple[str, float]:
    toks = torch.tensor(list(itertools.product(range(N_AA), repeat=length)))
    e = toy.expected_energy(toks, lam)
    i = int(e.argmin())
    return decode_sequence(toks[i]), float(e[i])


def test_policy_converges_to_known_optimum(tmp_path):
    cfg = _cfg(tmp_path)
    comp = build_components(cfg)
    toy = comp.energy.terms[0].model
    best_seq, best_e = _brute_force_best(toy, 2, cfg.conformation.guidance_strength)

    result, run_dir = run_design(cfg)

    # consensus (argmax) sequence of the final policy equals the analytic optimum
    ckpt = torch.load(run_dir / "checkpoints" / "final.pt", weights_only=False)
    logits = ckpt["policy"]["logits"]
    consensus = decode_sequence(logits.argmax(-1))
    assert consensus == best_seq
    # and the stochastic best-of-run reward is close to the analytic optimum (-E[U])
    assert result.best_reward > -best_e - 1.0
    # entropy fell well below uniform
    h0 = result.history[0]["policy_entropy"]
    assert result.history[-1]["policy_entropy"] < 0.6 * h0


def test_run_directory_contents_and_no_overwrite(tmp_path):
    cfg = _cfg(tmp_path, iters=3)
    _, d1 = run_design(cfg)
    _, d2 = run_design(cfg)
    assert d1 != d2
    for name in (
        "config.yaml",
        "metadata.json",
        "metrics.csv",
        "sequences.csv",
        "probabilities.csv",
    ):
        assert (d1 / name).exists(), name
    assert (d1 / "checkpoints" / "final.pt").exists()
    import json

    meta = json.loads((d1 / "metadata.json").read_text())
    assert meta["seed"] == 1 and "torch" in meta["dependencies"]


def test_reproducible_given_seed(tmp_path):
    r1, _ = run_design(_cfg(tmp_path, iters=5))
    r2, _ = run_design(_cfg(tmp_path, iters=5))
    assert r1.best_sequence == r2.best_sequence
    assert r1.best_reward == pytest.approx(r2.best_reward)
