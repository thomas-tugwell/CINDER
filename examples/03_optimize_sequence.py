"""Toy end-to-end S -> X -> U -> R -> S optimisation with a brute-force optimum; CPU only."""

import itertools
import tempfile

import torch

from cinder.config import config_from_dict
from cinder.experiments.runner import build_components, run_design
from cinder.types import N_AA, decode_sequence

with tempfile.TemporaryDirectory() as out:
    cfg = config_from_dict(
        {
            "output_dir": out,
            "system": {"binder_length": 2},
            "conformation": {"n_steps": 40},
            "optimization": {"iterations": 120, "learning_rate": 0.15},
        }
    )
    toy = build_components(cfg).energy.terms[0].model
    toks = torch.tensor(list(itertools.product(range(N_AA), repeat=2)))
    best = decode_sequence(toks[int(toy.expected_energy(toks).argmin())])
    result, _ = run_design(cfg)
    print(f"analytic optimum {best}; sampled best {result.best_sequence}")
