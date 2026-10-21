"""Reproduce the numbers in the README 'Validation' section. CPU: ~1 min; TMol rows optional."""

import itertools
import math
import tempfile

import torch

from cinder.config import config_from_dict
from cinder.conformations import EnergyGuidance, LangevinSampler
from cinder.conformations.normalizing_flow import (
    ConditionalFlow,
    NormalizingFlowSampler,
    train_flow,
)
from cinder.energy.toy import DoubleWellEnergy, GaussianMixtureEnergy, HarmonicEnergy
from cinder.experiments.runner import build_components, run_design
from cinder.systems import DesignSpace, MolecularSystem
from cinder.types import N_AA, SequenceBatch, decode_sequence


def system(dim):
    return MolecularSystem("t", DesignSpace(2, "AA", [0, 1]), torch.zeros(dim))


def seq(n=1):
    return SequenceBatch(torch.zeros(n, 2, dtype=torch.long))


rows = []
g = lambda s: torch.Generator().manual_seed(s)  # noqa: E731

lam, k = 2.0, 3.0
x = (
    LangevinSampler.from_energy(HarmonicEnergy(k), strength=lam, step_size=0.002, n_steps=1000)
    .sample(system(1), seq(), 5000, g(0))
    .coordinates
)
rows.append(("Langevin, harmonic: variance", f"{x.var():.4f}", f"{1 / (lam * k):.4f}"))

a, lam = 2.0, 3.0


class N01:
    def score(self, s, c, q, t=None):
        return -c


x = (
    LangevinSampler(
        EnergyGuidance(HarmonicEnergy(1.0, center=a), lam, N01()), step_size=0.005, n_steps=1500
    )
    .sample(system(1), seq(), 4000, g(1))
    .coordinates
)
rows.append(("Guided product of Gaussians: mean", f"{x.mean():.4f}", f"{lam * a / (1 + lam):.4f}"))
rows.append(("Guided product of Gaussians: variance", f"{x.var():.4f}", f"{1 / (1 + lam):.4f}"))

x = (
    LangevinSampler.from_energy(DoubleWellEnergy(), step_size=0.01, n_steps=3000)
    .sample(system(2), seq(), 1500, g(2))
    .coordinates[0]
)
rows.append(("Double well: fraction in right well", f"{(x[:, 0] > 0).float().mean():.3f}", "0.500"))

torch.manual_seed(0)
fl = ConditionalFlow(2, 40, 4, 32)
h = train_flow(fl, HarmonicEnergy(2.0, torch.tensor([1.0, -1.0])), system(2), seq(), 1.0, steps=600)
nlz = -math.log(2 * math.pi / 2.0)
rows.append(
    ("Flow, Gaussian: final loss vs bound -log Z", f"{sum(h[-50:]) / 50:.4f}", f"{nlz:.4f} (bound)")
)

torch.manual_seed(1)
fl = ConditionalFlow(2, 40, 6, 64)
means = torch.tensor([[-1.5, 0.0], [1.5, 0.0]])
h = train_flow(fl, GaussianMixtureEnergy(means), system(2), seq(), 1.0, steps=1500)
xs = NormalizingFlowSampler(fl).sample(system(2), seq(), 4000).coordinates[0]
rows.append(("Flow, bimodal: final loss vs -log Z", f"{sum(h[-100:]) / 100:.4f}", "0 (bound)"))
rows.append(
    ("Flow, bimodal: fraction in right mode", f"{(xs[:, 0] > 0).float().mean():.3f}", "0.500")
)

with tempfile.TemporaryDirectory() as out:
    cfg = config_from_dict(
        {
            "seed": 1,
            "output_dir": out,
            "system": {"binder_length": 2},
            "conformation": {"n_samples": 8, "step_size": 0.05, "n_steps": 40},
            "optimization": {"batch_size": 32, "learning_rate": 0.15, "iterations": 120},
        }
    )
    toy = build_components(cfg).energy.terms[0].model
    toks = torch.tensor(list(itertools.product(range(N_AA), repeat=2)))
    e = toy.expected_energy(toks)
    best = decode_sequence(toks[int(e.argmin())])
    res, run = run_design(cfg)
    cons = decode_sequence(
        torch.load(run / "checkpoints" / "final.pt", weights_only=False)["policy"]["logits"].argmax(
            -1
        )
    )
    rows.append(("Policy: consensus sequence", cons, f"{best} (brute force over 400)"))
    last = sum(h["reward_mean"] for h in res.history[-10:]) / 10
    rows.append(
        (
            "Policy: mean reward, last 10 iterations",
            f"{last:.3f}",
            f"{-float(e.min()):.3f} (analytic optimum)",
        )
    )
    rows.append(
        (
            "Policy: entropy first -> last iteration",
            f"{res.history[0]['policy_entropy']:.2f} -> {res.history[-1]['policy_entropy']:.2f}",
            "decreasing",
        )
    )

print("| Check | Measured | Expected |\n|---|---|---|")
for r in rows:
    print("| " + " | ".join(r) + " |")


from cinder.energy.tmol import tmol_available  # noqa: E402

if tmol_available():
    from pathlib import Path

    from cinder.energy.base import finite_difference_gradient
    from cinder.energy.tmol import load_structure_system
    from cinder.energy.tmol_design import build_tmol_design_energy

    pdb = Path(__file__).resolve().parents[1] / "tests/fixtures/barnase_barstar_AD.pdb"
    cfg = config_from_dict(
        {
            "system": {
                "kind": "structure",
                "structure_path": str(pdb),
                "target_chains": ["A"],
                "binder_chains": ["D"],
            },
            "energy": {"backend": "tmol", "strain_weight": 0.1},
        }
    )
    sy = load_structure_system(cfg.system)
    en = build_tmol_design_energy(cfg.energy, sy)
    sq = SequenceBatch(sy.design_space.native_tokens()[None])
    x0 = sy.reference_coordinates[None].clone()
    out = en.evaluate(sy, x0, sq)
    print("\nTMol, barnase/barstar (PDB 1BRS), native pose:")
    print("| Term | Weighted value |\n|---|---|")
    for n, v in out.terms.items():
        print(f"| {n} | {float(v):.2f} |")
    gr = en.gradient(sy, x0, sq)
    top = gr.abs().reshape(-1).topk(6).indices.tolist()
    fd = finite_difference_gradient(
        lambda c: float(en.energy(sy, c, sq)[0]), x0, eps=2e-2, indices=top
    )
    print("\n| Coordinate | Autograd dU/dx | Finite difference |\n|---|---|---|")
    for i in top:
        a, f = float(gr.reshape(-1)[i]), float(fd.reshape(-1)[i])
        print(f"| atom {i // 3}, axis {'xyz'[i % 3]} | {a:.3f} | {f:.3f} |")
