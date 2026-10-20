"""Validate Langevin sampling against the analytic Gaussian N(0, 1/(lambda k)); CPU only."""

import torch

from cinder.conformations import LangevinSampler
from cinder.energy.toy import HarmonicEnergy
from cinder.systems import DesignSpace, MolecularSystem
from cinder.types import SequenceBatch

lam, k = 2.0, 3.0
system = MolecularSystem("toy", DesignSpace(1, "A", [0]), torch.zeros(1))
sampler = LangevinSampler.from_energy(
    HarmonicEnergy(k), strength=lam, step_size=0.002, n_steps=1000
)
x = sampler.sample(system, SequenceBatch(torch.zeros(1, 1, dtype=torch.long)), 5000).coordinates
print(f"sample variance {x.var():.4f}  analytic {1 / (lam * k):.4f}")
