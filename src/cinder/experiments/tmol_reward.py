"""TMol mutant-ensemble reward: a different packed pose, and ensemble, for every sequence.

For each full binder sequence ``S`` (cached):

1. build its packed full-atom pose with :class:`~cinder.energy.tmol_mutation.TMolSequenceBuilder`
   (same repack region for every sequence, native included);
2. sample ``K`` poses around it by Langevin dynamics in the six rigid-body jump DOFs of the
   binder (internal geometry fixed), guided by the design energy of *that* sequence's pose;
3. evaluate the design energy of every pose and reduce with an aggregator (``mean`` by default).

``n_steps = 0`` with ``init_std = 0`` degenerates to the packed pose itself, i.e. the
single-structure baseline for the ensemble-versus-single comparison.

Wiring layer: it may know concrete classes, so TMol is imported lazily by its caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from cinder.config import EnergyConfig
from cinder.conformations.langevin import LangevinSampler
from cinder.energy.parametrized import ParametrizedEnergy
from cinder.energy.tmol_design import build_design_energy_for_context
from cinder.energy.tmol_mutation import TMolSequenceBuilder
from cinder.objectives.ensemble import Aggregator, mean_aggregate
from cinder.systems.rigid_body import RigidBodyParametrization
from cinder.systems.specification import MolecularSystem
from cinder.types import ObjectiveResult, SequenceBatch, decode_sequence, encode_sequence

log = logging.getLogger("cinder.tmol_reward")


@dataclass(frozen=True)
class MutantEnsembleSettings:
    """Sampling settings for the per-sequence jump-space ensemble."""

    n_samples: int = 8
    guidance_strength: float = 2.0  # lambda; constant (a fixed moderate value = a real ensemble)
    step_size: float = 1e-4
    n_steps: int = 60
    init_std: float = 0.3  # initial jump noise in Å-equivalent units
    max_grad_norm: float | None = 2000.0
    cache: bool = True  # reuse an ensemble when the same sequence is sampled again

    def __post_init__(self) -> None:
        if self.n_samples < 1 or self.n_steps < 0 or self.step_size <= 0:
            raise ValueError("n_samples >= 1, n_steps >= 0 and step_size > 0 are required")


class TMolMutantReward:
    """:class:`~cinder.types.RewardModel` backed by TMol mutant poses and jump-space ensembles."""

    def __init__(
        self,
        builder: TMolSequenceBuilder,
        energy_config: EnergyConfig,
        settings: MutantEnsembleSettings,
        aggregator: Aggregator = mean_aggregate,
        generator: torch.Generator | None = None,
    ) -> None:
        self.builder = builder
        self.energy_config = energy_config
        self.settings = settings
        self.aggregator = aggregator
        self.generator = generator
        self._cache: dict[str, tuple[Tensor, dict[str, Tensor]]] = {}

    def ensemble(self, sequence: str) -> tuple[Tensor, dict[str, Tensor]]:
        """Design energies ``[K]`` and per-term energies of ``sequence``'s ensemble."""
        if self.settings.cache and sequence in self._cache:
            return self._cache[sequence]
        st = self.settings
        msys: MolecularSystem = self.builder.system_for(sequence)
        tokens = encode_sequence(sequence)
        context: Any = msys.extras["tmol"]
        binder_atoms: Any = msys.extras["binder_atoms"]
        energy = build_design_energy_for_context(self.energy_config, msys, context, tokens)
        param = RigidBodyParametrization(msys.reference_coordinates, binder_atoms)
        sampler = LangevinSampler.from_energy(
            ParametrizedEnergy(energy, param, msys),
            strength=st.guidance_strength,
            step_size=st.step_size,
            n_steps=st.n_steps,
            init_std=st.init_std,
            max_grad_norm=st.max_grad_norm,
        )
        seq1 = SequenceBatch(tokens[None])
        q = sampler.sample(param.dof_system(msys), seq1, st.n_samples, self.generator)
        x = param.to_cartesian(q.coordinates[0])  # [K, n_atoms, 3]
        out = energy.evaluate(msys, x, SequenceBatch(tokens[None].repeat(st.n_samples, 1)))
        result = (out.total.detach(), {k: v.detach() for k, v in out.terms.items()})
        if st.cache:
            self._cache[sequence] = result
        return result

    def __call__(self, system: MolecularSystem, sequence: SequenceBatch) -> ObjectiveResult:
        per_seq = [self.ensemble(decode_sequence(row)) for row in sequence.tokens]
        energies = torch.stack([e for e, _ in per_seq])  # [B, K]
        names = per_seq[0][1].keys()
        terms = {n: torch.stack([t[n] for _, t in per_seq]) for n in names}
        return ObjectiveResult(reward=-self.aggregator(energies), energies=energies, terms=terms)
