"""Shared set-up for TMol studies on one complex (scans, refinement, table design).

Every study script needs the same things: the complex loaded twice (a probe with the native design
space to find the interface, then a system whose designable positions are the ones the study
mutates), a packer-backed :class:`TMolSequenceBuilder`, and scorers. :func:`load_study` builds them
once so the scripts stay thin. Wiring layer: imports TMol-dependent modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from cinder.config import EnergyConfig, ExperimentConfig, config_from_dict
from cinder.energy.tmol import load_structure_system
from cinder.energy.tmol_design import propose_design_positions
from cinder.energy.tmol_mutation import TMolSequenceBuilder
from cinder.experiments.tmol_reward import MutantEnsembleSettings, TMolMutantReward
from cinder.systems.specification import MolecularSystem
from cinder.types import RewardModel, SequenceBatch, encode_sequence


@dataclass
class TMolStudy:
    """A loaded complex ready for mutation studies."""

    config: ExperimentConfig
    probe: MolecularSystem  # native design space; its reference coordinates are the crystal pose
    system: MolecularSystem  # design positions = those requested (default: every binder residue)
    builder: TMolSequenceBuilder
    interface: list[int]  # first/second-shell binder positions of the native pose
    seed: int
    _single: RewardModel | None = field(default=None, repr=False)

    @property
    def native(self) -> str:
        return self.system.design_space.native

    @property
    def energy_config(self) -> EnergyConfig:
        return self.config.energy

    def reward(self, settings: MutantEnsembleSettings) -> TMolMutantReward:
        """A seeded mutant-ensemble reward over this study's builder."""
        gen = torch.Generator().manual_seed(self.seed)
        return TMolMutantReward(self.builder, self.config.energy, settings, generator=gen)

    def single_structure_reward(self) -> RewardModel:
        """The packed pose itself (no sampling): the single-structure baseline."""
        return self.reward(MutantEnsembleSettings(n_samples=1, n_steps=0, init_std=0.0))

    def single_structure_energy(self, sequence: str) -> float:
        """Design energy of ``sequence`` packed on the native backbone (lower is better)."""
        if self._single is None:
            self._single = self.single_structure_reward()
        batch = SequenceBatch(encode_sequence(sequence)[None])
        return -float(self._single(self.system, batch).reward[0])


def load_study(
    pdb: str,
    target_chains: Sequence[str],
    binder_chains: Sequence[str],
    *,
    design_positions: Sequence[int] | None = None,
    shells: str = "both",
    seed: int = 0,
    repack: str = "neighbors",
    repack_cutoff: float = 6.0,
    energy: dict[str, Any] | None = None,
) -> TMolStudy:
    """Load ``pdb`` and build everything a mutation study needs.

    ``design_positions``: 0-based binder indices the study may mutate (default: all of them, so any
    mutation can be built with one fixed repack region). ``energy`` overrides entries of the energy
    config (default: TMol backend, no restraint, ``strain_weight`` 0.1).
    """
    base: dict[str, Any] = {
        "seed": seed,
        "system": {
            "kind": "structure",
            "structure_path": pdb,
            "target_chains": list(target_chains),
            "binder_chains": list(binder_chains),
        },
        "energy": {
            "backend": "tmol",
            "restraint_weight": 0.0,
            "strain_weight": 0.1,
            **(energy or {}),
        },
    }
    probe = load_structure_system(config_from_dict(base).system)
    positions = (
        list(range(probe.design_space.length))
        if design_positions is None
        else sorted(set(design_positions))
    )
    config = config_from_dict({**base, "system": {**base["system"], "design_positions": positions}})
    system = load_structure_system(config.system)
    return TMolStudy(
        config=config,
        probe=probe,
        system=system,
        builder=TMolSequenceBuilder(system, repack=repack, cutoff=repack_cutoff, seed=seed),
        interface=propose_design_positions(probe, shells),
        seed=seed,
    )
