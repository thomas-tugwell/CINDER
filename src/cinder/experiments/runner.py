"""Build components from a resolved config and run experiments.

This is the only module that wires concrete implementations together; the library classes it
instantiates know nothing about each other beyond the Protocols in :mod:`cinder.types`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from cinder.config import ConfigError, ExperimentConfig
from cinder.conformations.langevin import LangevinSampler
from cinder.energy.composite import CompositeEnergy, EnergyTerm
from cinder.energy.toy import ToySequenceEnergy
from cinder.experiments.logging import collect_metadata, create_run_dir, write_run_header
from cinder.objectives.ensemble import EnergyEnsembleObjective, make_aggregator
from cinder.optimization.trainer import SequenceTrainer, TrainerSettings, TrainResult
from cinder.sequence.constraints import policy_from_design_space
from cinder.systems.design_space import DesignSpace
from cinder.systems.specification import MolecularSystem
from cinder.types import ConformationModel, SequenceBatch
from cinder.utils.device import resolve_device
from cinder.utils.seed import make_generator, seed_everything

log = logging.getLogger("cinder.runner")


@dataclass
class Components:
    system: MolecularSystem
    energy: CompositeEnergy
    sampler: ConformationModel


def build_system(cfg: ExperimentConfig) -> MolecularSystem:
    sc = cfg.system
    if sc.kind == "toy":
        length = sc.binder_length
        native = sc.native_sequence or "A" * length
        positions = sc.design_positions if sc.design_positions is not None else list(range(length))
        space = DesignSpace(length, native, positions, dict(sc.allowed_aa), sc.forbidden_aa)
        return MolecularSystem(
            name=sc.name,
            design_space=space,
            reference_coordinates=torch.zeros(sc.toy_dim),
            target_chains=sc.target_chains,
            binder_chains=sc.binder_chains,
        )
    from cinder.energy.tmol import load_structure_system  # lazy: optional dependency

    return load_structure_system(sc, cfg.device)


def build_energy(cfg: ExperimentConfig, system: MolecularSystem) -> CompositeEnergy:
    ec = cfg.energy
    if ec.backend == "toy_sequence":
        toy = ToySequenceEnergy(system.binder_length, dim=cfg.system.toy_dim, seed=ec.toy_seed)
        return CompositeEnergy([EnergyTerm("toy", toy, 1.0)])
    if ec.backend == "tmol":
        from cinder.energy.tmol import build_tmol_design_energy  # lazy: optional dependency

        return build_tmol_design_energy(ec, system)
    raise ConfigError(f"unknown energy backend {ec.backend!r}")


def build_sampler(cfg: ExperimentConfig, energy: CompositeEnergy) -> ConformationModel:
    cc = cfg.conformation
    if cc.backend == "langevin":
        return LangevinSampler.from_energy(
            energy,
            strength=cc.guidance_strength,
            step_size=cc.step_size,
            n_steps=cc.n_steps,
            init_std=cc.init_std,
            max_grad_norm=cc.max_grad_norm,
        )
    raise ConfigError(
        f"conformation backend {cc.backend!r} is not available through the runner yet "
        "(see docs/models.md for status)"
    )


def build_components(cfg: ExperimentConfig) -> Components:
    system = build_system(cfg)
    if cfg.system.kind == "toy":
        system = system.to(resolve_device(cfg.device))
    energy = build_energy(cfg, system)
    return Components(system, energy, build_sampler(cfg, energy))


def run_design(cfg: ExperimentConfig, run_dir: Path | None = None) -> tuple[TrainResult, Path]:
    """Run sequence optimisation and write the full run directory."""
    seed_everything(cfg.seed, cfg.deterministic)
    gen = make_generator(cfg.seed)
    comp = build_components(cfg)
    run_dir = run_dir or create_run_dir(cfg.output_dir, cfg.name)
    write_run_header(run_dir, cfg.to_dict(), collect_metadata(cfg.seed, comp.system.structure_path))
    if comp.system.structure_sha256 is not None:
        log.info("input structure sha256=%s", comp.system.structure_sha256)

    policy = policy_from_design_space(comp.system.design_space, cfg.sequence.init)
    oc = cfg.optimization
    objective = EnergyEnsembleObjective(
        comp.energy, make_aggregator(oc.aggregator, **oc.aggregator_options)
    )
    settings = TrainerSettings(
        batch_size=oc.batch_size,
        n_samples=cfg.conformation.n_samples,
        learning_rate=oc.learning_rate,
        iterations=oc.iterations,
        entropy_weight=cfg.sequence.entropy_weight,
        normalize_advantage=oc.normalize_advantage,
        checkpoint_every=oc.checkpoint_every,
        log_probs_every=oc.log_probs_every,
    )
    trainer = SequenceTrainer(comp.system, policy, comp.sampler, objective, settings, run_dir, gen)
    result = trainer.train()
    log.info("done: best=%s reward=%.4f run=%s", result.best_sequence, result.best_reward, run_dir)
    return result, run_dir


def score_native(cfg: ExperimentConfig) -> dict[str, float]:
    """Energy (and components) of the native sequence at the reference coordinates."""
    seed_everything(cfg.seed, cfg.deterministic)
    comp = build_components(cfg)
    seq = SequenceBatch(comp.system.design_space.native_tokens()[None])
    x = comp.system.reference_coordinates[None]
    out = comp.energy.evaluate(comp.system, x, seq)
    return {"total": float(out.total[0]), **{k: float(v[0]) for k, v in out.terms.items()}}


def sample_ensemble(cfg: ExperimentConfig) -> dict[str, float]:
    """Sample an ensemble for the native sequence and report energy statistics."""
    seed_everything(cfg.seed, cfg.deterministic)
    gen = make_generator(cfg.seed)
    comp = build_components(cfg)
    seq = SequenceBatch(comp.system.design_space.native_tokens()[None])
    confs = comp.sampler.sample(comp.system, seq, cfg.conformation.n_samples, gen)
    u = comp.energy.energy(comp.system, confs.flatten(), seq.repeat_interleave(confs.n_samples))
    return {"n": float(u.numel()), "energy_mean": float(u.mean()), "energy_std": float(u.std())}
