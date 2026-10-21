import pytest
import torch

from cinder.config import ConfigError, config_from_dict
from cinder.experiments.runner import build_energy, build_reward, build_system, sample_ensemble
from cinder.objectives.ensemble import SampledEnsembleReward
from cinder.optimization.trainer import SequenceTrainer, TrainerSettings
from cinder.sequence import policy_from_design_space
from cinder.types import RewardModel, SequenceBatch


def _cfg(**overrides):
    base = {
        "system": {"binder_length": 2},
        "conformation": {"n_samples": 4, "n_steps": 5},
        "optimization": {"iterations": 2, "batch_size": 4},
    }
    for k, v in overrides.items():
        base[k] = {**base.get(k, {}), **v}
    return config_from_dict(base)


def _components(cfg):
    system = build_system(cfg)
    return system, build_energy(cfg, system)


def test_toy_backend_builds_a_sampled_ensemble_reward():
    cfg = _cfg()
    system, energy = _components(cfg)
    reward = build_reward(cfg, system, energy, torch.Generator().manual_seed(0))
    assert isinstance(reward, SampledEnsembleReward) and isinstance(reward, RewardModel)
    out = reward(system, SequenceBatch(system.design_space.native_tokens()[None].repeat(3, 1)))
    assert out.reward.shape == (3,) and out.energies.shape == (3, 4)


def test_rigid_langevin_requires_a_tmol_structure_system():
    cfg = _cfg(conformation={"backend": "rigid_langevin"})
    system, energy = _components(cfg)
    with pytest.raises(ConfigError, match="TMol structure"):
        build_reward(cfg, system, energy)


def test_cartesian_langevin_with_tmol_energy_is_rejected():
    cfg = _cfg(energy={"backend": "tmol"})
    system = build_system(_cfg())
    with pytest.raises(ConfigError, match="rigid_langevin"):
        build_reward(cfg, system, energy=None)  # type: ignore[arg-type]


def test_unknown_backend_is_rejected():
    cfg = _cfg(conformation={"backend": "diffusion"})
    system, energy = _components(cfg)
    with pytest.raises(ConfigError, match="not available"):
        build_reward(cfg, system, energy)


def test_sample_ensemble_reports_ensemble_statistics():
    stats = sample_ensemble(_cfg())
    assert stats["n"] == 4.0 and stats["energy_std"] >= 0.0


def test_trainer_only_depends_on_the_reward_model_interface():
    """A hand-written reward (no conformations, no energy) is enough to train a policy."""
    cfg = _cfg()
    system, _ = _components(cfg)
    policy = policy_from_design_space(system.design_space)
    target = system.design_space.native_design_tokens()

    class Match:
        def __call__(self, system, sequence):
            from cinder.types import ObjectiveResult

            full = sequence.tokens[:, system.design_space.positions]
            hits = (full == target).float().sum(1)
            return ObjectiveResult(reward=hits, energies=-hits[:, None])

    trainer = SequenceTrainer(
        system, policy, Match(), TrainerSettings(batch_size=64, learning_rate=0.3, iterations=60)
    )
    result = trainer.train()
    assert result.history[-1]["reward_mean"] > result.history[0]["reward_mean"]
