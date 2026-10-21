"""TMol mutant-ensemble reward on barnase chain A / barstar chain D (1BRS). Needs TMol."""

from pathlib import Path

import pytest
import torch

from cinder.config import config_from_dict
from cinder.energy.tmol import load_structure_system, tmol_available
from cinder.energy.tmol_mutation import TMolSequenceBuilder
from cinder.experiments.runner import run_design
from cinder.experiments.tmol_reward import MutantEnsembleSettings, TMolMutantReward
from cinder.types import RewardModel, SequenceBatch, decode_sequence

pytestmark = [
    pytest.mark.tmol,
    pytest.mark.skipif(not tmol_available(), reason="tmol not installed"),
]

PDB = Path(__file__).resolve().parents[1] / "fixtures" / "barnase_barstar_AD.pdb"
POS = 33


def _cfg(**conformation):
    return config_from_dict(
        {
            "name": "tmol_reward_test",
            "seed": 0,
            "system": {
                "kind": "structure",
                "structure_path": str(PDB),
                "target_chains": ["A"],
                "binder_chains": ["D"],
                "design_positions": [29, POS],
            },
            "energy": {"backend": "tmol", "restraint_weight": 0.0},
            "conformation": {"backend": "rigid_langevin", **conformation},
            "optimization": {"iterations": 2, "batch_size": 2, "learning_rate": 0.1},
        }
    )


@pytest.fixture(scope="module")
def system():
    return load_structure_system(_cfg().system)


def _reward(system, settings, seed=0):
    builder = TMolSequenceBuilder(system, repack="neighbors", cutoff=6.0, seed=seed)
    return TMolMutantReward(
        builder, _cfg().energy, settings, generator=torch.Generator().manual_seed(seed)
    )


def _batch(system, *seqs):
    from cinder.types import encode_sequence

    return SequenceBatch(torch.stack([encode_sequence(s) for s in seqs]))


def test_single_structure_mode_equals_the_packed_pose_energy(system):
    """n_steps=0, init_std=0, K=1 is the single-structure baseline."""
    single = MutantEnsembleSettings(n_samples=1, n_steps=0, init_std=0.0)
    reward = _reward(system, single)
    native = system.design_space.native
    out = reward(system, _batch(system, native))
    assert isinstance(reward, RewardModel)
    scored = reward.builder.score(native)  # TMol-weighted term values of the packed pose
    assert float(out.energies[0, 0]) == pytest.approx(
        scored["interface"] + scored["repulsion"], rel=1e-4
    )  # design energy with strain_weight=0, restraint_weight=0
    assert float(out.reward[0]) == pytest.approx(-float(out.energies[0, 0]))


def test_ensemble_has_spread_is_cached_and_deterministic(system):
    ens = MutantEnsembleSettings(n_samples=3, n_steps=15, init_std=0.4, step_size=1e-4)
    native = system.design_space.native
    a, b = _reward(system, ens, seed=1), _reward(system, ens, seed=1)
    out = a(system, _batch(system, native))
    assert out.energies.shape == (1, 3) and float(out.energies.std()) > 0
    assert set(out.terms) >= {"interface", "repulsion"} and out.terms["interface"].shape == (1, 3)
    assert torch.equal(a(system, _batch(system, native)).energies, out.energies)  # cached
    assert torch.equal(b(system, _batch(system, native)).energies, out.energies)  # seeded


def test_different_sequences_get_different_rewards(system):
    ens = MutantEnsembleSettings(n_samples=2, n_steps=5, init_std=0.2)
    reward = _reward(system, ens)
    native = system.design_space.native
    other = native[:POS] + ("A" if native[POS] != "A" else "N") + native[POS + 1 :]
    out = reward(system, _batch(system, native, other))
    assert out.reward.shape == (2,)
    assert float(out.reward[0]) != float(out.reward[1])


def test_end_to_end_design_loop_runs_with_tmol(tmp_path):
    """`cinder design` machinery with TMol: policy -> mutant poses -> ensembles -> REINFORCE."""
    cfg = _cfg(n_samples=2, n_steps=5, init_std=0.2, step_size=1e-4)
    cfg.output_dir = str(tmp_path)
    result, run_dir = run_design(cfg)
    assert len(result.history) == 2
    assert all(torch.isfinite(torch.tensor(h["reward_mean"])) for h in result.history)
    assert (run_dir / "metrics.csv").exists() and (run_dir / "sequences.csv").exists()
    assert len(result.best_sequence) == 87
    assert decode_sequence(torch.zeros(1, dtype=torch.long)) == "A"  # sanity: codec unchanged


def test_mutant_scan_compares_single_structure_and_ensemble(system):
    from cinder.validation import apply_mutations, scan_mutants

    native = system.design_space.native
    spec_a = f"{native[29]}29A" if native[29] != "A" else f"{native[29]}29G"
    spec_b = f"{native[POS]}{POS}N" if native[POS] != "N" else f"{native[POS]}{POS}A"
    mutants = {"m1": spec_a, "m2": spec_b, "wt_like": f"{native[29]}29{native[29]}"}
    models = {
        "single": _reward(system, MutantEnsembleSettings(n_samples=1, n_steps=0, init_std=0.0)),
        "ensemble": _reward(
            system, MutantEnsembleSettings(n_samples=3, n_steps=10, init_std=0.3, step_size=1e-4)
        ),
    }
    rows = scan_mutants(system, models, mutants)
    by = {r.name: r for r in rows}
    assert by["m1"].sequence == apply_mutations(native, spec_a)
    assert by["wt_like"].delta["single"] == pytest.approx(0.0, abs=1e-4)  # identical to wild type
    assert by["m1"].spread["single"] == 0.0 and by["m1"].spread["ensemble"] > 0
    assert all(torch.isfinite(torch.tensor(r.delta["ensemble"])) for r in rows)
