"""TMol adapter checks on a real complex (barnase chain A / barstar chain D, from PDB 1BRS).

Run with ``pytest -m tmol``. Skipped when TMol is not installed.
"""

from pathlib import Path

import pytest
import torch

from cinder.config import config_from_dict
from cinder.energy.base import finite_difference_gradient
from cinder.energy.tmol import (
    INTERFACE_TERMS,
    REPULSION_TERMS,
    STRAIN_TERMS,
    SequenceMismatchError,
    TMolGroupEnergy,
    build_tmol_design_energy,
    load_structure_system,
    tmol_available,
)
from cinder.types import SequenceBatch

pytestmark = [
    pytest.mark.tmol,
    pytest.mark.skipif(not tmol_available(), reason="tmol not installed"),
]

PDB = Path(__file__).resolve().parents[1] / "fixtures" / "barnase_barstar_AD.pdb"


@pytest.fixture(scope="module")
def cfg():
    return config_from_dict(
        {
            "system": {
                "kind": "structure",
                "structure_path": str(PDB),
                "target_chains": ["A"],
                "binder_chains": ["D"],
                "design_positions": [29, 33, 34],
            },
            "energy": {"backend": "tmol", "strain_weight": 0.1, "restraint_weight": 0.5},
        }
    )


@pytest.fixture(scope="module")
def system(cfg):
    return load_structure_system(cfg.system)


@pytest.fixture(scope="module")
def energy(cfg, system):
    return build_tmol_design_energy(cfg.energy, system)


def _native(system, n=1):
    return SequenceBatch(system.design_space.native_tokens()[None].repeat(n, 1))


def test_loading_metadata(system):
    assert system.design_space.length == 87
    assert system.binder_length == 87 and system.structure_sha256
    assert system.coord_shape[1] == 3
    assert system.design_space.native.startswith("KKAVING")  # barstar


def test_energy_reproducible_and_finite(system, energy):
    x = system.reference_coordinates[None]
    a = energy.evaluate(system, x, _native(system))
    b = energy.evaluate(system, x, _native(system))
    assert torch.equal(a.total, b.total)
    assert torch.isfinite(a.total).all()
    assert set(a.terms) == {"interface", "repulsion", "strain", "restraint"}
    assert float(a.terms["interface"]) < 0  # a native tight complex is net attractive
    assert float(a.terms["restraint"]) == 0.0  # at the reference pose


def test_interface_decomposition_matches_manual(system, energy):
    ctx = system.extras["tmol"]
    x = system.reference_coordinates[None]
    rows = ctx.term_rows(INTERFACE_TERMS)
    manual = (
        ctx.scores("complex", x)[rows].sum()
        - ctx.scores("target", ctx.split(x, "target"))[rows].sum()
        - ctx.scores("binder", ctx.split(x, "binder"))[rows].sum()
    )
    model = TMolGroupEnergy(ctx, INTERFACE_TERMS, system.design_space.native_tokens(), True)
    assert float(model.energy(system, x, _native(system))) == pytest.approx(float(manual))
    # row order check: the repulsion-only group equals weight-adjusted fa_ljrep row
    rep = TMolGroupEnergy(ctx, REPULSION_TERMS, system.design_space.native_tokens(), False)
    assert float(rep.energy(system, x, _native(system))) == pytest.approx(
        float(ctx.scores("complex", x)[ctx.term_rows(["fa_ljrep"])].sum())
    )
    # groups are disjoint
    assert not set(INTERFACE_TERMS) & set(REPULSION_TERMS)
    assert not set(INTERFACE_TERMS) & set(STRAIN_TERMS)


def test_term_row_order_matches_isolated_score_function(system):
    """Zero all weights but fa_ljrep in a fresh score function: total == our fa_ljrep row."""
    import tmol
    from tmol import ParameterDatabase, ScoreType

    ctx = system.extras["tmol"]
    sf = tmol.beta2016_score_function(ctx.device, param_db=ParameterDatabase.get_default())
    for st in sf.all_score_types():
        sf.set_weight(st, 1.0 if st == ScoreType.fa_ljrep else 0.0)
    module = sf.render_whole_pose_scoring_module(ctx.poses["complex"])
    x = system.reference_coordinates[None]
    only_rep = float(module(x))
    row = ctx.scores("complex", x)[ctx.term_rows(["fa_ljrep"])].sum()
    w = float(ctx.sfxn.weights_tensor()[ctx.term_rows(["fa_ljrep"])[0]])
    assert only_rep * w == pytest.approx(float(row), rel=1e-4)


def test_batch_matches_single(system, energy):
    x = system.reference_coordinates[None]
    single = energy.energy(system, x, _native(system))
    batch = energy.energy(system, x.repeat(3, 1, 1), _native(system, 3))
    assert torch.allclose(batch, single.repeat(3), atol=1e-3)


def test_gradient_matches_finite_differences(system, energy):
    """Autograd dU/dX vs central differences on the highest-gradient interface atoms."""
    x = system.reference_coordinates[None].clone()
    seq = _native(system)
    g = energy.gradient(system, x, seq)
    assert torch.isfinite(g).all()
    top = g.abs().reshape(-1).topk(6).indices.tolist()

    def fn(c):
        return float(energy.energy(system, c, seq)[0])

    fd = finite_difference_gradient(fn, x, eps=2e-2, indices=top)
    for i in top:
        a, f = float(g.reshape(-1)[i]), float(fd.reshape(-1)[i])
        assert a == pytest.approx(f, rel=0.05, abs=0.05), (i, a, f)


def test_mutated_sequence_is_rejected(system, energy):
    seq = _native(system)
    seq.tokens[0, 5] = (seq.tokens[0, 5] + 1) % 20
    with pytest.raises(SequenceMismatchError):
        energy.energy(system, system.reference_coordinates[None], seq)


def test_local_ensemble_is_physically_sensible(system, cfg):
    """Milestone 3: restrained, TMol-guided Langevin around the reference stays near it
    without catastrophic geometry (energies finite, bounded RMSD, restraint pulls back)."""
    from cinder.conformations import LangevinSampler

    system = load_structure_system(
        config_from_dict(
            {**cfg.to_dict(), "system": {**cfg.system.__dict__, "movable": "binder"}}
        ).system
    )
    energy = build_tmol_design_energy(cfg.energy, system)
    sampler = LangevinSampler.from_energy(
        energy, strength=2.0, step_size=1e-4, n_steps=30, init_std=0.0
    )
    seq = _native(system)
    g = torch.Generator().manual_seed(0)
    out = sampler.sample(system, seq, 3, g).coordinates[0]  # [3, atoms, 3]
    assert torch.isfinite(out).all()
    ref = system.reference_coordinates
    moved = (out - ref).norm(dim=-1)  # [3, atoms]
    frozen = system.move_mask[:, 0] == 0
    assert moved[:, frozen].max() == 0  # target atoms did not move
    binder = system.extras["binder_atoms"]
    rmsd = moved[:, binder].pow(2).mean(-1).sqrt()
    assert ((rmsd > 0.02) & (rmsd < 0.5)).all()
    out_terms = energy.evaluate(system, out, SequenceBatch(seq.tokens.repeat(3, 1))).terms
    ref_terms = energy.evaluate(system, ref[None], seq).terms
    # stays bound and clash-free; only the stiff covalent term thermalises (equipartition)
    assert (out_terms["interface"] - ref_terms["interface"]).abs().max() < 15
    assert out_terms["repulsion"].max() < ref_terms["repulsion"] + 20
    assert (out_terms["restraint"] < 50).all()
