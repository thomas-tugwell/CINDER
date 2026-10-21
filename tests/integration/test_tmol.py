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
    load_structure_system,
    tmol_available,
    write_pdb,
)
from cinder.energy.tmol_design import build_tmol_design_energy
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
    model = TMolGroupEnergy(ctx, INTERFACE_TERMS, system.design_space.native_tokens(), "interface")
    assert float(model.energy(system, x, _native(system))) == pytest.approx(float(manual))
    # row order check: the repulsion-only group equals weight-adjusted fa_ljrep row
    rep = TMolGroupEnergy(ctx, REPULSION_TERMS, system.design_space.native_tokens(), "complex")
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


def test_write_pdb_roundtrip_preserves_atom_order_and_coordinates(system, tmp_path):
    """A written pose re-reads with the same atom count and (to PDB precision) coordinates."""
    import tmol

    x = system.reference_coordinates.clone()
    path = write_pdb(system, x, tmp_path / "ref.pdb")
    assert path.exists()
    reread = tmol.pose_stack_from_pdb(str(path), system.reference_coordinates.device)
    assert reread.coords.shape[1] == x.shape[0]
    assert torch.allclose(reread.coords[0], x, atol=2e-3)  # PDB stores 3 decimals


def test_write_pdb_perturbed_differs_and_validates_shape(system, tmp_path):
    x = system.reference_coordinates.clone()
    x[system.extras["binder_atoms"]] += 1.0
    a = write_pdb(system, system.reference_coordinates, tmp_path / "a.pdb").read_text()
    b = write_pdb(system, x, tmp_path / "b.pdb").read_text()
    assert a != b and a.count("ATOM") == b.count("ATOM")
    with pytest.raises(ValueError, match="expected coordinates"):
        write_pdb(system, x[:10], tmp_path / "bad.pdb")


def test_ca_atom_indices_point_at_ca_atoms(system, tmp_path):
    ca = system.extras["binder_ca_atoms"]
    assert len(ca) == system.design_space.length  # one CA per binder residue
    path = write_pdb(system, system.reference_coordinates, tmp_path / "ref.pdb")
    names = [ln[12:16].strip() for ln in path.read_text().splitlines() if ln.startswith("ATOM")]
    assert all(names[int(i)] == "CA" for i in ca)
    assert len(system.extras["target_ca_atoms"]) == 108


def test_rigid_body_gradient_matches_finite_differences_on_real_complex(system, energy):
    """dU/dq (6 binder jump DOFs, chained through TMol autograd) vs central differences."""
    from cinder.energy.parametrized import ParametrizedEnergy
    from cinder.systems.rigid_body import RigidBodyParametrization

    param = RigidBodyParametrization(system.reference_coordinates, system.extras["binder_atoms"])
    wrapped = ParametrizedEnergy(energy, param, system)
    dof_system = param.dof_system(system)
    seq = _native(system)
    q = torch.zeros(1, 6)
    q[0, :3] = torch.tensor([0.05, -0.04, 0.03])  # a small offset from the reference
    g = wrapped.gradient(dof_system, q, seq)
    assert g.shape == (1, 6) and torch.isfinite(g).all()

    def fn(v):
        return float(wrapped.energy(dof_system, v, seq)[0])

    fd = finite_difference_gradient(fn, q, eps=2e-2)
    assert torch.allclose(g, fd, rtol=0.05, atol=0.5), (g, fd)
    # moving the binder rigidly never changes its internal geometry -> strain term is invariant
    moved = param.to_cartesian(torch.tensor([[1.0, 0.5, -0.5, 2.0, -1.0, 1.0]]))
    a = energy.evaluate(system, system.reference_coordinates[None], seq).terms["strain"]
    b = energy.evaluate(system, moved, seq).terms["strain"]
    assert float(b) == pytest.approx(float(a), rel=1e-3)


def test_hbond_and_elec_terms_are_separate_weighted_extras(cfg, system):
    """hbond/elec weights add emphasis on top of ``interface``; default weights change nothing."""
    base = build_tmol_design_energy(cfg.energy, system)
    seq, x = _native(system), system.reference_coordinates[None]
    ref = base.evaluate(system, x, seq)
    assert "hbond" not in ref.terms and "elec" not in ref.terms  # weight 0 -> not added

    boosted_cfg = config_from_dict(
        {
            **cfg.to_dict(),
            "energy": {**cfg.energy.__dict__, "hbond_weight": 2.0, "elec_weight": 0.5},
        }
    )
    boosted = build_tmol_design_energy(boosted_cfg.energy, system).evaluate(system, x, seq)
    ctx = system.extras["tmol"]
    from cinder.energy.tmol import group_energies

    raw = group_energies(ctx, system.design_space.native_tokens())
    hb = float(raw["hbond"].energy(system, x, seq)[0])
    el = float(raw["elec"].energy(system, x, seq)[0])
    assert float(boosted.terms["hbond"]) == pytest.approx(2.0 * hb)
    assert float(boosted.terms["elec"]) == pytest.approx(0.5 * el)
    assert float(boosted.total) == pytest.approx(float(ref.total) + 2.0 * hb + 0.5 * el, rel=1e-4)
    assert hb < 0  # the native interface has net-favourable hydrogen bonding
