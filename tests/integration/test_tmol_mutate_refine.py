"""Noise -> mutate -> gradient refinement on barnase A / barstar D (1BRS). Needs TMol."""

from pathlib import Path

import pytest
import torch

from cinder.config import config_from_dict
from cinder.energy.tmol import load_structure_system, tmol_available
from cinder.energy.tmol_mutation import TMolSequenceBuilder
from cinder.experiments.mutate_refine import RefineSettings, StructureNoiser, run_mutate_refine
from cinder.validation import NoiseSpec, apply_mutations

pytestmark = [
    pytest.mark.tmol,
    pytest.mark.skipif(not tmol_available(), reason="tmol not installed"),
]

PDB = Path(__file__).resolve().parents[1] / "fixtures" / "barnase_barstar_AD.pdb"
FAST = RefineSettings(stage_a_steps=250, stage_b_steps=60, quench_steps=120)


@pytest.fixture(scope="module")
def setup():
    base = {
        "system": {
            "kind": "structure",
            "structure_path": str(PDB),
            "target_chains": ["A"],
            "binder_chains": ["D"],
        },
        "energy": {"backend": "tmol", "restraint_weight": 0.0, "strain_weight": 0.1},
    }
    probe = load_structure_system(config_from_dict(base).system)
    length = probe.design_space.length
    cfg = config_from_dict(
        {**base, "system": {**base["system"], "design_positions": list(range(length))}}
    )
    system = load_structure_system(cfg.system)
    builder = TMolSequenceBuilder(system, repack="neighbors", cutoff=6.0)
    return system, builder, StructureNoiser(system), cfg


def _bond_lengths(system, x):
    pose = system.extras["tmol"].poses["complex"]
    bts = pose.packed_block_types.active_block_types
    off = pose.block_coord_offset[0].long()
    lo, hi = system.extras["binder_residue_range"]
    pairs = []
    for res in range(lo, hi):
        idx = bts[int(pose.block_type_ind[0, res])].atom_to_idx
        pairs += [(int(off[res]) + idx["N"], int(off[res]) + idx["CA"])]
        pairs += [(int(off[res]) + idx["CA"], int(off[res]) + idx["C"])]
    p = torch.tensor(pairs)
    return (x[p[:, 0]] - x[p[:, 1]]).norm(dim=-1)


def test_each_noise_type_does_what_it_claims(setup):
    system, _, noiser, _ = setup
    ref, binder = system.reference_coordinates, system.extras["binder_atoms"]
    target = system.extras["target_atoms"]
    g = lambda: torch.Generator().manual_seed(0)  # noqa: E731
    assert torch.equal(noiser.apply(NoiseSpec(), g()), ref)  # no noise -> crystal pose

    for spec in (
        NoiseSpec(torsion_deg=1.0),
        NoiseSpec(rigid_rotation_deg=3.0, rigid_shift=1.0),
        NoiseSpec(cartesian_sigma=0.1),
    ):
        x = noiser.apply(spec, g())
        moved = (x - ref).norm(dim=-1)
        assert float(moved[target].max()) < 1e-3, spec.label()  # target is never noised
        assert float(moved[binder].max()) > 0.05, spec.label()
        assert torch.equal(x, noiser.apply(spec, g())), spec.label()  # seeded -> reproducible

    ref_len = _bond_lengths(system, ref)
    # torsion and rigid noise keep covalent bond lengths; Cartesian noise distorts them
    torsion = noiser.apply(NoiseSpec(torsion_deg=2.0), g())
    rigid = noiser.apply(NoiseSpec(rigid_rotation_deg=5.0, rigid_shift=2.0), g())
    cart = noiser.apply(NoiseSpec(cartesian_sigma=0.1), g())
    assert torch.allclose(_bond_lengths(system, torsion), ref_len, atol=1e-3)
    assert torch.allclose(_bond_lengths(system, rigid), ref_len, atol=1e-3)
    assert float((_bond_lengths(system, cart) - ref_len).abs().max()) > 0.05


def test_pipeline_noisy_start_is_repaired_by_gradients(setup, tmp_path):
    system, builder, noiser, cfg = setup
    native = system.design_space.native
    mutation = f"{native[28]}28A" if native[28] != "A" else f"{native[28]}28G"  # an interface site
    seq = apply_mutations(native, mutation)
    noise = NoiseSpec(
        torsion_deg=1.0, rigid_rotation_deg=2.0, rigid_shift=1.0, cartesian_sigma=0.05
    )
    res = run_mutate_refine(
        system, builder, noiser, cfg.energy, FAST, noise, seq, torch.Generator().manual_seed(0)
    )
    assert [s.name for s in res.snapshots] == [
        "baseline", "baseline_relaxed", "start", "stage_a", "final",
    ]  # fmt: skip
    assert res.sequence == seq and res.baseline.ca_rmsd < 1e-3  # baseline sits on the native CAs
    # the noisy mutant starts strained and clashing; refinement removes both
    assert res.start.terms["total"] > res.baseline.terms["total"] + 50
    assert res.final.terms["total"] < res.start.terms["total"] - 100
    assert res.final.terms["repulsion"] < res.start.terms["repulsion"]
    assert res.final.terms["internal"] < res.start.terms["internal"]
    # and it stays near the native backbone with the fold intact
    assert res.final.ca_rmsd < 3.0 and res.final.ca_aligned < 1.5
    assert res.final.terms["total"] - res.baseline_relaxed.terms["total"] == pytest.approx(
        res.energy_gap
    )
    # every stage can be written out for inspection
    out = builder.write_pdb(res.mutant, tmp_path / "start.pdb")
    assert out.read_text().count("ATOM") == res.system.reference_coordinates.shape[0]


def test_control_without_noise_starts_at_the_baseline_and_runs_are_deterministic(setup):
    system, builder, noiser, cfg = setup
    native = system.design_space.native
    seq = apply_mutations(native, f"{native[32]}32A" if native[32] != "A" else f"{native[32]}32G")
    a = run_mutate_refine(
        system,
        builder,
        noiser,
        cfg.energy,
        FAST,
        NoiseSpec(),
        seq,
        torch.Generator().manual_seed(1),
    )
    b = run_mutate_refine(
        system,
        builder,
        noiser,
        cfg.energy,
        FAST,
        NoiseSpec(),
        seq,
        torch.Generator().manual_seed(1),
    )
    assert a.start.terms["total"] == pytest.approx(a.baseline.terms["total"], abs=1e-3)
    assert a.final.terms["total"] == pytest.approx(b.final.terms["total"], abs=1e-3)  # seeded
    assert a.final.terms["total"] < a.baseline.terms["total"] + 5  # relaxing never hurts much
