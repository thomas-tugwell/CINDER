"""Sequence-specific poses via mutation + repacking (barnase chain A / barstar chain D, 1BRS).

Run with ``pytest -m tmol``. Skipped when TMol is not installed.
"""

from pathlib import Path

import pytest
import torch

from cinder.config import config_from_dict
from cinder.energy.tmol import _ca_atom_indices, load_structure_system, tmol_available
from cinder.energy.tmol_mutation import MutationError, TMolSequenceBuilder

pytestmark = [
    pytest.mark.tmol,
    pytest.mark.skipif(not tmol_available(), reason="tmol not installed"),
]

PDB = Path(__file__).resolve().parents[1] / "fixtures" / "barnase_barstar_AD.pdb"
POS = 33  # a design position inside the interface region


@pytest.fixture(scope="module")
def system():
    cfg = config_from_dict(
        {
            "system": {
                "kind": "structure",
                "structure_path": str(PDB),
                "target_chains": ["A"],
                "binder_chains": ["D"],
                "design_positions": [29, POS, 34],
            },
            "energy": {"backend": "tmol"},
        }
    )
    return load_structure_system(cfg.system)


@pytest.fixture(scope="module")
def builder(system):
    return TMolSequenceBuilder(system, repack="neighbors", cutoff=6.0)


def mutate(seq: str, pos: int, aa: str) -> str:
    return seq[:pos] + aa + seq[pos + 1 :]


def name3_at(mutant, system, pos):
    block = system.extras["binder_residue_range"][0] + pos
    bts = mutant.complex.packed_block_types.active_block_types
    return bts[int(mutant.complex.block_type_ind[0, block])].name3


def test_native_through_packer_keeps_identity_and_atom_count(system, builder):
    native = system.design_space.native
    mutant = builder.build(native)
    assert mutant.coordinates.shape == system.reference_coordinates.shape
    assert torch.isfinite(mutant.coordinates).all()


@pytest.mark.parametrize("aa,name3", [("W", "TRP"), ("A", "ALA"), ("G", "GLY"), ("L", "LEU")])
def test_mutation_produces_requested_residue_and_changes_atom_count(system, builder, aa, name3):
    native = system.design_space.native
    if native[POS] == aa:
        pytest.skip("already native")
    mutant = builder.build(mutate(native, POS, aa))
    assert name3_at(mutant, system, POS) == name3
    assert mutant.coordinates.shape[0] != system.reference_coordinates.shape[0] or aa == "L"
    # all other residues keep their identity
    other = system.design_space.positions[0]
    assert name3_at(mutant, system, other) == name3_at(builder.build(native), system, other)


def test_atom_count_ordering_gly_lt_ala_lt_trp(system, builder):
    native = system.design_space.native
    n = {aa: builder.build(mutate(native, POS, aa)).coordinates.shape[0] for aa in "GAW"}
    assert n["G"] < n["A"] < n["W"]


def test_backbone_is_fixed_by_mutation(system, builder):
    """The packer moves side chains only: every CA stays where it was."""
    native = system.design_space.native
    mutant = builder.build(mutate(native, POS, "W" if native[POS] != "W" else "A"))
    ref = system.reference_coordinates
    lo, hi = system.extras["binder_residue_range"]
    new_ca = _ca_atom_indices(mutant.complex, lo, hi)
    old_ca = system.extras["binder_ca_atoms"]
    assert torch.allclose(mutant.coordinates[new_ca], ref[old_ca], atol=1e-3)
    t_lo, t_hi = system.extras["target_residue_range"]
    assert torch.allclose(
        mutant.coordinates[_ca_atom_indices(mutant.complex, t_lo, t_hi)],
        ref[system.extras["target_ca_atoms"]],
        atol=1e-3,
    )


def test_build_is_cached_and_deterministic(system):
    native = system.design_space.native
    a = TMolSequenceBuilder(system, seed=3)
    b = TMolSequenceBuilder(system, seed=3)
    seq = mutate(native, POS, "A" if native[POS] != "A" else "L")
    assert a.build(seq) is a.build(seq)
    assert a.score(seq) == b.score(seq)


def test_score_terms_are_finite_and_complete(system, builder):
    s = builder.score(system.design_space.native)
    assert set(s) == {"interface", "repulsion", "strain", "hbond", "elec", "internal", "n_atoms"}
    assert all(torch.isfinite(torch.tensor(v)) for v in s.values())


def test_subposes_share_complex_coordinates(system, builder):
    native = system.design_space.native
    mutant = builder.build(mutate(native, POS, "W" if native[POS] != "W" else "A"))
    ctx = mutant.context
    assert torch.allclose(ctx.poses["binder"].coords[0], mutant.coordinates[ctx.atoms["binder"]])
    assert torch.allclose(ctx.poses["target"].coords[0], mutant.coordinates[ctx.atoms["target"]])


def test_validation_errors(system, builder):
    native = system.design_space.native
    with pytest.raises(ValueError, match="length"):
        builder.build(native[:-1])
    fixed_pos = next(i for i in range(len(native)) if i not in system.design_space.positions)
    with pytest.raises(ValueError, match="non-design"):
        builder.build(mutate(native, fixed_pos, "W" if native[fixed_pos] != "W" else "A"))
    with pytest.raises(ValueError):
        TMolSequenceBuilder(system, repack="bogus")


def test_identity_check_fails_loudly(system, builder):
    """If the packer ever returned a different residue than requested, we raise, not continue."""
    native = system.design_space.native
    packed = builder.build(native).complex  # holds the native residue at POS
    wrong = mutate(native, POS, "W" if native[POS] != "W" else "A")
    with pytest.raises(MutationError, match="requested"):
        builder._check_identities(packed, wrong)


def test_build_from_a_perturbed_backbone_packs_on_that_backbone(system, builder, tmp_path):
    """``build_from`` packs the mutant on supplied (perturbed) coordinates, uncached; the backbone
    CA positions stay at the perturbed values rather than the crystal ones."""
    native = system.design_space.native
    seq = mutate(native, POS, "A" if native[POS] != "A" else "L")
    ref = system.reference_coordinates
    noisy = ref.clone()
    binder = system.extras["binder_atoms"]
    noisy[binder] += 0.4  # a rigid 0.4 Å shift of the whole binder (keeps geometry)
    mutant = builder.build_from(seq, noisy)
    assert mutant is not builder.build(seq) and name3_at(mutant, system, POS) in {"ALA", "LEU"}
    lo, hi = system.extras["binder_residue_range"]
    new_ca = _ca_atom_indices(mutant.complex, lo, hi)
    old_ca = system.extras["binder_ca_atoms"]
    assert torch.allclose(mutant.coordinates[new_ca], noisy[old_ca], atol=1e-3)
    assert not torch.allclose(mutant.coordinates[new_ca], ref[old_ca], atol=1e-3)
    with pytest.raises(ValueError, match="coordinates must be"):
        builder.build_from(seq, noisy[:10])
    out = builder.write_pdb(mutant, tmp_path / "m.pdb")
    assert out.read_text().count("ATOM") == mutant.coordinates.shape[0]
    cached = builder.write_pdb(seq, tmp_path / "c.pdb")  # string form uses the native backbone
    assert cached.exists()


def test_mutant_system_has_binder_only_move_mask(system, builder):
    native = system.design_space.native
    msys = builder.system_for(mutate(native, POS, "A" if native[POS] != "A" else "L"))
    mask = msys.move_mask[:, 0]
    assert int(mask.sum()) == len(msys.extras["binder_atoms"])
    assert float(mask[msys.extras["target_atoms"]].sum()) == 0.0
