"""Shared study set-up and pairwise tables built from real TMol energies. Needs TMol."""

from pathlib import Path

import pytest
import torch

from cinder.energy.tmol import tmol_available
from cinder.energy.tmol_roles import polar_roles
from cinder.experiments.tmol_study import load_study
from cinder.objectives.pairwise_table import contact_pairs, tables_from_scorer
from cinder.types import AA_TO_INDEX, encode_sequence

pytestmark = [
    pytest.mark.tmol,
    pytest.mark.skipif(not tmol_available(), reason="tmol not installed"),
]

PDB = str(Path(__file__).resolve().parents[1] / "fixtures" / "barnase_barstar_AD.pdb")


@pytest.fixture(scope="module")
def study():
    return load_study(PDB, ["A"], ["D"], shells="first", seed=0)


def test_load_study_exposes_probe_system_interface_and_builder(study):
    length = study.probe.design_space.length
    assert length == 87 and study.system.design_space.positions == list(range(length))
    assert study.probe.design_space.positions == list(range(length))
    assert set(study.interface) >= {28, 32, 34, 38, 73}  # barstar hotspots (see the shell tests)
    assert study.native == study.system.design_space.native
    assert study.builder.system is study.system and study.energy_config.restraint_weight == 0.0
    restricted = load_study(PDB, ["A"], ["D"], design_positions=[34, 28, 28])
    assert restricted.system.design_space.positions == [28, 34]  # sorted, de-duplicated


def test_single_structure_energy_matches_the_weighted_term_sum_and_is_cached(study):
    seq = study.native
    scored = study.builder.score(seq)  # TMol-weighted raw terms of the packed pose
    expected = scored["interface"] + scored["repulsion"] + 0.1 * scored["strain"]
    assert study.single_structure_energy(seq) == pytest.approx(expected, rel=1e-4)
    assert study._single is study._single and study._single is not None  # one reward reused
    assert study.single_structure_energy(seq) == study.single_structure_energy(seq)


def test_pairwise_table_reproduces_real_single_and_double_substitutions_exactly(study):
    """Inclusion-exclusion makes one- and two-body substitutions exact by construction; this checks
    the bookkeeping against real packed-structure energies (3 mutations are *not* exact)."""
    positions = [34, 38]  # Asp35 and Asp39: in contact across the binder
    native = study.native
    roles = polar_roles(study.probe.extras["tmol"], None)
    lo = int(study.probe.extras["binder_residue_range"][0])
    pairs = contact_pairs(
        study.probe.reference_coordinates,
        roles.residue_of_atom,
        roles.heavy_atoms,
        [lo + p for p in positions],
        8.0,
    )
    assert pairs == [(0, 1)]
    alphabet = "ANE"
    table = tables_from_scorer(study.single_structure_energy, native, positions, pairs, alphabet)

    def true(changes):
        seq = list(native)
        for pos, aa in changes:
            seq[pos] = aa
        return "".join(seq)

    for changes in ([(34, "A")], [(38, "N")], [(34, "A"), (38, "E")], [(34, "E"), (38, "A")]):
        seq = true(changes)
        toks = encode_sequence(seq)[positions][None]
        assert float(table.energy(toks)[0]) == pytest.approx(
            study.single_structure_energy(seq), abs=1e-3
        )
    assert table.e1[0, AA_TO_INDEX["D"]] == 0.0 and table.e0 == pytest.approx(
        study.single_structure_energy(native)
    )
    assert table.e1.shape == (2, 20) and torch.isfinite(table.e2).all()
