import math

import pytest
import torch

from cinder.validation.skempi import (
    ddg_kcal,
    read_single_mutants,
    residue_index_map,
    to_scan_rows,
)

HEADER = (
    "#Pdb;Mutation(s)_PDB;Mutation(s)_cleaned;iMutation_Location(s);Hold_out_type;"
    "Hold_out_proteins;Affinity_mut (M);Affinity_mut_parsed;Affinity_wt (M);Affinity_wt_parsed;"
    "Reference;Protein 1;Protein 2;Temperature\n"
)


def _row(entry, mut, kd_mut, kd_wt, temp="298 (assumed)"):
    return f"{entry};{mut};{mut};COR;x;x;{kd_mut};{kd_mut};{kd_wt};{kd_wt};1;a;b;{temp}\n"


def _pdb_line(resi, name, chain, icode=" "):
    return f"ATOM  {1:5d}  CA  {name} {chain}{resi:4d}{icode}   0.000   0.000   0.000  1.00  0.00\n"


def test_ddg_sign_and_scale():
    assert ddg_kcal(1e-9, 1e-9) == 0.0
    assert ddg_kcal(1e-8, 1e-9) == pytest.approx(0.0019872041 * 298 * math.log(10))
    assert ddg_kcal(1e-10, 1e-9) < 0  # tighter binding is negative
    with pytest.raises(ValueError):
        ddg_kcal(0.0, 1e-9)


def test_read_filters_entry_chain_multi_and_averages_repeats(tmp_path):
    csv_path = tmp_path / "s.csv"
    csv_path.write_text(
        HEADER
        + _row("1ABC_A_B", "LB5A", 1e-8, 1e-9)
        + _row("1ABC_A_B", "LB5A", 1e-7, 1e-9)  # repeated measurement
        + _row("1ABC_A_B", "LB5A,KB6A", 1e-6, 1e-9)  # multi-mutant: skipped
        + _row("1ABC_A_B", "LA7A", 1e-6, 1e-9)  # other chain: skipped
        + _row("1ABC_A_B", "KB6A", "", 1e-9)  # no affinity: skipped
        + _row("9XYZ_A_B", "LB5A", 1e-6, 1e-9)  # other entry: skipped
    )
    got = read_single_mutants(csv_path, "1abc_a_b", "B")
    assert len(got) == 1 and got[0].n_measurements == 2
    expected = (ddg_kcal(1e-8, 1e-9) + ddg_kcal(1e-7, 1e-9)) / 2
    assert got[0].ddg == pytest.approx(expected)


def test_index_map_and_scan_rows(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        _pdb_line(10, "LEU", "B")
        + _pdb_line(11, "LYS", "B")
        + _pdb_line(11, "GLY", "B", icode="A")
        + _pdb_line(12, "ALA", "B")
        + _pdb_line(1, "TRP", "A")
    )
    idx = residue_index_map(pdb, "B")
    assert idx == {"10": (0, "L"), "11": (1, "K"), "11A": (2, "G"), "12": (3, "A")}
    csv_path = tmp_path / "s.csv"
    csv_path.write_text(
        HEADER
        + _row("1ABC_A_B", "LB10A", 1e-8, 1e-9)
        + _row("1ABC_A_B", "GB11AS", 1e-8, 1e-9)
        + _row("1ABC_A_B", "KB10A", 1e-8, 1e-9)  # wrong wild-type letter
        + _row("1ABC_A_B", "AB99S", 1e-8, 1e-9)  # absent residue
    )
    rows, skipped = to_scan_rows(read_single_mutants(csv_path, "1ABC_A_B", "B"), idx)
    assert {r["mutation"] for r in rows} == {"L0A", "G2S"}
    assert len(skipped) == 2


def test_roc_auc_and_enrichment():
    from cinder.validation import roc_auc, top_fraction_enrichment

    assert roc_auc([3, 2, 1, 0], [True, True, False, False]) == 1.0
    assert roc_auc([0, 1, 2, 3], [True, True, False, False]) == 0.0
    assert roc_auc([1, 1, 1, 1], [True, False, True, False]) == 0.5
    assert math.isnan(roc_auc([1, 2], [True, True]))
    truth = [5.0, 4.0, 1.0, 0.0, -1.0]
    assert top_fraction_enrichment([9, 8, 3, 2, 1], truth, 0.4) == 1.0
    assert top_fraction_enrichment([1, 2, 3, 8, 9], truth, 0.4) == 0.0


def test_paired_bootstrap_detects_a_better_score_and_a_tie():
    from cinder.validation import paired_bootstrap_spearman_difference

    g = torch.Generator().manual_seed(0)
    truth = torch.randn(200, generator=g)
    good = (truth + 0.3 * torch.randn(200, generator=g)).tolist()
    bad = torch.randn(200, generator=g).tolist()
    diff, lo, hi = paired_bootstrap_spearman_difference(good, bad, truth.tolist())
    assert diff > 0.5 and lo > 0.3
    diff, lo, hi = paired_bootstrap_spearman_difference(good, good, truth.tolist())
    assert diff == 0.0 and lo == hi == 0.0
    with pytest.raises(ValueError):
        paired_bootstrap_spearman_difference([1.0], [1.0], [1.0])
