from pathlib import Path

import pytest

from cinder.validation.datasets import (
    ComplexSpec,
    chain_breaks,
    extract_chains,
    fetch_pdb,
    helical_fraction,
    load_manifest,
    resolve_complex,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "barnase_barstar_AD.pdb"


def _line(serial, name, res, chain, resi, x, element, altloc=" "):
    return (
        f"ATOM  {serial:5d} {name:<4s}{altloc}{res} {chain}{resi:4d}    "
        f"{x:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00          {element:>2s}"
    )


def test_spec_validation():
    with pytest.raises(ValueError):
        ComplexSpec("../x", ["A"], ["B"])
    with pytest.raises(ValueError):
        ComplexSpec("2ZTA", ["A"], ["A"])
    with pytest.raises(ValueError):
        ComplexSpec("2ZTA", [], ["B"])
    assert ComplexSpec("2ZTA", ["A"], ["B"]).chains == ["A", "B"]


def test_fetch_rejects_bad_id_without_network(tmp_path):
    with pytest.raises(ValueError):
        fetch_pdb("not-an-id", tmp_path)


def test_fetch_uses_cache(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2ZTA.pdb").write_text("cached")
    assert fetch_pdb("2ZTA", tmp_path).read_text() == "cached"  # no network needed


def test_extract_chains_filters(tmp_path):
    src = tmp_path / "in.pdb"
    src.write_text(
        "\n".join(
            [
                _line(1, " CA ", "ALA", "A", 1, 1.0, "C"),
                _line(2, " CA ", "ALA", "A", 1, 9.0, "C", altloc="B"),  # dropped altloc
                _line(3, " H  ", "ALA", "A", 1, 2.0, "H"),  # hydrogen dropped
                _line(4, " CA ", "GLY", "B", 1, 3.0, "C"),
                _line(5, " CA ", "GLY", "C", 1, 4.0, "C"),  # unrequested chain
                "ENDMDL",
                _line(6, " CA ", "ALA", "A", 1, 5.0, "C"),  # second model ignored
            ]
        )
    )
    out = extract_chains(src, ["B", "A"], tmp_path / "out.pdb").read_text().splitlines()
    atoms = [ln for ln in out if ln.startswith("ATOM")]
    assert [ln[21] for ln in atoms] == ["B", "A"]  # requested order
    assert len(atoms) == 2 and atoms[1][16] == " "
    with pytest.raises(ValueError, match="not found"):
        extract_chains(src, ["Z"], tmp_path / "z.pdb")


def test_load_manifest_and_repo_manifest(tmp_path):
    specs = load_manifest(ROOT / "configs" / "benchmarks" / "helical_complexes.yaml")
    assert [s.pdb_id for s in specs] == ["2ZTA", "1CDL", "1G5J"]
    bad = tmp_path / "m.yaml"
    bad.write_text("complexes:\n  - {id: 2ZTA, target_chains: [A], binder_chains: [B], oops: 1}\n")
    with pytest.raises(ValueError, match="unknown manifest keys"):
        load_manifest(bad)


def test_helical_fraction_fixture():
    pytest.importorskip("biotite")
    frac = helical_fraction(FIXTURE)
    assert set(frac) == {"A", "D"} and all(0.0 <= v <= 1.0 for v in frac.values())


def test_chain_breaks_detects_gap(tmp_path):
    f = tmp_path / "g.pdb"
    f.write_text(
        "\n".join(
            [
                _line(1, " CA ", "ALA", "A", 1, 0.0, "C"),
                _line(2, " CA ", "ALA", "A", 2, 3.8, "C"),
                _line(3, " CA ", "ALA", "A", 9, 40.0, "C"),  # residues 3-8 missing
                _line(4, " CA ", "GLY", "B", 1, 0.0, "C"),
                _line(5, " CA ", "GLY", "B", 2, 3.8, "C"),
            ]
        )
    )
    assert chain_breaks(f) == {"A": [(2, 9, 36.2)]}
    assert chain_breaks(FIXTURE) == {}


def test_resolve_complex_from_path_manifest_errors_and_breaks(tmp_path):
    manifest = ROOT / "configs" / "benchmarks" / "helical_complexes.yaml"
    path, t, b = resolve_complex(None, manifest, tmp_path, FIXTURE, ["A"], ["D"])
    assert path == str(FIXTURE) and t == ["A"] and b == ["D"]
    with pytest.raises(ValueError, match="not in"):
        resolve_complex("ZZZZ", manifest, tmp_path, FIXTURE, ["A"], ["D"])
    gap = tmp_path / "gap.pdb"
    gap.write_text(
        "\n".join(
            [
                _line(1, " CA ", "ALA", "A", 1, 0.0, "C"),
                _line(2, " CA ", "ALA", "A", 9, 40.0, "C"),
            ]
        )
    )
    with pytest.raises(ValueError, match="chain breaks"):
        resolve_complex(None, manifest, tmp_path, gap, ["A"], ["B"])
    assert resolve_complex(None, manifest, tmp_path, gap, ["A"], ["B"], allow_breaks=True)[
        0
    ] == str(gap)
