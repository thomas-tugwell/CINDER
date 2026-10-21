import pytest

from cinder.validation import NoiseSpec


def test_defaults_are_no_noise_and_labels_are_informative():
    assert NoiseSpec().is_none and NoiseSpec().label() == "no noise"
    n = NoiseSpec(torsion_deg=1.0, rigid_rotation_deg=2.0, rigid_shift=1.5, cartesian_sigma=0.05)
    assert not n.is_none
    assert n.label() == "torsion σ=1° + rigid 2°/1.5 Å + cart σ=0.05 Å"
    assert n.slug() == "t1_r2_s1.5_c0.05"
    assert NoiseSpec(cartesian_sigma=0.1).label() == "cart σ=0.1 Å"


def test_negative_magnitudes_are_rejected_and_spec_is_hashable():
    for kwargs in ({"torsion_deg": -1}, {"rigid_shift": -0.1}, {"cartesian_sigma": -2}):
        with pytest.raises(ValueError, match=">= 0"):
            NoiseSpec(**kwargs)
    assert len({NoiseSpec(torsion_deg=1.0), NoiseSpec(torsion_deg=1.0)}) == 1  # frozen
