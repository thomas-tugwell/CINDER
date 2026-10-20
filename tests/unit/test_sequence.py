import pytest
import torch

from cinder.sequence import CategoricalPolicy, policy_from_design_space
from cinder.systems import DesignSpace
from cinder.types import AA_TO_INDEX, N_AA, SequenceBatch, decode_sequence, encode_sequence


def test_encode_decode_roundtrip():
    s = "ACDEFGHIKLMNPQRSTVWY"
    assert decode_sequence(encode_sequence(s)) == s
    with pytest.raises(ValueError):
        encode_sequence("AXB")


def test_design_space_masks_and_assembly():
    ds = DesignSpace(5, "AAAAA", [1, 3], allowed={1: "WYF"}, forbidden="C")
    m = ds.mask()
    assert m.shape == (2, N_AA)
    assert m[0].sum() == 3 and m[0, AA_TO_INDEX["W"]]
    assert m[1].sum() == 19 and not m[1, AA_TO_INDEX["C"]]
    full = ds.assemble(torch.tensor([[AA_TO_INDEX["W"], AA_TO_INDEX["D"]]]))
    assert decode_sequence(full[0]) == "AWADA"


def test_design_space_validation():
    with pytest.raises(ValueError):
        DesignSpace(3, "AA", [0])
    with pytest.raises(ValueError):
        DesignSpace(3, "AAA", [5])
    with pytest.raises(ValueError):
        DesignSpace(3, "AAA", [0], allowed={1: "A"})
    with pytest.raises(ValueError):
        DesignSpace(3, "AAA", [0], allowed={0: "C"}, forbidden="C")


def test_policy_never_samples_masked_and_normalizes():
    ds = DesignSpace(3, "AAA", [0, 1, 2], allowed={0: "WY", 1: "D"}, forbidden="P")
    pol = policy_from_design_space(ds)
    g = torch.Generator().manual_seed(0)
    s = pol.sample(2000, g)
    assert set(s.tokens[:, 0].tolist()) <= {AA_TO_INDEX["W"], AA_TO_INDEX["Y"]}
    assert (s.tokens[:, 1] == AA_TO_INDEX["D"]).all()
    assert (s.tokens[:, 2] != AA_TO_INDEX["P"]).all()
    assert torch.allclose(pol.probs().sum(-1), torch.ones(3))


def test_log_prob_matches_manual_and_entropy():
    pol = CategoricalPolicy(torch.ones(2, N_AA, dtype=torch.bool))
    toks = torch.tensor([[0, 5]])
    assert pol.log_prob(toks).item() == pytest.approx(2 * -torch.log(torch.tensor(20.0)).item())
    assert pol.entropy().item() == pytest.approx(2 * torch.log(torch.tensor(20.0)).item(), rel=1e-5)


def test_masked_entropy_counts_only_allowed():
    mask = torch.zeros(1, N_AA, dtype=torch.bool)
    mask[0, :4] = True
    assert CategoricalPolicy(mask).entropy().item() == pytest.approx(
        torch.log(torch.tensor(4.0)).item(), rel=1e-5
    )


def test_native_init_biases_native():
    ds = DesignSpace(2, "WY", [0, 1])
    pol = policy_from_design_space(ds, "native", native_bias=3.0)
    assert pol.argmax_tokens().tolist() == [AA_TO_INDEX["W"], AA_TO_INDEX["Y"]]
    with pytest.raises(ValueError):
        policy_from_design_space(ds, "bogus")


def test_sequence_batch_validation():
    with pytest.raises(ValueError):
        SequenceBatch(torch.zeros(3, dtype=torch.long))
    assert SequenceBatch(torch.zeros(2, 3, dtype=torch.long)).repeat_interleave(4).tokens.shape == (
        8,
        3,
    )
