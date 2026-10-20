import pytest
import torch

from cinder.conformations.diffusion import DiffusionConformationModel
from cinder.conformations.external.bioemu2 import (
    BackendUnavailableError,
    BioEmu2Adapter,
    bioemu_available,
)
from cinder.conformations.flow_matching import FlowMatchingConformationModel
from cinder.conformations.guidance import (
    energy_guided_score,
    energy_guided_velocity,
    linear_schedule,
)
from cinder.types import ConformationModel, SequenceBatch, UnsupportedOperation

from ..conftest import make_toy_system


def test_stubs_satisfy_protocol_and_fail_clearly():
    seq = SequenceBatch(torch.zeros(1, 2, dtype=torch.long))
    sys_ = make_toy_system()
    for model in (DiffusionConformationModel(), FlowMatchingConformationModel()):
        assert isinstance(model, ConformationModel)
        with pytest.raises(NotImplementedError):
            model.sample(sys_, seq, 2)
    with pytest.raises(UnsupportedOperation):
        DiffusionConformationModel().log_prob(sys_, seq, torch.zeros(1, 2))


@pytest.mark.skipif(bioemu_available(), reason="bioemu installed")
def test_bioemu_fails_gracefully():
    with pytest.raises(BackendUnavailableError):
        BioEmu2Adapter()


def test_guidance_formulas_and_schedule():
    s, g = torch.tensor([1.0]), torch.tensor([4.0])
    assert energy_guided_score(s, g, 0.5).item() == pytest.approx(-1.0)
    assert energy_guided_velocity(s, g, 0.25).item() == pytest.approx(0.0)
    f = linear_schedule(0.0, 2.0)
    assert (f(0.0), f(0.5), f(1.0), f(None), f(5.0)) == (0.0, 1.0, 2.0, 0.0, 2.0)
