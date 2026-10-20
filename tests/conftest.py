import pytest
import torch

from cinder.systems import DesignSpace, MolecularSystem


def make_toy_system(n_positions: int = 2, dim: int = 2) -> MolecularSystem:
    ds = DesignSpace(
        length=n_positions, native="A" * n_positions, positions=list(range(n_positions))
    )
    return MolecularSystem(name="toy", design_space=ds, reference_coordinates=torch.zeros(dim))


@pytest.fixture
def toy_system() -> MolecularSystem:
    return make_toy_system()
