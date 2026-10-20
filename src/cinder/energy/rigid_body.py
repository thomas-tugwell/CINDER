"""Energy as a function of rigid-body jump DOFs, ``U(q) = U_base(X(q))``."""

from __future__ import annotations

from torch import Tensor

from cinder.energy.base import autograd_gradient
from cinder.systems.rigid_body import RigidBodyParametrization
from cinder.systems.specification import MolecularSystem
from cinder.types import EnergyModel, SequenceBatch


class RigidBodyEnergy:
    """Wrap any coordinate-space :class:`~cinder.types.EnergyModel` so its argument is the 6-DOF
    jump vector. ``gradient`` is dU/dq by the chain rule through the (differentiable) transform
    and the base energy's own gradient path (autograd through TMol for the TMol backend).

    ``system`` passed to ``energy``/``gradient`` is ignored in favour of ``cartesian_system``, so
    the wrapper can be driven with the 6-DOF view produced by ``param.dof_system``.
    """

    def __init__(
        self,
        base: EnergyModel,
        param: RigidBodyParametrization,
        cartesian_system: MolecularSystem,
    ) -> None:
        self.base = base
        self.param = param
        self.cartesian_system = cartesian_system

    def energy(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        x = self.param.to_cartesian(coordinates)
        return self.base.energy(self.cartesian_system, x, sequence)

    def gradient(
        self, system: MolecularSystem, coordinates: Tensor, sequence: SequenceBatch
    ) -> Tensor:
        return autograd_gradient(lambda q: self.energy(system, q, sequence), coordinates)
