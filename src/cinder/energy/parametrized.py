"""Energy as a function of generalized coordinates ``q``: ``U(q) = U_base(X(q))``."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from cinder.energy.base import autograd_gradient
from cinder.systems.specification import MolecularSystem
from cinder.types import EnergyModel, SequenceBatch


@runtime_checkable
class CoordinateParametrization(Protocol):
    """A differentiable map from DOFs ``q`` ``[N, k]`` to Cartesian coordinates ``[N, n, 3]``.

    Implemented by :class:`~cinder.systems.rigid_body.RigidBodyParametrization` (6 jump DOFs),
    :class:`~cinder.energy.tmol_torsion.TMolTorsionParametrization` (torsions) and
    :class:`~cinder.systems.rigid_body.ComposedParametrization` (several at once).
    """

    @property
    def n_dofs(self) -> int: ...

    def to_cartesian(self, q: Tensor) -> Tensor: ...

    def dof_system(self, system: MolecularSystem) -> MolecularSystem: ...


class ParametrizedEnergy:
    """Wrap any coordinate-space :class:`~cinder.types.EnergyModel` so its argument is ``q``.

    ``gradient`` is dU/dq by the chain rule through the (differentiable) parametrization and the
    base energy's own gradient path (autograd through TMol for the TMol backend). ``system``
    passed to ``energy`` / ``gradient`` is ignored in favour of ``cartesian_system``, so the
    wrapper can be driven with the DOF view produced by ``param.dof_system``.
    """

    def __init__(
        self,
        base: EnergyModel,
        param: CoordinateParametrization,
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
