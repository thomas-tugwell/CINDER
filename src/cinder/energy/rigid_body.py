"""Energy as a function of rigid-body jump DOFs (a :class:`ParametrizedEnergy`)."""

from __future__ import annotations

from cinder.energy.parametrized import ParametrizedEnergy

# Kept as the descriptive name for the rigid-body case; the implementation is generic.
RigidBodyEnergy = ParametrizedEnergy

__all__ = ["ParametrizedEnergy", "RigidBodyEnergy"]
