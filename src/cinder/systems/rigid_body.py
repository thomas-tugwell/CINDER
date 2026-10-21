"""Rigid-body (jump) degrees of freedom for one group of atoms, e.g. the binder.

A pose is described by a 6-vector ``q = (t_x, t_y, t_z, r_x, r_y, r_z)``: a translation ``t`` in
Å and a rotation vector (axis * angle) expressed in *Å-equivalent* units, ``r = omega * s``,
where ``s`` (``rotation_scale``, default: the radius of gyration of the moving atoms) converts
radians to Å. This keeps isotropic noise and a single step size sensible across all six
dimensions. The map is

    x' = R(omega) (x - c) + c + t        for the moving atoms,

with ``c`` the reference centroid of the moving atoms; all other atoms are unchanged. Internal
geometry of the moving group is preserved exactly, there are no bond-stretch DOFs to heat, and
the map is differentiable, so gradients of any coordinate-space energy chain through it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor

from cinder.systems.specification import MolecularSystem

N_RIGID_DOF = 6


def rotation_from_vector(omega: Tensor) -> Tensor:
    """Rotation matrices ``[N, 3, 3]`` from rotation vectors ``[N, 3]`` (axis * radians).

    Rodrigues' formula with a Taylor expansion near zero so values and gradients stay finite
    at the identity.
    """
    theta2 = omega.pow(2).sum(-1, keepdim=True)  # [N, 1]
    small = theta2 < 1e-6
    safe = torch.where(small, torch.ones_like(theta2), theta2)
    theta = safe.sqrt()
    a = torch.where(small, 1 - theta2 / 6 + theta2.pow(2) / 120, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24 + theta2.pow(2) / 720, (1 - torch.cos(theta)) / safe)
    zero = torch.zeros_like(omega[:, 0])
    k = torch.stack(
        [
            torch.stack([zero, -omega[:, 2], omega[:, 1]], -1),
            torch.stack([omega[:, 2], zero, -omega[:, 0]], -1),
            torch.stack([-omega[:, 1], omega[:, 0], zero], -1),
        ],
        -2,
    )  # [N, 3, 3]
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(k)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


class RigidBodyParametrization:
    """Map jump DOFs ``q`` ``[N, 6]`` to full Cartesian coordinates ``[N, n_atoms, 3]``."""

    def __init__(
        self,
        reference: Tensor,
        moving_atoms: Tensor,
        rotation_scale: float | None = None,
    ) -> None:
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError("reference must be [n_atoms, 3]")
        if moving_atoms.numel() == 0:
            raise ValueError("moving_atoms is empty")
        self.reference = reference.detach().clone()
        self.moving_atoms = moving_atoms.long()
        moving = self.reference[self.moving_atoms]
        self.centroid = moving.mean(0)
        if rotation_scale is None:
            rotation_scale = float((moving - self.centroid).pow(2).sum(-1).mean().sqrt())
        self.rotation_scale = max(float(rotation_scale), 1.0)

    @property
    def n_dofs(self) -> int:
        return N_RIGID_DOF

    def apply(self, coords: Tensor, q: Tensor) -> Tensor:
        """Apply the jumps ``q`` ``[N, 6]`` to ``coords`` ``[N, n_atoms, 3]`` (any coordinates, not
        only the reference): ``x' = R (x - c) + c + t`` on the moving atoms about the *reference*
        centroid ``c``. Differentiable in both arguments."""
        if q.ndim != 2 or q.shape[1] != N_RIGID_DOF:
            raise ValueError(f"q must be [N, {N_RIGID_DOF}], got {tuple(q.shape)}")
        t, omega = q[:, :3], q[:, 3:] / self.rotation_scale
        rot = rotation_from_vector(omega)  # [N, 3, 3]
        c = self.centroid.to(q)
        moved = (coords[:, self.moving_atoms] - c) @ rot.transpose(1, 2) + c + t[:, None, :]
        out = coords.clone()
        out[:, self.moving_atoms] = moved
        return out

    def to_cartesian(self, q: Tensor) -> Tensor:
        """``q`` ``[N, 6]`` -> coordinates ``[N, n_atoms, 3]`` (differentiable in ``q``)."""
        if q.ndim != 2 or q.shape[1] != N_RIGID_DOF:
            raise ValueError(f"q must be [N, {N_RIGID_DOF}], got {tuple(q.shape)}")
        ref = self.reference.to(q)
        return self.apply(ref.expand(q.shape[0], *ref.shape), q)

    def dofs(
        self,
        rotation_deg: float,
        translation: float,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """A random jump of the given rotation angle (deg) and translation length (Å), in the
        same draw order as :func:`cinder.validation.perturb.rigid_body_perturb`."""
        axis = torch.randn(3, generator=generator, dtype=dtype)
        direction = torch.randn(3, generator=generator, dtype=dtype)
        t = direction / direction.norm() * translation
        omega = axis / axis.norm() * math.radians(rotation_deg)
        return torch.cat([t, omega * self.rotation_scale])

    def dof_system(self, system: MolecularSystem) -> MolecularSystem:
        """A copy of ``system`` whose coordinates are the 6 jump DOFs (reference = identity).

        Lets any sampler written for ``MolecularSystem`` (e.g. Langevin) run in jump space. The
        original system is kept in ``extras['cartesian_system']``.
        """
        return replace(
            system,
            reference_coordinates=torch.zeros(N_RIGID_DOF),
            move_mask=None,
            extras={**system.extras, "cartesian_system": system},
        )


class ComposedParametrization:
    """Torsion-type DOFs followed by a rigid-body jump: ``q = [q_first (k), q_rigid (6)]``.

    ``first`` maps its DOFs to coordinates (e.g. binder torsions); the rigid jump is then applied
    on top about the reference centroid. Both parts stay differentiable.
    """

    def __init__(self, first, rigid: RigidBodyParametrization) -> None:  # noqa: ANN001
        self.first = first
        self.rigid = rigid

    @property
    def n_dofs(self) -> int:
        return int(self.first.n_dofs) + N_RIGID_DOF

    def to_cartesian(self, q: Tensor) -> Tensor:
        if q.ndim != 2 or q.shape[1] != self.n_dofs:
            raise ValueError(f"q must be [N, {self.n_dofs}], got {tuple(q.shape)}")
        k = int(self.first.n_dofs)
        return self.rigid.apply(self.first.to_cartesian(q[:, :k]), q[:, k:])

    def dof_system(self, system: MolecularSystem) -> MolecularSystem:
        return replace(
            system,
            reference_coordinates=torch.zeros(self.n_dofs),
            move_mask=None,
            extras={**system.extras, "cartesian_system": system},
        )
