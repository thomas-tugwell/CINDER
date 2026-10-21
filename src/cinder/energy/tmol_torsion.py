"""Torsion-space (phi / psi / chi) degrees of freedom for the binder, via TMol's kinematics.

``q`` holds changes of selected TMol bond torsions, in Å-equivalent units
(``q = angle_radians * angle_scale``) so isotropic noise and one step size suit all DOFs. The map
``q -> coordinates`` runs TMol's differentiable forward kinematics on the complex pose, so
gradients of any coordinate-space energy chain through it. Bond lengths and bond angles are held
fixed (no bond-stretch DOFs to heat); which atoms move follows TMol's fold forest, which roots each
chain at its first residue, so a binder torsion moves the binder downstream of it and never the
target.

Only ``energy/tmol*.py`` modules import TMol; this class has the same interface as
:class:`~cinder.systems.rigid_body.RigidBodyParametrization` (``to_cartesian``, ``dof_system``,
``n_dofs``) so :class:`~cinder.energy.parametrized.ParametrizedEnergy` wraps either.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from cinder.energy.tmol import TMolContext, _import_tmol
from cinder.systems.specification import MolecularSystem

DEFAULT_TORSIONS = ("phi", "psi")
CHI_TORSIONS = ("chi1", "chi2", "chi3", "chi4")


class TMolTorsionParametrization:
    """Differentiable map from binder torsion changes ``q`` ``[N, k]`` to coordinates."""

    def __init__(
        self,
        context: TMolContext,
        binder_range: tuple[int, int],
        torsion_names: tuple[str, ...] = DEFAULT_TORSIONS,
        angle_scale: float = 3.0,
    ) -> None:
        _import_tmol()
        from tmol.kinematics import (  # noqa: PLC0415
            FoldForest,
            MinimizerMap,
            MoveMap,
            PoseStackKinematicsModule,
        )
        from tmol.kinematics.compiled import inverse_kin  # noqa: PLC0415
        from tmol.pose import get_named_torsions, get_torsion_names  # noqa: PLC0415

        if angle_scale <= 0:
            raise ValueError("angle_scale must be > 0")
        pose = context.poses["complex"]
        self.angle_scale = float(angle_scale)
        self.torsion_names = tuple(torsion_names)
        self.reference = pose.coords[0].detach().clone()
        self.kin = PoseStackKinematicsModule(pose, FoldForest.reasonable_fold_forest(pose))
        kmd = self.kin.kmd
        self.ids = kmd.forest.id.to(torch.int64)

        move_map = MoveMap.from_pose_stack(pose)
        self.selected: list[tuple[int, str]] = []
        lo, hi = binder_range
        for block in range(lo, hi):
            names = get_torsion_names(pose, 0, block)
            values = get_named_torsions(pose, 0, block, None, degrees=True)
            for k, name in enumerate(names):
                if name in self.torsion_names and not math.isnan(float(values[name])):
                    move_map.set_move_named_torsion_for_blocks(0, block, k)
                    self.selected.append((block, name))
        if not self.selected:
            raise ValueError(f"no torsions named {self.torsion_names} found in the binder")

        dof_mask = MinimizerMap(pose, kmd, move_map).dof_mask
        kincoords = torch.zeros((kmd.forest.id.shape[0], 3), dtype=torch.float32)
        kincoords[1:] = pose.coords.view(-1, 3)[self.ids[1:]].cpu()
        self.full_dofs = inverse_kin(
            kincoords,
            kmd.forest.parent,
            kmd.forest.frame_x,
            kmd.forest.frame_y,
            kmd.forest.frame_z,
            kmd.forest.doftype,
        ).detach()
        self.flat_idx = dof_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        self.ref_values = self.full_dofs.reshape(-1)[self.flat_idx].clone()

    @property
    def n_dofs(self) -> int:
        return int(self.flat_idx.numel())

    def to_cartesian(self, q: Tensor) -> Tensor:
        """``q`` ``[N, k]`` (Å-equivalent units) -> coordinates ``[N, n_atoms, 3]``."""
        if q.ndim != 2 or q.shape[1] != self.n_dofs:
            raise ValueError(f"q must be [N, {self.n_dofs}], got {tuple(q.shape)}")
        out = []
        for row in q:
            dofs = self.full_dofs.to(q).clone()
            dofs.view(-1)[self.flat_idx] = self.ref_values.to(q) + row / self.angle_scale
            kin_coords = self.kin(dofs)
            coords = self.reference.to(q).clone()
            coords[self.ids[1:]] = kin_coords[1:].to(q.dtype)
            out.append(coords)
        return torch.stack(out)

    def dofs(
        self,
        sigma_deg: float,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Random torsion perturbation: every selected angle gets Gaussian noise of std
        ``sigma_deg`` degrees (returned in Å-equivalent units)."""
        noise = torch.randn(self.n_dofs, generator=generator, dtype=dtype)
        return noise * math.radians(sigma_deg) * self.angle_scale

    def dof_system(self, system: MolecularSystem) -> MolecularSystem:
        """A copy of ``system`` whose coordinates are the ``k`` torsion DOFs (reference = 0)."""
        extras: dict[str, Any] = {**system.extras, "cartesian_system": system}
        return replace(
            system,
            reference_coordinates=torch.zeros(self.n_dofs),
            move_mask=None,
            extras=extras,
        )
