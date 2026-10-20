"""Structure perturbations and metrics for recovery benchmarks (backend independent)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def rotation_matrix(axis: Tensor, angle_rad: float) -> Tensor:
    """Rodrigues rotation about ``axis`` (need not be normalised)."""
    a = axis / axis.norm()
    zero = torch.zeros((), dtype=a.dtype)
    k = torch.stack(
        [
            torch.stack([zero, -a[2], a[1]]),
            torch.stack([a[2], zero, -a[0]]),
            torch.stack([-a[1], a[0], zero]),
        ]
    )
    eye = torch.eye(3, dtype=a.dtype)
    return eye + math.sin(angle_rad) * k + (1 - math.cos(angle_rad)) * (k @ k)


def rigid_body_perturb(
    coords: Tensor,
    atoms: Tensor,
    rotation_deg: float,
    translation: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Rotate ``atoms`` by ``rotation_deg`` about their centroid (random axis) and shift them by
    ``translation`` along a random direction. Internal geometry is preserved exactly."""
    axis = torch.randn(3, generator=generator, dtype=coords.dtype)
    direction = torch.randn(3, generator=generator, dtype=coords.dtype)
    shift = direction / direction.norm() * translation
    centre = coords[atoms].mean(0)
    rot = rotation_matrix(axis, math.radians(rotation_deg)).to(coords)
    out = coords.clone()
    out[atoms] = (coords[atoms] - centre) @ rot.T + centre + shift.to(coords)
    return out


def cartesian_perturb(
    coords: Tensor, atoms: Tensor, sigma: float, generator: torch.Generator | None = None
) -> Tensor:
    """Independent Gaussian noise (std ``sigma``) on ``atoms``; this distorts bonds."""
    out = coords.clone()
    noise = torch.randn(out[atoms].shape, generator=generator, dtype=coords.dtype)
    out[atoms] += sigma * noise.to(coords)
    return out


def rmsd(coords: Tensor, reference: Tensor, atoms: Tensor | None = None) -> float:
    """Plain RMSD (no superposition) over ``atoms``; use when the frame is shared/fixed."""
    a, b = (coords, reference) if atoms is None else (coords[atoms], reference[atoms])
    return float((a - b).pow(2).sum(-1).mean().sqrt())


def kabsch_rmsd(coords: Tensor, reference: Tensor, atoms: Tensor | None = None) -> float:
    """RMSD after optimal rigid superposition (Kabsch, proper rotations only) over ``atoms``.

    Measures whether the internal fold is preserved, independent of where the group sits.
    """
    a, b = (coords, reference) if atoms is None else (coords[atoms], reference[atoms])
    a64, b64 = a.double(), b.double()
    a_c, b_c = a64 - a64.mean(0), b64 - b64.mean(0)
    u, _, vt = torch.linalg.svd(a_c.T @ b_c)
    d = torch.sign(torch.linalg.det(u @ vt))
    fix = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    rot = u @ fix @ vt
    return float((a_c @ rot - b_c).pow(2).sum(-1).mean().sqrt())
