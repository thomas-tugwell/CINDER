"""Description of structural noise applied to a binder, in every available degree of freedom."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseSpec:
    """Noise to apply to the binder, all independent and all optional.

    * ``torsion_deg``: std (degrees) of Gaussian noise added to every selected phi/psi angle.
      Independent angle noise compounds along a helix like a lever, so keep it near 1 degree.
    * ``rigid_rotation_deg`` / ``rigid_shift``: a random rigid-body rotation (degrees, about the
      binder centroid) and translation (Å).
    * ``cartesian_sigma``: std (Å) of independent noise on every binder atom (distorts bonds).
    """

    torsion_deg: float = 0.0
    rigid_rotation_deg: float = 0.0
    rigid_shift: float = 0.0
    cartesian_sigma: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(self.torsion_deg, self.rigid_rotation_deg, self.rigid_shift, self.cartesian_sigma)
            < 0
        ):
            raise ValueError("noise magnitudes must be >= 0")

    @property
    def is_none(self) -> bool:
        return not any(
            (self.torsion_deg, self.rigid_rotation_deg, self.rigid_shift, self.cartesian_sigma)
        )

    def label(self) -> str:
        if self.is_none:
            return "no noise"
        parts = []
        if self.torsion_deg:
            parts.append(f"torsion σ={self.torsion_deg:g}°")
        if self.rigid_rotation_deg or self.rigid_shift:
            parts.append(f"rigid {self.rigid_rotation_deg:g}°/{self.rigid_shift:g} Å")
        if self.cartesian_sigma:
            parts.append(f"cart σ={self.cartesian_sigma:g} Å")
        return " + ".join(parts)

    def slug(self) -> str:
        return (
            f"t{self.torsion_deg:g}_r{self.rigid_rotation_deg:g}_s{self.rigid_shift:g}"
            f"_c{self.cartesian_sigma:g}"
        )
