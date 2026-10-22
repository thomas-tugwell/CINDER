"""Assemble the TMol design energy ``U_design`` for a loaded complex.

Terms (each with its own weight, all logged separately):

* ``interface``, ``repulsion``, ``strain`` -- TMol term groups
  (:data:`cinder.energy.tmol.TERM_GROUPS`)
* ``hbond``, ``elec`` -- extra emphasis on TMol's ``hbond`` / ``fa_elec`` (already inside
  ``interface``; default weight 0). ``fa_elec`` stands in for a salt-bridge term.
* ``internal`` -- the binder's own energy (weight 0 by default; set > 0 for torsion/chi DOFs)
* ``restraint`` -- harmonic tether to the reference pose (backend independent)
* ``hbond_graph`` -- differentiable first/second-shell hydrogen-bond graph (weight 0 by default;
  inner weights from ``energy.options['hbond_graph']``)
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from cinder.energy.composite import CompositeEnergy
from cinder.energy.restraints import HarmonicRestraint
from cinder.energy.tmol import TMolContext, group_energies
from cinder.energy.tmol_roles import polar_roles
from cinder.objectives.hbond_graph import (
    HBondGraphConfig,
    HBondGraphEnergy,
    hard_hbond_report,
    shell_residues,
)
from cinder.objectives.interface import DesignEnergyWeights, build_design_energy
from cinder.systems.specification import MolecularSystem

if TYPE_CHECKING:
    from cinder.config import EnergyConfig


def graph_config_from_options(options: dict[str, Any]) -> HBondGraphConfig:
    """``energy.options['hbond_graph']`` -> :class:`HBondGraphConfig` (unknown keys are errors)."""
    raw = dict(options.get("hbond_graph", {}))
    valid = {f.name for f in fields(HBondGraphConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown energy.options.hbond_graph keys: {sorted(unknown)}")
    return HBondGraphConfig(**raw)


def build_hbond_graph_energy(
    context: TMolContext, options: dict[str, Any] | None = None
) -> HBondGraphEnergy:
    """The H-bond graph energy for the complex pose held by ``context``."""
    cfg = graph_config_from_options(options or {})
    region = float((options or {}).get("hbond_graph_region", 8.0))
    return HBondGraphEnergy(polar_roles(context, region), cfg)


def build_design_energy_for_context(
    ec: EnergyConfig,
    system: MolecularSystem,
    context: TMolContext,
    sequence_tokens: Tensor | None = None,
) -> CompositeEnergy:
    """``U_design`` for the complex pose held by ``context``.

    ``sequence_tokens`` ``[L]`` is the binder sequence the pose was built for (default: the
    native sequence); the energy terms verify it on every call.
    """
    tokens = system.design_space.native_tokens() if sequence_tokens is None else sequence_tokens
    models: dict[str, Any] = dict(group_energies(context, tokens))
    models["restraint"] = HarmonicRestraint(float(ec.options.get("restraint_k", 1.0)))
    if ec.hbond_graph_weight > 0:
        models["hbond_graph"] = build_hbond_graph_energy(context, ec.options)
    weights = DesignEnergyWeights(
        interface=ec.interface_weight,
        repulsion=ec.repulsion_weight,
        strain=ec.strain_weight,
        restraint=ec.restraint_weight,
        hbond=ec.hbond_weight,
        elec=ec.elec_weight,
        hbond_graph=ec.hbond_graph_weight,
        internal=ec.internal_weight,
    )
    return build_design_energy(models, weights)


def build_tmol_design_energy(ec: EnergyConfig, system: MolecularSystem) -> CompositeEnergy:
    """``U_design`` for the native complex of ``system``."""
    ctx: TMolContext = system.extras["tmol"]  # type: ignore[assignment]
    return build_design_energy_for_context(ec, system, ctx)


def propose_design_positions(
    system: MolecularSystem,
    shells: str = "both",
    region_cutoff: float = 8.0,
    contact_cutoff: float = 4.5,
) -> list[int]:
    """Binder positions (0-based, binder-local) in the native pose's hydrogen-bond shells.

    ``shells``: ``first`` (binder residues bonding across the interface), ``second`` (binder
    residues that bond to a first-shell node but not across the interface) or ``both``. Uses the
    independent hard-threshold counter on the native coordinates. ``contact`` instead takes binder
    residues with a heavy atom within ``contact_cutoff`` Å of the target, which also finds
    hydrophobic interfaces that have no cross-interface hydrogen bond (e.g. BH3 peptides).
    Intended as a *starting list* to inspect and edit, not a final design space.
    """
    if shells not in {"first", "second", "both", "contact"}:
        raise ValueError(f"shells must be 'first', 'second', 'both' or 'contact', got {shells!r}")
    context: TMolContext = system.extras["tmol"]  # type: ignore[assignment]
    binder_range: Any = system.extras["binder_residue_range"]
    lo, hi = int(binder_range[0]), int(binder_range[1])
    roles = polar_roles(context, region_cutoff)
    if shells == "contact":
        owner = roles.residue_of_atom[roles.heavy_atoms]
        xyz = system.reference_coordinates[roles.heavy_atoms]
        in_binder = (owner >= lo) & (owner < hi)
        close = torch.cdist(xyz[in_binder], xyz[~in_binder]).min(dim=1).values < contact_cutoff
        return sorted({int(r) - lo for r in owner[in_binder][close]})
    report = hard_hbond_report(system.reference_coordinates, roles)
    first, second = shell_residues(report, roles, chain=1)
    chosen = {"first": first, "second": second, "both": first + second}[shells]
    return sorted(r - lo for r in chosen if lo <= r < hi)
