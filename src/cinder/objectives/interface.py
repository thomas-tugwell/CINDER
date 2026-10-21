"""Configuration of the composite design energy ``U_design``.

``U_design = w_interface U_interface + w_rep U_rep + w_strain U_strain + w_restraint U_restraint``.
Which backend provides each named term lives in the energy layer; this module only holds the
weights and builds a :class:`~cinder.energy.composite.CompositeEnergy` from named term models.
This is a *design surrogate*, deliberately not called a binding free energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cinder.energy.composite import CompositeEnergy, EnergyTerm
from cinder.types import EnergyModel


@dataclass
class DesignEnergyWeights:
    interface: float = 1.0
    repulsion: float = 1.0
    strain: float = 0.0
    restraint: float = 0.5
    hbond: float = 0.0  # extra emphasis on top of ``interface`` (which already contains hbond)
    elec: float = 0.0  # extra emphasis on top of ``interface`` (fa_elec stands in for salt bridges)
    hbond_graph: float = 0.0  # differentiable first/second-shell H-bond graph
    internal: float = 0.0  # the binder's own energy (keeps flexible binders folded)
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "interface": self.interface,
            "repulsion": self.repulsion,
            "strain": self.strain,
            "restraint": self.restraint,
            "hbond": self.hbond,
            "elec": self.elec,
            "hbond_graph": self.hbond_graph,
            "internal": self.internal,
            **self.extra,
        }


def build_design_energy(
    term_models: dict[str, EnergyModel], weights: DesignEnergyWeights
) -> CompositeEnergy:
    """Compose the provided named terms using ``weights`` (zero-weight terms are dropped)."""
    w = weights.as_dict()
    unknown = [n for n in term_models if n not in w]
    if unknown:
        raise ValueError(f"no weight configured for terms {unknown}")
    terms = [EnergyTerm(n, m, w[n]) for n, m in term_models.items() if w[n] != 0.0]
    return CompositeEnergy(terms)
