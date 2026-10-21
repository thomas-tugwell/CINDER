"""Dataclass configuration with strict YAML loading.

Unknown keys are errors (typos should not silently change an experiment). A config file may
list ``defaults`` -- paths of fragments (relative to the config file's parent's parent, i.e.
the ``configs/`` root, or to the file itself) that are deep-merged underneath it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    pass


@dataclass
class SystemConfig:
    kind: str = "toy"  # "toy" | "structure"
    name: str = "toy"
    structure_path: str | None = None
    target_chains: list[str] = field(default_factory=lambda: ["A"])
    binder_chains: list[str] = field(default_factory=lambda: ["B"])
    binder_length: int = 3
    native_sequence: str | None = None
    design_positions: list[int] | None = None  # None -> all binder positions
    allowed_aa: dict[int, str] = field(default_factory=dict)
    forbidden_aa: str = ""
    toy_dim: int = 2
    movable: str = "all"  # "all" | "binder": which atoms conformational sampling may move

    def validate(self) -> None:
        if self.kind not in {"toy", "structure"}:
            raise ConfigError(f"system.kind must be 'toy' or 'structure', got {self.kind!r}")
        if self.kind == "structure" and not self.structure_path:
            raise ConfigError("system.structure_path is required for kind='structure'")
        if self.binder_length < 1:
            raise ConfigError("system.binder_length must be >= 1")


@dataclass
class EnergyConfig:
    backend: str = "toy_sequence"  # "toy_sequence" | "tmol"
    interface_weight: float = 1.0
    repulsion_weight: float = 1.0
    strain_weight: float = 0.0
    restraint_weight: float = 0.5
    hbond_weight: float = 0.0  # extra weight on the hbond term (already inside interface)
    elec_weight: float = 0.0  # extra weight on fa_elec (stands in for salt bridges)
    hbond_graph_weight: float = 0.0  # first/second-shell H-bond graph (see options['hbond_graph'])
    internal_weight: float = 0.0  # binder-alone energy; use > 0 with torsion/chi DOFs
    toy_seed: int = 0
    options: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.backend not in {"toy_sequence", "tmol"}:
            raise ConfigError(f"unknown energy.backend {self.backend!r}")
        extra = (self.hbond_weight, self.elec_weight, self.hbond_graph_weight, self.internal_weight)
        if min(extra) < 0:
            raise ConfigError("hbond/elec/hbond_graph/internal weights must be >= 0")


@dataclass
class ConformationConfig:
    backend: str = "langevin"  # "langevin" | "normalizing_flow" | "diffusion" | ...
    n_samples: int = 8
    guidance_strength: float = 1.0  # lambda
    step_size: float = 0.02
    n_steps: int = 300
    init_std: float = 1.0
    max_grad_norm: float | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.n_samples < 1:
            raise ConfigError("conformation.n_samples must be >= 1")
        if self.guidance_strength < 0:
            raise ConfigError("conformation.guidance_strength must be >= 0")


@dataclass
class SequenceConfig:
    backend: str = "categorical"  # "categorical" | "autoregressive"
    init: str = "uniform"  # "uniform" | "native"
    entropy_weight: float = 0.01

    def validate(self) -> None:
        if self.backend not in {"categorical", "autoregressive"}:
            raise ConfigError("sequence.backend must be 'categorical' or 'autoregressive'")
        if self.init not in {"uniform", "native"}:
            raise ConfigError("sequence.init must be 'uniform' or 'native'")
        if self.entropy_weight < 0:
            raise ConfigError("sequence.entropy_weight must be >= 0")


@dataclass
class OptimizationConfig:
    batch_size: int = 32
    learning_rate: float = 0.05
    iterations: int = 200
    aggregator: str = "mean"
    aggregator_options: dict[str, float] = field(default_factory=dict)
    normalize_advantage: bool = False
    checkpoint_every: int = 50
    log_probs_every: int = 10

    def validate(self) -> None:
        if self.batch_size < 1 or self.iterations < 0 or self.learning_rate <= 0:
            raise ConfigError("optimization: batch_size>=1, iterations>=0, learning_rate>0")


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 0
    deterministic: bool = False
    device: str = "cpu"
    output_dir: str = "runs"
    system: SystemConfig = field(default_factory=SystemConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    conformation: ConformationConfig = field(default_factory=ConformationConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def validate(self) -> None:
        for part in (self.system, self.energy, self.conformation, self.sequence, self.optimization):
            part.validate()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown config key(s) under '{path or 'root'}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        hint = hints[key]
        if dataclasses.is_dataclass(hint) and isinstance(hint, type):
            if not isinstance(value, dict):
                raise ConfigError(f"'{path + key}' must be a mapping")
            kwargs[key] = _build(hint, value, f"{path}{key}.")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def _load_raw(path: Path) -> dict[str, Any]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping")
    merged: dict[str, Any] = {}
    for d in raw.pop("defaults", []):
        candidates = [path.parent / f"{d}.yaml", path.parent.parent / f"{d}.yaml"]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise ConfigError(f"default fragment {d!r} not found next to {path}")
        merged = _deep_merge(merged, _load_raw(found))
    return _deep_merge(merged, raw)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    cfg = _build(ExperimentConfig, data)
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load, merge, validate. ``overrides`` is a nested dict deep-merged last."""
    raw = _load_raw(Path(path))
    if overrides:
        raw = _deep_merge(raw, overrides)
    return config_from_dict(raw)
