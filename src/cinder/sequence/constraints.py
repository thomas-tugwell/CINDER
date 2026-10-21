"""Build policies from a :class:`~cinder.systems.design_space.DesignSpace`."""

from __future__ import annotations

import torch
from torch import Tensor

from cinder.sequence.autoregressive import AutoregressivePolicy
from cinder.sequence.categorical import CategoricalPolicy
from cinder.systems.design_space import DesignSpace
from cinder.types import N_AA


def _initial_logits(space: DesignSpace, init: str, native_bias: float) -> Tensor:
    """``init='uniform'`` -> equal logits; ``'native'`` -> ``native_bias`` added to the native AA
    (when it is allowed at that position)."""
    mask = space.mask()
    logits = torch.zeros(mask.shape)
    if init == "native":
        native = space.native_design_tokens()
        for i, aa in enumerate(native.tolist()):
            if mask[i, aa]:
                logits[i, aa] = native_bias
    elif init != "uniform":
        raise ValueError(f"unknown init {init!r}; expected 'uniform' or 'native'")
    assert logits.shape[1] == N_AA
    return logits


def policy_from_design_space(
    space: DesignSpace,
    init: str = "uniform",
    native_bias: float = 2.0,
    backend: str = "categorical",
) -> CategoricalPolicy | AutoregressivePolicy:
    """Build a policy over ``space``: ``backend`` is ``categorical`` (independent positions) or
    ``autoregressive`` (each position conditioned on earlier ones)."""
    logits = _initial_logits(space, init, native_bias)
    if backend == "categorical":
        return CategoricalPolicy(space.mask(), logits)
    if backend == "autoregressive":
        return AutoregressivePolicy(space.mask(), logits)
    raise ValueError(
        f"unknown sequence backend {backend!r}; expected 'categorical' or 'autoregressive'"
    )
