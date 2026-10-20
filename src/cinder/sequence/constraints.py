"""Build policies from a :class:`~cinder.systems.design_space.DesignSpace`."""

from __future__ import annotations

import torch

from cinder.sequence.categorical import CategoricalPolicy
from cinder.systems.design_space import DesignSpace
from cinder.types import N_AA


def policy_from_design_space(
    space: DesignSpace, init: str = "uniform", native_bias: float = 2.0
) -> CategoricalPolicy:
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
    return CategoricalPolicy(mask, logits)
