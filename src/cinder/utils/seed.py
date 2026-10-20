"""Seeding helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch. ``deterministic`` also requests deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
