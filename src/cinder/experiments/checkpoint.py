"""Policy/optimizer checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch


class _Stateful(Protocol):
    """Anything with torch-style ``state_dict`` / ``load_state_dict`` (modules, policies)."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> Any: ...


def save_checkpoint(
    path: Path,
    policy: _Stateful,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    extra: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: Path, policy: _Stateful, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
