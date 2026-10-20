"""Capability report: what is installed and what can run here."""

from __future__ import annotations

import platform
from importlib import metadata, util
from typing import Any

import torch


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def run_doctor() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    tmol_ok = util.find_spec("tmol") is not None
    bioemu_ok = util.find_spec("bioemu") is not None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "mps_available": torch.backends.mps.is_available(),
        "tmol": {"available": tmol_ok, "version": _version("tmol")},
        "bioemu": {"available": bioemu_ok, "version": _version("bioemu")},
    }


def format_report(report: dict[str, Any]) -> str:
    def yn(d: dict[str, Any]) -> str:
        return f"{d['version']}" if d["available"] else "not installed"

    lines = [
        f"Python        {report['python']}",
        f"Platform      {report['platform']}",
        f"PyTorch       {report['torch']}",
        f"CUDA          {'yes' if report['cuda_available'] else 'no'}"
        + (f" ({report['gpu']})" if report["gpu"] else ""),
        f"MPS           {'yes' if report['mps_available'] else 'no'}",
        f"TMol          {yn(report['tmol'])}",
        f"BioEmu2       {yn(report['bioemu'])}",
    ]
    return "\n".join(lines)
