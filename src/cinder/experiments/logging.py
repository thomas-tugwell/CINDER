"""Run directories, metadata capture and CSV metric logging."""

from __future__ import annotations

import csv
import datetime as dt
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from cinder.utils.io import sha256_file, write_json

log = logging.getLogger("cinder")


def create_run_dir(output_dir: str | Path, name: str) -> Path:
    """Create a fresh ``<output_dir>/<timestamp>_<name>[_n]`` directory; never reuses one."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = f"{stamp}_{name}"
    for n in range(1000):
        path = root / (base if n == 0 else f"{base}_{n}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        (path / "checkpoints").mkdir()
        (path / "structures").mkdir()
        return path
    raise RuntimeError(f"could not create a unique run directory under {root}")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _pkg_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_metadata(seed: int, structure_path: str | Path | None = None) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            n: _pkg_version(n) for n in ("torch", "numpy", "pyyaml", "tmol", "bioemu", "cinder")
        },
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "mps_available": torch.backends.mps.is_available(),
        "seed": seed,
        "input_structure_sha256": (
            sha256_file(structure_path)
            if structure_path and Path(structure_path).exists()
            else None
        ),
        "created": dt.datetime.now().isoformat(),
    }


def write_run_header(run_dir: Path, config_dict: dict[str, Any], metadata: dict[str, Any]) -> None:
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
    write_json(run_dir / "metadata.json", metadata)


class CsvLogger:
    """Append-only CSV with a header fixed by the first row (later rows must share keys)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fields: list[str] | None = None

    def write(self, row: dict[str, Any]) -> None:
        new = self._fields is None
        if new:
            self._fields = list(row)
        elif set(row) != set(self._fields or []):
            raise ValueError(f"row keys changed: {sorted(set(row) ^ set(self._fields or []))}")
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields or [])
            if new:
                w.writeheader()
            w.writerow(row)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
