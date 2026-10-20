"""Small IO helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
