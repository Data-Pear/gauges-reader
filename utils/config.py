from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
