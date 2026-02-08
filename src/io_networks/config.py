from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and enforce required top-level sections."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")

    required_sections = {"project", "icio", "paths"}
    missing = sorted(required_sections - set(data.keys()))
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    year_range = data.get("icio", {}).get("year_range", {})
    start = year_range.get("start")
    end = year_range.get("end")
    if start is None or end is None:
        raise ValueError("config.icio.year_range must include start and end.")
    if int(start) > int(end):
        raise ValueError("config.icio.year_range.start must be <= end.")

    return data
