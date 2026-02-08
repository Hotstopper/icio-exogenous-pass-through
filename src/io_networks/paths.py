from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    raw = Path(cfg["paths"]["raw"])
    return {
        "raw": raw,
        "interim": Path(cfg["paths"]["interim"]),
        "processed": Path(cfg["paths"]["processed"]),
        "matrices": Path(cfg["paths"]["matrices"]),
        "outputs": Path(cfg["paths"]["outputs"]),
    }


def ensure_output_dirs(paths: dict[str, Path]) -> Path:
    eda_dir = paths["outputs"] / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    return eda_dir
