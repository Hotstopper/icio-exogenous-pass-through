from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from io_networks.paths import ensure_output_dirs, resolve_paths


def _extract_year(filename: str) -> int | None:
    match = re.search(r"(19\d{2}|20\d{2})", filename)
    return int(match.group(1)) if match else None


def run_eda(cfg: dict[str, Any]) -> Path:
    paths = resolve_paths(cfg)
    eda_dir = ensure_output_dirs(paths)

    variant = "extended" if cfg["icio"].get("extended", False) else "regular"
    raw_dir = paths["raw"] / variant

    year_start = int(cfg["icio"]["year_range"]["start"])
    year_end = int(cfg["icio"]["year_range"]["end"])

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    rows: list[dict[str, Any]] = []
    for file_path in sorted(raw_dir.glob("*.csv")):
        year = _extract_year(file_path.name)
        if year is None or year < year_start or year > year_end:
            continue

        df = pd.read_csv(file_path)
        rows.append(
            {
                "year": year,
                "file": file_path.name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "null_cells": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )

    if not rows:
        raise ValueError(
            f"No CSV files matched year range {year_start}-{year_end} in {raw_dir}."
        )

    summary = pd.DataFrame(rows).sort_values(["year", "file"]).reset_index(drop=True)
    output_file = eda_dir / "summary_by_year.csv"
    summary.to_csv(output_file, index=False)
    return output_file
