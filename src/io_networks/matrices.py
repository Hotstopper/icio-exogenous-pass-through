from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from io_networks.paths import resolve_paths

SPECIAL_ROWS = {"TLS", "VA", "OUT"}
ZERO_TOL = 1e-12


def _extract_year(filename: str) -> int | None:
    match = re.search(r"(19\d{2}|20\d{2})", filename)
    return int(match.group(1)) if match else None


def _git_commit() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return output.strip()
    except Exception:
        return "unknown"


def _build_a_for_frame(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    if "V1" not in df.columns:
        raise ValueError("Required row-label column V1 is missing.")
    if "OUT" not in df.columns:
        raise ValueError("Required OUT column is missing.")

    row_labels = df["V1"].astype("string").fillna("")
    production_mask = ~row_labels.isin(SPECIAL_ROWS)
    sector_labels = row_labels.loc[production_mask].tolist()

    if not sector_labels:
        raise ValueError("No production sectors found after excluding TLS/VA/OUT rows.")

    missing_sector_cols = [label for label in sector_labels if label not in df.columns]
    if missing_sector_cols:
        raise ValueError(
            "Missing intermediate columns for production sectors. "
            f"Examples: {missing_sector_cols[:5]}"
        )

    z = (
        df.loc[production_mask, sector_labels]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    x = pd.to_numeric(df.loc[production_mask, "OUT"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    zero_out_mask = x <= ZERO_TOL

    a = np.zeros_like(z, dtype=float)
    valid_cols = ~zero_out_mask
    if np.any(valid_cols):
        a[:, valid_cols] = z[:, valid_cols] / x[valid_cols]

    return a, sector_labels, x, zero_out_mask


def build_yearly_a(cfg: dict[str, Any]) -> Path:
    paths = resolve_paths(cfg)

    variant = "extended" if cfg["icio"].get("extended", False) else "regular"
    raw_dir = paths["raw"] / variant

    year_start = int(cfg["icio"]["year_range"]["start"])
    year_end = int(cfg["icio"]["year_range"]["end"])

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    out_dir = paths["matrices"] / "A" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[int, Path]] = []
    for file_path in sorted(raw_dir.glob("*.csv")):
        year = _extract_year(file_path.name)
        if year is None or year < year_start or year > year_end:
            continue
        files.append((year, file_path))

    if not files:
        raise ValueError(
            f"No CSV files matched year range {year_start}-{year_end} in {raw_dir}."
        )

    build_time = datetime.now(timezone.utc).isoformat()
    commit_hash = _git_commit()

    summary_rows: list[dict[str, Any]] = []

    for year, file_path in files:
        df = pd.read_csv(file_path)
        a, sector_labels, x, zero_out_mask = _build_a_for_frame(df)

        npz_path = out_dir / f"A_{year}.npz"
        np.savez_compressed(npz_path, A=a)

        meta_df = pd.DataFrame(
            {
                "year": year,
                "variant": variant,
                "sector": sector_labels,
                "out": x,
                "zero_out": zero_out_mask,
            }
        )
        meta_path = out_dir / f"A_{year}_meta.parquet"
        meta_df.to_parquet(meta_path, index=False)

        summary_rows.append(
            {
                "year": year,
                "variant": variant,
                "source_file": file_path.name,
                "n_sectors": int(len(sector_labels)),
                "zero_out_count": int(zero_out_mask.sum()),
                "npz_file": npz_path.name,
                "meta_file": meta_path.name,
                "nan_count_in_A": int(np.isnan(a).sum()),
                "inf_count_in_A": int(np.isinf(a).sum()),
                "colsum_min": float(np.sum(a, axis=0).min()),
                "colsum_max": float(np.sum(a, axis=0).max()),
                "build_timestamp_utc": build_time,
                "git_commit": commit_hash,
                "policy": "zero_column_when_out_leq_tol",
                "zero_tol": ZERO_TOL,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("year").reset_index(drop=True)
    summary_path = out_dir / "build_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return summary_path