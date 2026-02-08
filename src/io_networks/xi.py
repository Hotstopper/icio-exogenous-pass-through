from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from io_networks.paths import resolve_paths


def _extract_year(filename: str) -> int | None:
    match = re.search(r"(19\d{2}|20\d{2})", filename)
    return int(match.group(1)) if match else None


def _git_commit() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return output.strip()
    except Exception:
        return "unknown"


def _country_from_label(label: str) -> str:
    if "_" not in label:
        return ""
    return label.split("_", 1)[0]


def _raw_file_map(raw_dir: Path, year_start: int, year_end: int) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for file_path in sorted(raw_dir.glob("*.csv")):
        year = _extract_year(file_path.name)
        if year is None or year < year_start or year > year_end:
            continue
        mapping[year] = file_path
    return mapping


def _load_hfce_frame(raw_file: Path, countries: list[str]) -> tuple[pd.DataFrame, set[str]]:
    header = pd.read_csv(raw_file, nrows=0).columns.tolist()
    wanted = [f"{country}_HFCE" for country in countries]
    present = [c for c in wanted if c in header]

    if not present:
        return pd.DataFrame(), set()

    frame = pd.read_csv(raw_file, usecols=["V1", *present])
    frame = frame.set_index("V1")
    return frame, set(present)


def build_xi(cfg: dict[str, Any]) -> Path:
    paths = resolve_paths(cfg)
    variant = "extended" if cfg["icio"].get("extended", False) else "regular"

    year_start = int(cfg["icio"]["year_range"]["start"])
    year_end = int(cfg["icio"]["year_range"]["end"])

    blocks_dir = paths["matrices"] / "blocks" / variant
    if not blocks_dir.exists():
        raise FileNotFoundError(f"Blocks directory not found: {blocks_dir}. Run build-blocks first.")

    raw_dir = paths["raw"] / variant
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    out_dir = paths["processed"] / "xi" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_by_year = _raw_file_map(raw_dir, year_start, year_end)

    build_time = datetime.now(timezone.utc).isoformat()
    commit_hash = _git_commit()

    xi_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    weights_rows: list[dict[str, Any]] = []

    for year in range(year_start, year_end + 1):
        blocks_npz = blocks_dir / f"blocks_{year}.npz"
        blocks_meta = blocks_dir / f"blocks_{year}_meta.parquet"
        raw_file = raw_by_year.get(year)

        if not blocks_npz.exists() or not blocks_meta.exists() or raw_file is None:
            continue

        arr = np.load(blocks_npz)
        required_arrays = {"tau", "tau_dir", "tau_amp"}
        missing = required_arrays - set(arr.files)
        if missing:
            raise ValueError(
                f"Year {year}: blocks file {blocks_npz.name} is missing arrays {sorted(missing)}. "
                "Re-run build-blocks to refresh artifacts."
            )

        tau = arr["tau"]
        tau_dir = arr["tau_dir"]
        tau_amp = arr["tau_amp"]

        meta = pd.read_parquet(blocks_meta)
        n_meta = meta[meta["group"] == "N"].sort_values("index_in_group").reset_index(drop=True)
        labels_n = n_meta["label"].astype(str).tolist()

        if len(labels_n) != len(tau):
            raise ValueError(
                f"Year {year}: label count {len(labels_n)} does not match tau length {len(tau)}"
            )

        tau_s = pd.Series(tau, index=labels_n, dtype=float)
        tau_dir_s = pd.Series(tau_dir, index=labels_n, dtype=float)
        tau_amp_s = pd.Series(tau_amp, index=labels_n, dtype=float)

        countries = sorted({_country_from_label(label) for label in labels_n if _country_from_label(label)})

        hfce_frame, hfce_cols_present = _load_hfce_frame(raw_file, countries)

        for country in countries:
            labels_country = [label for label in labels_n if label.startswith(f"{country}_")]
            hfce_col = f"{country}_HFCE"
            has_hfce_column = hfce_col in hfce_cols_present

            if not has_hfce_column:
                xi_rows.append(
                    {
                        "year": year,
                        "country": country,
                        "variant": variant,
                        "xi": np.nan,
                        "xi_dir": np.nan,
                        "xi_amp": np.nan,
                        "identity_residual": np.nan,
                        "n_sectors_used": len(labels_country),
                        "status": "missing_hfce_column",
                        "build_timestamp_utc": build_time,
                        "git_commit": commit_hash,
                    }
                )
                diag_rows.append(
                    {
                        "year": year,
                        "country": country,
                        "variant": variant,
                        "hfce_column": hfce_col,
                        "has_hfce_column": False,
                        "weight_sum_raw": np.nan,
                        "weight_sum_norm": np.nan,
                        "n_positive_weights": 0,
                        "n_zero_weights": len(labels_country),
                        "n_negative_clipped": np.nan,
                        "status": "missing_hfce_column",
                        "build_timestamp_utc": build_time,
                        "git_commit": commit_hash,
                    }
                )
                continue

            raw_weights = pd.to_numeric(
                hfce_frame.reindex(labels_country)[hfce_col], errors="coerce"
            ).fillna(0.0)

            negative_count = int((raw_weights < 0).sum())
            raw_weights = raw_weights.clip(lower=0.0)
            raw_sum = float(raw_weights.sum())

            if raw_sum <= 0.0:
                xi_rows.append(
                    {
                        "year": year,
                        "country": country,
                        "variant": variant,
                        "xi": np.nan,
                        "xi_dir": np.nan,
                        "xi_amp": np.nan,
                        "identity_residual": np.nan,
                        "n_sectors_used": len(labels_country),
                        "status": "zero_hfce_mass",
                        "build_timestamp_utc": build_time,
                        "git_commit": commit_hash,
                    }
                )
                diag_rows.append(
                    {
                        "year": year,
                        "country": country,
                        "variant": variant,
                        "hfce_column": hfce_col,
                        "has_hfce_column": True,
                        "weight_sum_raw": raw_sum,
                        "weight_sum_norm": np.nan,
                        "n_positive_weights": 0,
                        "n_zero_weights": len(labels_country),
                        "n_negative_clipped": negative_count,
                        "status": "zero_hfce_mass",
                        "build_timestamp_utc": build_time,
                        "git_commit": commit_hash,
                    }
                )
                continue

            weights_norm = raw_weights / raw_sum

            tau_c = tau_s.reindex(labels_country).to_numpy(dtype=float)
            tau_dir_c = tau_dir_s.reindex(labels_country).to_numpy(dtype=float)
            tau_amp_c = tau_amp_s.reindex(labels_country).to_numpy(dtype=float)
            w = weights_norm.to_numpy(dtype=float)

            xi = float(np.dot(w, tau_c))
            xi_dir = float(np.dot(w, tau_dir_c))
            xi_amp = float(np.dot(w, tau_amp_c))
            residual = float(xi - (xi_dir + xi_amp))

            xi_rows.append(
                {
                    "year": year,
                    "country": country,
                    "variant": variant,
                    "xi": xi,
                    "xi_dir": xi_dir,
                    "xi_amp": xi_amp,
                    "identity_residual": residual,
                    "n_sectors_used": len(labels_country),
                    "status": "ok",
                    "build_timestamp_utc": build_time,
                    "git_commit": commit_hash,
                }
            )

            diag_rows.append(
                {
                    "year": year,
                    "country": country,
                    "variant": variant,
                    "hfce_column": hfce_col,
                    "has_hfce_column": True,
                    "weight_sum_raw": raw_sum,
                    "weight_sum_norm": float(weights_norm.sum()),
                    "n_positive_weights": int((weights_norm > 0).sum()),
                    "n_zero_weights": int((weights_norm == 0).sum()),
                    "n_negative_clipped": negative_count,
                    "status": "ok",
                    "build_timestamp_utc": build_time,
                    "git_commit": commit_hash,
                }
            )

            for label, wr, wn, t, td, ta in zip(
                labels_country,
                raw_weights.to_numpy(dtype=float),
                w,
                tau_c,
                tau_dir_c,
                tau_amp_c,
                strict=True,
            ):
                weights_rows.append(
                    {
                        "year": year,
                        "country": country,
                        "variant": variant,
                        "sector_label": label,
                        "weight_raw": float(wr),
                        "weight_norm": float(wn),
                        "tau": float(t),
                        "tau_dir": float(td),
                        "tau_amp": float(ta),
                        "contrib_xi": float(wn * t),
                        "contrib_xi_dir": float(wn * td),
                        "contrib_xi_amp": float(wn * ta),
                        "build_timestamp_utc": build_time,
                        "git_commit": commit_hash,
                    }
                )

    if not xi_rows:
        raise ValueError("No xi rows were generated. Check blocks/raw inputs and year range.")

    xi_df = pd.DataFrame(xi_rows).sort_values(["year", "country"]).reset_index(drop=True)
    diag_df = pd.DataFrame(diag_rows).sort_values(["year", "country"]).reset_index(drop=True)
    weights_df = pd.DataFrame(weights_rows).sort_values(["year", "country", "sector_label"]).reset_index(drop=True)

    xi_path = out_dir / "xi_by_country_year.parquet"
    diag_path = out_dir / "weights_diagnostics.parquet"
    weights_path = out_dir / "weights_by_country_sector.parquet"

    xi_df.to_parquet(xi_path, index=False)
    diag_df.to_parquet(diag_path, index=False)
    weights_df.to_parquet(weights_path, index=False)

    summary_df = (
        xi_df.groupby("year", as_index=False)
        .agg(
            n_countries=("country", "nunique"),
            n_ok=("status", lambda s: int((s == "ok").sum())),
            xi_mean=("xi", "mean"),
            xi_min=("xi", "min"),
            xi_max=("xi", "max"),
            max_abs_identity_residual=("identity_residual", lambda s: float(np.nanmax(np.abs(s)))),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    summary_path = out_dir / "xi_summary.csv"
    try:
        summary_df.to_csv(summary_path, index=False)
    except PermissionError:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary_path = out_dir / f"xi_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)

    return xi_path
