from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as spla

from io_networks.paths import resolve_paths

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


def _load_quarterly_real_gdp_growth(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Quarterly real GDP CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"country", "period", "year", "quarter", "real_gdp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Quarterly real GDP input is missing columns: {sorted(missing)}")

    out = df.loc[:, ["country", "period", "year", "quarter", "real_gdp"]].copy()
    out["country"] = out["country"].astype(str).str.strip().str.upper()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["quarter"] = pd.to_numeric(out["quarter"], errors="coerce")
    out["real_gdp"] = pd.to_numeric(out["real_gdp"], errors="coerce")
    out = out.dropna(subset=["country", "period", "year", "quarter", "real_gdp"]).copy()
    out["year"] = out["year"].astype(int)
    out["quarter"] = out["quarter"].astype(int)
    out = out.sort_values(["country", "year", "quarter"]).reset_index(drop=True)

    if (out["real_gdp"] <= 0.0).any():
        raise ValueError("Quarterly real GDP must be strictly positive to compute log differences.")

    out["quarterly_real_gdp_growth"] = (
        out.groupby("country")["real_gdp"].transform(lambda s: np.log(s).diff())
    )
    return out[["country", "period", "year", "quarter", "quarterly_real_gdp_growth"]]


def build_c_diff_quarterly(cfg: dict[str, Any]) -> Path:
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

    quarterly_gdp_path = paths["processed"] / "real_gdp" / "quarterly_real_gdp.csv"
    quarterly_growth = _load_quarterly_real_gdp_growth(quarterly_gdp_path)

    out_dir = paths["processed"] / "c_diff" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_by_year = _raw_file_map(raw_dir, year_start, year_end)

    build_time = datetime.now(timezone.utc).isoformat()
    commit_hash = _git_commit()

    c_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    weights_rows: list[dict[str, Any]] = []

    for year in range(year_start, year_end + 1):
        blocks_npz = blocks_dir / f"blocks_{year}.npz"
        blocks_meta = blocks_dir / f"blocks_{year}_meta.parquet"
        raw_file = raw_by_year.get(year)

        if not blocks_npz.exists() or not blocks_meta.exists() or raw_file is None:
            continue

        quarters = quarterly_growth.loc[quarterly_growth["year"] == year].copy()
        if quarters.empty:
            continue
        quarter_keys = (
            quarters.loc[:, ["year", "quarter", "period"]]
            .drop_duplicates()
            .sort_values(["year", "quarter"])
            .reset_index(drop=True)
        )

        arr = np.load(blocks_npz)
        required_arrays = {"A_NN", "c_N", "v_over_out_N"}
        missing = required_arrays - set(arr.files)
        if missing:
            raise ValueError(
                f"Year {year}: blocks file {blocks_npz.name} is missing arrays {sorted(missing)}. "
                "Re-run build-blocks to refresh artifacts."
            )

        a_nn = arr["A_NN"]
        c_n = arr["c_N"]
        v_over_out_n = arr["v_over_out_N"]
        c_n_quarter = c_n / 4.0

        meta = pd.read_parquet(blocks_meta)
        n_meta = meta[meta["group"] == "N"].sort_values("index_in_group").reset_index(drop=True)
        labels_n = n_meta["label"].astype(str).tolist()

        if len(labels_n) != len(c_n_quarter):
            raise ValueError(
                f"Year {year}: label count {len(labels_n)} does not match c_N length {len(c_n_quarter)}"
            )
        if len(labels_n) != len(v_over_out_n):
            raise ValueError(
                f"Year {year}: label count {len(labels_n)} does not match v_over_out_N length {len(v_over_out_n)}"
            )

        countries = sorted({_country_from_label(label) for label in labels_n if _country_from_label(label)})
        hfce_frame, hfce_cols_present = _load_hfce_frame(raw_file, countries)

        m = sparse.eye(a_nn.shape[0], format="csr") - sparse.csr_matrix(a_nn).transpose().tocsr()
        country_label_index = {
            country: [i for i, label in enumerate(labels_n) if label.startswith(f"{country}_")]
            for country in countries
        }

        for quarter_row in quarter_keys.itertuples(index=False):
            period = str(quarter_row.period)
            quarter = int(quarter_row.quarter)
            quarter_slice = quarters.loc[quarters["period"] == period, ["country", "quarterly_real_gdp_growth"]]
            growth_by_country = {
                str(country): float(growth) if pd.notna(growth) else np.nan
                for country, growth in quarter_slice.itertuples(index=False, name=None)
            }
            observed_growth_country = {
                country for country, growth in growth_by_country.items() if pd.notna(growth)
            }

            gdp_growth_n = np.full(len(labels_n), np.nan, dtype=float)
            for country, idxs in country_label_index.items():
                g = growth_by_country.get(country, np.nan)
                if idxs:
                    gdp_growth_n[idxs] = g

            gdp_growth_n_filled = np.where(np.isnan(gdp_growth_n), 0.0, gdp_growth_n)
            rhs = v_over_out_n * gdp_growth_n_filled
            c_gdp_q_n = np.full(len(labels_n), np.nan, dtype=float)
            c_gdp_solve_success = True
            c_gdp_solve_note = "ok"
            c_gdp_residual_inf = np.nan
            try:
                c_gdp_q_n = spla.spsolve(m, rhs)
                c_gdp_q_n = np.asarray(c_gdp_q_n, dtype=float)
                c_gdp_residual_inf = float(np.linalg.norm(m.dot(c_gdp_q_n) - rhs, ord=np.inf))
            except Exception as exc:
                c_gdp_solve_success = False
                c_gdp_solve_note = str(exc)

            c_diff_q_n = c_n_quarter - c_gdp_q_n
            c_diff_s = pd.Series(c_diff_q_n, index=labels_n, dtype=float)
            c_n_quarter_s = pd.Series(c_n_quarter, index=labels_n, dtype=float)
            c_gdp_q_s = pd.Series(c_gdp_q_n, index=labels_n, dtype=float)

            for country in countries:
                labels_country = [label for label in labels_n if label.startswith(f"{country}_")]
                hfce_col = f"{country}_HFCE"
                has_hfce_column = hfce_col in hfce_cols_present

                if not has_hfce_column:
                    c_rows.append(
                        {
                            "year": year,
                            "quarter": quarter,
                            "period": period,
                            "country": country,
                            "variant": variant,
                            "c_diff": np.nan,
                            "n_sectors_used": len(labels_country),
                            "status": "missing_hfce_column",
                            "build_timestamp_utc": build_time,
                            "git_commit": commit_hash,
                        }
                    )
                    diag_rows.append(
                        {
                            "year": year,
                            "quarter": quarter,
                            "period": period,
                            "country": country,
                            "variant": variant,
                            "hfce_column": hfce_col,
                            "has_hfce_column": False,
                            "weight_sum_raw": np.nan,
                            "weight_sum_norm": np.nan,
                            "n_positive_weights": 0,
                            "n_zero_weights": len(labels_country),
                            "n_negative_clipped": np.nan,
                            "c_gdp_solve_success": c_gdp_solve_success,
                            "c_gdp_solve_note": c_gdp_solve_note,
                            "c_gdp_residual_inf": c_gdp_residual_inf,
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
                    c_rows.append(
                        {
                            "year": year,
                            "quarter": quarter,
                            "period": period,
                            "country": country,
                            "variant": variant,
                            "c_diff": np.nan,
                            "n_sectors_used": len(labels_country),
                            "status": "zero_hfce_mass",
                            "build_timestamp_utc": build_time,
                            "git_commit": commit_hash,
                        }
                    )
                    diag_rows.append(
                        {
                            "year": year,
                            "quarter": quarter,
                            "period": period,
                            "country": country,
                            "variant": variant,
                            "hfce_column": hfce_col,
                            "has_hfce_column": True,
                            "weight_sum_raw": raw_sum,
                            "weight_sum_norm": np.nan,
                            "n_positive_weights": 0,
                            "n_zero_weights": len(labels_country),
                            "n_negative_clipped": negative_count,
                            "c_gdp_solve_success": c_gdp_solve_success,
                            "c_gdp_solve_note": c_gdp_solve_note,
                            "c_gdp_residual_inf": c_gdp_residual_inf,
                            "status": "zero_hfce_mass",
                            "build_timestamp_utc": build_time,
                            "git_commit": commit_hash,
                        }
                    )
                    continue

                weights_norm = raw_weights / raw_sum
                c_diff_country = c_diff_s.reindex(labels_country).to_numpy(dtype=float)
                c_country_quarter = c_n_quarter_s.reindex(labels_country).to_numpy(dtype=float)
                c_gdp_country = c_gdp_q_s.reindex(labels_country).to_numpy(dtype=float)
                w = weights_norm.to_numpy(dtype=float)

                has_observed_quarterly_growth = country in observed_growth_country

                if not has_observed_quarterly_growth:
                    c_rows.append(
                        {
                            "year": year,
                            "quarter": quarter,
                            "period": period,
                            "country": country,
                            "variant": variant,
                            "c_diff": np.nan,
                            "n_sectors_used": len(labels_country),
                            "status": "missing_quarterly_real_gdp_growth",
                            "build_timestamp_utc": build_time,
                            "git_commit": commit_hash,
                        }
                    )
                    diag_rows.append(
                        {
                            "year": year,
                            "quarter": quarter,
                            "period": period,
                            "country": country,
                            "variant": variant,
                            "hfce_column": hfce_col,
                            "has_hfce_column": True,
                            "weight_sum_raw": raw_sum,
                            "weight_sum_norm": float(weights_norm.sum()),
                            "n_positive_weights": int((weights_norm > 0).sum()),
                            "n_zero_weights": int((weights_norm == 0).sum()),
                            "n_negative_clipped": negative_count,
                            "c_gdp_solve_success": c_gdp_solve_success,
                            "c_gdp_solve_note": c_gdp_solve_note,
                            "c_gdp_residual_inf": c_gdp_residual_inf,
                            "status": "missing_quarterly_real_gdp_growth",
                            "build_timestamp_utc": build_time,
                            "git_commit": commit_hash,
                        }
                    )
                    continue

                if np.isnan(c_diff_country).any():
                    c_rows.append(
                        {
                            "year": year,
                            "quarter": quarter,
                            "period": period,
                            "country": country,
                            "variant": variant,
                            "c_diff": np.nan,
                            "n_sectors_used": len(labels_country),
                            "status": "missing_c_diff_vector",
                            "build_timestamp_utc": build_time,
                            "git_commit": commit_hash,
                        }
                    )
                    diag_rows.append(
                        {
                            "year": year,
                            "quarter": quarter,
                            "period": period,
                            "country": country,
                            "variant": variant,
                            "hfce_column": hfce_col,
                            "has_hfce_column": True,
                            "weight_sum_raw": raw_sum,
                            "weight_sum_norm": float(weights_norm.sum()),
                            "n_positive_weights": int((weights_norm > 0).sum()),
                            "n_zero_weights": int((weights_norm == 0).sum()),
                            "n_negative_clipped": negative_count,
                            "c_gdp_solve_success": c_gdp_solve_success,
                            "c_gdp_solve_note": c_gdp_solve_note,
                            "c_gdp_residual_inf": c_gdp_residual_inf,
                            "status": "missing_c_diff_vector",
                            "build_timestamp_utc": build_time,
                            "git_commit": commit_hash,
                        }
                    )
                    continue

                c_diff_scalar = float(np.dot(w, c_diff_country))

                c_rows.append(
                    {
                        "year": year,
                        "quarter": quarter,
                        "period": period,
                        "country": country,
                        "variant": variant,
                        "c_diff": c_diff_scalar,
                        "n_sectors_used": len(labels_country),
                        "status": "ok",
                        "build_timestamp_utc": build_time,
                        "git_commit": commit_hash,
                    }
                )
                diag_rows.append(
                    {
                        "year": year,
                        "quarter": quarter,
                        "period": period,
                        "country": country,
                        "variant": variant,
                        "hfce_column": hfce_col,
                        "has_hfce_column": True,
                        "weight_sum_raw": raw_sum,
                        "weight_sum_norm": float(weights_norm.sum()),
                        "n_positive_weights": int((weights_norm > 0).sum()),
                        "n_zero_weights": int((weights_norm == 0).sum()),
                        "n_negative_clipped": negative_count,
                        "c_gdp_solve_success": c_gdp_solve_success,
                        "c_gdp_solve_note": c_gdp_solve_note,
                        "c_gdp_residual_inf": c_gdp_residual_inf,
                        "status": "ok",
                        "build_timestamp_utc": build_time,
                        "git_commit": commit_hash,
                    }
                )

                for label, wr, wn, c_val, c_gdp_val, c_diff_val in zip(
                    labels_country,
                    raw_weights.to_numpy(dtype=float),
                    w,
                    c_country_quarter,
                    c_gdp_country,
                    c_diff_country,
                    strict=True,
                ):
                    weights_rows.append(
                        {
                            "year": year,
                            "quarter": quarter,
                            "period": period,
                            "country": country,
                            "variant": variant,
                            "sector_label": label,
                            "weight_raw": float(wr),
                            "weight_norm": float(wn),
                            "c_N_over_4": float(c_val),
                            "quarterly_real_gdp_growth": float(growth_by_country.get(country, np.nan)),
                            "c_gdp_q_N": float(c_gdp_val),
                            "c_diff_q_N": float(c_diff_val),
                            "contrib_c_diff": float(wn * c_diff_val),
                            "build_timestamp_utc": build_time,
                            "git_commit": commit_hash,
                        }
                    )

    if not c_rows:
        raise ValueError(
            "No quarterly c_diff rows were generated. Check blocks, raw HFCE inputs, and quarterly real GDP."
        )

    c_df = pd.DataFrame(c_rows).sort_values(["year", "quarter", "country"]).reset_index(drop=True)
    diag_df = pd.DataFrame(diag_rows).sort_values(["year", "quarter", "country"]).reset_index(drop=True)
    if weights_rows:
        weights_df = pd.DataFrame(weights_rows).sort_values(
            ["year", "quarter", "country", "sector_label"]
        ).reset_index(drop=True)
    else:
        weights_df = pd.DataFrame(
            columns=[
                "year",
                "quarter",
                "period",
                "country",
                "variant",
                "sector_label",
                "weight_raw",
                "weight_norm",
                "c_N_over_4",
                "quarterly_real_gdp_growth",
                "c_gdp_q_N",
                "c_diff_q_N",
                "contrib_c_diff",
                "build_timestamp_utc",
                "git_commit",
            ]
        )

    c_path = out_dir / "c_diff_quarterly_by_country_period.parquet"
    diag_path = out_dir / "weights_diagnostics_quarterly.parquet"
    weights_path = out_dir / "weights_by_country_sector_quarterly.parquet"

    c_df.to_parquet(c_path, index=False)
    diag_df.to_parquet(diag_path, index=False)
    weights_df.to_parquet(weights_path, index=False)

    summary_df = (
        c_df.groupby(["year", "quarter", "period"], as_index=False)
        .agg(
            n_countries=("country", "nunique"),
            n_ok=("status", lambda s: int((s == "ok").sum())),
            c_diff_mean=("c_diff", "mean"),
            c_diff_min=("c_diff", "min"),
            c_diff_max=("c_diff", "max"),
        )
        .sort_values(["year", "quarter"])
        .reset_index(drop=True)
    )

    summary_path = out_dir / "c_diff_quarterly_summary.csv"
    try:
        summary_df.to_csv(summary_path, index=False)
    except PermissionError:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary_path = out_dir / f"c_diff_quarterly_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)

    return c_path
