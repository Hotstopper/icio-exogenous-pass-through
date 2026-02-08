from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from io_networks.paths import ensure_output_dirs, resolve_paths

FINAL_DEMAND_SUFFIXES = {"HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"}
SPECIAL_ROWS = {"TLS", "VA", "OUT"}
ZERO_TOL = 1e-12


def _extract_year(filename: str) -> int | None:
    match = re.search(r"(19\d{2}|20\d{2})", filename)
    return int(match.group(1)) if match else None


def _country_from_label(label: str) -> str | None:
    if "_" not in label:
        return None
    return label.split("_", 1)[0]


def _is_final_demand_column(col: str) -> bool:
    if "_" not in col:
        return False
    return col.split("_", 1)[1] in FINAL_DEMAND_SUFFIXES


def _build_country_column_maps(columns: list[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    domestic_transaction: dict[str, list[str]] = {}
    domestic_intermediate: dict[str, list[str]] = {}

    for col in columns:
        if col == "OUT" or "_" not in col:
            continue
        country = col.split("_", 1)[0]
        domestic_transaction.setdefault(country, []).append(col)
        if not _is_final_demand_column(col):
            domestic_intermediate.setdefault(country, []).append(col)

    return domestic_transaction, domestic_intermediate


def _append_check(
    checks: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    year: int,
    file_name: str,
    check_name: str,
    passed: bool,
    value: Any,
    detail: str,
) -> None:
    status = "pass" if passed else "fail"
    checks.append(
        {
            "year": year,
            "file": file_name,
            "check": check_name,
            "status": status,
            "value": value,
            "detail": detail,
        }
    )
    if not passed:
        issues.append(
            {
                "year": year,
                "file": file_name,
                "check": check_name,
                "value": value,
                "detail": detail,
            }
        )


def run_eda(cfg: dict[str, Any]) -> Path:
    paths = resolve_paths(cfg)
    eda_dir = ensure_output_dirs(paths)

    variant = "extended" if cfg["icio"].get("extended", False) else "regular"
    raw_dir = paths["raw"] / variant

    year_start = int(cfg["icio"]["year_range"]["start"])
    year_end = int(cfg["icio"]["year_range"]["end"])

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

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

    expected_years = set(range(year_start, year_end + 1))
    observed_years = {year for year, _ in files}
    missing_years = sorted(expected_years - observed_years)

    summary_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    baseline_columns: list[str] | None = None
    baseline_row_labels: list[str] | None = None

    for year, file_path in files:
        df = pd.read_csv(file_path)
        file_name = file_path.name

        if "V1" not in df.columns:
            raise ValueError(f"Required row-label column V1 is missing in {file_name}")

        row_labels = df["V1"].astype("string").fillna("")
        columns = list(df.columns)

        numeric_cols = [c for c in columns if c != "V1"]
        numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        transaction_cols = [c for c in numeric_cols if c != "OUT"]
        intermediate_cols = [c for c in transaction_cols if not _is_final_demand_column(c)]

        production_mask = ~row_labels.isin(SPECIAL_ROWS)

        duplicate_col_count = int(pd.Index(columns).duplicated().sum())
        duplicate_row_label_count = int(row_labels.duplicated().sum())
        null_cells = int(numeric.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        all_zero_transaction_rows = 0
        all_zero_intermediate_rows = 0
        all_zero_intermediate_cols = 0
        domestic_zero_transaction_rows = 0
        domestic_zero_intermediate_rows = 0
        out_missing_production_rows = 0
        out_nonpositive_production_rows = 0

        if transaction_cols:
            all_zero_transaction_rows = int(
                (numeric.loc[production_mask, transaction_cols].abs().sum(axis=1) <= ZERO_TOL).sum()
            )

        if intermediate_cols:
            prod_intermediate = numeric.loc[production_mask, intermediate_cols]
            all_zero_intermediate_rows = int((prod_intermediate.abs().sum(axis=1) <= ZERO_TOL).sum())
            all_zero_intermediate_cols = int((prod_intermediate.abs().sum(axis=0) <= ZERO_TOL).sum())

        if "OUT" in numeric.columns:
            out_series = numeric.loc[production_mask, "OUT"]
            out_missing_production_rows = int(out_series.isna().sum())
            out_nonpositive_production_rows = int((out_series <= ZERO_TOL).sum())

        domestic_transaction_map, domestic_intermediate_map = _build_country_column_maps(columns)
        production_countries = (
            row_labels.loc[production_mask]
            .map(_country_from_label)
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        for country in production_countries:
            country_row_mask = production_mask & row_labels.str.startswith(f"{country}_")

            tx_cols = domestic_transaction_map.get(country, [])
            if tx_cols:
                tx_sum = numeric.loc[country_row_mask, tx_cols].abs().sum(axis=1)
                domestic_zero_transaction_rows += int((tx_sum <= ZERO_TOL).sum())

            int_cols = domestic_intermediate_map.get(country, [])
            if int_cols:
                int_sum = numeric.loc[country_row_mask, int_cols].abs().sum(axis=1)
                domestic_zero_intermediate_rows += int((int_sum <= ZERO_TOL).sum())

        columns_match_baseline = True
        row_labels_match_baseline = True
        if baseline_columns is None:
            baseline_columns = columns
            baseline_row_labels = row_labels.tolist()
        else:
            columns_match_baseline = columns == baseline_columns
            row_labels_match_baseline = row_labels.tolist() == baseline_row_labels

        summary_rows.append(
            {
                "year": year,
                "file": file_name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "null_cells": null_cells,
                "duplicate_rows": duplicate_rows,
                "duplicate_column_names": duplicate_col_count,
                "duplicate_row_labels": duplicate_row_label_count,
                "all_zero_transaction_rows": all_zero_transaction_rows,
                "all_zero_intermediate_rows": all_zero_intermediate_rows,
                "all_zero_intermediate_columns": all_zero_intermediate_cols,
                "domestic_zero_transaction_rows": domestic_zero_transaction_rows,
                "domestic_zero_intermediate_rows": domestic_zero_intermediate_rows,
                "out_missing_production_rows": out_missing_production_rows,
                "out_nonpositive_production_rows": out_nonpositive_production_rows,
                "columns_match_baseline": columns_match_baseline,
                "row_labels_match_baseline": row_labels_match_baseline,
            }
        )

        _append_check(
            check_rows,
            issue_rows,
            year,
            file_name,
            "structure.unique_column_names",
            duplicate_col_count == 0,
            duplicate_col_count,
            "Duplicate column names in file.",
        )
        _append_check(
            check_rows,
            issue_rows,
            year,
            file_name,
            "structure.unique_row_labels",
            duplicate_row_label_count == 0,
            duplicate_row_label_count,
            "Duplicate V1 row labels in file.",
        )
        _append_check(
            check_rows,
            issue_rows,
            year,
            file_name,
            "structure.columns_match_baseline",
            columns_match_baseline,
            columns_match_baseline,
            "Column schema differs from baseline year in the selected range.",
        )
        _append_check(
            check_rows,
            issue_rows,
            year,
            file_name,
            "structure.row_labels_match_baseline",
            row_labels_match_baseline,
            row_labels_match_baseline,
            "Row labels differ from baseline year in the selected range.",
        )
        _append_check(
            check_rows,
            issue_rows,
            year,
            file_name,
            "accounting.has_out_column",
            "OUT" in numeric.columns,
            "OUT" in numeric.columns,
            "Expected OUT column is missing.",
        )
        _append_check(
            check_rows,
            issue_rows,
            year,
            file_name,
            "accounting.out_missing_production_rows",
            out_missing_production_rows == 0,
            out_missing_production_rows,
            "Production rows with missing OUT value.",
        )
        _append_check(
            check_rows,
            issue_rows,
            year,
            file_name,
            "zeros.all_zero_intermediate_rows",
            all_zero_intermediate_rows == 0,
            all_zero_intermediate_rows,
            "Rows with zero intermediate transactions across all countries.",
        )
        _append_check(
            check_rows,
            issue_rows,
            year,
            file_name,
            "zeros.all_zero_intermediate_columns",
            all_zero_intermediate_cols == 0,
            all_zero_intermediate_cols,
            "Columns with zero intermediate inflows across all production rows.",
        )
        _append_check(
            check_rows,
            issue_rows,
            year,
            file_name,
            "zeros.domestic_zero_intermediate_rows",
            domestic_zero_intermediate_rows == 0,
            domestic_zero_intermediate_rows,
            "Rows with zero domestic intermediate transactions.",
        )

    if missing_years:
        issue_rows.append(
            {
                "year": "all",
                "file": "all",
                "check": "structure.missing_year_files",
                "value": ",".join(str(y) for y in missing_years),
                "detail": "Expected years are missing from input files.",
            }
        )
        check_rows.append(
            {
                "year": "all",
                "file": "all",
                "check": "structure.missing_year_files",
                "status": "fail",
                "value": ",".join(str(y) for y in missing_years),
                "detail": "Expected years are missing from input files.",
            }
        )
    else:
        check_rows.append(
            {
                "year": "all",
                "file": "all",
                "check": "structure.missing_year_files",
                "status": "pass",
                "value": "",
                "detail": "All years in range have at least one file.",
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["year", "file"]).reset_index(drop=True)
    checks_df = pd.DataFrame(check_rows)
    issues_df = pd.DataFrame(issue_rows)

    summary_path = eda_dir / "summary_by_year.csv"
    checks_path = eda_dir / "check_results.csv"
    issues_path = eda_dir / "issues.csv"

    summary_df.to_csv(summary_path, index=False)
    checks_df.to_csv(checks_path, index=False)
    issues_df.to_csv(issues_path, index=False)

    return summary_path