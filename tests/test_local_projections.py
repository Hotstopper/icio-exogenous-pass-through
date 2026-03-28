from __future__ import annotations

import sys
from pathlib import Path
import shutil
import uuid

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from io_networks.local_projections import build_panel_lp_dataset, fit_panel_local_projections
from io_networks.local_projections import compare_horizon0_to_run_ols


def test_build_panel_lp_dataset_merges_optional_controls():
    temp_root = Path.cwd() / "outputs" / f"test_local_projections_{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        xi_df = pd.DataFrame(
            {
                "year": [2000, 2001, 2000, 2001],
                "country": ["AAA", "AAA", "BBB", "BBB"],
                "xi": [1.0, 1.2, 0.5, 0.7],
                "status": ["ok", "ok", "ok", "ok"],
            }
        )
        xi_path = temp_root / "xi.parquet"
        xi_df.to_parquet(xi_path, index=False)

        cpi_df = pd.DataFrame(
            {
                "FREQ": ["A", "A", "A", "A"],
                "MEASURE": ["CPI"] * 4,
                "METHODOLOGY": ["N"] * 4,
                "EXPENDITURE": ["_T"] * 4,
                "ADJUSTMENT": ["N"] * 4,
                "UNIT_MEASURE": ["PC"] * 4,
                "REF_AREA": ["AAA", "AAA", "BBB", "BBB"],
                "TIME_PERIOD": [2000, 2001, 2000, 2001],
                "OBS_VALUE": [2.0, 2.5, 1.0, 1.5],
            }
        )
        cpi_path = temp_root / "cpi.csv"
        cpi_df.to_csv(cpi_path, index=False)

        oil_df = pd.DataFrame(
            {
                "date": ["2000-12-31", "2001-12-31"],
                "pct_change": [0.10, 0.20],
            }
        )
        oil_path = temp_root / "oil.csv"
        oil_df.to_csv(oil_path, index=False)

        control_df = pd.DataFrame(
            {
                "country": ["AAA", "AAA", "BBB", "BBB"],
                "year": [2000, 2001, 2000, 2001],
                "z_control": [5.0, 6.0, 7.0, 8.0],
                "status": ["ok", "ok", "ok", "ok"],
            }
        )
        control_path = temp_root / "control.parquet"
        control_df.to_parquet(control_path, index=False)

        panel = build_panel_lp_dataset(
            xi_path=xi_path,
            cpi_path=cpi_path,
            oil_path=oil_path,
            controls=[
                {
                    "name": "z_control",
                    "path": control_path,
                    "value_column": "z_control",
                }
            ],
        )

        assert list(panel.columns) == [
            "country",
            "year",
            "xi",
            "oil_pct_change",
            "xi_x_oil",
            "cpi_pct_change",
            "z_control",
        ]
        assert len(panel) == 4
        assert panel["z_control"].notna().all()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_build_panel_lp_dataset_can_require_control_for_sample():
    temp_root = Path.cwd() / "outputs" / f"test_local_projections_{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        xi_df = pd.DataFrame(
            {
                "year": [2000, 2001, 2000, 2001],
                "country": ["AAA", "AAA", "BBB", "BBB"],
                "xi": [1.0, 1.2, 0.5, 0.7],
                "status": ["ok", "ok", "ok", "ok"],
            }
        )
        xi_path = temp_root / "xi.parquet"
        xi_df.to_parquet(xi_path, index=False)

        cpi_df = pd.DataFrame(
            {
                "FREQ": ["A", "A", "A", "A"],
                "MEASURE": ["CPI"] * 4,
                "METHODOLOGY": ["N"] * 4,
                "EXPENDITURE": ["_T"] * 4,
                "ADJUSTMENT": ["N"] * 4,
                "UNIT_MEASURE": ["PC"] * 4,
                "REF_AREA": ["AAA", "AAA", "BBB", "BBB"],
                "TIME_PERIOD": [2000, 2001, 2000, 2001],
                "OBS_VALUE": [2.0, 2.5, 1.0, 1.5],
            }
        )
        cpi_path = temp_root / "cpi.csv"
        cpi_df.to_csv(cpi_path, index=False)

        oil_df = pd.DataFrame(
            {
                "date": ["2000-12-31", "2001-12-31"],
                "pct_change": [0.10, 0.20],
            }
        )
        oil_path = temp_root / "oil.csv"
        oil_df.to_csv(oil_path, index=False)

        control_df = pd.DataFrame(
            {
                "country": ["AAA", "AAA", "BBB", "BBB"],
                "year": [2000, 2001, 2000, 2001],
                "z_control": [5.0, None, 7.0, 8.0],
                "status": ["ok", "ok", "ok", "ok"],
            }
        )
        control_path = temp_root / "control.parquet"
        control_df.to_parquet(control_path, index=False)

        panel = build_panel_lp_dataset(
            xi_path=xi_path,
            cpi_path=cpi_path,
            oil_path=oil_path,
            controls=[
                {
                    "name": "z_control",
                    "path": control_path,
                    "value_column": "z_control",
                    "required_for_sample": True,
                }
            ],
        )

        assert len(panel) == 3
        assert panel["z_control"].notna().all()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_fit_panel_local_projections_returns_all_horizons():
    countries = ["AAA", "BBB", "CCC"]
    years = list(range(2000, 2008))
    rows = []
    for country_idx, country in enumerate(countries):
        for year_idx, year in enumerate(years):
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "cpi_pct_change": 0.03 + 0.01 * year_idx + 0.005 * country_idx,
                    "xi_x_oil": 0.10 + 0.02 * year_idx + 0.01 * country_idx,
                    "c_diff": 0.01 + 0.003 * year_idx + 0.002 * country_idx,
                }
            )

    df = pd.DataFrame(
        rows
    )

    irf_df, coef_df, models = fit_panel_local_projections(
        df,
        horizons=2,
        y_lags=1,
        controls=["c_diff"],
        include_country_fe=True,
        include_year_fe=True,
    )

    assert irf_df["horizon"].tolist() == [0, 1, 2]
    assert set(irf_df["term"]) == {"xi_x_oil"}
    assert set(coef_df["horizon"]) == {0, 1, 2}
    assert set(models) == {0, 1, 2}


def test_build_panel_lp_dataset_keeps_quarterly_rows():
    temp_root = Path.cwd() / "outputs" / f"test_local_projections_{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        xi_df = pd.DataFrame(
            {
                "year": [2000, 2001, 2000, 2001],
                "country": ["AAA", "AAA", "BBB", "BBB"],
                "xi": [1.0, 1.2, 0.5, 0.7],
                "status": ["ok", "ok", "ok", "ok"],
            }
        )
        xi_path = temp_root / "xi.parquet"
        xi_df.to_parquet(xi_path, index=False)

        cpi_df = pd.DataFrame(
            {
                "FREQ": ["Q"] * 8,
                "MEASURE": ["CPI"] * 8,
                "METHODOLOGY": ["N"] * 8,
                "EXPENDITURE": ["_T"] * 8,
                "ADJUSTMENT": ["N"] * 8,
                "UNIT_MEASURE": ["PC"] * 8,
                "REF_AREA": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB"],
                "TIME_PERIOD": ["2000-Q1", "2000-Q2", "2001-Q1", "2001-Q2", "2000-Q1", "2000-Q2", "2001-Q1", "2001-Q2"],
                "OBS_VALUE": [2.0, 2.5, 3.0, 3.5, 1.0, 1.5, 2.0, 2.5],
            }
        )
        cpi_path = temp_root / "cpi.csv"
        cpi_df.to_csv(cpi_path, index=False)

        oil_df = pd.DataFrame(
            {
                "date": ["2000-03-31", "2000-06-30", "2001-03-31", "2001-06-30"],
                "pct_change": [0.10, 0.20, 0.30, 0.40],
            }
        )
        oil_path = temp_root / "oil.csv"
        oil_df.to_csv(oil_path, index=False)

        panel = build_panel_lp_dataset(
            xi_path=xi_path,
            cpi_path=cpi_path,
            oil_path=oil_path,
            cpi_freq="Q",
        )

        assert list(panel.columns) == [
            "country",
            "year",
            "period",
            "time_index",
            "xi",
            "oil_pct_change",
            "xi_x_oil",
            "cpi_pct_change",
        ]
        assert len(panel) == 8
        assert panel["period"].tolist()[:4] == ["2000-Q1", "2000-Q2", "2001-Q1", "2001-Q2"]
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_fit_panel_local_projections_uses_time_index_when_present():
    rows = []
    periods = ["2000-Q1", "2000-Q2", "2000-Q3", "2000-Q4", "2001-Q1", "2001-Q2"]
    for country_idx, country in enumerate(["AAA", "BBB", "CCC"]):
        for period_idx, period in enumerate(periods):
            year = int(period[:4])
            quarter = int(period[-1])
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "period": period,
                    "time_index": year * 4 + quarter - 1,
                    "cpi_pct_change": 0.03 + 0.01 * period_idx + 0.005 * country_idx,
                    "xi_x_oil": 0.10 + 0.02 * period_idx + 0.01 * country_idx,
                }
            )

    df = pd.DataFrame(rows)

    irf_df, coef_df, models = fit_panel_local_projections(
        df,
        horizons=2,
        y_lags=1,
        include_country_fe=True,
        include_year_fe=True,
        cov_type="hc3",
    )

    assert irf_df["horizon"].tolist() == [0, 1, 2]
    assert set(coef_df["horizon"]) == {0, 1, 2}
    assert set(models) == {0, 1, 2}


def test_compare_horizon0_to_run_ols_matches_exactly():
    countries = ["AAA", "BBB", "CCC"]
    years = list(range(2000, 2008))
    rows = []
    for country_idx, country in enumerate(countries):
        for year_idx, year in enumerate(years):
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "cpi_pct_change": 0.03 + 0.01 * year_idx + 0.005 * country_idx,
                    "xi_x_oil": 0.10 + 0.02 * year_idx + 0.01 * country_idx,
                }
            )

    df = pd.DataFrame(rows)
    comparison = compare_horizon0_to_run_ols(
        df,
        cpi_lags=1,
        include_country_fe=False,
        regressand="cpi",
        include_year_fe=False,
        cov_type="hc3",
    )

    assert abs(comparison.loc[0, "difference"]) < 1e-12
