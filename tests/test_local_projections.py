from __future__ import annotations

import sys
import warnings
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from io_networks.local_projections import (
    build_panel_lp_dataset,
    fit_cumulative_panel_local_projections,
    fit_cumulative_panel_local_projections_iv,
    fit_panel_local_projections,
    fit_panel_local_projections_iv,
)
from io_networks.local_projections import compare_horizon0_to_run_ols
from io_networks.regression import extract_inference_arrays, run_ols


class _DummyInferenceModel:
    def __init__(self) -> None:
        self.params = pd.Series([1.0, -2.0], index=["a", "b"])
        self.use_t = False
        self.df_resid = 10.0

    def cov_params(self) -> np.ndarray:
        return np.array([[4.0, 0.0], [0.0, -1.0]])


def test_extract_inference_arrays_silences_negative_covariance_diagonal_warning():
    model = _DummyInferenceModel()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inference = extract_inference_arrays(model)

    assert caught == []
    assert inference["std_err"][0] == 2.0
    assert np.isnan(inference["std_err"][1])
    assert np.isnan(inference["ci_low"][1])
    assert np.isnan(inference["ci_high"][1])


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


def test_build_panel_lp_dataset_can_lag_xi_by_one_year_for_interactions():
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

        news_df = pd.DataFrame(
            {
                "date": ["2000-03-31", "2000-06-30", "2001-03-31", "2001-06-30"],
                "period": ["2000Q1", "2000Q2", "2001Q1", "2001Q2"],
                "news_shock": [1.5, -2.0, 0.5, 3.0],
            }
        )
        news_path = temp_root / "kaenzig.csv"
        news_df.to_csv(news_path, index=False)

        panel = build_panel_lp_dataset(
            xi_path=xi_path,
            cpi_path=cpi_path,
            oil_path=oil_path,
            news_path=news_path,
            news_column="news_shock",
            cpi_freq="Q",
            lag_xi_by_one_year_for_xi_x_oil=True,
        )

        aaa_2000_q1 = panel.loc[(panel["country"] == "AAA") & (panel["period"] == "2000-Q1")].iloc[0]
        aaa_2001_q1 = panel.loc[(panel["country"] == "AAA") & (panel["period"] == "2001-Q1")].iloc[0]
        bbb_2001_q2 = panel.loc[(panel["country"] == "BBB") & (panel["period"] == "2001-Q2")].iloc[0]

        assert aaa_2000_q1["xi"] == 1.0
        assert pd.isna(aaa_2000_q1["xi_x_oil"])
        assert pd.isna(aaa_2000_q1["xi_x_news"])
        assert aaa_2001_q1["xi"] == 1.2
        assert aaa_2001_q1["xi_x_oil"] == 1.0 * 0.30
        assert aaa_2001_q1["xi_x_news"] == 1.0 * 0.5
        assert bbb_2001_q2["xi_x_oil"] == 0.5 * 0.40
        assert bbb_2001_q2["xi_x_news"] == 0.5 * 3.0
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_build_panel_lp_dataset_adds_quarterly_kaenzig_news_instrument():
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

        news_df = pd.DataFrame(
            {
                "date": ["2000-03-31", "2000-06-30", "2001-03-31", "2001-06-30"],
                "period": ["2000Q1", "2000Q2", "2001Q1", "2001Q2"],
                "2025M06": [1.5, -2.0, 0.5, 3.0],
            }
        )
        news_path = temp_root / "kaenzig.csv"
        news_df.to_csv(news_path, index=False)

        panel = build_panel_lp_dataset(
            xi_path=xi_path,
            cpi_path=cpi_path,
            oil_path=oil_path,
            news_path=news_path,
            cpi_freq="Q",
        )

        assert "news" in panel.columns
        assert "xi_x_news" in panel.columns
        q1_aaa = panel.loc[(panel["country"] == "AAA") & (panel["period"] == "2000-Q1")].iloc[0]
        q2_bbb = panel.loc[(panel["country"] == "BBB") & (panel["period"] == "2000-Q2")].iloc[0]
        assert q1_aaa["news"] == 1.5
        assert q1_aaa["xi_x_news"] == 1.5
        assert q2_bbb["news"] == -2.0
        assert q2_bbb["xi_x_news"] == -1.0
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_build_panel_lp_dataset_can_choose_news_column_from_multi_column_csv():
    temp_root = Path.cwd() / "outputs" / f"test_local_projections_{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        xi_df = pd.DataFrame(
            {
                "year": [2000, 2000],
                "country": ["AAA", "BBB"],
                "xi": [2.0, 0.5],
                "status": ["ok", "ok"],
            }
        )
        xi_path = temp_root / "xi.parquet"
        xi_df.to_parquet(xi_path, index=False)

        cpi_df = pd.DataFrame(
            {
                "FREQ": ["Q"] * 2,
                "MEASURE": ["CPI"] * 2,
                "METHODOLOGY": ["N"] * 2,
                "EXPENDITURE": ["_T"] * 2,
                "ADJUSTMENT": ["N"] * 2,
                "UNIT_MEASURE": ["PC"] * 2,
                "REF_AREA": ["AAA", "BBB"],
                "TIME_PERIOD": ["2000-Q1", "2000-Q1"],
                "OBS_VALUE": [2.0, 1.0],
            }
        )
        cpi_path = temp_root / "cpi.csv"
        cpi_df.to_csv(cpi_path, index=False)

        oil_df = pd.DataFrame({"date": ["2000-03-31"], "pct_change": [0.10]})
        oil_path = temp_root / "oil.csv"
        oil_df.to_csv(oil_path, index=False)

        news_df = pd.DataFrame(
            {
                "date": ["2000-03-31"],
                "period": ["2000Q1"],
                "surprise": [1.0],
                "news_shock": [3.0],
            }
        )
        news_path = temp_root / "kaenzig.csv"
        news_df.to_csv(news_path, index=False)

        panel = build_panel_lp_dataset(
            xi_path=xi_path,
            cpi_path=cpi_path,
            oil_path=oil_path,
            news_path=news_path,
            news_column="news_shock",
            cpi_freq="Q",
        )

        assert panel["news"].tolist() == [3.0, 3.0]
        assert panel["xi_x_news"].tolist() == [6.0, 1.5]
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


def test_run_ols_supports_two_way_clustered_standard_errors():
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

    model, coef_df = run_ols(
        pd.DataFrame(rows),
        cpi_lags=1,
        include_country_fe=True,
        regressand="cpi",
        cov_type="cluster_country_time",
    )

    assert float(model.nobs) > 0
    assert "std_err_cluster_country_time" in coef_df.columns


def test_fit_panel_local_projections_iv_returns_all_horizons_and_terms():
    rows = []
    periods = ["2000-Q1", "2000-Q2", "2000-Q3", "2000-Q4", "2001-Q1", "2001-Q2", "2001-Q3", "2001-Q4"]
    for country_idx, country in enumerate(["AAA", "BBB", "CCC", "DDD"]):
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
                    "xi_x_news": 0.12 + 0.015 * period_idx + 0.008 * country_idx,
                }
            )

    df = pd.DataFrame(rows)

    irf_df, coef_df, models = fit_panel_local_projections_iv(
        df,
        horizons=2,
        y_lags=1,
        shock_lags=1,
        include_country_fe=True,
        include_year_fe=False,
        cov_type="cluster_country",
    )

    assert irf_df["horizon"].tolist() == [0, 1, 2]
    assert set(irf_df["term"]) == {"xi_x_oil"}
    assert set(coef_df["horizon"]) == {0, 1, 2}
    assert set(models) == {0, 1, 2}
    horizon0_terms = set(coef_df.loc[coef_df["horizon"] == 0, "term"])
    assert "xi_x_oil" in horizon0_terms
    assert "xi_x_oil_lag1" in horizon0_terms


def test_fit_panel_local_projections_supports_two_way_clustered_standard_errors():
    rows = []
    periods = ["2000-Q1", "2000-Q2", "2000-Q3", "2000-Q4", "2001-Q1", "2001-Q2", "2001-Q3", "2001-Q4"]
    for country_idx, country in enumerate(["AAA", "BBB", "CCC", "DDD"]):
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

    irf_df, coef_df, models = fit_panel_local_projections(
        pd.DataFrame(rows),
        horizons=2,
        y_lags=1,
        include_country_fe=True,
        include_year_fe=True,
        cov_type="cluster_country_time",
    )

    assert irf_df["horizon"].tolist() == [0, 1, 2]
    assert "std_err_cluster_country_time" in coef_df.columns
    assert set(models) == {0, 1, 2}


def test_fit_panel_local_projections_iv_supports_two_way_clustered_standard_errors():
    rows = []
    periods = ["2000-Q1", "2000-Q2", "2000-Q3", "2000-Q4", "2001-Q1", "2001-Q2", "2001-Q3", "2001-Q4"]
    for country_idx, country in enumerate(["AAA", "BBB", "CCC", "DDD"]):
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
                    "xi_x_news": 0.12 + 0.015 * period_idx + 0.008 * country_idx,
                }
            )

    irf_df, coef_df, models = fit_panel_local_projections_iv(
        pd.DataFrame(rows),
        horizons=2,
        y_lags=1,
        shock_lags=1,
        include_country_fe=True,
        include_year_fe=False,
        cov_type="cluster_country_time",
    )

    assert irf_df["horizon"].tolist() == [0, 1, 2]
    assert "std_err_cluster_country_time" in coef_df.columns
    assert set(models) == {0, 1, 2}


def test_fit_cumulative_panel_local_projections_supports_two_way_clustered_standard_errors():
    rows = []
    periods = [f"2000-Q{q}" for q in [1, 2, 3, 4]] + [f"2001-Q{q}" for q in [1, 2, 3, 4]] + [f"2002-Q{q}" for q in [1, 2, 3, 4]]
    for country_idx, country in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        for period_idx, period in enumerate(periods):
            year = int(period[:4])
            quarter = int(period[-1])
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "period": period,
                    "time_index": year * 4 + quarter - 1,
                    "cpi_pct_change": 0.02 + 0.004 * period_idx + 0.002 * country_idx,
                    "xi_x_oil": 0.10 + 0.01 * period_idx + 0.005 * country_idx,
                }
            )

    df = pd.DataFrame(rows)
    for horizon in range(3):
        df[f"cum_cpi_lead{horizon}"] = (
            df.groupby("country")["cpi_pct_change"]
            .transform(lambda series, h=horizon: sum(series.shift(-step) for step in range(h + 1)))
        )

    irf_df, coef_df, models = fit_cumulative_panel_local_projections(
        df,
        horizons=2,
        y_lags=1,
        include_country_fe=True,
        include_time_fe=True,
        cov_type="cluster_country_time",
    )

    assert irf_df["horizon"].tolist() == [0, 1, 2]
    assert "std_err_cluster_country_time" in coef_df.columns
    assert set(models) == {0, 1, 2}


def test_fit_cumulative_panel_local_projections_iv_preserves_irf_shape():
    rows = []
    periods = [f"2000-Q{q}" for q in [1, 2, 3, 4]] + [f"2001-Q{q}" for q in [1, 2, 3, 4]] + [f"2002-Q{q}" for q in [1, 2, 3, 4]]
    for country_idx, country in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        for period_idx, period in enumerate(periods):
            year = int(period[:4])
            quarter = int(period[-1])
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "period": period,
                    "time_index": year * 4 + quarter - 1,
                    "cpi_pct_change": 0.02 + 0.004 * period_idx + 0.002 * country_idx,
                    "xi_x_oil": 0.10 + 0.01 * period_idx + 0.005 * country_idx,
                    "xi_x_news": 0.08 + 0.012 * period_idx + 0.004 * country_idx,
                }
            )

    df = pd.DataFrame(rows)
    for horizon in range(3):
        df[f"cum_cpi_lead{horizon}"] = (
            df.groupby("country")["cpi_pct_change"]
            .transform(lambda series, h=horizon: sum(series.shift(-step) for step in range(h + 1)))
        )

    irf_df, coef_df, _ = fit_cumulative_panel_local_projections_iv(
        df,
        horizons=2,
        y_lags=1,
        shock_lags=1,
        include_country_fe=True,
        include_time_fe=False,
        cov_type="cluster_country",
    )

    required_irf_cols = {"horizon", "term", "coef", "ci_low_95", "ci_high_95", "nobs"}
    assert required_irf_cols.issubset(irf_df.columns)
    assert set(irf_df["term"]) == {"xi_x_oil"}
    assert "xi_x_oil_lag1" in set(coef_df.loc[coef_df["horizon"] == 0, "term"])


def test_fit_cumulative_panel_local_projections_iv_supports_two_way_clustered_standard_errors():
    rows = []
    periods = [f"2000-Q{q}" for q in [1, 2, 3, 4]] + [f"2001-Q{q}" for q in [1, 2, 3, 4]] + [f"2002-Q{q}" for q in [1, 2, 3, 4]]
    for country_idx, country in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        for period_idx, period in enumerate(periods):
            year = int(period[:4])
            quarter = int(period[-1])
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "period": period,
                    "time_index": year * 4 + quarter - 1,
                    "cpi_pct_change": 0.02 + 0.004 * period_idx + 0.002 * country_idx,
                    "xi_x_oil": 0.10 + 0.01 * period_idx + 0.005 * country_idx,
                    "xi_x_news": 0.08 + 0.012 * period_idx + 0.004 * country_idx,
                }
            )

    df = pd.DataFrame(rows)
    for horizon in range(3):
        df[f"cum_cpi_lead{horizon}"] = (
            df.groupby("country")["cpi_pct_change"]
            .transform(lambda series, h=horizon: sum(series.shift(-step) for step in range(h + 1)))
        )

    irf_df, coef_df, models = fit_cumulative_panel_local_projections_iv(
        df,
        horizons=2,
        y_lags=1,
        shock_lags=1,
        include_country_fe=True,
        include_time_fe=False,
        cov_type="cluster_country_time",
    )

    assert irf_df["horizon"].tolist() == [0, 1, 2]
    assert "std_err_cluster_country_time" in coef_df.columns
    assert set(models) == {0, 1, 2}
