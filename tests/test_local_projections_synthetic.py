from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from io_networks.local_projections import fit_panel_local_projections


def _make_synthetic_panel(
    *,
    n_countries: int = 30,
    n_periods: int = 80,
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    countries = [f"C{i:02d}" for i in range(n_countries)]
    oil = rng.normal(0.0, 0.8, size=n_periods)
    years = 2000 + np.arange(n_periods) // 4
    quarters = 1 + np.arange(n_periods) % 4
    xis = {country: float(rng.uniform(0.5, 1.5)) for country in countries}
    country_fe = {country: float(rng.normal(0.0, 0.02)) for country in countries}

    rows: list[dict[str, float | int | str]] = []
    for country in countries:
        y_lag1 = 0.0
        y_lag2 = 0.0
        shock_lag1 = 0.0
        for t in range(n_periods):
            shock = xis[country] * oil[t]
            eps = float(rng.normal(0.0, 0.05))
            cpi = 0.45 * y_lag1 + 0.20 * y_lag2 + 0.30 * shock + 0.10 * shock_lag1 + country_fe[country] + eps
            year = int(years[t])
            quarter = int(quarters[t])
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "period": f"{year}-Q{quarter}",
                    "time_index": int(year * 4 + quarter - 1),
                    "xi": xis[country],
                    "oil_pct_change": float(oil[t]),
                    "xi_x_oil": float(shock),
                    "cpi_pct_change": float(cpi),
                }
            )
            y_lag2 = y_lag1
            y_lag1 = cpi
            shock_lag1 = shock
    return pd.DataFrame(rows)


def _add_structured_missing_cpi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = out["country"].isin(["C00", "C01", "C02", "C03"]) & out["period"].str.endswith(("Q3", "Q4"))
    out.loc[mask, "cpi_pct_change"] = np.nan
    return out


def _add_cumulative_targets(df: pd.DataFrame, *, horizons: int) -> pd.DataFrame:
    out = df.copy().sort_values(["country", "time_index"]).reset_index(drop=True)
    for horizon in range(horizons + 1):
        target_col = f"cum_cpi_lead{horizon}"
        out[target_col] = 0.0
        for step in range(horizon + 1):
            shifted = out.groupby("country")["cpi_pct_change"].shift(-step)
            out[target_col] = shifted if step == 0 else out[target_col] + shifted
    return out


def _fit_cumulative_lp(
    df: pd.DataFrame,
    *,
    horizons: int,
    y_lags: int,
) -> pd.DataFrame:
    target_cols = [f"cum_cpi_lead{h}" for h in range(horizons + 1)]
    work = df[["country", "time_index", "cpi_pct_change", "xi_x_oil", *target_cols]].copy()
    work = work.sort_values(["country", "time_index"]).reset_index(drop=True)

    y_lag_cols: list[str] = []
    for lag in range(1, y_lags + 1):
        col = f"cpi_pct_change_lag{lag}"
        work[col] = work.groupby("country")["cpi_pct_change"].shift(lag)
        y_lag_cols.append(col)

    x_cols = ["xi_x_oil", *y_lag_cols]
    country_fe = pd.get_dummies(work["country"], prefix="country_fe", drop_first=True, dtype=float)
    fe_cols = country_fe.columns.tolist()

    rows: list[dict[str, float | int]] = []
    for horizon in range(horizons + 1):
        dep_col = f"cum_cpi_lead{horizon}"
        design = pd.concat([work[["country", dep_col]], work[x_cols], country_fe], axis=1).dropna().reset_index(drop=True)
        y = pd.to_numeric(design[dep_col], errors="coerce")
        x = sm.add_constant(design[x_cols + fe_cols], has_constant="add")
        model = sm.OLS(y, x, missing="raise").fit(cov_type="cluster", cov_kwds={"groups": design["country"]})
        rows.append(
            {
                "horizon": horizon,
                "coef": float(model.params["xi_x_oil"]),
                "nobs": float(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def _identity_diagnostic(df: pd.DataFrame, *, horizons: int = 20, y_lags: int = 2) -> pd.DataFrame:
    noncum_irf, _, _ = fit_panel_local_projections(
        df,
        y_col="cpi_pct_change",
        shock_col="xi_x_oil",
        horizons=horizons,
        y_lags=y_lags,
        shock_lags=0,
        controls=[],
        control_lags=0,
        include_country_fe=True,
        include_year_fe=False,
        cov_type="cluster_country",
    )
    cum_irf = _fit_cumulative_lp(_add_cumulative_targets(df, horizons=horizons), horizons=horizons, y_lags=y_lags)

    out = noncum_irf[["horizon", "coef", "nobs"]].rename(columns={"coef": "beta_noncum", "nobs": "noncum_nobs"})
    out = out.merge(cum_irf.rename(columns={"coef": "cum_beta", "nobs": "cum_nobs"}), on="horizon", how="inner")
    out["cum_beta_diff"] = out["cum_beta"].diff()
    out["identity_diff"] = out["cum_beta_diff"] - out["beta_noncum"]
    return out


def test_synthetic_identity_mismatch_exists_even_without_missing_data():
    diagnostic = _identity_diagnostic(_make_synthetic_panel(), horizons=20, y_lags=2)
    max_abs_identity_diff = float(diagnostic.loc[diagnostic["horizon"] >= 1, "identity_diff"].abs().max())

    assert max_abs_identity_diff > 0.05


def test_synthetic_missingness_is_not_the_main_driver_of_identity_mismatch():
    balanced = _identity_diagnostic(_make_synthetic_panel(), horizons=20, y_lags=2)
    missing = _identity_diagnostic(_add_structured_missing_cpi(_make_synthetic_panel()), horizons=20, y_lags=2)

    balanced_max_abs = float(balanced.loc[balanced["horizon"] >= 1, "identity_diff"].abs().max())
    missing_max_abs = float(missing.loc[missing["horizon"] >= 1, "identity_diff"].abs().max())

    assert abs(missing_max_abs - balanced_max_abs) < 0.02
