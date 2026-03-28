from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from io_networks import regression


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix == ".csv":
        return pd.read_csv(table_path)
    raise ValueError(f"Unsupported table format for {table_path}. Use .csv or .parquet.")


def _coerce_panel_keys(
    df: pd.DataFrame,
    *,
    country_col: str | None = None,
    year_col: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    subset: list[str] = []
    if country_col is not None:
        out[country_col] = out[country_col].astype(str).str.strip()
        subset.append(country_col)
    if year_col is not None:
        out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
        subset.append(year_col)
    out = out.dropna(subset=subset)
    if year_col is not None:
        out[year_col] = out[year_col].astype(int)
    return out


def _validate_unique_keys(df: pd.DataFrame, *, keys: list[str], label: str) -> None:
    dup_count = int(df.duplicated(keys).sum())
    if dup_count > 0:
        raise ValueError(f"{label} has {dup_count} duplicate rows for keys {keys}.")


def load_generic_control(
    *,
    path: str | Path,
    name: str,
    value_column: str,
    merge_keys: list[str] | None = None,
    merge_key_columns: dict[str, str] | None = None,
    country_column: str = "country",
    year_column: str = "year",
    status_column: str | None = "status",
    ok_status: str | None = "ok",
) -> pd.DataFrame:
    if merge_keys is None:
        merge_keys = ["country", "year"]
    key_column_map = dict(merge_key_columns or {})

    raw = _read_table(path)
    required_keys = {key_column_map.get(key, country_column if key == "country" else year_column if key == "year" else key) for key in merge_keys}
    required = required_keys | {value_column}
    if status_column is not None:
        required.add(status_column)
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Control '{name}' is missing columns: {sorted(missing)}")

    out = raw.copy()
    if value_column != name and name in out.columns and name not in merge_keys:
        out = out.drop(columns=[name])
    rename_map: dict[str, str] = {}
    if "country" in merge_keys:
        rename_map[key_column_map.get("country", country_column)] = "country"
    if "year" in merge_keys:
        rename_map[key_column_map.get("year", year_column)] = "year"
    for key in merge_keys:
        if key in {"country", "year"}:
            continue
        source_col = key_column_map.get(key, key)
        if source_col != key:
            rename_map[source_col] = key
    rename_map[value_column] = name
    out = out.rename(columns=rename_map)

    if status_column is not None and ok_status is not None:
        out = out.loc[out[status_column] == ok_status].copy()

    selected_cols = merge_keys + [name]
    out = out[selected_cols].copy()
    if "country" in merge_keys or "year" in merge_keys:
        out = _coerce_panel_keys(
            out,
            country_col="country" if "country" in merge_keys else None,
            year_col="year" if "year" in merge_keys else None,
        )
    out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.dropna(subset=[name]).reset_index(drop=True)
    _validate_unique_keys(out, keys=merge_keys, label=f"Control '{name}'")
    return out


def build_panel_lp_dataset(
    *,
    xi_path: str | Path = regression.XI_DEFAULT,
    cpi_path: str | Path = regression.CPI_DEFAULT,
    oil_path: str | Path | None = None,
    cpi_freq: str = "A",
    controls: list[dict[str, Any]] | None = None,
    exclude_countries: list[str] | None = None,
    sample_start: int | None = None,
    sample_end: int | None = None,
) -> pd.DataFrame:
    if cpi_freq not in regression.OIL_DEFAULT_BY_FREQ:
        raise ValueError(f"Unsupported cpi_freq '{cpi_freq}'.")

    xi_df = regression.load_xi(Path(xi_path))
    collapse_to_yearly = cpi_freq == "A"
    cpi_df = regression.load_cpi_pct_change(Path(cpi_path), freq=cpi_freq, collapse_to_yearly=collapse_to_yearly)
    oil_source = Path(oil_path) if oil_path is not None else regression.OIL_DEFAULT_BY_FREQ[cpi_freq]
    oil_df = regression.load_oil_pct_change(oil_source, freq=cpi_freq, collapse_to_yearly=collapse_to_yearly)

    if collapse_to_yearly:
        panel = xi_df.merge(cpi_df, on=["country", "year"], how="inner", validate="many_to_one")
        panel = panel.merge(oil_df, on=["year"], how="inner", validate="many_to_one")
        panel = panel.sort_values(["country", "year"]).reset_index(drop=True)
        panel_keys = ["country", "year"]
    else:
        panel = cpi_df.merge(xi_df, on=["country", "year"], how="inner", validate="many_to_one")
        panel = panel.merge(
            oil_df[["period", "time_index", "oil_pct_change"]],
            on=["period", "time_index"],
            how="inner",
            validate="many_to_one",
        )
        panel = panel.sort_values(["country", "time_index"]).reset_index(drop=True)
        panel_keys = ["country", "period"]
    panel["xi_x_oil"] = panel["xi"] * panel["oil_pct_change"]

    for spec in controls or []:
        control_name = str(spec["name"])
        control_df = load_generic_control(
            path=spec["path"],
            name=control_name,
            value_column=spec.get("value_column", control_name),
            merge_keys=list(spec.get("merge_keys", ["country", "year"])),
            merge_key_columns=spec.get("merge_key_columns"),
            country_column=spec.get("country_column", "country"),
            year_column=spec.get("year_column", "year"),
            status_column=spec.get("status_column", "status"),
            ok_status=spec.get("ok_status", "ok"),
        )
        merge_keys = list(spec.get("merge_keys", ["country", "year"]))
        panel = panel.merge(control_df, on=merge_keys, how="left", validate="many_to_one")
        if spec.get("required_for_sample", False):
            panel = panel.dropna(subset=[control_name]).copy()

    if exclude_countries:
        panel = panel.loc[~panel["country"].isin(exclude_countries)].copy()

    if sample_start is not None:
        panel = panel.loc[panel["year"] >= int(sample_start)].copy()
    if sample_end is not None:
        panel = panel.loc[panel["year"] <= int(sample_end)].copy()

    keep_cols = ["country", "year"]
    if not collapse_to_yearly:
        keep_cols.extend(["period", "time_index"])
    keep_cols.extend(["xi", "oil_pct_change", "xi_x_oil", "cpi_pct_change"])
    for spec in controls or []:
        keep_cols.append(str(spec["name"]))

    sort_cols = ["country", "year"] if collapse_to_yearly else ["country", "time_index"]
    panel = panel[keep_cols].sort_values(sort_cols).reset_index(drop=True)
    _validate_unique_keys(panel, keys=panel_keys, label="Panel LP dataset")
    return panel


def panel_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    required = {"country", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel diagnostics require columns: {sorted(required)}")

    diag = {
        "rows": int(len(df)),
        "countries": int(df["country"].nunique()),
        "year_min": int(df["year"].min()) if not df.empty else None,
        "year_max": int(df["year"].max()) if not df.empty else None,
        "avg_years_per_country": float(df.groupby("country")["year"].nunique().mean()) if not df.empty else None,
    }
    if "period" in df.columns:
        diag["period_min"] = str(df["period"].min()) if not df.empty else None
        diag["period_max"] = str(df["period"].max()) if not df.empty else None
        diag["avg_periods_per_country"] = float(df.groupby("country")["period"].nunique().mean()) if not df.empty else None
    return pd.DataFrame([diag])


def build_cumulative_targets(
    df: pd.DataFrame,
    *,
    y_col: str = "cpi_pct_change",
    horizons: int = 8,
    annual_cpi_df: pd.DataFrame | None = None,
    use_annual_override: bool = False,
) -> pd.DataFrame:
    out = df.copy().sort_values(["country", "time_index"]).reset_index(drop=True)
    if "period" not in out.columns:
        raise ValueError("Quarterly cumulative LP requires a 'period' column.")
    if "time_index" not in out.columns:
        raise ValueError("Quarterly cumulative LP requires a 'time_index' column.")

    period_parts = out["period"].astype(str).str.extract(r"^(\d{4})-Q([1-4])$")
    out["period_year"] = pd.to_numeric(period_parts[0], errors="coerce")
    out["period_quarter"] = pd.to_numeric(period_parts[1], errors="coerce")
    if out[["period_year", "period_quarter"]].isna().any().any():
        raise ValueError("Could not parse year/quarter from the quarterly period labels.")
    out["period_year"] = out["period_year"].astype(int)
    out["period_quarter"] = out["period_quarter"].astype(int)

    annual_lookup = annual_cpi_df.copy() if annual_cpi_df is not None else None

    for horizon in range(horizons + 1):
        target_col = f"cum_cpi_lead{horizon}"
        out[target_col] = 0.0
        for step in range(horizon + 1):
            shifted = out.groupby("country")[y_col].shift(-step)
            out[target_col] = shifted if step == 0 else out[target_col] + shifted

        if use_annual_override and annual_lookup is not None and (horizon + 1) % 4 == 0:
            end_year_col = f"end_year_h{horizon}"
            end_quarter_col = f"end_quarter_h{horizon}"
            out[end_year_col] = out.groupby("country")["period_year"].shift(-horizon)
            out[end_quarter_col] = out.groupby("country")["period_quarter"].shift(-horizon)
            aligned = (out["period_quarter"] == 1) & (out[end_quarter_col] == 4)
            annual_rows: list[str] = []
            years_in_window = (horizon + 1) // 4
            for year_offset in range(years_in_window):
                year_col = f"annual_year_h{horizon}_{year_offset}"
                value_col = f"annual_cpi_h{horizon}_{year_offset}"
                out[year_col] = out["period_year"] + year_offset
                merged = out[["country", year_col]].merge(
                    annual_lookup,
                    left_on=["country", year_col],
                    right_on=["country", "year"],
                    how="left",
                )
                out[value_col] = merged["annual_cpi_pct_change"].values
                annual_rows.append(value_col)
            annual_sum = out[annual_rows].sum(axis=1, min_count=years_in_window)
            out[target_col] = out[target_col].where(~aligned, annual_sum)

    return out


def build_legacy_ols_reference_dataset(
    *,
    xi_path: str | Path = regression.XI_DEFAULT,
    c_path: str | Path = regression.C_DEFAULT,
    c_diff_path: str | Path = regression.C_DIFF_DEFAULT,
    cpi_path: str | Path = regression.CPI_DEFAULT,
    oil_path: str | Path | None = None,
    cpi_freq: str = "A",
    exclude_countries: list[str] | None = None,
    remove_cpi_outliers: bool = False,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    xi_df = regression.load_xi(Path(xi_path))
    c_df = regression.load_c(Path(c_path))
    c_diff_df = regression.load_c_diff(Path(c_diff_path))
    cpi_df = regression.load_cpi_pct_change(Path(cpi_path), freq=cpi_freq)
    oil_source = Path(oil_path) if oil_path is not None else regression.OIL_DEFAULT_BY_FREQ[cpi_freq]
    oil_df = regression.load_oil_pct_change(oil_source, freq=cpi_freq)

    out = regression.prepare_regression_df(xi_df, c_df, cpi_df, oil_df, c_diff_df=c_diff_df)

    if remove_cpi_outliers:
        q1 = out["cpi_pct_change"].quantile(0.25)
        q3 = out["cpi_pct_change"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - float(iqr_multiplier) * iqr
        upper = q3 + float(iqr_multiplier) * iqr
        out = out.loc[out["cpi_pct_change"].between(lower, upper)].copy()

    if exclude_countries:
        out = regression.exclude_countries(out, list(exclude_countries))

    return out.sort_values(["country", "year"]).reset_index(drop=True)


def fit_panel_local_projections(
    df: pd.DataFrame,
    *,
    y_col: str = "cpi_pct_change",
    shock_col: str = "xi_x_oil",
    horizons: int = 4,
    y_lags: int = 1,
    shock_lags: int = 0,
    controls: list[str] | None = None,
    control_lags: int | dict[str, int] = 0,
    include_country_fe: bool = True,
    include_year_fe: bool = True,
    cov_type: str = "cluster_country",
    time_col: str | None = None,
    common_sample: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, Any]]:
    if horizons < 0:
        raise ValueError(f"horizons must be >= 0, got {horizons}")
    if y_lags < 0:
        raise ValueError(f"y_lags must be >= 0, got {y_lags}")
    if shock_lags < 0:
        raise ValueError(f"shock_lags must be >= 0, got {shock_lags}")

    controls = controls or []
    resolved_time_col = time_col or ("time_index" if "time_index" in df.columns else "year")
    required = {"country", resolved_time_col, y_col, shock_col, *controls}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing LP columns: {sorted(missing)}")

    keep_cols = ["country", resolved_time_col, y_col, shock_col, *controls]
    if "year" in df.columns and "year" not in keep_cols:
        keep_cols.append("year")
    if "period" in df.columns and "period" not in keep_cols:
        keep_cols.append("period")
    work = df[keep_cols].copy()
    work = work.sort_values(["country", resolved_time_col]).reset_index(drop=True)

    y_lag_cols: list[str] = []
    for lag in range(1, y_lags + 1):
        col = f"{y_col}_lag{lag}"
        work[col] = work.groupby("country")[y_col].shift(lag)
        y_lag_cols.append(col)

    shock_lag_cols: list[str] = []
    for lag in range(1, shock_lags + 1):
        col = f"{shock_col}_lag{lag}"
        work[col] = work.groupby("country")[shock_col].shift(lag)
        shock_lag_cols.append(col)

    control_cols: list[str] = []
    for control in controls:
        control_cols.append(control)
        lag_count = control_lags[control] if isinstance(control_lags, dict) else int(control_lags)
        if lag_count < 0:
            raise ValueError(f"Control lag count must be >= 0 for '{control}', got {lag_count}")
        for lag in range(1, lag_count + 1):
            col = f"{control}_lag{lag}"
            work[col] = work.groupby("country")[control].shift(lag)
            control_cols.append(col)

    irf_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    models: dict[int, Any] = {}

    dep_cols = [f"{y_col}_lead{h}" for h in range(horizons + 1)]
    for horizon in range(horizons + 1):
        dep_col = dep_cols[horizon]
        work[dep_col] = work.groupby("country")[y_col].shift(-horizon)

    x_cols = [shock_col, *shock_lag_cols, *y_lag_cols, *control_cols]
    if common_sample:
        sample_cols = [*x_cols, *dep_cols]
        work = work.dropna(subset=sample_cols).reset_index(drop=True)

    for horizon in range(horizons + 1):
        dep_col = dep_cols[horizon]

        x_base = work[x_cols].copy()
        fe_cols: list[str] = []

        if include_country_fe:
            country_fe = pd.get_dummies(
                work["country"],
                prefix="country_fe",
                drop_first=True,
                dtype=float,
            )
            x_base = pd.concat([x_base, country_fe], axis=1)
            fe_cols.extend(country_fe.columns.tolist())

        if include_year_fe:
            fe_source = work[resolved_time_col]
            fe_prefix = "year_fe" if resolved_time_col == "year" else "time_fe"
            year_fe = pd.get_dummies(
                fe_source,
                prefix=fe_prefix,
                drop_first=True,
                dtype=float,
            )
            x_base = pd.concat([x_base, year_fe], axis=1)
            fe_cols.extend(year_fe.columns.tolist())

        design_cols = ["country", dep_col]
        if "year" in work.columns:
            design_cols.append("year")
        if "period" in work.columns:
            design_cols.append("period")
        design_cols.append(resolved_time_col)
        design = pd.concat([work[design_cols], x_base], axis=1).dropna().reset_index(drop=True)
        if design.empty:
            raise ValueError(
                "No rows left for local projections after lead/lag construction and missing-data filters. "
                f"horizon={horizon}, y_lags={y_lags}, shock_lags={shock_lags}."
            )

        y = design[dep_col]
        x = sm.add_constant(design[x_cols + fe_cols], has_constant="add")
        base_model = sm.OLS(y, x, missing="raise")
        if cov_type == "cluster_country":
            if design["country"].nunique() < 2:
                raise ValueError("cluster_country requires at least two countries in the estimation sample.")
            if len(design) <= x.shape[1]:
                raise ValueError(
                    "cluster_country requires more estimation rows than regressors after fixed effects "
                    f"and lag construction. Got nobs={len(design)} and k={x.shape[1]} at horizon={horizon}."
                )
            model = base_model.fit(cov_type="cluster", cov_kwds={"groups": design["country"]})
            std_err_col = "std_err_cluster_country"
        elif cov_type == "hc3":
            model = base_model.fit(cov_type="HC3")
            std_err_col = "std_err_hc3"
        else:
            raise ValueError(f"Unsupported cov_type '{cov_type}'. Use 'cluster_country' or 'hc3'.")

        models[horizon] = model
        ci = model.conf_int()
        coef_df = pd.DataFrame(
            {
                "horizon": horizon,
                "term": model.params.index,
                "coef": model.params.values,
                std_err_col: model.bse.values,
                "t": model.tvalues.values,
                "p_value": model.pvalues.values,
                "ci_low_95": ci.iloc[:, 0].values,
                "ci_high_95": ci.iloc[:, 1].values,
                "nobs": float(model.nobs),
            }
        )
        coef_rows.extend(coef_df.to_dict(orient="records"))

        shock_row = coef_df.loc[coef_df["term"] == shock_col]
        if shock_row.empty:
            raise ValueError(f"Shock term '{shock_col}' was not present in horizon {horizon} results.")

        irf_rows.append(
            {
                "horizon": horizon,
                "term": shock_col,
                "coef": float(shock_row["coef"].iloc[0]),
                std_err_col: float(shock_row[std_err_col].iloc[0]),
                "t": float(shock_row["t"].iloc[0]),
                "p_value": float(shock_row["p_value"].iloc[0]),
                "ci_low_95": float(shock_row["ci_low_95"].iloc[0]),
                "ci_high_95": float(shock_row["ci_high_95"].iloc[0]),
                "nobs": float(model.nobs),
                "n_countries": int(design["country"].nunique()),
            }
        )

    return pd.DataFrame(irf_rows), pd.DataFrame(coef_rows), models


def fit_cumulative_panel_local_projections(
    df: pd.DataFrame,
    *,
    y_col: str = "cpi_pct_change",
    shock_col: str = "xi_x_oil",
    horizons: int = 8,
    y_lags: int = 4,
    shock_lags: int = 0,
    controls: list[str] | None = None,
    control_lags: int | dict[str, int] = 0,
    include_country_fe: bool = True,
    include_time_fe: bool = True,
    cov_type: str = "cluster_country",
    common_sample: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, Any]]:
    controls = controls or []
    required = {"country", "time_index", y_col, shock_col, *controls}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing cumulative LP columns: {sorted(missing)}")

    target_cols = [f"cum_cpi_lead{h}" for h in range(horizons + 1)]
    missing_targets = sorted(set(target_cols) - set(df.columns))
    if missing_targets:
        raise ValueError(f"Missing cumulative target columns: {missing_targets}")

    work = df[["country", "time_index", "year", "period", y_col, shock_col, *controls, *target_cols]].copy()
    work = work.sort_values(["country", "time_index"]).reset_index(drop=True)

    y_lag_cols: list[str] = []
    for lag in range(1, y_lags + 1):
        col = f"{y_col}_lag{lag}"
        work[col] = work.groupby("country")[y_col].shift(lag)
        y_lag_cols.append(col)

    shock_lag_cols: list[str] = []
    for lag in range(1, shock_lags + 1):
        col = f"{shock_col}_lag{lag}"
        work[col] = work.groupby("country")[shock_col].shift(lag)
        shock_lag_cols.append(col)

    control_cols: list[str] = []
    for control in controls:
        control_cols.append(control)
        lag_count = control_lags[control] if isinstance(control_lags, dict) else int(control_lags)
        if lag_count < 0:
            raise ValueError(f"Control lag count must be >= 0 for '{control}', got {lag_count}")
        for lag in range(1, lag_count + 1):
            col = f"{control}_lag{lag}"
            work[col] = work.groupby("country")[control].shift(lag)
            control_cols.append(col)

    x_cols = [shock_col, *shock_lag_cols, *y_lag_cols, *control_cols]
    if common_sample:
        work = work.dropna(subset=[*x_cols, *target_cols]).reset_index(drop=True)

    irf_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    models: dict[int, Any] = {}

    for horizon in range(horizons + 1):
        dep_col = f"cum_cpi_lead{horizon}"
        x_base = work[x_cols].copy()
        fe_cols: list[str] = []

        if include_country_fe:
            country_fe = pd.get_dummies(work["country"], prefix="country_fe", drop_first=True, dtype=float)
            x_base = pd.concat([x_base, country_fe], axis=1)
            fe_cols.extend(country_fe.columns.tolist())

        if include_time_fe:
            time_fe = pd.get_dummies(work["time_index"], prefix="time_fe", drop_first=True, dtype=float)
            x_base = pd.concat([x_base, time_fe], axis=1)
            fe_cols.extend(time_fe.columns.tolist())

        design = pd.concat([work[["country", "year", "period", dep_col]], x_base], axis=1).dropna().reset_index(drop=True)
        if design.empty:
            raise ValueError(
                "No rows left for cumulative local projections after target, lag, and missing-data filters. "
                f"horizon={horizon}, y_lags={y_lags}, shock_lags={shock_lags}."
            )

        y = pd.to_numeric(design[dep_col], errors="coerce")
        x = sm.add_constant(design[x_cols + fe_cols], has_constant="add")
        base_model = sm.OLS(y, x, missing="raise")
        if cov_type == "cluster_country":
            if design["country"].nunique() < 2:
                raise ValueError("cluster_country requires at least two countries in the estimation sample.")
            if len(design) <= x.shape[1]:
                raise ValueError(
                    "cluster_country requires more estimation rows than regressors after fixed effects "
                    f"and lag construction. Got nobs={len(design)} and k={x.shape[1]} at horizon={horizon}."
                )
            model = base_model.fit(cov_type="cluster", cov_kwds={"groups": design["country"]})
            std_err_col = "std_err_cluster_country"
        elif cov_type == "hc3":
            model = base_model.fit(cov_type="HC3")
            std_err_col = "std_err_hc3"
        else:
            raise ValueError(f"Unsupported cov_type '{cov_type}'.")

        models[horizon] = model
        ci = model.conf_int()
        coef_df = pd.DataFrame(
            {
                "horizon": horizon,
                "term": model.params.index,
                "coef": model.params.values,
                std_err_col: model.bse.values,
                "t": model.tvalues.values,
                "p_value": model.pvalues.values,
                "ci_low_95": ci.iloc[:, 0].values,
                "ci_high_95": ci.iloc[:, 1].values,
                "nobs": float(model.nobs),
            }
        )
        coef_rows.extend(coef_df.to_dict(orient="records"))

        shock_row = coef_df.loc[coef_df["term"] == shock_col]
        irf_rows.append(
            {
                "horizon": horizon,
                "term": shock_col,
                "coef": float(shock_row["coef"].iloc[0]),
                std_err_col: float(shock_row[std_err_col].iloc[0]),
                "t": float(shock_row["t"].iloc[0]),
                "p_value": float(shock_row["p_value"].iloc[0]),
                "ci_low_95": float(shock_row["ci_low_95"].iloc[0]),
                "ci_high_95": float(shock_row["ci_high_95"].iloc[0]),
                "nobs": float(model.nobs),
                "n_countries": int(design["country"].nunique()),
            }
        )

    return pd.DataFrame(irf_rows), pd.DataFrame(coef_rows), models


def compare_horizon0_to_run_ols(
    df: pd.DataFrame,
    *,
    cpi_lags: int = 1,
    include_country_fe: bool = False,
    regressand: str = "cpi",
    include_year_fe: bool = False,
    cov_type: str = "hc3",
) -> pd.DataFrame:
    model_ols, coef_ols = regression.run_ols(
        df,
        cpi_lags=cpi_lags,
        include_country_fe=include_country_fe,
        regressand=regressand,
    )

    required_cols = ["country", "year", "cpi_pct_change", "xi_x_oil"]
    lp_df = df[required_cols].copy()
    controls: list[str] = []
    if regressand == "cpi_on_xi_x_oil_and_c_diff":
        lp_df["c_diff"] = pd.to_numeric(df["c_diff"], errors="coerce")
        controls = ["c_diff"]

    irf_df, _, models = fit_panel_local_projections(
        lp_df,
        y_col="cpi_pct_change",
        shock_col="xi_x_oil",
        horizons=0,
        y_lags=cpi_lags,
        shock_lags=0,
        controls=controls,
        include_country_fe=include_country_fe,
        include_year_fe=include_year_fe,
        cov_type=cov_type,
    )

    ols_coef = float(coef_ols.loc[coef_ols["term"] == "xi_x_oil", "coef"].iloc[0])
    lp_coef = float(irf_df.loc[irf_df["horizon"] == 0, "coef"].iloc[0])

    return pd.DataFrame(
        [
            {
                "metric": "xi_x_oil_coef",
                "run_ols": ols_coef,
                "lp_h0": lp_coef,
                "difference": lp_coef - ols_coef,
                "run_ols_nobs": float(model_ols.nobs),
                "lp_h0_nobs": float(models[0].nobs),
            }
        ]
    )


def plot_irf(
    irf_df: pd.DataFrame,
    *,
    title: str | None = None,
    ax: Any = None,
) -> Any:
    required = {"horizon", "coef", "ci_low_95", "ci_high_95"}
    missing = required - set(irf_df.columns)
    if missing:
        raise ValueError(f"IRF plot requires columns: {sorted(required)}")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    plot_df = irf_df.sort_values("horizon").reset_index(drop=True)
    ax.plot(plot_df["horizon"], plot_df["coef"], marker="o", linewidth=2)
    ax.fill_between(
        plot_df["horizon"],
        plot_df["ci_low_95"],
        plot_df["ci_high_95"],
        alpha=0.2,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Response of CPI")
    ax.set_title(title or "Panel local projections: CPI response to xi_x_oil")
    return ax
