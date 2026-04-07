from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


XI_DEFAULT = Path('data/processed/xi/regular/xi_by_country_year.parquet')
C_DEFAULT = Path('data/processed/c/regular/c_by_country_year.parquet')
C_DIFF_DEFAULT = Path('data/processed/c_diff/regular/c_diff_by_country_year.parquet')
CPI_DEFAULT = Path('data/processed/cpi/annual_cpi.csv')
OUT_DEFAULT = Path('outputs/regression')
OIL_DEFAULT_BY_FREQ: dict[str, Path] = {
    'A': Path('data/processed/oil/annual_oil.csv'),
    'M': Path('data/processed/oil/monthly_oil.csv'),
    'Q': Path('data/processed/oil/quarterly_oil.csv'),
}

SUPPORTED_COV_TYPES = {'hc3', 'cluster_country', 'cluster_country_time'}


def _validate_cluster_design(
    design: pd.DataFrame,
    x_cols: list[str],
    *,
    cov_type: str,
    time_col: str | None = None,
) -> None:
    if cov_type == 'cluster_country':
        if design['country'].nunique() < 2:
            raise ValueError('cluster_country requires at least two countries in the estimation sample.')
    elif cov_type == 'cluster_country_time':
        if time_col is None:
            raise ValueError('cluster_country_time requires a time_col.')
        if design['country'].nunique() < 2:
            raise ValueError('cluster_country_time requires at least two countries in the estimation sample.')
        if design[time_col].nunique() < 2:
            raise ValueError(
                f"cluster_country_time requires at least two unique values in '{time_col}' "
                'in the estimation sample.'
            )
    else:
        raise ValueError(
            f"Unsupported clustered cov_type '{cov_type}'. "
            "Use 'cluster_country' or 'cluster_country_time'."
        )

    if len(design) <= len(x_cols):
        raise ValueError(
            f'{cov_type} requires more estimation rows than regressors after fixed effects '
            f'and lag construction. Got nobs={len(design)} and k={len(x_cols)}.'
        )


def _fit_linear_model_with_covariance(
    base_model,
    design: pd.DataFrame,
    x_cols: list[str],
    *,
    cov_type: str,
    time_col: str | None = None,
):
    if cov_type == 'hc3':
        return base_model.fit(cov_type='HC3'), 'std_err_hc3'
    if cov_type == 'cluster_country':
        _validate_cluster_design(design, x_cols, cov_type=cov_type)
        return (
            base_model.fit(cov_type='cluster', cov_kwds={'groups': design['country']}),
            'std_err_cluster_country',
        )
    if cov_type == 'cluster_country_time':
        _validate_cluster_design(design, x_cols, cov_type=cov_type, time_col=time_col)
        country_codes = pd.factorize(design['country'])[0].astype(np.int64, copy=False)
        time_codes = pd.factorize(design[time_col])[0].astype(np.int64, copy=False)
        groups = np.column_stack([country_codes, time_codes])
        return (
            base_model.fit(cov_type='cluster', cov_kwds={'groups': groups}),
            'std_err_cluster_country_time',
        )
    raise ValueError(
        f"Unsupported cov_type '{cov_type}'. "
        "Use 'hc3', 'cluster_country', or 'cluster_country_time'."
    )


def load_xi(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {'year', 'country', 'xi', 'status'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing xi columns: {sorted(missing)}')

    df = df.loc[df['status'] == 'ok', ['year', 'country', 'xi']].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['xi'] = pd.to_numeric(df['xi'], errors='coerce')
    df = df.dropna(subset=['year', 'country', 'xi'])
    df['year'] = df['year'].astype(int)
    return df


def load_c(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {'year', 'country', 'c', 'status'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing c columns: {sorted(missing)}')

    df = df.loc[df['status'] == 'ok', ['year', 'country', 'c']].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['c'] = pd.to_numeric(df['c'], errors='coerce')
    df = df.dropna(subset=['year', 'country', 'c'])
    df['year'] = df['year'].astype(int)
    return df


def load_c_diff(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {'year', 'country', 'c_diff', 'status'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing c_diff columns: {sorted(missing)}')

    df = df.loc[df['status'] == 'ok', ['year', 'country', 'c_diff']].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['c_diff'] = pd.to_numeric(df['c_diff'], errors='coerce')
    df = df.dropna(subset=['year', 'country', 'c_diff'])
    df['year'] = df['year'].astype(int)
    return df


def _extract_year(series: pd.Series, freq: str) -> pd.Series:
    if freq in {'M', 'A'}:
        num_year = pd.to_numeric(series, errors='coerce')
        dt_year = pd.to_datetime(series, errors='coerce').dt.year
        return num_year.fillna(dt_year)
    if freq == 'Q':
        series_str = series.astype(str)
        str_year = pd.to_numeric(series_str.str.extract(r'^(\d{4})-Q[1-4]$')[0], errors='coerce')
        dt_year = pd.to_datetime(series.where(str_year.isna()), errors='coerce').dt.year
        return str_year.fillna(dt_year)
    raise ValueError(f"Unsupported freq '{freq}'.")


def _extract_subperiod(series: pd.Series, freq: str) -> pd.Series:
    if freq == 'A':
        return pd.Series(1, index=series.index, dtype='Int64')
    if freq == 'Q':
        series_str = series.astype(str)
        str_quarter = pd.to_numeric(series_str.str.extract(r'^(\d{4})-Q([1-4])$')[1], errors='coerce')
        dt_quarter = pd.to_datetime(series.where(str_quarter.isna()), errors='coerce').dt.quarter
        return str_quarter.fillna(dt_quarter).astype('Int64')
    if freq == 'M':
        dt = pd.to_datetime(series, errors='coerce')
        return dt.dt.month.astype('Int64')
    raise ValueError(f"Unsupported freq '{freq}'.")


def _build_period_columns(series: pd.Series, freq: str) -> pd.DataFrame:
    year = _extract_year(series, freq=freq)
    subperiod = _extract_subperiod(series, freq=freq)
    out = pd.DataFrame({'year': year, 'subperiod': subperiod})
    out = out.dropna(subset=['year', 'subperiod']).copy()
    out['year'] = out['year'].astype(int)
    out['subperiod'] = out['subperiod'].astype(int)
    if freq == 'A':
        out['period'] = out['year'].astype(str)
        out['time_index'] = out['year']
    elif freq == 'Q':
        out['period'] = out['year'].astype(str) + '-Q' + out['subperiod'].astype(str)
        out['time_index'] = out['year'] * 4 + out['subperiod'] - 1
    elif freq == 'M':
        out['period'] = out['year'].astype(str) + '-M' + out['subperiod'].astype(str).str.zfill(2)
        out['time_index'] = out['year'] * 12 + out['subperiod'] - 1
    else:
        raise ValueError(f"Unsupported freq '{freq}'.")
    return out[['year', 'subperiod', 'period', 'time_index']]


def load_cpi_pct_change(path: Path, freq: str = 'A', collapse_to_yearly: bool = True) -> pd.DataFrame:
    if freq not in {'A', 'M', 'Q'}:
        raise ValueError(f"Unsupported cpi freq '{freq}'. Use 'A', 'M', or 'Q'.")

    usecols = [
        'FREQ',
        'MEASURE',
        'METHODOLOGY',
        'EXPENDITURE',
        'ADJUSTMENT',
        'UNIT_MEASURE',
        'REF_AREA',
        'TIME_PERIOD',
        'OBS_VALUE',
    ]
    df = pd.read_csv(path, usecols=usecols)

    mask = (
        (df['FREQ'] == freq)
        & (df['MEASURE'] == 'CPI')
        & (df['METHODOLOGY'] == 'N')
        & (df['EXPENDITURE'] == '_T')
        & (df['ADJUSTMENT'] == 'N')
        & (df['UNIT_MEASURE'] == 'PC')
    )

    out = df.loc[mask, ['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']].rename(
        columns={
            'REF_AREA': 'country',
            'TIME_PERIOD': 'time_period',
            'OBS_VALUE': 'cpi_pct_change',
        }
    )

    out['cpi_pct_change'] = pd.to_numeric(out['cpi_pct_change'], errors='coerce')
    # Convert percent units to proportion units for regression use.
    out['cpi_pct_change'] = out['cpi_pct_change'] / 100.0
    out = out.dropna(subset=['country', 'time_period', 'cpi_pct_change']).reset_index(drop=True)

    period_cols = _build_period_columns(out['time_period'], freq=freq)
    out = pd.concat([out[['country', 'cpi_pct_change']].reset_index(drop=True), period_cols.reset_index(drop=True)], axis=1)

    if collapse_to_yearly and freq in {'M', 'Q'}:
        out = (
            out.groupby(['country', 'year'], as_index=False)
            .agg(cpi_pct_change=('cpi_pct_change', 'mean'))
        )
        dup_count = int(out.duplicated(['country', 'year']).sum())
        if dup_count > 0:
            raise ValueError(f'CPI filtered data has {dup_count} duplicate country-year rows.')
        return out

    key_cols = ['country', 'year'] if collapse_to_yearly else ['country', 'period']
    dup_count = int(out.duplicated(key_cols).sum())
    if dup_count > 0:
        raise ValueError(f'CPI filtered data has {dup_count} duplicate rows for keys {key_cols}.')

    return out


def load_oil_pct_change(path: Path, freq: str = 'A', collapse_to_yearly: bool = True) -> pd.DataFrame:
    if freq not in {'A', 'M', 'Q'}:
        raise ValueError(f"Unsupported oil freq '{freq}'. Use 'A', 'M', or 'Q'.")

    usecols = ['date', 'pct_change']
    df = pd.read_csv(path, usecols=usecols)

    out = df.rename(columns={'pct_change': 'oil_pct_change'}).copy()
    out['oil_pct_change'] = pd.to_numeric(out['oil_pct_change'], errors='coerce')
    out = out.dropna(subset=['date', 'oil_pct_change']).reset_index(drop=True)

    period_cols = _build_period_columns(out['date'], freq=freq)
    out = pd.concat([period_cols.reset_index(drop=True), out[['oil_pct_change']].reset_index(drop=True)], axis=1)

    if collapse_to_yearly and freq in {'M', 'Q'}:
        out = out.groupby('year', as_index=False).agg(oil_pct_change=('oil_pct_change', 'mean'))
        dup_count = int(out.duplicated(['year']).sum())
        if dup_count > 0:
            raise ValueError(f'Oil data has {dup_count} duplicate year rows.')
        return out

    key_cols = ['year'] if collapse_to_yearly else ['period']
    dup_count = int(out.duplicated(key_cols).sum())
    if dup_count > 0:
        raise ValueError(f'Oil data has {dup_count} duplicate rows for keys {key_cols}.')

    return out


def prepare_regression_df(
    xi_df: pd.DataFrame,
    c_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    c_diff_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = xi_df.merge(cpi_df, on=['country', 'year'], how='inner', validate='many_to_one')
    merged = merged.merge(c_df, on=['country', 'year'], how='inner', validate='many_to_one')
    if c_diff_df is not None:
        merged = merged.merge(c_diff_df, on=['country', 'year'], how='inner', validate='many_to_one')
    merged = merged.merge(oil_df, on=['year'], how='inner', validate='many_to_one')
    merged = merged.sort_values(['country', 'year']).reset_index(drop=True)
    merged['xi_x_oil'] = merged['xi'] * merged['oil_pct_change']
    cols = ['country', 'year', 'xi', 'c']
    if 'c_diff' in merged.columns:
        cols.append('c_diff')
    cols += ['oil_pct_change', 'xi_x_oil', 'cpi_pct_change']
    merged = merged[cols]
    merged = merged.reset_index(drop=True)
    return merged


def exclude_countries(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    if not countries:
        return df.copy()
    out = df.loc[~df['country'].isin(countries)].copy()
    return out.reset_index(drop=True)


def run_ols(
    df: pd.DataFrame,
    cpi_lags: int = 1,
    include_country_fe: bool = True,
    regressand: str = 'cpi',
    cov_type: str = 'hc3',
):
    if cpi_lags < 0:
        raise ValueError(f'cpi_lags must be >= 0, got {cpi_lags}')
    if regressand not in {
        'cpi',
        'cpi_minus_xi_x_oil',
        'cpi_minus_c',
        'cpi_minus_xi_x_oil_c',
        'cpi_minus_xi_x_oil_c_diff',
        'cpi_on_xi_x_oil_and_c_diff',
    }:
        raise ValueError(
            "Unsupported regressand "
            f"'{regressand}'. Use 'cpi', 'cpi_minus_xi_x_oil', 'cpi_minus_c', "
            "'cpi_minus_xi_x_oil_c', 'cpi_minus_xi_x_oil_c_diff', "
            "or 'cpi_on_xi_x_oil_and_c_diff'."
        )

    required = {'country', 'year', 'xi_x_oil', 'cpi_pct_change'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing regression columns: {sorted(missing)}')

    work = df[['country', 'year', 'xi_x_oil', 'cpi_pct_change']].copy()
    if regressand in {'cpi_minus_c', 'cpi_minus_xi_x_oil_c'}:
        if 'c' not in df.columns:
            raise ValueError("Missing regression column: ['c']")
        work['c'] = pd.to_numeric(df['c'], errors='coerce')
    if regressand == 'cpi_minus_xi_x_oil_c_diff':
        if 'c_diff' not in df.columns:
            raise ValueError("Missing regression column: ['c_diff']")
        work['c_diff'] = pd.to_numeric(df['c_diff'], errors='coerce')
    if regressand == 'cpi_on_xi_x_oil_and_c_diff':
        if 'c_diff' not in df.columns:
            raise ValueError("Missing regression column: ['c_diff']")
        work['c_diff'] = pd.to_numeric(df['c_diff'], errors='coerce')

    if regressand == 'cpi_minus_xi_x_oil':
        work['regressand'] = work['cpi_pct_change'] - work['xi_x_oil']
    elif regressand == 'cpi_minus_c':
        work['regressand'] = work['cpi_pct_change'] - work['c']
    elif regressand == 'cpi_minus_xi_x_oil_c':
        work['regressand'] = work['cpi_pct_change'] - work['xi_x_oil'] - work['c']
    elif regressand == 'cpi_minus_xi_x_oil_c_diff':
        work['regressand'] = work['cpi_pct_change'] - work['xi_x_oil'] - work['c_diff']
    else:
        work['regressand'] = work['cpi_pct_change']
    work = work.sort_values(['country', 'year']).reset_index(drop=True)

    lag_cols: list[str] = []
    for lag in range(1, cpi_lags + 1):
        col = f'cpi_pct_change_lag{lag}'
        work[col] = work.groupby('country')['cpi_pct_change'].shift(lag)
        lag_cols.append(col)

    x_cols = ['xi_x_oil']
    if regressand == 'cpi_on_xi_x_oil_and_c_diff':
        x_cols.append('c_diff')
    x_cols += lag_cols
    x_base = work[x_cols].copy()

    fe_cols: list[str] = []
    if include_country_fe:
        # Country fixed effects: one omitted baseline country to avoid multicollinearity.
        fe = pd.get_dummies(work['country'], prefix='country_fe', drop_first=True, dtype=float)
        fe_cols = fe.columns.tolist()
        x_base = pd.concat([x_base, fe], axis=1)

    design = pd.concat([work[['country', 'year', 'regressand']], x_base], axis=1).dropna()
    if design.empty:
        raise ValueError(
            'No rows left for OLS after applying lag and missing-data filters. '
            f'cpi_lags={cpi_lags}, include_country_fe={include_country_fe}.'
        )

    y = design['regressand']
    x = sm.add_constant(design[x_cols + fe_cols], has_constant='add')
    model, std_err_col = _fit_linear_model_with_covariance(
        sm.OLS(y, x, missing='raise'),
        design,
        list(x.columns),
        cov_type=cov_type,
        time_col='year',
    )

    ci = model.conf_int()
    coef = pd.DataFrame(
        {
            'term': model.params.index,
            'coef': model.params.values,
            std_err_col: model.bse.values,
            't': model.tvalues.values,
            'p_value': model.pvalues.values,
            'ci_low_95': ci.iloc[:, 0].values,
            'ci_high_95': ci.iloc[:, 1].values,
        }
    )
    return model, coef


def save_outputs(
    df: pd.DataFrame,
    coef_df: pd.DataFrame,
    model,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / 'cpi_xi_oil_regression_data.csv'
    coef_path = out_dir / 'cpi_xi_oil_ols_coefficients.csv'
    summary_path = out_dir / 'cpi_xi_oil_ols_summary.txt'
    metrics_path = out_dir / 'cpi_xi_oil_ols_metrics.csv'

    df.to_csv(merged_path, index=False)
    coef_df.to_csv(coef_path, index=False)
    summary_path.write_text(model.summary().as_text(), encoding='utf-8')

    metrics = pd.DataFrame(
        [
            {
                'nobs': float(model.nobs),
                'r_squared': float(model.rsquared),
                'adj_r_squared': float(model.rsquared_adj),
                'f_stat': float(model.fvalue) if model.fvalue is not None else None,
                'f_p_value': float(model.f_pvalue) if model.f_pvalue is not None else None,
            }
        ]
    )
    metrics.to_csv(metrics_path, index=False)

    return {
        'merged_data': merged_path,
        'coefficients': coef_path,
        'summary': summary_path,
        'metrics': metrics_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='OLS regression of annual CPI percentage change on xi * oil change with configurable CPI/oil frequency, CPI lags, and optional country FE'
    )
    parser.add_argument('--xi-path', type=Path, default=XI_DEFAULT)
    parser.add_argument('--c-path', type=Path, default=C_DEFAULT)
    parser.add_argument('--c-diff-path', type=Path, default=C_DIFF_DEFAULT)
    parser.add_argument('--cpi-path', type=Path, default=CPI_DEFAULT)
    parser.add_argument('--cpi-freq', choices=['A', 'M', 'Q'], default='A')
    parser.add_argument('--oil-path', type=Path, default=None)
    parser.add_argument('--cpi-lags', type=int, default=1)
    parser.add_argument('--include-country-fe', dest='include_country_fe', action='store_true')
    parser.add_argument('--no-country-fe', dest='include_country_fe', action='store_false')
    parser.set_defaults(include_country_fe=True)
    parser.add_argument('--cov-type', choices=sorted(SUPPORTED_COV_TYPES), default='hc3')
    parser.add_argument(
        '--regressand',
        choices=[
            'cpi',
            'cpi_minus_xi_x_oil',
            'cpi_minus_c',
            'cpi_minus_xi_x_oil_c',
            'cpi_minus_xi_x_oil_c_diff',
            'cpi_on_xi_x_oil_and_c_diff',
        ],
        default='cpi',
    )
    parser.add_argument('--exclude-argentina', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=OUT_DEFAULT)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.regressand == 'cpi_minus_c' and args.cpi_freq != 'A':
        raise ValueError("regressand='cpi_minus_c' is only supported for annual frequency (cpi_freq='A').")
    if args.regressand == 'cpi_minus_xi_x_oil_c_diff' and args.cpi_freq != 'A':
        raise ValueError(
            "regressand='cpi_minus_xi_x_oil_c_diff' is only supported for annual frequency (cpi_freq='A')."
        )

    xi_df = load_xi(args.xi_path)
    c_df = load_c(args.c_path)
    c_diff_needed = args.regressand in {'cpi_minus_xi_x_oil_c_diff', 'cpi_on_xi_x_oil_and_c_diff'}
    c_diff_df = load_c_diff(args.c_diff_path) if c_diff_needed else None
    cpi_df = load_cpi_pct_change(args.cpi_path, freq=args.cpi_freq)
    oil_path = args.oil_path if args.oil_path is not None else OIL_DEFAULT_BY_FREQ[args.cpi_freq]
    oil_df = load_oil_pct_change(oil_path, freq=args.cpi_freq)
    reg_df = prepare_regression_df(xi_df, c_df, cpi_df, oil_df, c_diff_df=c_diff_df)
    if args.exclude_argentina:
        reg_df = exclude_countries(reg_df, ['ARG'])

    if reg_df.empty:
        raise ValueError('No rows left after joining xi, CPI, and oil by country/year.')

    model, coef_df = run_ols(
        reg_df,
        cpi_lags=args.cpi_lags,
        include_country_fe=args.include_country_fe,
        regressand=args.regressand,
        cov_type=args.cov_type,
    )
    paths = save_outputs(reg_df, coef_df, model, args.out_dir)

    print(f'Rows used: {len(reg_df)}')
    print(f'Countries used: {reg_df["country"].nunique()}')
    print(f'Year span: {reg_df["year"].min()}-{reg_df["year"].max()}')
    print(f'cpi_freq: {args.cpi_freq}')
    print(f'c_path: {args.c_path}')
    if c_diff_needed:
        print(f'c_diff_path: {args.c_diff_path}')
    print(f'oil_path: {oil_path}')
    print(f'cpi_lags: {args.cpi_lags}')
    print(f'include_country_fe: {args.include_country_fe}')
    print(f'cov_type: {args.cov_type}')
    print(f'regressand: {args.regressand}')
    print(f'exclude_argentina: {args.exclude_argentina}')
    for label, path in paths.items():
        print(f'{label}: {path}')


if __name__ == '__main__':
    main()
