# io-networks

Dissertation pipeline for OECD ICIO input-output network analysis, country-level aggregation, and downstream econometric work.

## Overview

This repository does three connected jobs:

1. Builds yearly ICIO-derived matrix objects from OECD input-output tables.
2. Aggregates sector-level objects into country-year measures such as `xi`, `c`, and `c_diff`.
3. Supports OLS and panel local projection analysis using those derived measures plus CPI, oil, and other controls.
4. Supports LP-IV workflows that instrument `xi_x_oil` with news-based shock series such as Kaenzig oil supply surprises.

At a high level, the workflow is:

```text
raw ICIO tables
-> technical coefficients A
-> exogenous / non-exogenous block decomposition
-> country-year aggregates (xi, c, c_diff)
-> econometric panels and notebooks
```

## Installation

Requirements:
- Python 3.11+

Install from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

Optional development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Declared runtime dependencies include:
- `numpy`
- `pandas`
- `pyarrow`
- `scipy`
- `statsmodels`
- `pyyaml`
- `matplotlib`
- `adjustText`

The package CLI entrypoint is:

```powershell
io-net
```

implemented by `io_networks.cli:main`.

## Repository Layout

```text
.
|- config/
|  `- default.yaml
|- data/
|  |- processed/
|  |  |- c/
|  |  |- c_diff/
|  |  |- cpi/
|  |  |- exchange_rates/
|  |  |- policy_rates/
|  |  |- real_gdp/
|  |  |- oil/
|  |  `- xi/
|  `- reference/
|- notebooks/
|- outputs/
|- src/io_networks/
|  |- blocks.py
|  |- c.py
|  |- c_diff.py
|  |- cli.py
|  |- config.py
|  |- eda.py
|  |- local_projections.py
|  |- logging.py
|  |- matrices.py
|  |- paths.py
|  |- policy_rates.py
|  |- real_gdp.py
|  |- regression.py
|  |- viz.py
|  |- viz_xi.py
|  `- xi.py
|- tests/
|- local_projections.py
|- regression.py
`- pyproject.toml
```

Top-level shims:
- `local_projections.py` re-exports `src/io_networks/local_projections.py` for notebook-friendly imports.
- `regression.py` re-exports `src/io_networks/regression.py`.

## Configuration

Main config file:

```text
config/default.yaml
```

Important settings:
- `icio.extended`: switch between `regular` and `extended` ICIO variants
- `icio.year_range.start`, `icio.year_range.end`: years to process
- `sectors.exo_codes`: OECD sector codes treated as exogenous
- `lambda.method`: weighting method for the exogenous block
- `lambda.normalize`: whether to normalize exogenous weights
- `paths.raw`, `paths.interim`, `paths.processed`, `paths.matrices`, `paths.outputs`: root directories used by the pipeline

The default project name is `icio-exogenous-pass-through`.

## Main Pipeline

The build pipeline is exposed through:

```powershell
io-net --help
```

Core commands:

```powershell
io-net eda --config config/default.yaml
io-net build-a --config config/default.yaml
io-net build-blocks --config config/default.yaml
io-net build-xi --config config/default.yaml
io-net build-zeta --config config/default.yaml
io-net build-c --config config/default.yaml
io-net build-c-diff --config config/default.yaml
io-net build-real-gdp --config config/default.yaml
io-net build-output-gap --config config/default.yaml
io-net build-policy-rates --config config/default.yaml
```

Recommended order:

1. `io-net eda`
2. `io-net build-a`
3. `io-net build-blocks`
4. `io-net build-xi`
5. `io-net build-zeta`
6. `io-net build-c`
7. `io-net build-c-diff`
8. `io-net build-real-gdp`
9. `io-net build-output-gap`
10. `io-net build-policy-rates`

## Build Outputs

### `eda`

Validates the yearly raw ICIO files and writes diagnostics under `outputs/eda`.

Typical outputs:
- `summary_by_year.csv`
- `check_results.csv`
- `issues.csv`

### `build-a`

Builds yearly technical coefficient matrices `A = Z / x`.

Outputs under `data/matrices/A/<variant>`:
- `A_<year>.npz`
- `A_<year>_meta.parquet`
- `build_summary.csv`

Notes:
- columns with effectively zero gross output are zeroed out in `A`
- the summary includes matrix diagnostics and metadata file names

### `build-blocks`

Splits sectors into exogenous (`E`) and non-exogenous (`N`) groups, then computes the block objects used by the decomposition.

Outputs under `data/matrices/blocks/<variant>`:
- `blocks_<year>.npz`
- `blocks_<year>_meta.parquet`
- `blocks_summary.csv`
- `country_c_<year>.parquet`
- `country_c_gdp_<year>.parquet`

Stored arrays include:
- `A_NN`
- `A_EN`
- `tau`
- `tau_dir`
- `tau_amp`
- `kappa`
- `chi`
- `lambda_E`
- `v_over_out_N`
- `pct_change_v_N`
- `c_N`
- `gdp_growth_N`
- `c_gdp_N`
- index arrays such as `idx_N` and `idx_E`

### `build-xi`

Aggregates sector-level `tau` objects and exogenous-sector `lambda` weights to country-year
`xi` measures using HFCE weights normalized over all sectors within a country-year.

For each country-year, let `w_N` be normalized HFCE weights on non-exogenous sectors and `w_E`
be normalized HFCE weights on exogenous sectors, such that their entries sum to 1 jointly across
all sectors. Then the builder computes:

- `xi_dir = w_E' lambda`
- `xi_amp_1 = w_N' tau_dir`
- `xi_amp_2 = w_N' tau_amp`
- `xi_amp = xi_amp_1 + xi_amp_2 = w_N' tau`
- `xi = xi_dir + xi_amp`

Outputs under `data/processed/xi/<variant>`:
- `xi_by_country_year.parquet`
- `weights_diagnostics.parquet`
- `weights_by_country_sector.parquet`
- `xi_summary.csv`

Important output fields include:
- `country`
- `year`
- `xi`
- `xi_dir` (exogenous-sector contribution, `w_E' lambda`)
- `xi_amp_1` (direct non-exogenous contribution, `w_N' tau_dir`)
- `xi_amp_2` (amplification-tail contribution, `w_N' tau_amp`)
- `xi_amp` (non-exogenous-sector contribution, `w_N' tau`)
- `identity_residual`
- `status`
- build timestamp and git commit metadata

Possible `status` values include `ok`, `missing_hfce_column`, and `zero_hfce_mass`.

### `build-zeta`

Aggregates sector-level `chi` objects to country-year `zeta` using HFCE weights.

Outputs under `data/processed/zeta/<variant>`:
- `zeta_by_country_year.parquet`
- `weights_diagnostics.parquet`
- `weights_by_country_sector.parquet`
- `zeta_summary.csv`

Possible `status` values include `ok`, `missing_hfce_column`, `zero_hfce_mass`, and `missing_chi_vector`.

### `build-c`

Aggregates `c_N` to country-year `c` using HFCE weights.

Outputs under `data/processed/c/<variant>`:
- `c_by_country_year.parquet`
- `weights_diagnostics.parquet`
- `weights_by_country_sector.parquet`
- `c_summary.csv`

Possible `status` values include `ok`, `missing_hfce_column`, `zero_hfce_mass`, and `missing_c_vector`.

### `build-c-diff`

Aggregates `(c_N - c_gdp_N)` to country-year `c_diff` using HFCE weights.

Outputs under `data/processed/c_diff/<variant>`:
- `c_diff_by_country_year.parquet`
- `weights_diagnostics.parquet`
- `weights_by_country_sector.parquet`
- `c_diff_summary.csv`

Possible `status` values include `ok`, `missing_hfce_column`, `zero_hfce_mass`, and `missing_c_diff_vector`.

### `build-real-gdp`

Builds a unified quarterly real GDP table from raw IMF and country-source files.

Outputs under `data/processed/real_gdp/`:
- `quarterly_real_gdp.csv`

### `build-policy-rates`

Builds the quarterly policy-rate panel used as an optional control in the LP and LP-IV notebooks.

Outputs under `data/processed/policy_rates/`:
- `quarterly_policy_rates.csv`

### Kaenzig Oil Surprise Tables

The repository also supports processing Kaenzig oil supply surprise vintages into monthly and quarterly tables.

Processed outputs under `data/processed/oil/` include:
- `kaenzig_oil_supply_surprises_monthly.csv`
- `kaenzig_oil_supply_surprises_quarterly.csv`

The quarterly file can contain one or more candidate instrument columns. The LP-IV workflow now lets you choose which one to use explicitly.

## Local Projections and LP-IV

Notebook-first workflows live under `notebooks/`.

Core notebooks include:
- `panel_local_projections.ipynb`
- `panel_cumulative_local_projections.ipynb`
- `panel_local_projections_iv.ipynb`
- `panel_cumulative_local_projections_iv.ipynb`

The shared implementation lives in:
- `src/io_networks/local_projections.py`

The IV notebooks estimate 2SLS local projections with:
- endogenous regressor: `xi_x_oil`
- instrument: `xi_x_news`

`xi_x_news` is constructed inside the panel builder as:

```text
xi_x_news = xi * news
```

where `news` is loaded from the selected column of the quarterly Kaenzig CSV.

### Choosing the Instrument Column

At the top of each IV notebook, set:

```python
news_column = "surprise"
```

or:

```python
news_column = "news_shock"
```

Then the notebook passes `news_column` into `build_panel_lp_dataset(...)`, which normalizes the selected source column to the internal `news` field and builds `xi_x_news` from it.

This means you do not need to rename columns in the shared code when switching instruments; only the notebook setting changes.

Important output fields include:
- `country`
- `country_name`
- `period`
- `year`
- `quarter`
- `real_gdp`

### `build-output-gap`

Builds quarterly log real GDP and HP-filtered output gaps from `quarterly_real_gdp.csv`.

Outputs under `data/processed/real_gdp/`:
- `quarterly_output_gap.csv`

Important output fields include:
- `country`
- `country_name`
- `period`
- `year`
- `quarter`
- `real_gdp`
- `log_real_gdp`
- `hp_trend_log_real_gdp`
- `output_gap`
- `hp_filter_status`

### `build-policy-rates`

Builds quarterly policy-rate controls from the BIS monthly central-bank policy-rate export.

Outputs under `data/processed/policy_rates/`:
- `quarterly_policy_rates.csv`

The quarterly series is the simple mean of the three monthly observations in each quarter. For euro adopters, missing observations from the adoption quarter onward are filled with the Euro area aggregate rate.

## Regression Workflow

`src/io_networks/regression.py` provides utilities to:
- load `xi`, `c`, `c_diff`, CPI, and oil series
- merge them into econometric panels
- estimate OLS specifications with optional country fixed effects
- save merged data, coefficient tables, summary text, and model metrics

Default input locations:
- `xi`: `data/processed/xi/regular/xi_by_country_year.parquet`
- `c`: `data/processed/c/regular/c_by_country_year.parquet`
- `c_diff`: `data/processed/c_diff/regular/c_diff_by_country_year.parquet`
- CPI: `data/processed/cpi/annual_cpi.csv`
- oil: chosen by CPI frequency from `annual_oil.csv`, `quarterly_oil.csv`, or `monthly_oil.csv`

Supported CPI and oil frequencies:
- annual: `A`
- quarterly: `Q`
- monthly: `M`

Supported regressands:
- `cpi`
- `cpi_minus_xi_x_oil`
- `cpi_minus_c`
- `cpi_minus_xi_x_oil_c`
- `cpi_minus_xi_x_oil_c_diff`
- `cpi_on_xi_x_oil_and_c_diff`

CLI usage:

```powershell
python -m io_networks.regression --help
```

Useful flags:
- `--xi-path`
- `--c-path`
- `--c-diff-path`
- `--cpi-path`
- `--cpi-freq`
- `--oil-path`
- `--cpi-lags`
- `--include-country-fe` / `--no-country-fe`
- `--cov-type` (`hc3`, `cluster_country`, or `cluster_country_time`)
- `--regressand`
- `--exclude-argentina`
- `--out-dir`

Default regression outputs are written under `outputs/regression`:
- merged regression data
- coefficient table
- text summary
- model metrics CSV

## Panel Local Projections

`src/io_networks/local_projections.py` supports:
- building panel LP datasets from `xi`, CPI, oil, and optional controls
- annual and quarterly panels
- horizon-by-horizon local projections with country and year or time fixed effects
- `HC3`, clustered-by-country, or two-way clustered-by-country-and-time inference
- cumulative quarterly CPI targets
- horizon-0 comparisons against the OLS implementation
- impulse-response plotting

Key public helpers include:
- `build_panel_lp_dataset(...)`
- `panel_diagnostics(...)`
- `build_cumulative_targets(...)`
- `fit_panel_local_projections(...)`
- `fit_cumulative_panel_local_projections(...)`
- `compare_horizon0_to_run_ols(...)`
- `plot_irf(...)`

The LP dataset builder now supports generic controls with:
- custom merge keys
- custom source column names via `merge_key_columns`
- optional sample restriction using `required_for_sample`

That is what powers the quarterly controls in the notebooks, including:
- source file: `data/processed/exchange_rates/quarterly_exchange_rates.csv`
- source value column: `log_diff`
- merged control name inside LP panels: `exchange_rate`
- source file: `data/processed/real_gdp/quarterly_output_gap.csv`
- source value column: `output_gap`
- merged control name inside LP panels: `output_gap`
- source file: `data/processed/policy_rates/quarterly_policy_rates.csv`
- source value column: `policy_rate`
- merged control name inside LP panels: `policy_rate`

## Notebooks

The `notebooks/` directory mixes visualization, exploratory work, and econometric workflows.

Current notebooks include:
- `naive_ols.ipynb`
- `naive_ols_2.ipynb`
- `lagged_xi_oil_cpi_minus_c_diff_ols.ipynb`
- `panel_local_projections.ipynb`
- `panel_cumulative_local_projections.ipynb`
- `panel_cumulative_local_projections_oil2.ipynb`
- `panel_cumulative_local_projections_oil2_euro.ipynb`
- `panel_cumulative_local_projections_iv_oil2.ipynb`
- `tau_bubble_chart.ipynb`
- `tau_bubble_all_countries.ipynb`
- `chi_bubble_chart.ipynb`
- `chi_bubble_all_countries.ipynb`
- `xi_line_chart.ipynb`
- `xi_phase_map.ipynb`
- `zeta_line_chart.ipynb`
- `zeta_phase_map.ipynb`
- `oecd_stan.ipynb`

Current LP notebook conventions:
- both panel LP notebooks use quarterly CPI and quarterly oil
- optional controls are defined through `panel_controls`
- controls that enter the regression are listed separately in `regression_controls`
- the current quarterly controls are `exchange_rate`, `output_gap`, and `policy_rate`
- output-gap and policy-rate control files are built from raw source data via the CLI commands above
- the two `oil2` cumulative LP notebooks now replace the plain `oil2` regressor with `zeta_x_oil2`
- in those notebooks, annual `zeta` is merged by `country` and `year`, lagged one year within country, and then interacted with quarterly `oil_pct_change ** 2`

Visualization helpers live mainly in:
- `src/io_networks/viz.py`
- `src/io_networks/viz_xi.py`
- `src/io_networks/viz_zeta.py`

These support:
- single-country `tau` bubble charts
- single-country `chi` bubble charts
- all-country bubble and boxen comparisons
- xi line charts
- xi phase-map style plots
- zeta line charts
- zeta phase-map style plots using lagged `zeta` on the x-axis and current `zeta` on the y-axis

## Testing and Linting

The repository is configured for `pytest` and `ruff`.

Run:

```powershell
pytest
ruff check .
```

Current automated LP coverage includes:
- optional control merges
- control-based sample restriction
- quarterly panel support
- automatic use of `time_index` in quarterly LPs
- horizon coverage in fitted LP models
- horizon-0 equivalence between LP and OLS in the compatible setup
- synthetic checks around identity mismatch and common-sample behavior

## Troubleshooting

- If a downstream build step cannot find inputs, run the upstream steps first.
- If `build-xi`, `build-zeta`, `build-c`, or `build-c-diff` returns non-`ok` statuses, inspect the corresponding diagnostics parquet files and summary CSVs.
- If a control merge fails in LP code, check both the merge keys and `merge_key_columns` against the source file headers.
- If local projections fail after lead and lag construction, reduce horizons or lag counts, or inspect missingness after merges.
- If clustered LP inference fails, make sure the estimation sample still contains at least two countries and more rows than regressors after fixed effects and lags are added.
- If quarterly cumulative LP targets behave unexpectedly, verify whether annual CPI override is enabled for full-year windows.
