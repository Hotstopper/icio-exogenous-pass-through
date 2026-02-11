# io-networks

Dissertation pipeline for OECD ICIO input-output network analysis.

This repository builds:
- quality checks for yearly ICIO tables,
- yearly technical-coefficient matrices `A`,
- exogenous/non-exogenous block objects and `tau` decomposition,
- country-year `xi` metrics (including directional and amplification parts),
- notebook-ready visualizations for `tau` and `xi`.

## What This Repo Does

The workflow is organized as a reproducible pipeline:
1. `eda`: validates raw ICIO yearly files and writes diagnostics.
2. `build-a`: builds yearly technical coefficient matrices `A = Z / x`.
3. `build-blocks`: partitions sectors into exogenous (`E`) and non-exogenous (`N`) sets, then computes:
   - `A_NN`
   - `A_EN`
   - `tau_dir`
   - `tau_amp`
   - `tau = tau_dir + tau_amp`
4. `build-xi`: aggregates sector-level `tau` objects to country-year `xi` using HFCE weights:
   - `xi`
   - `xi_dir`
   - `xi_amp`
   - identity residual `xi - (xi_dir + xi_amp)`

Visualization helpers in `src/io_networks/viz.py` and `src/io_networks/viz_xi.py` support:
- sector bubble charts for `tau` variants,
- country line charts for `xi`, `xi_dir`, `xi_amp`, and `share = xi_amp / xi`,
- phase maps with time-colored trajectories, ratio modes, background countries, and flow arrows.

## Repository Structure

```text
.
├─ config/
│  └─ default.yaml
├─ data/
│  ├─ raw/
│  │  ├─ regular/
│  │  └─ extended/
│  ├─ matrices/
│  │  ├─ A/
│  │  └─ blocks/
│  ├─ processed/
│  │  └─ xi/
│  └─ reference/
│     ├─ country_codes.csv
│     └─ sector_codes.csv
├─ notebooks/
│  ├─ tau_bubble_chart.ipynb
│  ├─ tau_bubble_all_countries.ipynb
│  ├─ xi_line_chart.ipynb
│  └─ xi_phase_map.ipynb
├─ outputs/
│  └─ eda/
├─ src/io_networks/
│  ├─ cli.py
│  ├─ eda.py
│  ├─ matrices.py
│  ├─ blocks.py
│  ├─ xi.py
│  ├─ viz.py
│  └─ viz_xi.py
└─ pyproject.toml
```

## Requirements

- Python 3.11+
- Core dependencies are managed in `pyproject.toml`:
  - `numpy`, `pandas`, `pyarrow`, `scipy`, `statsmodels`, `pyyaml`, `matplotlib`, `adjustText`

## Installation

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

Optional dev tools:

```powershell
python -m pip install -e ".[dev]"
```

## Configuration

Main config: `config/default.yaml`

Important fields:
- `icio.extended`: switch between `data/raw/regular` and `data/raw/extended`
- `icio.year_range.start`, `icio.year_range.end`: analysis window (default `1995` to `2022`)
- `sectors.exo_codes`: exogenous OECD sector codes (default `["B06", "C19"]`)
- `lambda.method`: currently `uniform`
- `lambda.normalize`: whether to normalize `lambda` to sum to 1
- `paths`: raw/interim/processed/matrices/outputs roots

## CLI Commands

Entrypoint script:

```powershell
io-net --help
```

Commands:

```powershell
io-net eda --config config/default.yaml
io-net build-a --config config/default.yaml
io-net build-blocks --config config/default.yaml
io-net build-xi --config config/default.yaml
```

Typical run order:
1. `io-net eda`
2. `io-net build-a`
3. `io-net build-blocks`
4. `io-net build-xi`

## Pipeline Outputs

### 1) EDA (`outputs/eda`)

- `summary_by_year.csv`: row/column counts, duplicate/null/zero diagnostics, schema consistency flags.
- `check_results.csv`: pass/fail checks by year.
- `issues.csv`: failed checks and missing-year issues.

### 2) A Matrices (`data/matrices/A/<variant>`)

Per year:
- `A_<year>.npz` with array `A`
- `A_<year>_meta.parquet` with sector labels, `OUT`, and `zero_out`

Aggregate:
- `build_summary.csv`

### 3) Block Objects (`data/matrices/blocks/<variant>`)

Per year:
- `blocks_<year>.npz` containing:
  - `A_NN`, `A_EN`
  - `tau`, `tau_dir`, `tau_amp`
  - `idx_N`, `idx_E`
  - `lambda_E`
- `blocks_<year>_meta.parquet` with N/E labels and index mappings

Aggregate:
- `blocks_summary.csv` (solver diagnostics, spectral radius estimate, condition estimate, means, NaN counts)

### 4) Xi Outputs (`data/processed/xi/<variant>`)

- `xi_by_country_year.parquet`:
  - `year`, `country`, `xi`, `xi_dir`, `xi_amp`, `identity_residual`, status fields
- `weights_diagnostics.parquet`:
  - HFCE availability, normalization stats, clipped negatives
- `weights_by_country_sector.parquet`:
  - raw/normalized weights and contribution decomposition
- `xi_summary.csv`:
  - yearly aggregate diagnostics

## Notebooks

### `notebooks/tau_bubble_chart.ipynb`

Uses:
- `prepare_country_bubble_data(...)`
- `plot_country_bubble(...)`

Supports metrics:
- `tau`
- `tau_dir`
- `tau_amp`
- `tau_amp2` (with `tau_amp` tails)

### `notebooks/tau_bubble_all_countries.ipynb`

Uses:
- `prepare_all_countries_bubble_data(...)`
- `plot_all_countries_bubble(...)`
- `plot_all_countries_boxen(...)`

Supports metrics:
- `tau`
- `tau_dir`
- `tau_amp`
- `tau_amp2`

Design notes:
- boxen-style distribution view by sector (recommended),
- optional light point overlay for raw dispersion context,
- all-country bubble view remains available when needed.

### `notebooks/xi_line_chart.ipynb`

Uses:
- `load_xi_data(...)`
- `plot_xi_country_lines(...)`

Supports line metrics:
- `xi`
- `xi_dir`
- `xi_amp`
- `share = xi_amp / xi`

Common controls:
- highlighted vs background countries,
- `linear` / `log` / `symlog` y-scales,
- endpoint labels and reserved label space,
- serif typography.

### `notebooks/xi_phase_map.ipynb`

Uses:
- `select_phase_map_countries(...)`
- `plot_xi_phase_map(...)`

Phase-map design:
- x-axis: `xi`
- y-axis ratio mode:
  - amplification: `xi / xi_dir`
  - share: `xi_amp / xi`
- time-colored trajectory (configurable colormap, including truncated `Oranges`)
- optional grayscale background countries
- optional flow arrows
- configurable temporal downsampling (e.g., every 3 or 5 years)
- optional axis clipping (e.g., `x in [0.01, 0.1]`, share `y in [0.4, 1.0]`)

## Method Notes

- `build-a` sets a whole column of `A` to zero when `OUT <= 1e-12` for that sector-year.
- `build-blocks` currently supports uniform `lambda` over exogenous sectors.
- `build-xi` clips negative HFCE weights to zero before normalization.
- `xi` identity is tracked explicitly through `identity_residual`.

## Testing and Linting

`pyproject.toml` is configured for:
- `pytest` (`tests` path)
- `ruff` (line length 100, rules `E`, `F`, `I`)

Run:

```powershell
pytest
ruff check .
```

Note: there may be no test files yet under `tests/`.

## Troubleshooting

- `FileNotFoundError` on blocks/A/xi paths:
  - run upstream pipeline steps first (`build-a` -> `build-blocks` -> `build-xi`).
- `build-xi` missing HFCE columns:
  - output rows are marked with non-`ok` status; inspect `weights_diagnostics.parquet`.
- Log-scale plot errors:
  - ensure plotted metric has strictly positive values when using `yscale='log'`.

## Entry Point

The package exposes:

```powershell
io-net
```

implemented by `io_networks.cli:main`.
