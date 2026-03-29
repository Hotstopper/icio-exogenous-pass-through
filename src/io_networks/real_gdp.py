from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from io_networks.config import load_config
from io_networks.paths import resolve_paths


IMF_WIDE_ID_COLUMNS = [
    "SERIES_CODE",
    "COUNTRY",
    "INDICATOR",
    "PRICE_TYPE",
    "S_ADJUSTMENT",
    "TYPE_OF_TRANSFORMATION",
    "FREQUENCY",
]

ANGOLA_QUARTER_MAP = {
    "I TRIM": 1,
    "II TRIM": 2,
    "III TRIM": 3,
    "IV TRIM": 4,
}

UAE_QUARTER_MAP = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
}


def _format_period(year: pd.Series, quarter: pd.Series) -> pd.Series:
    return year.astype(int).astype(str) + "-Q" + quarter.astype(int).astype(str)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["country"] = out["country"].astype(str).str.strip()
    out["country_name"] = out["country_name"].astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["quarter"] = pd.to_numeric(out["quarter"], errors="coerce")
    out["real_gdp"] = pd.to_numeric(out["real_gdp"], errors="coerce")

    out = out.dropna(subset=["country", "country_name", "year", "quarter", "real_gdp"]).copy()
    out["year"] = out["year"].astype(int)
    out["quarter"] = out["quarter"].astype(int)
    out = out.loc[out["quarter"].between(1, 4)].copy()
    out["period"] = _format_period(out["year"], out["quarter"])

    key_cols = ["country", "period"]
    dup_count = int(out.duplicated(key_cols).sum())
    if dup_count > 0:
        raise ValueError(f"Real GDP output has {dup_count} duplicate rows for keys {key_cols}.")

    out = out.sort_values(["country", "year", "quarter"]).reset_index(drop=True)
    return out[["country", "country_name", "period", "year", "quarter", "real_gdp"]]


def _load_imf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    quarter_cols = [col for col in df.columns if isinstance(col, str) and col.count("-Q") == 1]

    mask = (
        (df["INDICATOR"] == "Gross domestic product (GDP)")
        & (df["PRICE_TYPE"] == "Constant prices")
        & (df["S_ADJUSTMENT"] == "Seasonally adjusted (SA)")
        & (df["TYPE_OF_TRANSFORMATION"] == "Domestic currency")
        & (df["FREQUENCY"] == "Quarterly")
    )

    long = (
        df.loc[mask, [*IMF_WIDE_ID_COLUMNS, *quarter_cols]]
        .melt(
            id_vars=IMF_WIDE_ID_COLUMNS,
            value_vars=quarter_cols,
            var_name="period",
            value_name="real_gdp",
        )
        .dropna(subset=["real_gdp"])
        .copy()
    )

    period_parts = long["period"].astype(str).str.extract(r"^(?P<year>\d{4})-Q(?P<quarter>[1-4])$")
    long["year"] = pd.to_numeric(period_parts["year"], errors="coerce")
    long["quarter"] = pd.to_numeric(period_parts["quarter"], errors="coerce")
    long["country"] = long["SERIES_CODE"].astype(str).str.split(".").str[0].str.strip()
    long["country_name"] = long["COUNTRY"]

    return long[["country", "country_name", "year", "quarter", "real_gdp"]]


def _load_uae(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = (
        (df["MEASURE"] == "TOT_GDP")
        & (df["QGDP_SYS"] == "CON")
        & (df["QGDP_UNIT"] == "VAL")
        & (df["FREQ"] == "Q")
    )

    out = df.loc[mask, ["Reference area", "TIME_PERIOD", "QUARTER", "OBS_VALUE"]].copy()
    out["country"] = "ARE"
    out["country_name"] = out["Reference area"]
    out["year"] = pd.to_numeric(out["TIME_PERIOD"], errors="coerce")
    out["quarter"] = out["QUARTER"].map(UAE_QUARTER_MAP)
    out["real_gdp"] = out["OBS_VALUE"]

    return out[["country", "country_name", "year", "quarter", "real_gdp"]]


def _load_angola(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=1, dtype=str)
    df = df.rename(
        columns={
            df.columns[0]: "year",
            df.columns[1]: "quarter_label",
            df.columns[2]: "real_gdp",
        }
    )

    out = df[["year", "quarter_label", "real_gdp"]].copy()
    out["year"] = out["year"].replace("", pd.NA).ffill()
    out["quarter"] = out["quarter_label"].astype(str).str.strip().map(ANGOLA_QUARTER_MAP)
    out["real_gdp"] = (
        out["real_gdp"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )
    out["country"] = "AGO"
    out["country_name"] = "Angola"

    return out[["country", "country_name", "year", "quarter", "real_gdp"]]


def build_real_gdp_table(config_path: str | Path = Path("config/default.yaml")) -> Path:
    cfg = load_config(config_path)
    paths = resolve_paths(cfg)

    raw_dir = paths["raw"] / "real_gdp"
    output_dir = paths["processed"] / "real_gdp"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        _load_imf(raw_dir / "imf_real_gdp_quarterly.csv"),
        _load_uae(raw_dir / "uae_quarterly_real_gdp.csv"),
        _load_angola(raw_dir / "angola_quarterly_real_gdp.csv"),
    ]
    combined = _finalize(pd.concat(frames, ignore_index=True))

    output_path = output_dir / "quarterly_real_gdp.csv"
    combined.to_csv(output_path, index=False)
    return output_path


def build_output_gap_table(
    config_path: str | Path = Path("config/default.yaml"),
    *,
    min_obs: int = 32,
    hp_lambda: int = 1600,
) -> Path:
    cfg = load_config(config_path)
    paths = resolve_paths(cfg)

    input_path = paths["processed"] / "real_gdp" / "quarterly_real_gdp.csv"
    output_dir = paths["processed"] / "real_gdp"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {"country", "country_name", "period", "year", "quarter", "real_gdp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Quarterly real GDP input is missing columns: {sorted(missing)}")

    df = df.sort_values(["country", "year", "quarter"]).reset_index(drop=True).copy()
    df["n_obs_country"] = (
        df.groupby("country")["real_gdp"].transform(lambda s: int(s.notna().sum())).astype(int)
    )
    df["log_real_gdp"] = np.nan
    df["hp_trend_log_real_gdp"] = np.nan
    df["output_gap"] = np.nan
    df["hp_filter_status"] = "ok"

    for country, group in df.groupby("country", sort=False):
        idx = group.index
        real_gdp = pd.to_numeric(group["real_gdp"], errors="coerce")
        valid = real_gdp.notna()
        n_obs = int(valid.sum())

        if n_obs < min_obs:
            df.loc[idx, "hp_filter_status"] = "too_few_observations"
            continue

        if (real_gdp.loc[valid] <= 0).any():
            df.loc[idx, "hp_filter_status"] = "non_positive_gdp"
            continue

        log_real_gdp = np.log(real_gdp.loc[valid].to_numpy(dtype=float))
        cycle, trend = hpfilter(log_real_gdp, lamb=hp_lambda)

        valid_idx = real_gdp.loc[valid].index
        df.loc[valid_idx, "log_real_gdp"] = log_real_gdp
        df.loc[valid_idx, "hp_trend_log_real_gdp"] = trend
        df.loc[valid_idx, "output_gap"] = cycle

    output_path = output_dir / "quarterly_output_gap.csv"
    df.to_csv(output_path, index=False)
    return output_path
