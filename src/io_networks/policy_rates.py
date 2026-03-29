from __future__ import annotations

from pathlib import Path

import pandas as pd

from io_networks.config import load_config
from io_networks.paths import resolve_paths

RAW_FILENAME = "bis_dp_search_export_policy_rate.csv"
OUTPUT_FILENAME = "quarterly_policy_rates.csv"
EURO_AREA_NAME = "Euro area"
EURO_AREA_CODE = "EA19"
COUNTRY_NAME_OVERRIDES = {
    "Hong Kong SAR": "Hong Kong",
    "Korea": "South Korea",
    "Kuwait": "Kuwait",
    "North Macedonia": "North Macedonia",
    "Serbia": "Serbia",
}
COUNTRY_CODE_OVERRIDES = {
    "Kuwait": "KWT",
    "North Macedonia": "MKD",
    "Serbia": "SRB",
}
EURO_ADOPTION_STARTS = {
    "AUT": "1999Q1",
    "BEL": "1999Q1",
    "DEU": "1999Q1",
    "ESP": "1999Q1",
    "FRA": "1999Q1",
    "GRC": "2001Q1",
    "ITA": "1999Q1",
    "NLD": "1999Q1",
    "PRT": "1999Q1",
    "EST": "2011Q1",
    "IRL": "2002Q1",
    "FIN": "2002Q1",
    "LTU": "2015Q1",
    "LUX": "1999Q1",
    "LVA": "2014Q1",
    "MLT": "2008Q1",
    "SVK": "2009Q1",
    "SVN": "2007Q1",
}


def _extract_country_metadata(raw: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    metadata = raw[["REF_AREA:Reference area"]].drop_duplicates().copy()
    ref_area_parts = metadata["REF_AREA:Reference area"].astype(str).str.split(":", n=1, expand=True)
    metadata["country_name_raw"] = ref_area_parts[1].str.strip()
    metadata["country_name"] = metadata["country_name_raw"].replace(COUNTRY_NAME_OVERRIDES)
    metadata.loc[metadata["country_name_raw"] == EURO_AREA_NAME, "country"] = EURO_AREA_CODE

    reference = pd.read_csv(reference_path)
    reference["country_name"] = reference["country_name"].astype(str).str.strip()
    reference["country"] = reference["country_code"].astype(str).str.strip()

    metadata = metadata.merge(
        reference[["country_name", "country"]],
        how="left",
        on="country_name",
        validate="m:1",
        suffixes=("", "_ref"),
    )
    if "country_ref" in metadata.columns:
        metadata["country"] = metadata["country"].fillna(metadata["country_ref"])
        metadata = metadata.drop(columns=["country_ref"])
    metadata["country"] = metadata["country"].fillna(metadata["country_name"].map(COUNTRY_CODE_OVERRIDES))

    missing = sorted(metadata.loc[metadata["country"].isna(), "country_name_raw"].unique())
    if missing:
        raise ValueError(f"Missing ISO3 mappings for policy-rate countries: {missing}")

    return metadata[["REF_AREA:Reference area", "country"]]


def _apply_euro_area_fill(quarterly: pd.DataFrame) -> pd.DataFrame:
    euro_area = quarterly.loc[quarterly["country"] == EURO_AREA_CODE, ["year", "qtr", "policy_rate"]].copy()
    if euro_area.empty:
        raise ValueError("Euro area policy-rate series was not found in the raw BIS file.")

    filled_frames: list[pd.DataFrame] = []
    for country, start in EURO_ADOPTION_STARTS.items():
        adoption_period = pd.Period(start, freq="Q")
        target = quarterly.loc[quarterly["country"] == country, ["country", "year", "qtr", "policy_rate"]].copy()
        merged = euro_area.merge(target, how="left", on=["year", "qtr"], suffixes=("_ea", ""))
        mask = (merged["year"] > adoption_period.year) | (
            (merged["year"] == adoption_period.year) & (merged["qtr"] >= adoption_period.quarter)
        )
        merged = merged.loc[mask, ["year", "qtr", "policy_rate_ea", "policy_rate"]].copy()
        merged["country"] = country
        merged["policy_rate"] = merged["policy_rate"].fillna(merged["policy_rate_ea"])
        filled_frames.append(merged[["country", "year", "qtr", "policy_rate"]])

    keep = quarterly.loc[~quarterly["country"].isin({EURO_AREA_CODE, *EURO_ADOPTION_STARTS.keys()})].copy()
    return pd.concat([keep, *filled_frames], ignore_index=True, sort=False)


def build_policy_rate_table(config_path: str | Path = Path("config/default.yaml")) -> Path:
    cfg = load_config(config_path)
    paths = resolve_paths(cfg)

    raw_dir = paths["raw"] / "policy_rates"
    output_dir = paths["processed"] / "policy_rates"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_dir / RAW_FILENAME, header=5)
    country_meta = _extract_country_metadata(raw, paths["raw"].parent / "reference" / "country_codes.csv")

    id_columns = [
        "DATAFLOW_ID:Dataflow ID",
        "KEY:Timeseries Key",
        "FREQ:Frequency",
        "REF_AREA:Reference area",
        "Unit",
        "Unit multiplier",
        "TIME_PERIOD:Period",
    ]
    date_columns = [col for col in raw.columns if pd.to_datetime(col, format="%d/%m/%Y", errors="coerce") is not pd.NaT]

    long = raw[[*id_columns, *date_columns]].melt(
        id_vars=id_columns,
        value_vars=date_columns,
        var_name="date",
        value_name="policy_rate",
    )
    long["date"] = pd.to_datetime(long["date"], format="%d/%m/%Y", errors="coerce")
    long["policy_rate"] = pd.to_numeric(long["policy_rate"], errors="coerce")
    long = long.dropna(subset=["date", "policy_rate"]).copy()
    long = long.merge(country_meta, on="REF_AREA:Reference area", how="inner", validate="m:1")

    long["year"] = long["date"].dt.year
    long["qtr"] = long["date"].dt.quarter

    quarterly = (
        long.groupby(["country", "year", "qtr"], as_index=False)
        .agg(policy_rate=("policy_rate", "mean"), months_in_quarter=("date", "nunique"))
    )
    quarterly = quarterly.loc[quarterly["months_in_quarter"] == 3].copy()
    quarterly = _apply_euro_area_fill(quarterly)
    quarterly["quarter"] = (
        quarterly["year"].astype(int).astype(str) + "-Q" + quarterly["qtr"].astype(int).astype(str)
    )
    quarterly = quarterly.sort_values(["country", "year", "qtr"]).reset_index(drop=True)

    output = quarterly[["quarter", "country", "policy_rate"]]
    output_path = output_dir / OUTPUT_FILENAME
    output.to_csv(output_path, index=False)
    return output_path
