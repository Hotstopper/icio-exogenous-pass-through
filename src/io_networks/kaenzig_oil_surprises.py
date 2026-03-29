from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from io_networks.paths import resolve_paths

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CELL_REF_RE = re.compile(r"([A-Z]+)")
VINTAGE_RE = re.compile(r"(\d{4}M\d{2})")
MONTHLY_DATE_RE = re.compile(r"^(\d{4})M(\d{2})$")


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(text_node.text or "" for text_node in si.iterfind(".//a:t", NS))
        for si in root.findall("a:si", NS)
    ]


def _sheet_targets(zf: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: f"xl/{rel.attrib['Target']}" for rel in rels}

    out: dict[str, str] = {}
    for sheet in workbook.findall(".//a:sheet", NS):
        rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
        out[sheet.attrib["name"]] = rel_map[rel_id]
    return out


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    value_node = cell.find("a:v", NS)
    if value_node is None:
        return None
    raw = value_node.text
    if raw is None:
        return None
    if cell.attrib.get("t") == "s":
        return shared_strings[int(raw)]
    return raw


def _monthly_date_to_timestamp(value: str) -> pd.Timestamp:
    match = MONTHLY_DATE_RE.match(value)
    if match is None:
        raise ValueError(f"Unsupported monthly date format: {value!r}")
    year, month = match.groups()
    return pd.Timestamp(year=int(year), month=int(month), day=1)


def extract_monthly_surprise_series(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        shared_strings = _load_shared_strings(zf)
        sheet_target = _sheet_targets(zf).get("Monthly")
        if sheet_target is None:
            raise ValueError(f"Workbook {path} does not contain a 'Monthly' sheet.")
        worksheet = ET.fromstring(zf.read(sheet_target))

    rows: list[dict[str, str | float | pd.Timestamp]] = []
    for row in worksheet.findall(".//a:sheetData/a:row", NS)[1:]:
        values: dict[str, str | None] = {}
        for cell in row.findall("a:c", NS):
            ref = cell.attrib.get("r", "")
            match = CELL_REF_RE.match(ref)
            if match is None:
                continue
            values[match.group(1)] = _cell_value(cell, shared_strings)

        raw_date = values.get("A")
        raw_series = values.get("B")
        raw_news_shock = values.get("C")
        if raw_date is None or raw_series is None:
            continue

        rows.append(
            {
                "date": _monthly_date_to_timestamp(raw_date),
                "period": raw_date,
                "surprise": pd.to_numeric(raw_series, errors="coerce"),
                "news_shock": pd.to_numeric(raw_news_shock, errors="coerce"),
            }
        )

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["date", "surprise"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No monthly oil supply surprise rows found in {path}.")
    return out


def _vintage_from_path(path: Path) -> str:
    match = VINTAGE_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Could not infer vintage from filename: {path.name}")
    return match.group(1)


def build_monthly_vintage_table(raw_dir: Path, keep_only_vintage: str | None = None) -> pd.DataFrame:
    files = sorted(
        path
        for path in raw_dir.glob("*.xls*")
        if path.is_file() and not path.name.startswith("~$")
    )
    if not files:
        raise FileNotFoundError(f"No Excel files found in {raw_dir}")

    combined: pd.DataFrame | None = None
    for path in files:
        vintage = _vintage_from_path(path)
        series = extract_monthly_surprise_series(path).rename(
            columns={
                "surprise": f"{vintage}_surprise",
                "news_shock": f"{vintage}_news_shock",
            }
        )[["date", "period", f"{vintage}_surprise", f"{vintage}_news_shock"]]
        if combined is None:
            combined = series
            continue
        combined = combined.merge(series, on=["date", "period"], how="outer", sort=True)

    assert combined is not None
    combined = combined.sort_values("date").reset_index(drop=True)
    value_cols = sorted(col for col in combined.columns if col not in {"date", "period"})
    if keep_only_vintage is not None:
        keep_cols = [f"{keep_only_vintage}_surprise", f"{keep_only_vintage}_news_shock"]
        missing = [col for col in keep_cols if col not in value_cols]
        if missing:
            raise ValueError(f"Requested vintage {keep_only_vintage!r} not found in raw files.")
        value_cols = keep_cols
    vintage_cols = ["date", "period", *value_cols]
    return combined[vintage_cols]


def build_quarterly_vintage_table(monthly_df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [col for col in monthly_df.columns if col not in {"date", "period"}]
    quarterly = monthly_df.copy()
    quarterly["date"] = quarterly["date"].dt.to_period("Q").dt.end_time.dt.normalize()
    quarterly["period"] = quarterly["date"].dt.to_period("Q").astype(str)
    quarterly = quarterly.groupby(["date", "period"], as_index=False)[value_cols].sum(min_count=3)
    quarterly = quarterly.sort_values("date").reset_index(drop=True)
    return quarterly[["date", "period", *value_cols]]


def build_kaenzig_oil_surprise_tables(cfg: dict) -> tuple[Path, Path]:
    paths = resolve_paths(cfg)
    raw_dir = paths["raw"] / "kaenzig_oil_surprises"
    output_dir = paths["processed"] / "oil"
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly = build_monthly_vintage_table(raw_dir, keep_only_vintage="2025M06")
    monthly = monthly.rename(
        columns={
            "2025M06_surprise": "surprise",
            "2025M06_news_shock": "news_shock",
        }
    )
    quarterly = build_quarterly_vintage_table(monthly)

    monthly_path = output_dir / "kaenzig_oil_supply_surprises_monthly.csv"
    quarterly_path = output_dir / "kaenzig_oil_supply_surprises_quarterly.csv"

    monthly_to_write = monthly.copy()
    monthly_to_write["date"] = monthly_to_write["date"].dt.strftime("%Y-%m-%d")
    monthly_to_write.to_csv(monthly_path, index=False)

    quarterly_to_write = quarterly.copy()
    quarterly_to_write["date"] = quarterly_to_write["date"].dt.strftime("%Y-%m-%d")
    quarterly_to_write.to_csv(quarterly_path, index=False)

    return monthly_path, quarterly_path
