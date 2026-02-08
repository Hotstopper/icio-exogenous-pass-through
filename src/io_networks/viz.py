from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from io_networks.paths import resolve_paths

_ALLOWED_METRICS = {"tau", "tau_dir", "tau_amp"}


def _resolve_variant(cfg: dict[str, Any], variant: str | None) -> str:
    if variant:
        return variant
    return "extended" if cfg["icio"].get("extended", False) else "regular"


def _country_from_label(label: str) -> str:
    if "_" not in label:
        return ""
    return label.split("_", 1)[0]


def _sector_code_from_label(label: str) -> str:
    if "_" not in label:
        return ""
    return label.split("_", 1)[1]


def _sector_bucket(sector_code: str) -> str:
    if not sector_code:
        return "Other"
    lead = sector_code[0]
    if lead == "A":
        return "Agriculture"
    if lead in {"B", "C", "D", "E", "F"}:
        return "Manufacturing"
    if lead in {"G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"}:
        return "Services"
    return "Other"


def _load_reference_table(path: Path, key_col: str, value_col: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[key_col, value_col])
    df = pd.read_csv(path)
    if key_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[key_col, value_col])
    return df[[key_col, value_col]].dropna().drop_duplicates(subset=[key_col])


def _load_sector_reference(path: Path) -> pd.DataFrame:
    cols = ["sector_code", "sector_name", "sector_abbreviation"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    available = [c for c in cols if c in df.columns]
    if "sector_code" not in available:
        return pd.DataFrame(columns=cols)
    out = df[available].dropna(subset=["sector_code"]).drop_duplicates(subset=["sector_code"])
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[cols]


def _metric_label(metric: str) -> str:
    labels = {
        "tau": r"$\tau$",
        "tau_dir": r"$\tau_{dir}$",
        "tau_amp": r"$\tau_{amp}$",
    }
    return labels.get(metric, metric)


def prepare_country_bubble_data(
    cfg: dict[str, Any],
    country: str,
    year: int,
    metric: str = "tau",
    variant: str | None = None,
) -> pd.DataFrame:
    """Return country-year sector rows for bubble plotting of tau metrics.

    Bubble size uses sector output (OUT) from A metadata as a GDP proxy.
    """
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {sorted(_ALLOWED_METRICS)}")

    country = country.upper()
    use_variant = _resolve_variant(cfg, variant)
    paths = resolve_paths(cfg)

    blocks_dir = paths["matrices"] / "blocks" / use_variant
    a_dir = paths["matrices"] / "A" / use_variant
    ref_dir = Path("data/reference")

    blocks_npz = blocks_dir / f"blocks_{year}.npz"
    blocks_meta = blocks_dir / f"blocks_{year}_meta.parquet"
    a_meta = a_dir / f"A_{year}_meta.parquet"

    if not blocks_npz.exists() or not blocks_meta.exists():
        raise FileNotFoundError(
            f"Missing blocks files for {year} in {blocks_dir}. Run build-blocks first."
        )
    if not a_meta.exists():
        raise FileNotFoundError(f"Missing A metadata file: {a_meta}")

    arr = np.load(blocks_npz)
    if metric not in arr.files:
        raise ValueError(f"Metric {metric} not found in {blocks_npz.name}. Available: {arr.files}")

    tau = arr[metric]

    meta = pd.read_parquet(blocks_meta)
    n_meta = meta[meta["group"] == "N"].sort_values("index_in_group").reset_index(drop=True)
    labels_n = n_meta["label"].astype(str).tolist()

    if len(labels_n) != len(tau):
        raise ValueError(
            f"Mismatch between labels ({len(labels_n)}) and {metric} length ({len(tau)}) for year {year}."
        )

    df = pd.DataFrame(
        {
            "label": labels_n,
            metric: tau,
        }
    )
    df["country_code"] = df["label"].map(_country_from_label)
    df["sector_code"] = df["label"].map(_sector_code_from_label)

    df = df[df["country_code"] == country].copy()
    if df.empty:
        raise ValueError(f"No non-exogenous rows found for country={country}, year={year}.")

    a_meta_df = pd.read_parquet(a_meta)[["sector", "out"]].rename(columns={"sector": "label", "out": "out_proxy"})
    df = df.merge(a_meta_df, on="label", how="left")
    df["out_proxy"] = pd.to_numeric(df["out_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)

    sector_ref = _load_sector_reference(ref_dir / "sector_codes.csv")
    country_ref = _load_reference_table(ref_dir / "country_codes.csv", "country_code", "country_name")

    if not sector_ref.empty:
        df = df.merge(sector_ref, on="sector_code", how="left")
    else:
        df["sector_name"] = pd.NA
        df["sector_abbreviation"] = pd.NA

    if not country_ref.empty:
        df = df.merge(country_ref, on="country_code", how="left")
    else:
        df["country_name"] = pd.NA

    df["sector_bucket"] = df["sector_code"].map(_sector_bucket)

    bucket_order = {"Agriculture": 0, "Manufacturing": 1, "Services": 2, "Other": 3}
    df["_bucket_order"] = df["sector_bucket"].map(bucket_order).fillna(99).astype(int)
    df = df.sort_values(["_bucket_order", "sector_code", "label"]).reset_index(drop=True)
    df["x"] = np.arange(len(df), dtype=float)

    # Bubble-size helper columns for notebook plotting.
    max_out = float(df["out_proxy"].max())
    if max_out > 0:
        df["bubble_size_norm"] = df["out_proxy"] / max_out
    else:
        df["bubble_size_norm"] = 0.0
    df["bubble_size"] = 100.0 + 1400.0 * df["bubble_size_norm"]

    df["year"] = int(year)
    df["variant"] = use_variant
    df["metric"] = metric

    return df.drop(columns=["_bucket_order"])


def plot_country_bubble(
    df: pd.DataFrame,
    metric: str,
    title: str | None = None,
    label_top_n: int = 20,
    ax: Any = None,
) -> Any:
    """Plot bubble chart from prepared country bubble dataframe.

    Expects output from prepare_country_bubble_data.
    """
    if metric not in df.columns:
        raise ValueError(f"metric column '{metric}' not in dataframe")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it in your .venv.") from exc
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    work = df.copy()
    metric_label = _metric_label(metric)
    metric_values = pd.to_numeric(work[metric], errors="coerce")
    bubble_sizes = pd.to_numeric(work.get("bubble_size"), errors="coerce")
    valid_mask = (
        np.isfinite(work["x"].to_numpy(dtype=float))
        & np.isfinite(metric_values.to_numpy(dtype=float))
        & np.isfinite(bubble_sizes.to_numpy(dtype=float))
    )
    plot = work.loc[valid_mask].copy()
    if plot.empty:
        raise ValueError(f"No finite data available to plot for metric '{metric}'.")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(plot["x"], plot[metric], s=plot["bubble_size"], alpha=0.55)

    # Label highest values for readability.
    texts = []
    if label_top_n > 0:
        top = plot.nlargest(min(label_top_n, len(plot)), metric)
        for _, row in top.iterrows():
            label = row.get("sector_abbreviation")
            if pd.isna(label) or not str(label).strip():
                label = row["sector_code"]
            txt = ax.text(
                float(row["x"]),
                float(row[metric]),
                str(label),
                fontsize=8,
                fontfamily="serif",
            )
            texts.append(txt)
        if adjust_text is not None and texts:
            ax.figure.canvas.draw()
            x_vals = plot["x"].to_numpy(dtype=float)
            y_vals = pd.to_numeric(plot[metric], errors="coerce").to_numpy(dtype=float)
            params = dict(
                texts=texts,
                x=x_vals,
                y=y_vals,
                ax=ax,
                avoid_self=True,
                ensure_inside_axes=True,
                expand_axes=False,
                only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                expand_text=(1.02, 1.08),
                expand_points=(1.05, 1.1),
                force_text=0.12,
                force_points=0.15,
                max_move=(8, 8),
                iter_lim=200,
            )
            adjust_text(**params)

    bucket_centers = plot.groupby("sector_bucket", as_index=False)["x"].mean()
    ax.set_xticks(bucket_centers["x"])
    ax.set_xticklabels(bucket_centers["sector_bucket"], fontfamily="serif", fontsize=12)

    for bucket in ["Agriculture", "Manufacturing", "Services", "Other"]:
        sub = plot[plot["sector_bucket"] == bucket]
        if len(sub) == 0:
            continue
        ax.axvline(float(sub["x"].max()) + 0.5, linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("")
    ax.set_ylabel(metric_label, fontfamily="serif", fontsize=14)
    if title:
        ax.set_title(title, fontfamily="serif", fontsize=18)
    else:
        country = str(work["country_code"].iloc[0]) if len(work) else ""
        country_name = str(work["country_name"].iloc[0]) if len(work) and "country_name" in work.columns else ""
        if not country_name or country_name.lower() == "nan":
            country_name = country
        year = int(work["year"].iloc[0]) if len(work) else 0
        ax.set_title(f"{country_name} {year}: {metric_label}", fontfamily="serif")

    for tick in ax.get_yticklabels():
        tick.set_fontfamily("serif")

    return ax
