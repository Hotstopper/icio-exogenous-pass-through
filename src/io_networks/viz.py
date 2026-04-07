from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from io_networks.paths import resolve_paths

_ALLOWED_METRICS = {"tau", "tau_dir", "tau_amp", "tau_amp2", "chi"}


def _load_tau_arrays(
    blocks_npz: Path,
    blocks_meta: Path,
    metric: str,
) -> tuple[pd.DataFrame, str, np.ndarray, np.ndarray | None]:
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {sorted(_ALLOWED_METRICS)}")
    if not blocks_npz.exists() or not blocks_meta.exists():
        year = blocks_npz.stem.replace("blocks_", "")
        raise FileNotFoundError(
            f"Missing blocks files in {blocks_npz.parent} for year {year}. "
            "Run build-blocks first."
        )

    arr = np.load(blocks_npz)
    if metric == "tau_amp2":
        required = {"tau", "tau_amp"}
        missing = sorted(required.difference(set(arr.files)))
        if missing:
            raise ValueError(
                f"Metric mode tau_amp2 requires arrays {sorted(required)} in {blocks_npz.name}. "
                f"Missing: {missing}. Available: {arr.files}"
            )
        metric_col = "tau"
        tau_values = arr[metric_col]
        tau_amp_tail = arr["tau_amp"]
    else:
        if metric not in arr.files:
            raise ValueError(
                f"Metric {metric} not found in {blocks_npz.name}. "
                f"Available: {arr.files}"
            )
        metric_col = metric
        tau_values = arr[metric_col]
        tau_amp_tail = None

    meta = pd.read_parquet(blocks_meta)
    n_meta = meta[meta["group"] == "N"].sort_values("index_in_group").reset_index(drop=True)
    labels_n = n_meta["label"].astype(str).tolist()

    if len(labels_n) != len(tau_values):
        raise ValueError(
            f"Mismatch between labels ({len(labels_n)}) and "
            f"{metric_col} length ({len(tau_values)}) for {blocks_npz.stem}."
        )
    if tau_amp_tail is not None and len(labels_n) != len(tau_amp_tail):
        raise ValueError(
            f"Mismatch between labels ({len(labels_n)}) and tau_amp length ({len(tau_amp_tail)}) "
            f"for {blocks_npz.stem}."
        )

    return n_meta, metric_col, tau_values, tau_amp_tail


def _resolve_variant(cfg: dict[str, Any], variant: str | None) -> str:
    if variant:
        return variant
    return "extended" if cfg["icio"].get("extended", False) else "regular"


def _extract_year(filename: str) -> int | None:
    match = re.search(r"(19\d{2}|20\d{2})", filename)
    return int(match.group(1)) if match else None


def _raw_file_for_year(raw_dir: Path, year: int) -> Path | None:
    for file_path in sorted(raw_dir.glob("*.csv")):
        if _extract_year(file_path.name) == year:
            return file_path
    return None


def _load_country_hfce_proxy(raw_file: Path, country: str) -> pd.DataFrame:
    hfce_col = f"{country}_HFCE"
    header = pd.read_csv(raw_file, nrows=0).columns.tolist()
    if hfce_col not in header:
        return pd.DataFrame(columns=["label", "hfce_proxy"])

    frame = pd.read_csv(raw_file, usecols=["V1", hfce_col]).rename(columns={"V1": "label"})
    frame["hfce_proxy"] = pd.to_numeric(frame[hfce_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return frame[["label", "hfce_proxy"]]


def _load_all_countries_hfce_proxy(raw_file: Path) -> pd.DataFrame:
    header = pd.read_csv(raw_file, nrows=0).columns.tolist()
    hfce_cols = [col for col in header if col.endswith("_HFCE")]
    if not hfce_cols:
        return pd.DataFrame(columns=["label", "hfce_proxy"])

    frame = pd.read_csv(raw_file, usecols=["V1", *hfce_cols]).rename(columns={"V1": "label"})
    hfce_values = frame[hfce_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["hfce_proxy"] = hfce_values.sum(axis=1)
    return frame[["label", "hfce_proxy"]]


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
    if lead in {"B", "C"}:
        return "Manufacturing"
    if lead in {"D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"}:
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
        "tau_amp2": r"$\tau$ with $\tau_{amp}$ tails",
        "chi": r"$\chi$",
    }
    return labels.get(metric, metric)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")

    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[mask]
    w = w[mask]
    if len(v) == 0:
        return float("nan")

    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) - 0.5 * w
    cdf = cdf / np.sum(w)
    return float(np.interp(q, cdf, v))


def prepare_country_bubble_data(
    cfg: dict[str, Any],
    country: str,
    year: int,
    metric: str = "tau",
    variant: str | None = None,
) -> pd.DataFrame:
    """Return country-year sector rows for bubble plotting of tau metrics.

    Bubble size uses sector HFCE expenditure for the selected country-year.
    """
    country = country.upper()
    use_variant = _resolve_variant(cfg, variant)
    paths = resolve_paths(cfg)

    blocks_dir = paths["matrices"] / "blocks" / use_variant
    a_dir = paths["matrices"] / "A" / use_variant
    raw_dir = paths["raw"] / use_variant
    ref_dir = Path("data/reference")

    blocks_npz = blocks_dir / f"blocks_{year}.npz"
    blocks_meta = blocks_dir / f"blocks_{year}_meta.parquet"
    a_meta = a_dir / f"A_{year}_meta.parquet"
    raw_file = _raw_file_for_year(raw_dir, year)

    if not a_meta.exists():
        raise FileNotFoundError(f"Missing A metadata file: {a_meta}")
    if raw_file is None:
        raise FileNotFoundError(f"Missing raw ICIO file for year {year} in {raw_dir}")

    n_meta, metric_col, tau_values, tau_amp_tail = _load_tau_arrays(blocks_npz, blocks_meta, metric)
    labels_n = n_meta["label"].astype(str).tolist()

    df = pd.DataFrame(
        {
            "label": labels_n,
            metric: tau_values,
        }
    )
    if tau_amp_tail is not None:
        df["tau_amp_tail"] = tau_amp_tail
    df["country_code"] = df["label"].map(_country_from_label)
    df["sector_code"] = df["label"].map(_sector_code_from_label)

    df = df[df["country_code"] == country].copy()
    if df.empty:
        raise ValueError(f"No non-exogenous rows found for country={country}, year={year}.")

    a_meta_df = pd.read_parquet(a_meta)[["sector", "out"]].rename(
        columns={"sector": "label", "out": "out_proxy"}
    )
    df = df.merge(a_meta_df, on="label", how="left")
    df["out_proxy"] = pd.to_numeric(df["out_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    hfce_proxy_df = _load_country_hfce_proxy(raw_file, country)
    df = df.merge(hfce_proxy_df, on="label", how="left")
    df["hfce_proxy"] = pd.to_numeric(df["hfce_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["bubble_size_proxy"] = df["hfce_proxy"]

    sector_ref = _load_sector_reference(ref_dir / "sector_codes.csv")
    country_ref = _load_reference_table(
        ref_dir / "country_codes.csv",
        "country_code",
        "country_name",
    )

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
    max_proxy = float(df["bubble_size_proxy"].max())
    if max_proxy > 0:
        df["bubble_size_norm"] = df["bubble_size_proxy"] / max_proxy
    else:
        df["bubble_size_norm"] = 0.0
    df["bubble_size"] = 100.0 + 1900.0 * df["bubble_size_norm"]

    df["year"] = int(year)
    df["variant"] = use_variant
    df["metric"] = metric

    return df.drop(columns=["_bucket_order"])


def prepare_all_countries_bubble_data(
    cfg: dict[str, Any],
    year: int,
    metric: str = "tau",
    variant: str | None = None,
    bubble_size: float = 18.0,
) -> pd.DataFrame:
    """Return all-country non-exogenous rows for year-level tau bubble plotting.

    Bubble size is constant to avoid GDP-size clutter when plotting all countries.
    """
    use_variant = _resolve_variant(cfg, variant)
    paths = resolve_paths(cfg)

    blocks_dir = paths["matrices"] / "blocks" / use_variant
    a_dir = paths["matrices"] / "A" / use_variant
    raw_dir = paths["raw"] / use_variant
    ref_dir = Path("data/reference")

    blocks_npz = blocks_dir / f"blocks_{year}.npz"
    blocks_meta = blocks_dir / f"blocks_{year}_meta.parquet"
    a_meta = a_dir / f"A_{year}_meta.parquet"
    raw_file = _raw_file_for_year(raw_dir, year)

    n_meta, metric_col, tau_values, tau_amp_tail = _load_tau_arrays(blocks_npz, blocks_meta, metric)

    df = n_meta[["label"]].copy()
    df[metric] = tau_values
    if tau_amp_tail is not None:
        df["tau_amp_tail"] = tau_amp_tail

    df["country_code"] = df["label"].map(_country_from_label)
    df["sector_code"] = df["label"].map(_sector_code_from_label)

    sector_ref = _load_sector_reference(ref_dir / "sector_codes.csv")
    country_ref = _load_reference_table(
        ref_dir / "country_codes.csv",
        "country_code",
        "country_name",
    )

    if not sector_ref.empty:
        df = df.merge(sector_ref, on="sector_code", how="left")
    else:
        df["sector_name"] = pd.NA
        df["sector_abbreviation"] = pd.NA

    if not country_ref.empty:
        df = df.merge(country_ref, on="country_code", how="left")
    else:
        df["country_name"] = pd.NA

    if a_meta.exists():
        a_meta_df = pd.read_parquet(a_meta)[["sector", "out"]].rename(
            columns={"sector": "label", "out": "out_proxy"}
        )
        df = df.merge(a_meta_df, on="label", how="left")
        df["out_proxy"] = (
            pd.to_numeric(df["out_proxy"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
    else:
        df["out_proxy"] = 0.0

    if raw_file is not None:
        hfce_proxy_df = _load_all_countries_hfce_proxy(raw_file)
        df = df.merge(hfce_proxy_df, on="label", how="left")
        df["hfce_proxy"] = (
            pd.to_numeric(df["hfce_proxy"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
    else:
        df["hfce_proxy"] = 0.0

    df["sector_bucket"] = df["sector_code"].map(_sector_bucket)
    bucket_order = {"Agriculture": 0, "Manufacturing": 1, "Services": 2, "Other": 3}
    df["_bucket_order"] = df["sector_bucket"].map(bucket_order).fillna(99).astype(int)
    df = df.sort_values(
        ["_bucket_order", "sector_code", "country_code", "label"]
    ).reset_index(drop=True)

    sector_positions = (
        df[["sector_code", "_bucket_order"]]
        .drop_duplicates()
        .sort_values(["_bucket_order", "sector_code"])
        .reset_index(drop=True)
    )
    sector_positions["x_base"] = np.arange(len(sector_positions), dtype=float)
    df = df.merge(sector_positions[["sector_code", "x_base"]], on="sector_code", how="left")

    df["x"] = df["x_base"]
    df["bubble_size_norm"] = 0.0
    df["bubble_size"] = float(bubble_size)
    df["year"] = int(year)
    df["variant"] = use_variant
    df["metric"] = metric
    df["metric_col"] = metric_col

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
    if metric == "tau_amp2" and "tau_amp_tail" not in df.columns:
        raise ValueError("tau_amp2 mode requires 'tau_amp_tail' in dataframe.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it in your .venv.") from exc
    work = df.copy()
    metric_label = _metric_label(metric)
    metric_values = pd.to_numeric(work[metric], errors="coerce")
    bubble_sizes = pd.to_numeric(work.get("bubble_size"), errors="coerce")
    valid_mask = (
        np.isfinite(work["x"].to_numpy(dtype=float))
        & np.isfinite(metric_values.to_numpy(dtype=float))
        & np.isfinite(bubble_sizes.to_numpy(dtype=float))
    )
    plot = work.loc[valid_mask].copy().reset_index(drop=True)
    if plot.empty:
        raise ValueError(f"No finite data available to plot for metric '{metric}'.")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    if metric == "tau_amp2":
        tails = pd.to_numeric(plot["tau_amp_tail"], errors="coerce").fillna(0.0).clip(lower=0.0)
        y_top = pd.to_numeric(plot[metric], errors="coerce").to_numpy(dtype=float)
        y_bottom = y_top - tails.to_numpy(dtype=float)
        ax.vlines(
            plot["x"].to_numpy(dtype=float),
            y_bottom,
            y_top,
            colors="C0",
            linewidth=1,
            alpha=0.7,
            zorder=1,
        )

    ax.scatter(plot["x"], plot[metric], s=plot["bubble_size"], alpha=0.55)

    # Label placement: keep text close to own bubble, avoid other bubbles and prior labels.
    if label_top_n > 0:
        top = plot.nlargest(min(label_top_n, len(plot)), metric)
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        trans = ax.transData
        axes_bbox = ax.get_window_extent(renderer=renderer)
        px_per_pt = ax.figure.dpi / 72.0

        x_all = plot["x"].to_numpy(dtype=float)
        y_all = pd.to_numeric(plot[metric], errors="coerce").to_numpy(dtype=float)
        s_all = pd.to_numeric(plot["bubble_size"], errors="coerce").to_numpy(dtype=float)
        centers = trans.transform(np.column_stack([x_all, y_all]))
        radii_px = np.sqrt(np.maximum(s_all, 0.0) / np.pi) * px_per_pt
        placed_bboxes = []

        def _rect_circle_overlap(rect, cx, cy, r):
            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
            nx = np.clip(cx, x0, x1)
            ny = np.clip(cy, y0, y1)
            return (cx - nx) ** 2 + (cy - ny) ** 2 < r**2

        angles_deg = [25, -25, 155, -155, 60, -60, 120, -120, 90, -90, 0, 180]
        offset_factors = [0.75, 0.9, 1.05]
        for _, row in top.iterrows():
            label = row.get("sector_abbreviation")
            if pd.isna(label) or not str(label).strip():
                label = row["sector_code"]
            bubble_size = float(row.get("bubble_size", 0.0))
            bubble_radius_pt = np.sqrt(max(bubble_size, 0.0) / np.pi)
            base_offset_pt = bubble_radius_pt + 0.5
            row_idx = int(row.name)
            best = None

            for factor in offset_factors:
                off = base_offset_pt * factor
                for ang_deg in angles_deg:
                    ang = np.deg2rad(ang_deg)
                    dx = off * np.cos(ang)
                    dy = off * np.sin(ang)
                    ha = "left" if dx >= 0 else "right"
                    va = "bottom" if dy >= 0 else "top"
                    candidate = ax.annotate(
                        str(label),
                        xy=(float(row["x"]), float(row[metric])),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=8,
                        fontfamily="serif",
                        ha=ha,
                        va=va,
                        clip_on=True,
                        alpha=0.0,
                    )
                    bbox = candidate.get_window_extent(renderer=renderer).expanded(1.02, 1.08)
                    candidate.remove()

                    label_overlap = sum(1 for b in placed_bboxes if bbox.overlaps(b))
                    bubble_overlap = 0.0
                    for j, ((cx, cy), r) in enumerate(zip(centers, radii_px)):
                        if _rect_circle_overlap(bbox, cx, cy, r):
                            bubble_overlap += 0.5 if j == row_idx else 1.0
                    contains_start = axes_bbox.contains(*bbox.get_points()[0])
                    contains_end = axes_bbox.contains(*bbox.get_points()[1])
                    outside = 0 if contains_start and contains_end else 1
                    dist_penalty = np.hypot(dx, dy)
                    cx_label = 0.5 * (bbox.x0 + bbox.x1)
                    cy_label = 0.5 * (bbox.y0 + bbox.y1)
                    own_cx, own_cy = centers[row_idx]
                    own_dist = np.hypot(cx_label - own_cx, cy_label - own_cy)
                    own_edge_gap = max(0.0, own_dist - radii_px[row_idx])
                    target_gap = 4.0
                    own_gap_penalty = abs(own_edge_gap - target_gap)
                    other_dists = [
                        np.hypot(cx_label - ocx, cy_label - ocy)
                        for j, (ocx, ocy) in enumerate(centers)
                        if j != row_idx
                    ]
                    nearest_other = min(other_dists) if other_dists else np.inf
                    closer_to_other_penalty = max(0.0, own_dist - nearest_other)
                    score = (
                        4000 * outside
                        + 800 * label_overlap
                        + 220 * bubble_overlap
                        + 12 * own_gap_penalty
                        + 120 * closer_to_other_penalty
                        + 0.8 * dist_penalty
                    )

                    if best is None or score < best[0]:
                        best = (score, dx, dy, ha, va, bbox)

            _, dx, dy, ha, va, bbox = best
            final_txt = ax.annotate(
                str(label),
                xy=(float(row["x"]), float(row[metric])),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                fontfamily="serif",
                ha=ha,
                va=va,
                clip_on=True,
            )
            placed_bboxes.append(
                final_txt.get_window_extent(renderer=renderer).expanded(1.02, 1.08)
            )

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
    if title is not None:
        ax.set_title(title, fontfamily="serif", fontsize=18)

    for tick in ax.get_yticklabels():
        tick.set_fontfamily("serif")

    return ax


def plot_all_countries_bubble(
    df: pd.DataFrame,
    metric: str,
    title: str | None = None,
    alpha: float = 0.22,
    jitter: float = 0.18,
    ax: Any = None,
) -> Any:
    """Plot all-country tau bubbles with constant, small, transparent markers."""
    if metric not in df.columns:
        raise ValueError(f"metric column '{metric}' not in dataframe")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it in your .venv.") from exc

    work = df.copy()
    metric_values = pd.to_numeric(work[metric], errors="coerce")
    x_values = pd.to_numeric(work.get("x"), errors="coerce")
    bubble_sizes = pd.to_numeric(work.get("bubble_size", 18.0), errors="coerce").fillna(18.0)
    valid_mask = (
        np.isfinite(x_values.to_numpy(dtype=float))
        & np.isfinite(metric_values.to_numpy(dtype=float))
        & np.isfinite(bubble_sizes.to_numpy(dtype=float))
    )
    plot = work.loc[valid_mask].copy().reset_index(drop=True)
    if plot.empty:
        raise ValueError(f"No finite data available to plot for metric '{metric}'.")

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 7))

    if jitter > 0:
        stable_order = plot["country_code"].fillna("").astype(str).sort_values().unique().tolist()
        if len(stable_order) <= 1:
            offsets = {stable_order[0]: 0.0} if stable_order else {}
        else:
            offsets = {
                code: float(np.linspace(-jitter, jitter, len(stable_order))[i])
                for i, code in enumerate(stable_order)
            }
        plot["x_plot"] = (
            plot["x"].to_numpy(dtype=float)
            + plot["country_code"].map(offsets).fillna(0.0)
        )
    else:
        plot["x_plot"] = plot["x"].to_numpy(dtype=float)

    ax.scatter(
        plot["x_plot"],
        plot[metric],
        s=plot["bubble_size"],
        alpha=float(alpha),
        color="C0",
        linewidths=0,
    )

    bucket_centers = plot.groupby("sector_bucket", as_index=False)["x"].mean()
    ax.set_xticks(bucket_centers["x"])
    ax.set_xticklabels(bucket_centers["sector_bucket"], fontfamily="serif", fontsize=12)

    for bucket in ["Agriculture", "Manufacturing", "Services", "Other"]:
        sub = plot[plot["sector_bucket"] == bucket]
        if len(sub) == 0:
            continue
        ax.axvline(float(sub["x"].max()) + 0.5, linestyle="--", linewidth=1, alpha=0.45)

    metric_label = _metric_label(metric)
    ax.set_xlabel("")
    ax.set_ylabel(metric_label, fontfamily="serif", fontsize=14)
    if title is not None:
        ax.set_title(title, fontfamily="serif", fontsize=17)

    for tick in ax.get_yticklabels():
        tick.set_fontfamily("serif")

    return ax


def plot_all_countries_boxen(
    df: pd.DataFrame,
    metric: str,
    title: str | None = None,
    group_col: str = "sector_code",
    bucket_col: str = "sector_bucket",
    mark_buckets: bool = True,
    yscale: str = "linear",
    flip_axes: bool = False,
    metric_clip_quantiles: tuple[float, float] | None = None,
    metric_decimals: int = 6,
    weighted: bool = False,
    weight_col: str = "out_proxy",
    fit_median_regression: bool = False,
    report_r2: bool = True,
    regression_color: str = "#b22222",
    show_points: bool = True,
    point_alpha: float = 0.08,
    point_size: float = 6.0,
    ax: Any = None,
) -> Any:
    """Plot all-country metric distributions by group (boxen-style)."""
    if metric not in df.columns:
        raise ValueError(f"metric column '{metric}' not in dataframe")
    if group_col not in df.columns:
        raise ValueError(f"group column '{group_col}' not in dataframe")
    if weighted and weight_col not in df.columns:
        raise ValueError(f"weight column '{weight_col}' not in dataframe")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it in your .venv.") from exc

    work = df.copy()
    metric_values = pd.to_numeric(work[metric], errors="coerce")
    valid_mask = np.isfinite(metric_values.to_numpy(dtype=float))
    if yscale == "log":
        valid_mask &= (metric_values.to_numpy(dtype=float) > 0.0)
    valid_mask &= work[group_col].notna().to_numpy(dtype=bool)
    if weighted:
        weights = pd.to_numeric(work[weight_col], errors="coerce").to_numpy(dtype=float)
        valid_mask &= np.isfinite(weights) & (weights > 0.0)
    plot = work.loc[valid_mask].copy().reset_index(drop=True)
    if plot.empty:
        raise ValueError(f"No finite data available to plot for metric '{metric}'.")

    if "x_base" in plot.columns:
        order_df = (
            plot[[group_col, "x_base", bucket_col]]
            .drop_duplicates(subset=[group_col])
            .sort_values(["x_base", group_col])
        )
        order = order_df[group_col].astype(str).tolist()
    else:
        order = sorted(plot[group_col].astype(str).dropna().unique().tolist())
    if not order:
        raise ValueError(f"No recognized values available in '{group_col}' for plotting.")

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 7))

    used_seaborn = False
    median_positions: list[float] = []
    median_values: list[float] = []
    if not weighted:
        try:
            import seaborn as sns

            sns.boxenplot(
                data=plot,
                x=metric if flip_axes else group_col,
                y=group_col if flip_axes else metric,
                order=order,
                ax=ax,
                color="C0",
                linewidth=1.0,
                k_depth="proportion",
            )
            if show_points:
                sns.stripplot(
                    data=plot,
                    x=metric if flip_axes else group_col,
                    y=group_col if flip_axes else metric,
                    order=order,
                    ax=ax,
                    color="0.25",
                    alpha=float(point_alpha),
                    size=float(point_size),
                    jitter=0.24,
                )
            used_seaborn = True
        except ImportError:
            pass

    if weighted:
        stats = []
        for grp in order:
            sub = plot.loc[plot[group_col].astype(str) == grp]
            vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            w = pd.to_numeric(sub[weight_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
            vals = vals[mask]
            w = w[mask]
            if len(vals) == 0:
                continue
            q10 = _weighted_quantile(vals, w, 0.10)
            q25 = _weighted_quantile(vals, w, 0.25)
            q50 = _weighted_quantile(vals, w, 0.50)
            q75 = _weighted_quantile(vals, w, 0.75)
            q90 = _weighted_quantile(vals, w, 0.90)
            stats.append(
                {
                    "label": str(grp),
                    "whislo": float(q10),
                    "q1": float(q25),
                    "med": float(q50),
                    "q3": float(q75),
                    "whishi": float(q90),
                    "fliers": [],
                }
            )
            median_positions.append(float(len(stats) - 1))
            median_values.append(float(q50))

        if not stats:
            raise ValueError(
                f"No non-empty weighted groups available to plot for metric '{metric}'."
            )

        positions = np.arange(len(stats), dtype=float)
        ax.bxp(
            stats,
            positions=positions,
            widths=0.6,
            showfliers=False,
            vert=not flip_axes,
            patch_artist=True,
            boxprops={"facecolor": "#8eb8e5", "alpha": 0.65, "edgecolor": "C0"},
            medianprops={"color": "#1f2a44", "linewidth": 1.6},
            whiskerprops={"color": "C0", "linewidth": 1.0},
            capprops={"color": "C0", "linewidth": 1.0},
        )
        if flip_axes:
            ax.set_yticks(positions)
            ax.set_yticklabels([s["label"] for s in stats], fontfamily="serif", fontsize=12)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels([s["label"] for s in stats], fontfamily="serif", fontsize=12)

        if show_points:
            rng = np.random.default_rng(0)
            for i, grp in enumerate([s["label"] for s in stats]):
                vals = (
                    pd.to_numeric(
                        plot.loc[plot[group_col].astype(str) == grp, metric],
                        errors="coerce",
                    )
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if len(vals) == 0:
                    continue
                if flip_axes:
                    y = i + rng.uniform(-0.12, 0.12, size=len(vals))
                    ax.scatter(
                        vals,
                        y,
                        s=float(point_size) ** 2,
                        alpha=float(point_alpha),
                        color="0.2",
                        linewidths=0,
                        zorder=1,
                    )
                else:
                    x = i + rng.uniform(-0.12, 0.12, size=len(vals))
                    ax.scatter(
                        x,
                        vals,
                        s=float(point_size) ** 2,
                        alpha=float(point_alpha),
                        color="0.2",
                        linewidths=0,
                        zorder=1,
                    )
    elif not used_seaborn:
        grouped = [
            pd.to_numeric(plot.loc[plot[group_col].astype(str) == bucket, metric], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
            for bucket in order
        ]
        grouped = [g for g in grouped if len(g) > 0]
        if not grouped:
            raise ValueError(f"No non-empty groups available to plot for metric '{metric}'.")

        positions = np.arange(len(grouped), dtype=float)
        ax.boxplot(
            grouped,
            positions=positions,
            widths=0.6,
            vert=not flip_axes,
            patch_artist=True,
            boxprops={"facecolor": "#8eb8e5", "alpha": 0.65, "edgecolor": "C0"},
            medianprops={"color": "#1f2a44", "linewidth": 1.4},
            whiskerprops={"color": "C0", "linewidth": 1.0},
            capprops={"color": "C0", "linewidth": 1.0},
            flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.15, "markeredgewidth": 0},
        )
        if flip_axes:
            ax.set_yticks(positions)
            ax.set_yticklabels(order, fontfamily="serif", fontsize=12)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(order, fontfamily="serif", fontsize=12)

        if show_points:
            rng = np.random.default_rng(0)
            for i, grp in enumerate(order):
                vals = (
                    pd.to_numeric(
                        plot.loc[plot[group_col].astype(str) == grp, metric],
                        errors="coerce",
                    )
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if len(vals) == 0:
                    continue
                x = i + rng.uniform(-0.12, 0.12, size=len(vals))
                if flip_axes:
                    y = i + rng.uniform(-0.12, 0.12, size=len(vals))
                    ax.scatter(
                        vals,
                        y,
                        s=float(point_size) ** 2,
                        alpha=float(point_alpha),
                        color="0.2",
                        linewidths=0,
                        zorder=1,
                    )
                else:
                    ax.scatter(
                        x,
                        vals,
                        s=float(point_size) ** 2,
                        alpha=float(point_alpha),
                        color="0.2",
                        linewidths=0,
                        zorder=1,
                    )
        median_positions = [float(i) for i in range(len(order))]
        median_values = [
            float(
                pd.to_numeric(
                    plot.loc[plot[group_col].astype(str) == grp, metric],
                    errors="coerce",
                ).median()
            )
            for grp in order
        ]
    else:
        median_positions = [float(i) for i in range(len(order))]
        median_values = [
            float(
                pd.to_numeric(
                    plot.loc[plot[group_col].astype(str) == grp, metric],
                    errors="coerce",
                ).median()
            )
            for grp in order
        ]

    metric_label = _metric_label(metric)
    if flip_axes:
        ax.set_xlabel(metric_label, fontfamily="serif", fontsize=14)
        ax.set_ylabel("")
        ax.set_xscale(yscale)
        if group_col in {"sector_code", "sector_abbreviation"}:
            for tick in ax.get_yticklabels():
                tick.set_fontfamily("serif")
                tick.set_fontsize(9)
    else:
        ax.set_xlabel("")
        ax.set_ylabel(metric_label, fontfamily="serif", fontsize=14)
        ax.set_yscale(yscale)
        if group_col in {"sector_code", "sector_abbreviation"}:
            ax.tick_params(axis="x", labelrotation=90)
            for tick in ax.get_xticklabels():
                tick.set_fontfamily("serif")
                tick.set_fontsize(9)

    def _plain_number(x: float, _pos: int) -> str:
        text = f"{x:.{int(metric_decimals)}f}"
        text = text.rstrip("0").rstrip(".")
        return "0" if text in {"-0", ""} else text

    if flip_axes:
        ax.xaxis.set_major_formatter(FuncFormatter(_plain_number))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(_plain_number))

    if metric_clip_quantiles is not None:
        q_lo, q_hi = metric_clip_quantiles
        q_lo = float(q_lo)
        q_hi = float(q_hi)
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("metric_clip_quantiles must satisfy 0 <= q_lo < q_hi <= 1.")
        clip_vals = pd.to_numeric(plot[metric], errors="coerce").to_numpy(dtype=float)
        clip_vals = clip_vals[np.isfinite(clip_vals)]
        if yscale == "log":
            clip_vals = clip_vals[clip_vals > 0.0]
        if len(clip_vals) > 0:
            lo = float(np.quantile(clip_vals, q_lo))
            hi = float(np.quantile(clip_vals, q_hi))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                if flip_axes:
                    ax.set_xlim(lo, hi)
                else:
                    ax.set_ylim(lo, hi)

    if mark_buckets and bucket_col in plot.columns:
        order_df = (
            plot[[group_col, bucket_col] + (["x_base"] if "x_base" in plot.columns else [])]
            .dropna(subset=[group_col, bucket_col])
            .drop_duplicates(subset=[group_col])
        )
        if "x_base" in order_df.columns:
            order_df = order_df.sort_values(["x_base", group_col]).reset_index(drop=True)
        else:
            order_df[group_col] = order_df[group_col].astype(str)
            order_df = order_df.sort_values([group_col]).reset_index(drop=True)

        buckets = order_df[bucket_col].astype(str).tolist()
        start = 0
        segments: list[tuple[str, int, int]] = []
        for i in range(1, len(buckets) + 1):
            if i == len(buckets) or buckets[i] != buckets[i - 1]:
                segments.append((buckets[start], start, i - 1))
                start = i

        for _, _, end in segments[:-1]:
            if flip_axes:
                ax.axhline(float(end) + 0.5, linestyle=":", linewidth=1.0, alpha=0.8, color="0.35")
            else:
                ax.axvline(float(end) + 0.5, linestyle=":", linewidth=1.0, alpha=0.8, color="0.35")

    # Optional trendline along group medians and corresponding R^2.
    med_pos = np.asarray(median_positions, dtype=float)
    med_val = np.asarray(median_values, dtype=float)
    med_mask = np.isfinite(med_pos) & np.isfinite(med_val)
    med_pos = med_pos[med_mask]
    med_val = med_val[med_mask]
    r2 = float("nan")
    if fit_median_regression and len(med_pos) >= 2:
        if yscale == "log":
            transform_mask = med_val > 0
            med_pos_fit = med_pos[transform_mask]
            med_val_fit = med_val[transform_mask]
            if len(med_pos_fit) >= 2:
                y_fit = np.log10(med_val_fit)
                coefs = np.polyfit(med_pos_fit, y_fit, 1)
                y_hat = np.polyval(coefs, med_pos_fit)
                ss_res = float(np.sum((y_fit - y_hat) ** 2))
                ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                med_grid = np.linspace(med_pos_fit.min(), med_pos_fit.max(), 200)
                pred = 10.0 ** np.polyval(coefs, med_grid)
                if flip_axes:
                    ax.plot(pred, med_grid, color=regression_color, linewidth=1.8, zorder=4)
                else:
                    ax.plot(med_grid, pred, color=regression_color, linewidth=1.8, zorder=4)
        else:
            coefs = np.polyfit(med_pos, med_val, 1)
            y_hat = np.polyval(coefs, med_pos)
            ss_res = float(np.sum((med_val - y_hat) ** 2))
            ss_tot = float(np.sum((med_val - np.mean(med_val)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            med_grid = np.linspace(med_pos.min(), med_pos.max(), 200)
            pred = np.polyval(coefs, med_grid)
            if flip_axes:
                ax.plot(pred, med_grid, color=regression_color, linewidth=1.8, zorder=4)
            else:
                ax.plot(med_grid, pred, color=regression_color, linewidth=1.8, zorder=4)

    ax._median_regression_r2 = r2
    if report_r2 and np.isfinite(r2):
        ax.text(
            0.99,
            0.98,
            f"$R^2={r2:.3f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontfamily="serif",
            color=regression_color,
        )

    if bucket_colors:
        legend_order = ["Agriculture", "Manufacturing", "Services", "Other"]
        handles = [
            Line2D(
                [0],
                [0],
                color=bucket_colors[name],
                marker="o",
                linestyle="-",
                linewidth=whisker_linewidth,
                markersize=max(float(dot_size) ** 0.5 / 1.6, 5.0),
                markerfacecolor=bucket_colors[name],
                markeredgecolor="white",
                label=name,
            )
            for name in legend_order
            if name in bucket_colors
        ]
        if handles:
            ax.legend(handles=handles, frameon=False, loc="best", prop={"family": "serif", "size": 10})

    if title is not None:
        ax.set_title(title, fontfamily="serif", fontsize=17)

    for tick in ax.get_yticklabels():
        tick.set_fontfamily("serif")

    return ax


def plot_all_countries_dot_whisker(
    df: pd.DataFrame,
    metric: str,
    title: str | None = None,
    group_col: str = "sector_code",
    bucket_col: str = "sector_bucket",
    mark_buckets: bool = True,
    yscale: str = "linear",
    flip_axes: bool = False,
    metric_clip_quantiles: tuple[float, float] | None = None,
    metric_decimals: int = 6,
    weighted: bool = False,
    weight_col: str = "out_proxy",
    sort_by: str = "group",
    fit_median_regression: bool = False,
    report_r2: bool = True,
    regression_color: str = "#b22222",
    point_color: str = "#1f4e79",
    whisker_color: str = "#8fb7dd",
    bucket_colors: dict[str, str] | None = None,
    dot_size: float = 52.0,
    whisker_linewidth: float = 2.0,
    ax: Any = None,
) -> Any:
    """Plot group medians with interquantile whiskers for all-country metric data."""
    if metric not in df.columns:
        raise ValueError(f"metric column '{metric}' not in dataframe")
    if group_col not in df.columns:
        raise ValueError(f"group column '{group_col}' not in dataframe")
    if weighted and weight_col not in df.columns:
        raise ValueError(f"weight column '{weight_col}' not in dataframe")
    if sort_by not in {"group", "median"}:
        raise ValueError("sort_by must be one of {'group', 'median'}")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it in your .venv.") from exc

    work = df.copy()
    metric_values = pd.to_numeric(work[metric], errors="coerce")
    valid_mask = np.isfinite(metric_values.to_numpy(dtype=float))
    if yscale == "log":
        valid_mask &= (metric_values.to_numpy(dtype=float) > 0.0)
    valid_mask &= work[group_col].notna().to_numpy(dtype=bool)
    if weighted:
        weights = pd.to_numeric(work[weight_col], errors="coerce").to_numpy(dtype=float)
        valid_mask &= np.isfinite(weights) & (weights > 0.0)
    plot = work.loc[valid_mask].copy().reset_index(drop=True)
    if plot.empty:
        raise ValueError(f"No finite data available to plot for metric '{metric}'.")

    stats = []
    for grp, sub in plot.groupby(group_col, sort=False):
        vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        bucket = (
            sub[bucket_col].dropna().astype(str).iloc[0]
            if bucket_col in sub.columns and sub[bucket_col].notna().any()
            else ""
        )
        x_base = (
            float(pd.to_numeric(sub["x_base"], errors="coerce").dropna().iloc[0])
            if "x_base" in sub.columns and pd.to_numeric(sub["x_base"], errors="coerce").notna().any()
            else float("nan")
        )
        if weighted:
            w = pd.to_numeric(sub[weight_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
            vals = vals[mask]
            w = w[mask]
            if len(vals) == 0:
                continue
            q10 = _weighted_quantile(vals, w, 0.10)
            q25 = _weighted_quantile(vals, w, 0.25)
            q50 = _weighted_quantile(vals, w, 0.50)
            q75 = _weighted_quantile(vals, w, 0.75)
            q90 = _weighted_quantile(vals, w, 0.90)
        else:
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            q10, q25, q50, q75, q90 = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])

        stats.append(
            {
                "label": str(grp),
                "bucket": bucket,
                "x_base": x_base,
                "q10": float(q10),
                "q25": float(q25),
                "q50": float(q50),
                "q75": float(q75),
                "q90": float(q90),
            }
        )

    if not stats:
        raise ValueError(f"No non-empty groups available to plot for metric '{metric}'.")

    stats_df = pd.DataFrame(stats)
    if sort_by == "median":
        stats_df = stats_df.sort_values(["q50", "label"], ascending=[False, True]).reset_index(drop=True)
    elif stats_df["x_base"].notna().any():
        stats_df = stats_df.sort_values(["x_base", "label"]).reset_index(drop=True)
    else:
        stats_df = stats_df.sort_values(["label"]).reset_index(drop=True)

    stats_df["position"] = np.arange(len(stats_df), dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 7))

    for row in stats_df.itertuples(index=False):
        pos = float(row.position)
        row_color = point_color
        if bucket_colors is not None:
            row_color = bucket_colors.get(str(row.bucket), row_color)
        if flip_axes:
            ax.hlines(pos, row.q25, row.q75, color=row_color, linewidth=whisker_linewidth, zorder=3)
            ax.scatter(row.q50, pos, s=dot_size, color=row_color, edgecolors="white", linewidths=0.8, zorder=4)
        else:
            ax.vlines(pos, row.q25, row.q75, color=row_color, linewidth=whisker_linewidth, zorder=3)
            ax.scatter(pos, row.q50, s=dot_size, color=row_color, edgecolors="white", linewidths=0.8, zorder=4)

    median_positions = stats_df["position"].to_numpy(dtype=float)
    median_values = stats_df["q50"].to_numpy(dtype=float)

    if flip_axes:
        ax.set_yticks(stats_df["position"])
        ax.set_yticklabels(stats_df["label"], fontfamily="serif", fontsize=11)
    else:
        ax.set_xticks(stats_df["position"])
        ax.set_xticklabels(stats_df["label"], fontfamily="serif", fontsize=11)

    metric_label = _metric_label(metric)
    if flip_axes:
        ax.set_xlabel(metric_label, fontfamily="serif", fontsize=14)
        ax.set_ylabel("")
        ax.set_xscale(yscale)
        if group_col in {"sector_code", "sector_abbreviation"}:
            for tick in ax.get_yticklabels():
                tick.set_fontfamily("serif")
                tick.set_fontsize(9)
    else:
        ax.set_xlabel("")
        ax.set_ylabel(metric_label, fontfamily="serif", fontsize=14)
        ax.set_yscale(yscale)
        if group_col in {"sector_code", "sector_abbreviation"}:
            ax.tick_params(axis="x", labelrotation=90)
            for tick in ax.get_xticklabels():
                tick.set_fontfamily("serif")
                tick.set_fontsize(9)

    def _plain_number(x: float, _pos: int) -> str:
        text = f"{x:.{int(metric_decimals)}f}"
        text = text.rstrip("0").rstrip(".")
        return "0" if text in {"-0", ""} else text

    if flip_axes:
        ax.xaxis.set_major_formatter(FuncFormatter(_plain_number))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(_plain_number))

    if metric_clip_quantiles is not None:
        q_lo, q_hi = metric_clip_quantiles
        q_lo = float(q_lo)
        q_hi = float(q_hi)
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("metric_clip_quantiles must satisfy 0 <= q_lo < q_hi <= 1.")
        clip_vals = pd.to_numeric(plot[metric], errors="coerce").to_numpy(dtype=float)
        clip_vals = clip_vals[np.isfinite(clip_vals)]
        if yscale == "log":
            clip_vals = clip_vals[clip_vals > 0.0]
        if len(clip_vals) > 0:
            lo = float(np.quantile(clip_vals, q_lo))
            hi = float(np.quantile(clip_vals, q_hi))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                if flip_axes:
                    ax.set_xlim(lo, hi)
                else:
                    ax.set_ylim(lo, hi)

    if mark_buckets and "bucket" in stats_df.columns:
        buckets = stats_df["bucket"].fillna("").astype(str).tolist()
        start = 0
        segments: list[tuple[str, int, int]] = []
        for i in range(1, len(buckets) + 1):
            if i == len(buckets) or buckets[i] != buckets[i - 1]:
                segments.append((buckets[start], start, i - 1))
                start = i

        for _, _, end in segments[:-1]:
            if flip_axes:
                ax.axhline(float(end) + 0.5, linestyle=":", linewidth=1.0, alpha=0.8, color="0.35")
            else:
                ax.axvline(float(end) + 0.5, linestyle=":", linewidth=1.0, alpha=0.8, color="0.35")

    r2 = float("nan")
    med_pos = np.asarray(median_positions, dtype=float)
    med_val = np.asarray(median_values, dtype=float)
    med_mask = np.isfinite(med_pos) & np.isfinite(med_val)
    med_pos = med_pos[med_mask]
    med_val = med_val[med_mask]
    if fit_median_regression and len(med_pos) >= 2:
        if yscale == "log":
            transform_mask = med_val > 0
            med_pos_fit = med_pos[transform_mask]
            med_val_fit = med_val[transform_mask]
            if len(med_pos_fit) >= 2:
                y_fit = np.log10(med_val_fit)
                coefs = np.polyfit(med_pos_fit, y_fit, 1)
                y_hat = np.polyval(coefs, med_pos_fit)
                ss_res = float(np.sum((y_fit - y_hat) ** 2))
                ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                med_grid = np.linspace(med_pos_fit.min(), med_pos_fit.max(), 200)
                pred = 10.0 ** np.polyval(coefs, med_grid)
                if flip_axes:
                    ax.plot(pred, med_grid, color=regression_color, linewidth=1.8, zorder=4)
                else:
                    ax.plot(med_grid, pred, color=regression_color, linewidth=1.8, zorder=4)
        else:
            coefs = np.polyfit(med_pos, med_val, 1)
            y_hat = np.polyval(coefs, med_pos)
            ss_res = float(np.sum((med_val - y_hat) ** 2))
            ss_tot = float(np.sum((med_val - np.mean(med_val)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            med_grid = np.linspace(med_pos.min(), med_pos.max(), 200)
            pred = np.polyval(coefs, med_grid)
            if flip_axes:
                ax.plot(pred, med_grid, color=regression_color, linewidth=1.8, zorder=4)
            else:
                ax.plot(med_grid, pred, color=regression_color, linewidth=1.8, zorder=4)

    ax._median_regression_r2 = r2
    ax._dot_whisker_stats = stats_df.copy()
    if report_r2 and np.isfinite(r2):
        ax.text(
            0.99,
            0.98,
            f"$R^2={r2:.3f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontfamily="serif",
            color=regression_color,
        )

    if bucket_colors:
        legend_order = ["Agriculture", "Manufacturing", "Services", "Other"]
        handles = [
            Line2D(
                [0],
                [0],
                color=bucket_colors[name],
                marker="o",
                linestyle="-",
                linewidth=whisker_linewidth,
                markersize=max(float(dot_size) ** 0.5 / 1.6, 5.0),
                markerfacecolor=bucket_colors[name],
                markeredgecolor="white",
                label=name,
            )
            for name in legend_order
            if name in bucket_colors
        ]
        if handles:
            ax.legend(
                handles=handles,
                frameon=False,
                loc="upper right",
                prop={"family": "serif", "size": 10},
            )

    if title is not None:
        ax.set_title(title, fontfamily="serif", fontsize=17)

    for tick in ax.get_yticklabels():
        tick.set_fontfamily("serif")

    return ax
