from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from io_networks.paths import resolve_paths

_ALLOWED_METRICS = {"tau", "tau_dir", "tau_amp", "tau_amp2"}


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
        "tau_amp2": r"$\tau$ with $\tau_{amp}$ tails",
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
    if metric == "tau_amp2":
        required = {"tau", "tau_amp"}
        missing = sorted(required.difference(set(arr.files)))
        if missing:
            raise ValueError(
                f"Metric mode tau_amp2 requires arrays {sorted(required)} in {blocks_npz.name}. "
                f"Missing: {missing}. Available: {arr.files}"
            )
        tau = arr["tau"]
        tau_amp_tail = arr["tau_amp"]
    else:
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
    if metric == "tau_amp2" and len(labels_n) != len(tau_amp_tail):
        raise ValueError(
            f"Mismatch between labels ({len(labels_n)}) and tau_amp length ({len(tau_amp_tail)}) for year {year}."
        )

    df = pd.DataFrame(
        {
            "label": labels_n,
            metric: tau,
        }
    )
    if metric == "tau_amp2":
        df["tau_amp_tail"] = tau_amp_tail
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
                    outside = 0 if axes_bbox.contains(*bbox.get_points()[0]) and axes_bbox.contains(*bbox.get_points()[1]) else 1
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
            placed_bboxes.append(final_txt.get_window_extent(renderer=renderer).expanded(1.02, 1.08))

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
