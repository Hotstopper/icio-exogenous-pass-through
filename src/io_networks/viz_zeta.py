from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from io_networks.paths import resolve_paths

_ALLOWED_METRICS = {"zeta", "zeta_lag"}


def _metric_label(metric: str) -> str:
    labels = {
        "zeta": r"$\zeta$",
        "zeta_lag": r"Lagged $\zeta$",
    }
    return labels.get(metric, metric)


def _resolve_variant(cfg: dict[str, Any], variant: str | None) -> str:
    if variant:
        return variant
    return "extended" if cfg["icio"].get("extended", False) else "regular"


def load_zeta_data(
    cfg: dict[str, Any],
    variant: str | None = None,
) -> pd.DataFrame:
    """Load country-year zeta table produced by build-zeta."""
    use_variant = _resolve_variant(cfg, variant)
    paths = resolve_paths(cfg)
    zeta_path = paths["processed"] / "zeta" / use_variant / "zeta_by_country_year.parquet"
    if not zeta_path.exists():
        raise FileNotFoundError(f"Missing zeta dataset: {zeta_path}. Run build-zeta first.")
    return pd.read_parquet(zeta_path)


def _prepare_zeta_frame(
    df: pd.DataFrame,
    only_ok: bool = True,
) -> pd.DataFrame:
    required = {"country", "year", "zeta"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for zeta plotting: {missing}")

    work = df.copy()
    work["country"] = work["country"].astype(str).str.upper()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["zeta"] = pd.to_numeric(work["zeta"], errors="coerce")

    if only_ok and "status" in work.columns:
        work = work[work["status"] == "ok"].copy()

    work = work.dropna(subset=["country", "year", "zeta"]).copy()
    work = work[np.isfinite(work["year"]) & np.isfinite(work["zeta"])].copy()
    work = work.sort_values(["country", "year"]).reset_index(drop=True)

    work["zeta_lag"] = work.groupby("country")["zeta"].shift(1)
    return work


def select_phase_map_countries(
    df: pd.DataFrame,
    base_countries: Iterable[str] = ("USA", "CHN"),
    only_ok: bool = True,
) -> list[str]:
    """Return placeholder countries using country-median zeta levels."""
    work = _prepare_zeta_frame(df=df, only_ok=only_ok)
    if work.empty:
        raise ValueError("No finite rows available for zeta country selection.")

    country_zeta = (
        work.groupby("country", as_index=False)["zeta"]
        .median()
        .sort_values("country")
        .reset_index(drop=True)
    )
    if country_zeta.empty:
        raise ValueError("No country-level zeta values available for selection.")

    available = set(country_zeta["country"].tolist())
    chosen: list[str] = []
    for country in base_countries:
        country_u = str(country).upper()
        if country_u in available and country_u not in chosen:
            chosen.append(country_u)

    pool = country_zeta[~country_zeta["country"].isin(chosen)].copy()
    if pool.empty:
        return chosen

    def _pick_nearest(target: float) -> str | None:
        if pool.empty:
            return None
        idx = (pool["zeta"] - target).abs().idxmin()
        return str(pool.loc[idx, "country"])

    q25 = float(pool["zeta"].quantile(0.25))
    q50 = float(pool["zeta"].quantile(0.50))
    q75 = float(pool["zeta"].quantile(0.75))

    for target in (q50, q25, q75):
        picked = _pick_nearest(target)
        if picked is not None and picked not in chosen:
            chosen.append(picked)
            pool = pool[pool["country"] != picked].copy()

    return chosen[:5]


def plot_zeta_country_lines(
    df: pd.DataFrame,
    metric: str = "zeta",
    highlight_countries: Iterable[str] | None = None,
    exclude_countries: Iterable[str] | None = None,
    only_ok: bool = True,
    title: str | None = None,
    yscale: str = "linear",
    symlog_linthresh: float = 1e-3,
    alpha_bg: float = 0.18,
    lw_bg: float = 0.8,
    lw_hi: float = 2.6,
    use_end_labels: bool = True,
    show_legend: bool = False,
    label_pad_years: float = 1.2,
    ax: Any = None,
) -> Any:
    """Plot country zeta lines with gray background and highlighted countries."""
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {sorted(_ALLOWED_METRICS)}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it in your .venv.") from exc

    work = _prepare_zeta_frame(df=df, only_ok=only_ok)
    if metric == "zeta_lag":
        work = work.dropna(subset=["zeta_lag"]).copy()
    if work.empty:
        raise ValueError("No data available to plot after filtering.")
    if yscale not in {"linear", "log", "symlog"}:
        raise ValueError("yscale must be one of: linear, log, symlog")
    if yscale == "log" and (work[metric] <= 0).any():
        raise ValueError(f"Cannot use log scale with non-positive '{metric}' values.")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 7))

    ex_set = set()
    if exclude_countries:
        ex_set = {str(country).upper() for country in exclude_countries}
        work = work[~work["country"].isin(ex_set)].copy()
        if work.empty:
            raise ValueError("No data available after applying exclude_countries.")

    hi_order = []
    if highlight_countries:
        seen = set()
        for country in highlight_countries:
            country_u = str(country).upper()
            if country_u in ex_set or country_u in seen:
                continue
            hi_order.append(country_u)
            seen.add(country_u)
    hi_set = set(hi_order)

    countries = sorted(work["country"].unique())
    bg_countries = [country for country in countries if country not in hi_set]

    for country in bg_countries:
        sub = work[work["country"] == country].sort_values("year")
        ax.plot(
            sub["year"].to_numpy(dtype=float),
            sub[metric].to_numpy(dtype=float),
            color="0.5",
            alpha=alpha_bg,
            linewidth=lw_bg,
            zorder=1,
        )

    if hi_order:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
            "color",
            ["C0", "C1", "C2", "C3", "C4"],
        )
        hi_data = work[work["country"].isin(hi_set)]
        hi_present = sorted(hi_data["country"].unique(), key=hi_order.index)
        hi_endpoints: list[tuple[str, float, float, str]] = []
        for i, country in enumerate(hi_present):
            sub = hi_data[hi_data["country"] == country].sort_values("year")
            color = cycle[i % len(cycle)]
            ax.plot(
                sub["year"].to_numpy(dtype=float),
                sub[metric].to_numpy(dtype=float),
                color=color,
                linewidth=lw_hi,
                zorder=3,
                label=country,
            )
            hi_endpoints.append(
                (
                    country,
                    float(sub["year"].iloc[-1]),
                    float(sub[metric].iloc[-1]),
                    color,
                )
            )
        if hi_present and show_legend:
            legend = ax.legend(frameon=False, ncol=min(4, len(hi_present)))
            for txt in legend.get_texts():
                txt.set_fontfamily("serif")
        if hi_endpoints and use_end_labels:
            span = float(work[metric].max() - work[metric].min())
            min_gap = 0.02 * span if span > 0 else 0.0
            ordered = sorted(hi_endpoints, key=lambda row: row[2])
            adjusted: list[tuple[str, float, float, str]] = []
            prev_y: float | None = None
            for country, x_last, y_last, color in ordered:
                y_adj = y_last if prev_y is None else max(y_last, prev_y + min_gap)
                adjusted.append((country, x_last, y_adj, color))
                prev_y = y_adj
            x_last_global = float(work["year"].max())
            x_label = x_last_global + 0.35 * label_pad_years
            for country, x_last, y_last, color in adjusted:
                ax.text(
                    x_label,
                    y_last,
                    country,
                    va="center",
                    ha="left",
                    color=color,
                    fontsize=10,
                    fontfamily="serif",
                    clip_on=False,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.75,
                        "pad": 0.15,
                    },
                    zorder=4,
                )
                ax.plot(
                    [x_last, x_label - 0.03 * label_pad_years],
                    [y_last, y_last],
                    color=color,
                    linewidth=0.8,
                    alpha=0.8,
                    zorder=3,
                )

    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.8)
    ax.set_xlabel("Year", fontfamily="serif")
    ax.set_ylabel(_metric_label(metric), fontfamily="serif")
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh)
    else:
        ax.set_yscale(yscale)
    if title:
        ax.set_title(title, fontfamily="serif")
    else:
        ax.set_title(f"{metric} by country over time", fontfamily="serif")

    for tick in ax.get_xticklabels():
        tick.set_fontfamily("serif")
    for tick in ax.get_yticklabels():
        tick.set_fontfamily("serif")

    years = pd.to_numeric(work["year"], errors="coerce")
    if years.notna().any():
        xmin = float(np.nanmin(years.to_numpy(dtype=float)))
        xmax = float(np.nanmax(years.to_numpy(dtype=float)))
        ax.set_xlim(xmin, xmax + label_pad_years)

    return ax


def plot_zeta_phase_map(
    df: pd.DataFrame,
    countries: Iterable[str],
    only_ok: bool = True,
    title: str | None = None,
    cmap: str = "viridis",
    lw: float = 2.6,
    point_size: float = 26.0,
    show_end_labels: bool = True,
    end_label_dx_pts: float = 10.0,
    end_label_dy_pts: float = 8.0,
    end_label_bbox_alpha: float = 0.8,
    show_background: bool = True,
    bg_color: str = "0.55",
    bg_alpha: float = 0.12,
    bg_lw: float = 0.9,
    show_flow_arrows: bool = False,
    flow_arrow_step: int = 4,
    flow_arrow_lw: float = 1.0,
    flow_arrow_scale: float = 11.0,
    ax: Any = None,
) -> Any:
    """Plot time-colored trajectories with x=lagged zeta and y=current zeta."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it in your .venv.") from exc

    work = _prepare_zeta_frame(df=df, only_ok=only_ok)
    work = work.dropna(subset=["zeta_lag"]).copy()
    if work.empty:
        raise ValueError("No finite rows available to plot zeta phase map.")

    wanted = []
    seen = set()
    for country in countries:
        country_u = str(country).upper()
        if country_u not in seen:
            wanted.append(country_u)
            seen.add(country_u)
    if not wanted:
        raise ValueError("Provide at least one country for zeta phase map.")

    plot = work[work["country"].isin(wanted)].copy()
    if plot.empty:
        raise ValueError("None of the requested countries are available in the filtered data.")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    years_all = plot["year"].to_numpy(dtype=float)
    norm = Normalize(vmin=float(np.nanmin(years_all)), vmax=float(np.nanmax(years_all)))
    cmap_obj = cmap if hasattr(cmap, "__call__") else plt.get_cmap(cmap)

    if show_background:
        bg = work[~work["country"].isin(wanted)].copy()
        for country in sorted(bg["country"].unique()):
            sub = bg[bg["country"] == country].sort_values("year")
            if len(sub) < 2:
                continue
            ax.plot(
                sub["zeta_lag"].to_numpy(dtype=float),
                sub["zeta"].to_numpy(dtype=float),
                color=bg_color,
                alpha=bg_alpha,
                linewidth=bg_lw,
                zorder=1,
            )

    for country in wanted:
        sub = plot[plot["country"] == country].sort_values("year").copy()
        if len(sub) == 0:
            continue

        x = sub["zeta_lag"].to_numpy(dtype=float)
        y = sub["zeta"].to_numpy(dtype=float)
        years = sub["year"].to_numpy(dtype=float)

        if len(sub) >= 2:
            pts = np.column_stack([x, y])
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap_obj, norm=norm, linewidths=lw, alpha=0.95, zorder=3)
            lc.set_array(years[1:])
            ax.add_collection(lc)

        ax.scatter(
            x,
            y,
            c=years,
            cmap=cmap_obj,
            norm=norm,
            s=point_size,
            edgecolors="none",
            alpha=0.95,
            zorder=4,
        )

        if show_flow_arrows and len(sub) >= 2:
            step = max(1, int(flow_arrow_step))
            for i in range(step, len(sub), step):
                x0, y0 = float(x[i - 1]), float(y[i - 1])
                x1, y1 = float(x[i]), float(y[i])
                if not (
                    np.isfinite(x0)
                    and np.isfinite(y0)
                    and np.isfinite(x1)
                    and np.isfinite(y1)
                ):
                    continue
                if x0 == x1 and y0 == y1:
                    continue
                color = cmap_obj(norm(float(years[i])))
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "lw": flow_arrow_lw,
                        "mutation_scale": flow_arrow_scale,
                        "alpha": 0.9,
                    },
                    zorder=5,
                )

        if show_end_labels:
            dy = end_label_dy_pts
            if len(sub) >= 2 and float(y[-1] - y[-2]) < 0:
                dy = -end_label_dy_pts
            ax.annotate(
                country,
                xy=(float(x[-1]), float(y[-1])),
                xytext=(end_label_dx_pts, dy),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=15,
                fontfamily="serif",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": end_label_bbox_alpha,
                    "pad": 0.2,
                },
                zorder=6,
            )

    ax.autoscale()
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_xlabel(r"Lagged $\zeta$", fontfamily="serif")
    ax.set_ylabel(_metric_label("zeta"), fontfamily="serif")
    if title:
        ax.set_title(title, fontfamily="serif")
    else:
        ax.set_title(r"Phase Map: $\zeta_t$ vs $\zeta_{t-1}$", fontfamily="serif")

    for tick in ax.get_xticklabels():
        tick.set_fontfamily("serif")
    for tick in ax.get_yticklabels():
        tick.set_fontfamily("serif")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Year", fontfamily="serif")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily("serif")

    return ax
