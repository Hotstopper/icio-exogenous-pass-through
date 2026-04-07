from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from io_networks.paths import resolve_paths

_ALLOWED_METRICS = {
    "xi",
    "xi_dir",
    "xi_amp_1",
    "xi_amp_2",
    "xi_amp",
    "share",
    "share_amp_1",
    "share_amp_2",
}
_ALLOWED_RATIO_MODES = {"amplification", "share"}
_ALLOWED_SHARE_METRICS = {"share", "share_amp_1", "share_amp_2"}


def _metric_label(metric: str) -> str:
    labels = {
        "xi": r"$\xi$",
        "xi_dir": r"$\xi_{dir}$",
        "xi_amp_1": r"$\xi_{amp,1}$",
        "xi_amp_2": r"$\xi_{amp,2}$",
        "xi_amp": r"$\xi_{amp}$",
        "share": r"$\xi_{amp} / \xi$",
        "share_amp_1": r"$\xi_{amp,1} / \xi$",
        "share_amp_2": r"$\xi_{amp,2} / \xi$",
    }
    return labels.get(metric, metric)


def _resolve_variant(cfg: dict[str, Any], variant: str | None) -> str:
    if variant:
        return variant
    return "extended" if cfg["icio"].get("extended", False) else "regular"


def load_xi_data(
    cfg: dict[str, Any],
    variant: str | None = None,
) -> pd.DataFrame:
    """Load country-year xi table produced by build-xi."""
    use_variant = _resolve_variant(cfg, variant)
    paths = resolve_paths(cfg)
    xi_path = paths["processed"] / "xi" / use_variant / "xi_by_country_year.parquet"
    if not xi_path.exists():
        raise FileNotFoundError(f"Missing xi dataset: {xi_path}. Run build-xi first.")
    return pd.read_parquet(xi_path)


def _resolve_share_metric(ratio_mode: str, share_metric: str | None) -> str | None:
    if ratio_mode not in _ALLOWED_RATIO_MODES:
        raise ValueError(f"ratio_mode must be one of {sorted(_ALLOWED_RATIO_MODES)}")
    if ratio_mode != "share":
        return None
    metric = "share" if share_metric is None else str(share_metric)
    if metric not in _ALLOWED_SHARE_METRICS:
        raise ValueError(f"share_metric must be one of {sorted(_ALLOWED_SHARE_METRICS)}")
    return metric


def _ratio_label(ratio_mode: str, share_metric: str | None = None) -> str:
    if ratio_mode == "amplification":
        return r"$\xi / \xi_{dir}$"
    resolved_share_metric = _resolve_share_metric(ratio_mode, share_metric)
    return _metric_label(resolved_share_metric or ratio_mode)


def _prepare_xi_ratio_frame(
    df: pd.DataFrame,
    ratio_mode: str,
    share_metric: str | None = None,
    only_ok: bool = True,
) -> pd.DataFrame:
    resolved_share_metric = _resolve_share_metric(ratio_mode, share_metric)
    required = {"country", "year", "xi", "xi_dir"}
    if ratio_mode == "share":
        if resolved_share_metric == "share":
            required.add("xi_amp")
        elif resolved_share_metric == "share_amp_1":
            required.add("xi_amp_1")
        elif resolved_share_metric == "share_amp_2":
            required.add("xi_amp_2")
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for ratio computation: {missing}")

    work = df.copy()
    work["country"] = work["country"].astype(str).str.upper()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["xi"] = pd.to_numeric(work["xi"], errors="coerce")
    work["xi_dir"] = pd.to_numeric(work["xi_dir"], errors="coerce")
    if "xi_amp" in work.columns:
        work["xi_amp"] = pd.to_numeric(work["xi_amp"], errors="coerce")
    if "xi_amp_1" in work.columns:
        work["xi_amp_1"] = pd.to_numeric(work["xi_amp_1"], errors="coerce")
    if "xi_amp_2" in work.columns:
        work["xi_amp_2"] = pd.to_numeric(work["xi_amp_2"], errors="coerce")

    if only_ok and "status" in work.columns:
        work = work[work["status"] == "ok"].copy()

    if ratio_mode == "amplification":
        denom = work["xi_dir"].replace(0.0, np.nan)
        work["ratio"] = work["xi"] / denom
    else:
        denom = work["xi"].replace(0.0, np.nan)
        numerator_col = {
            "share": "xi_amp",
            "share_amp_1": "xi_amp_1",
            "share_amp_2": "xi_amp_2",
        }[resolved_share_metric or "share"]
        work["ratio"] = work[numerator_col] / denom

    work = work.dropna(subset=["country", "year", "xi", "ratio"]).copy()
    work = work[np.isfinite(work["xi"]) & np.isfinite(work["ratio"])].copy()
    return work


def select_phase_map_countries(
    df: pd.DataFrame,
    ratio_mode: str = "share",
    share_metric: str | None = None,
    base_countries: Iterable[str] = ("USA", "CHN"),
    only_ok: bool = True,
) -> list[str]:
    """Return placeholder countries: USA, CHN, median, q25, q75 by country-level ratio."""
    work = _prepare_xi_ratio_frame(
        df=df,
        ratio_mode=ratio_mode,
        share_metric=share_metric,
        only_ok=only_ok,
    )
    if work.empty:
        raise ValueError("No finite rows available for phase-map country selection.")

    country_ratio = (
        work.groupby("country", as_index=False)["ratio"]
        .median()
        .sort_values("country")
        .reset_index(drop=True)
    )
    if country_ratio.empty:
        raise ValueError("No country-level ratios available for selection.")

    available = set(country_ratio["country"].tolist())
    chosen: list[str] = []
    for c in base_countries:
        cu = str(c).upper()
        if cu in available and cu not in chosen:
            chosen.append(cu)

    pool = country_ratio[~country_ratio["country"].isin(chosen)].copy()
    if pool.empty:
        return chosen

    def _pick_nearest(target: float) -> str | None:
        if pool.empty:
            return None
        idx = (pool["ratio"] - target).abs().idxmin()
        return str(pool.loc[idx, "country"])

    q25 = float(pool["ratio"].quantile(0.25))
    q50 = float(pool["ratio"].quantile(0.50))
    q75 = float(pool["ratio"].quantile(0.75))

    for target in (q50, q25, q75):
        picked = _pick_nearest(target)
        if picked is not None and picked not in chosen:
            chosen.append(picked)
            pool = pool[pool["country"] != picked].copy()

    return chosen[:5]


def plot_xi_phase_map(
    df: pd.DataFrame,
    countries: Iterable[str],
    ratio_mode: str = "share",
    share_metric: str | None = None,
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
    """Plot time-colored country trajectories with x=xi and y=selected ratio."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it in your .venv.") from exc

    work = _prepare_xi_ratio_frame(
        df=df,
        ratio_mode=ratio_mode,
        share_metric=share_metric,
        only_ok=only_ok,
    )
    if work.empty:
        raise ValueError("No finite rows available to plot phase map.")

    wanted = []
    seen = set()
    for c in countries:
        cu = str(c).upper()
        if cu not in seen:
            wanted.append(cu)
            seen.add(cu)
    if not wanted:
        raise ValueError("Provide at least one country for phase map.")

    plot = work[work["country"].isin(wanted)].copy()
    if plot.empty:
        raise ValueError("None of the requested countries are available in the filtered data.")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    years_all = plot["year"].to_numpy(dtype=float)
    norm = Normalize(vmin=float(np.nanmin(years_all)), vmax=float(np.nanmax(years_all)))
    if hasattr(cmap, "__call__"):
        cmap_obj = cmap
    else:
        cmap_obj = plt.get_cmap(cmap)

    if show_background:
        bg = work[~work["country"].isin(wanted)].copy()
        for country in sorted(bg["country"].unique()):
            sub = bg[bg["country"] == country].sort_values("year")
            if len(sub) < 2:
                continue
            ax.plot(
                sub["xi"].to_numpy(dtype=float),
                sub["ratio"].to_numpy(dtype=float),
                color=bg_color,
                alpha=bg_alpha,
                linewidth=bg_lw,
                zorder=1,
            )

    for country in wanted:
        sub = plot[plot["country"] == country].sort_values("year").copy()
        if len(sub) == 0:
            continue

        x = sub["xi"].to_numpy(dtype=float)
        y = sub["ratio"].to_numpy(dtype=float)
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
                if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
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
    ax.set_xlabel(_metric_label("xi"), fontfamily="serif")
    ax.set_ylabel(_ratio_label(ratio_mode, share_metric=share_metric), fontfamily="serif")
    if title:
        ax.set_title(title, fontfamily="serif")
    else:
        ax.set_title(
            rf"Phase Map: {_ratio_label(ratio_mode, share_metric=share_metric)} vs $\xi$",
            fontfamily="serif",
        )

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


def plot_xi_composition_map(
    df: pd.DataFrame,
    year: int,
    ratio_mode: str = "share",
    share_metric: str | None = None,
    highlight_countries: Iterable[str] | None = None,
    country_groups: dict[str, Iterable[str]] | None = None,
    exclude_countries: Iterable[str] | None = None,
    only_ok: bool = True,
    title: str | None = None,
    point_size_bg: float = 26.0,
    point_size_hi: float = 44.0,
    alpha_bg: float = 0.4,
    alpha_hi: float = 0.95,
    color_bg: str = "0.65",
    show_legend: bool = True,
    highlight_colors: dict[str, str] | None = None,
    highlight_label_fontsize: float = 12.0,
    label_dx_pts: float = 7.0,
    label_dy_pts: float = 7.0,
    label_bbox_alpha: float = 0.82,
    ax: Any = None,
) -> Any:
    """Plot a single-year cross-country scatter with x=xi and y=selected ratio."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it in your .venv.") from exc

    work = _prepare_xi_ratio_frame(
        df=df,
        ratio_mode=ratio_mode,
        share_metric=share_metric,
        only_ok=only_ok,
    )
    if work.empty:
        raise ValueError("No finite rows available to plot composition map.")

    year_value = int(year)
    plot = work[work["year"] == year_value].copy()
    if plot.empty:
        years = sorted(int(y) for y in work["year"].dropna().unique())
        raise ValueError(f"No rows available for year={year_value}. Available years: {years}")

    ex_set = set()
    if exclude_countries:
        ex_set = {str(c).upper() for c in exclude_countries}
        plot = plot[~plot["country"].isin(ex_set)].copy()
        if plot.empty:
            raise ValueError("No rows available after applying exclude_countries.")

    hi_order: list[str] = []
    if highlight_countries:
        seen = set()
        for c in highlight_countries:
            cu = str(c).upper()
            if cu in ex_set or cu in seen:
                continue
            hi_order.append(cu)
            seen.add(cu)

    hi_set = set(hi_order)
    bg = plot[~plot["country"].isin(hi_set)].copy()
    hi = plot[plot["country"].isin(hi_set)].copy()

    group_map: dict[str, str] = {}
    group_order: list[str] = []
    if country_groups:
        seen_groups = set()
        for group_name, members in country_groups.items():
            group_key = str(group_name)
            if group_key not in seen_groups:
                group_order.append(group_key)
                seen_groups.add(group_key)
            for country in members:
                cu = str(country).upper()
                if cu not in ex_set and cu not in group_map:
                    group_map[cu] = group_key

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
    hi_color_map = {str(country).upper(): color for country, color in (highlight_colors or {}).items()}

    if group_map:
        grouped = plot[plot["country"].isin(group_map)].copy()
        ungrouped = plot[~plot["country"].isin(group_map)].copy()
        hi_ungrouped = hi[~hi["country"].isin(group_map)].copy()
        bg_ungrouped = ungrouped[~ungrouped["country"].isin(hi_set)].copy()

        if not bg_ungrouped.empty:
            ax.scatter(
                bg_ungrouped["xi"].to_numpy(dtype=float),
                bg_ungrouped["ratio"].to_numpy(dtype=float),
                s=point_size_bg,
                color=color_bg,
                alpha=alpha_bg,
                edgecolors="none",
                zorder=2,
            )

        for i, group_name in enumerate(group_order):
            sub = grouped[grouped["country"].map(group_map) == group_name].copy()
            if sub.empty:
                continue
            color = cycle[i % len(cycle)]
            ax.scatter(
                sub["xi"].to_numpy(dtype=float),
                sub["ratio"].to_numpy(dtype=float),
                s=point_size_bg,
                color=color,
                alpha=alpha_hi,
                edgecolors="none",
                zorder=3,
                label=group_name,
            )

        handles, labels = ax.get_legend_handles_labels()
        if handles and show_legend:
            legend = ax.legend(frameon=False, title="Groups")
            if legend.get_title() is not None:
                legend.get_title().set_fontfamily("serif")
            for txt in legend.get_texts():
                txt.set_fontfamily("serif")

        if hi_order:
            hi_present = sorted(hi["country"].unique(), key=hi_order.index)
            hi_ungrouped_present = [country for country in hi_present if country not in group_map]
            for i, country in enumerate(hi_ungrouped_present):
                sub = hi_ungrouped[hi_ungrouped["country"] == country]
                if sub.empty:
                    continue
                color = hi_color_map.get(country, cycle[(len(group_order) + i) % len(cycle)])
                x = float(sub["xi"].iloc[0])
                y = float(sub["ratio"].iloc[0])
                ax.scatter(
                    [x],
                    [y],
                    s=point_size_hi,
                    color=color,
                    alpha=alpha_hi,
                    edgecolors="none",
                    zorder=4,
                )

            for country in hi_present:
                sub = hi[hi["country"] == country]
                if sub.empty:
                    continue
                x = float(sub["xi"].iloc[0])
                y = float(sub["ratio"].iloc[0])
                ax.annotate(
                    country,
                    xy=(x, y),
                    xytext=(label_dx_pts, label_dy_pts),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=highlight_label_fontsize,
                    fontfamily="serif",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": label_bbox_alpha,
                        "pad": 0.2,
                    },
                    zorder=5,
                )
    else:
        if not bg.empty:
            ax.scatter(
                bg["xi"].to_numpy(dtype=float),
                bg["ratio"].to_numpy(dtype=float),
                s=point_size_bg,
                color=color_bg,
                alpha=alpha_bg,
                edgecolors="none",
                zorder=2,
            )

        if not hi.empty:
            hi_present = sorted(hi["country"].unique(), key=hi_order.index)
            for i, country in enumerate(hi_present):
                sub = hi[hi["country"] == country]
                if sub.empty:
                    continue
                color = hi_color_map.get(country, cycle[i % len(cycle)])
                x = float(sub["xi"].iloc[0])
                y = float(sub["ratio"].iloc[0])
                ax.scatter(
                    [x],
                    [y],
                    s=point_size_hi,
                    color=color,
                    alpha=alpha_hi,
                    edgecolors="none",
                    zorder=4,
                )
                ax.annotate(
                    country,
                    xy=(x, y),
                    xytext=(label_dx_pts, label_dy_pts),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=highlight_label_fontsize,
                    fontfamily="serif",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": label_bbox_alpha,
                        "pad": 0.2,
                    },
                    zorder=5,
                )

    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_xlabel(_metric_label("xi"), fontfamily="serif")
    ax.set_ylabel(_ratio_label(ratio_mode, share_metric=share_metric), fontfamily="serif")
    if title:
        ax.set_title(title, fontfamily="serif")
    else:
        ax.set_title(
            rf"Composition Map: {_ratio_label(ratio_mode, share_metric=share_metric)} vs $\xi$ ({year_value})",
            fontfamily="serif",
        )

    for tick in ax.get_xticklabels():
        tick.set_fontfamily("serif")
    for tick in ax.get_yticklabels():
        tick.set_fontfamily("serif")

    return ax


def plot_xi_country_lines(
    df: pd.DataFrame,
    metric: str = "xi",
    highlight_countries: Iterable[str] | None = None,
    exclude_countries: Iterable[str] | None = None,
    only_ok: bool = True,
    title: str | None = None,
    yscale: str = "linear",
    symlog_linthresh: float = 1e-3,
    log_ymin: float = 0.01,
    log_bottom_pad_factor: float = 0.92,
    show_background: bool = True,
    alpha_bg: float = 0.18,
    lw_bg: float = 0.8,
    lw_hi: float = 2.6,
    use_end_labels: bool = True,
    end_label_fontsize: float = 11.0,
    show_legend: bool = False,
    label_pad_years: float = 1.2,
    reference_years: tuple[float, ...] = (2008.0, 2020.0),
    reference_year_labels: dict[float, str] | None = None,
    reference_label_y: float = 0.98,
    reference_label_x_offset: float = 0.12,
    reference_label_rotation: float = 90.0,
    reference_label_fontsize: float = 10.0,
    reference_label_color: str = "0.2",
    reference_label_ha: str = "left",
    reference_label_va: str = "top",
    ax: Any = None,
) -> Any:
    """Plot country xi lines with optional gray background and highlighted countries."""
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {sorted(_ALLOWED_METRICS)}")
    required_cols = {"country", "year"}
    if metric == "share":
        required_cols.update({"xi", "xi_amp"})
    elif metric == "share_amp_1":
        required_cols.update({"xi", "xi_amp_1"})
    elif metric == "share_amp_2":
        required_cols.update({"xi", "xi_amp_2"})
    else:
        required_cols.add(metric)
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, LogLocator
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install it in your .venv.") from exc

    work = df.copy()
    work["country"] = work["country"].astype(str).str.upper()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    if metric == "share":
        work["xi"] = pd.to_numeric(work["xi"], errors="coerce")
        work["xi_amp"] = pd.to_numeric(work["xi_amp"], errors="coerce")
        denom = work["xi"].replace(0.0, np.nan)
        work["share"] = work["xi_amp"] / denom
    elif metric == "share_amp_1":
        work["xi"] = pd.to_numeric(work["xi"], errors="coerce")
        work["xi_amp_1"] = pd.to_numeric(work["xi_amp_1"], errors="coerce")
        denom = work["xi"].replace(0.0, np.nan)
        work["share_amp_1"] = work["xi_amp_1"] / denom
    elif metric == "share_amp_2":
        work["xi"] = pd.to_numeric(work["xi"], errors="coerce")
        work["xi_amp_2"] = pd.to_numeric(work["xi_amp_2"], errors="coerce")
        denom = work["xi"].replace(0.0, np.nan)
        work["share_amp_2"] = work["xi_amp_2"] / denom
    else:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")

    if only_ok and "status" in work.columns:
        work = work[work["status"] == "ok"].copy()

    work = work.dropna(subset=["country", "year", metric]).copy()
    if work.empty:
        raise ValueError("No data available to plot after filtering.")
    if yscale not in {"linear", "log", "symlog"}:
        raise ValueError("yscale must be one of: linear, log, symlog")
    if yscale == "log":
        non_positive = int((work[metric] <= 0).sum())
        if non_positive > 0:
            raise ValueError(
                f"Cannot use log scale: found {non_positive} non-positive '{metric}' values. "
                "Use yscale='linear' or yscale='symlog'."
            )
        if log_ymin <= 0:
            raise ValueError("log_ymin must be > 0 for log scale.")
        if not (0 < log_bottom_pad_factor <= 1):
            raise ValueError("log_bottom_pad_factor must be in (0, 1].")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 7))

    ex_set = set()
    if exclude_countries:
        ex_set = {str(c).upper() for c in exclude_countries}
        work = work[~work["country"].isin(ex_set)].copy()
        if work.empty:
            raise ValueError("No data available after applying exclude_countries.")

    hi_order = []
    if highlight_countries:
        seen = set()
        for c in highlight_countries:
            cu = str(c).upper()
            if cu in ex_set:
                continue
            if cu not in seen:
                hi_order.append(cu)
                seen.add(cu)
    hi_set = set(hi_order)

    countries = sorted(work["country"].unique())
    bg_countries = [c for c in countries if c not in hi_set]

    if show_background:
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
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
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
            ordered = sorted(hi_endpoints, key=lambda t: t[2])
            adjusted: list[tuple[str, float, float, float, str]] = []
            prev_y: float | None = None
            for country, x_last, y_last, color in ordered:
                y_label = y_last if prev_y is None else max(y_last, prev_y + min_gap)
                adjusted.append((country, x_last, y_last, y_label, color))
                prev_y = y_label
            x_last_global = float(work["year"].max())
            x_label = x_last_global + 0.35 * label_pad_years
            for country, x_last, y_last, y_label, color in adjusted:
                ax.text(
                    x_label,
                    y_label,
                    country,
                    va="center",
                    ha="left",
                    color="black",
                    fontsize=end_label_fontsize,
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

                # Thin connector from series endpoint to label zone.
                ax.plot(
                    [x_last, x_label - 0.03 * label_pad_years],
                    [y_last, y_label],
                    color=color,
                    linewidth=0.8,
                    alpha=0.8,
                    zorder=3,
                )

    for ref_year in reference_years:
        ax.axvline(
            float(ref_year),
            color="0.1",
            linestyle="--",
            linewidth=2.0,
            alpha=0.35,
            zorder=2,
        )
        if reference_year_labels and float(ref_year) in reference_year_labels:
            ax.text(
                float(ref_year) + reference_label_x_offset,
                reference_label_y,
                str(reference_year_labels[float(ref_year)]),
                transform=ax.get_xaxis_transform(),
                ha=reference_label_ha,
                va=reference_label_va,
                rotation=reference_label_rotation,
                fontsize=reference_label_fontsize,
                fontfamily="serif",
                color=reference_label_color,
                clip_on=False,
                zorder=4,
            )

    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.8)
    ax.set_xlabel("Year", fontfamily="serif")
    ax.set_ylabel(_metric_label(metric), fontfamily="serif")
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh)
    else:
        ax.set_yscale(yscale)
    if yscale == "log":
        # Add denser log tick marks including mid-decade (e.g., 0.05) and plain decimal labels.
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 5.0)))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        def _fmt_log_major(y: float, _: Any) -> str:
            if y <= 0:
                return ""
            if y < 0.01:
                return f"{y:.3f}"
            return f"{y:.2f}"

        def _fmt_log_minor(y: float, _: Any) -> str:
            if y <= 0:
                return ""
            if y < 0.01:
                return f"{y:.3f}"
            return f"{y:.2f}"

        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_log_major))
        ax.yaxis.set_minor_formatter(FuncFormatter(_fmt_log_minor))
        ax.tick_params(axis="y", which="minor", labelsize=9)
        ax.grid(axis="y", which="minor", alpha=0.14, linewidth=0.5)
        ymax_data = float(np.nanmax(work[metric].to_numpy(dtype=float)))
        ymin_vis = log_ymin * log_bottom_pad_factor
        ymax_vis = ymax_data * 1.05
        if ymax_vis <= ymin_vis:
            ymax_vis = ymin_vis * 1.2
        ax.set_ylim(ymin_vis, ymax_vis)
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
