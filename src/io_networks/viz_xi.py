from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from io_networks.paths import resolve_paths

_ALLOWED_METRICS = {"xi", "xi_dir", "xi_amp"}


def _metric_label(metric: str) -> str:
    labels = {
        "xi": r"$\xi$",
        "xi_dir": r"$\xi_{dir}$",
        "xi_amp": r"$\xi_{amp}$",
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
    alpha_bg: float = 0.18,
    lw_bg: float = 0.8,
    lw_hi: float = 2.6,
    use_end_labels: bool = True,
    show_legend: bool = False,
    label_pad_years: float = 1.2,
    ax: Any = None,
) -> Any:
    """Plot country xi lines with gray background and highlighted countries."""
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {sorted(_ALLOWED_METRICS)}")
    required_cols = {"country", "year", metric}
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

                # Thin connector from series endpoint to label zone.
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
