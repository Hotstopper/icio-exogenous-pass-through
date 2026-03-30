from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as spla

from io_networks.paths import resolve_paths

ZERO_TOL = 1e-12


def _extract_year(filename: str) -> int | None:
    match = re.search(r"(19\d{2}|20\d{2})", filename)
    return int(match.group(1)) if match else None


def _git_commit() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return output.strip()
    except Exception:
        return "unknown"


def _sector_code(label: str) -> str:
    if "_" not in label:
        return ""
    return label.split("_", 1)[1]


def _country_from_label(label: str) -> str:
    if "_" not in label:
        return ""
    return label.split("_", 1)[0]


def _raw_file_map(raw_dir: Path, year_start: int, year_end: int) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for file_path in sorted(raw_dir.glob("*.csv")):
        year = _extract_year(file_path.name)
        if year is None or year < year_start or year > year_end:
            continue
        mapping[year] = file_path
    return mapping


def _compute_v_and_v_over_out(
    raw_file: Path, sector_labels: list[str]
) -> tuple[np.ndarray, np.ndarray, int]:
    frame = pd.read_csv(raw_file, usecols=["V1", *sector_labels]).set_index("V1")

    tls = pd.to_numeric(frame.reindex(["TLS"]).iloc[0], errors="coerce").fillna(0.0)
    va = pd.to_numeric(frame.reindex(["VA"]).iloc[0], errors="coerce").fillna(0.0)
    out = pd.to_numeric(frame.reindex(["OUT"]).iloc[0], errors="coerce").fillna(0.0)
    tls = tls.to_numpy(dtype=float)
    va = va.to_numpy(dtype=float)
    out = out.to_numpy(dtype=float)

    v = tls + va
    vc = np.zeros_like(v, dtype=float)
    nonzero_out = np.abs(out) > ZERO_TOL
    vc[nonzero_out] = v[nonzero_out] / out[nonzero_out]
    return v, vc, int((~nonzero_out).sum())


def _load_wb_real_gdp_growth(gdp_csv: Path, year_start: int, year_end: int) -> pd.Series:
    if not gdp_csv.exists():
        raise FileNotFoundError(f"World Bank real GDP growth CSV not found: {gdp_csv}")

    frame = pd.read_csv(gdp_csv, skiprows=4, dtype=str)
    year_cols = [col for col in frame.columns if re.fullmatch(r"\d{4}", str(col))]
    if not year_cols:
        raise ValueError(f"No year columns found in {gdp_csv}")

    long = frame.melt(
        id_vars=["Country Code", "Indicator Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp_growth",
    )
    long = long[long["Indicator Code"] == "NY.GDP.MKTP.KD.ZG"].copy()
    long["country"] = long["Country Code"].astype(str).str.strip().str.upper()
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["gdp_growth"] = pd.to_numeric(long["gdp_growth"], errors="coerce")
    # World Bank series NY.GDP.MKTP.KD.ZG is in percent units; convert to proportion units.
    long["gdp_growth"] = long["gdp_growth"] / 100.0
    long = long.dropna(subset=["year"]).copy()
    long["year"] = long["year"].astype(int)
    long = long[long["year"].between(year_start, year_end, inclusive="both")].copy()
    long = long.groupby(["country", "year"], as_index=False)["gdp_growth"].mean()

    return long.set_index(["country", "year"])["gdp_growth"]


def _build_lambda(n_exo: int, cfg: dict[str, Any]) -> np.ndarray:
    method = cfg.get("lambda", {}).get("method", "uniform")
    normalize = bool(cfg.get("lambda", {}).get("normalize", True))

    if n_exo <= 0:
        raise ValueError("Exogenous sector set E is empty. Check sectors.exo_codes.")

    if method != "uniform":
        raise ValueError(
            f"Unsupported lambda.method: {method}. Currently only 'uniform' is implemented."
        )

    lam = np.ones(n_exo, dtype=float)
    if normalize:
        lam = lam / lam.sum()
    return lam


def _spectral_radius_estimate(a_nn: sparse.csr_matrix) -> tuple[float, str]:
    n = a_nn.shape[0]
    if n == 0:
        return 0.0, "empty"

    try:
        eig = spla.eigs(a_nn, k=1, which="LM", return_eigenvectors=False, tol=1e-4, maxiter=500)
        return float(np.abs(eig[0])), "eigs"
    except Exception:
        bound = float(spla.norm(a_nn, ord=1))
        return bound, "one_norm_upper_bound"


def _condition_number_estimate(m: sparse.csr_matrix) -> float:
    # Dense condition number is too expensive for full ICIO dimensionality.
    n = m.shape[0]
    if n == 0:
        return 1.0
    if n > 1200:
        return float("nan")

    dense = m.toarray()
    return float(np.linalg.cond(dense))


def build_blocks(cfg: dict[str, Any]) -> Path:
    paths = resolve_paths(cfg)
    variant = "extended" if cfg["icio"].get("extended", False) else "regular"

    year_start = int(cfg["icio"]["year_range"]["start"])
    year_end = int(cfg["icio"]["year_range"]["end"])

    a_dir = paths["matrices"] / "A" / variant
    if not a_dir.exists():
        raise FileNotFoundError(f"A matrix directory not found: {a_dir}. Run build-a first.")

    raw_dir = paths["raw"] / variant
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    out_dir = paths["matrices"] / "blocks" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_by_year = _raw_file_map(raw_dir, year_start, year_end)
    gdp_csv = paths["raw"] / "world_bank_real_gdp_growth" / "world_bank_real_gdp_growth.csv"
    gdp_growth_by_country_year = _load_wb_real_gdp_growth(gdp_csv, year_start, year_end)

    exo_codes = set(cfg.get("sectors", {}).get("exo_codes", []))
    if not exo_codes:
        raise ValueError("config.sectors.exo_codes is empty. Define exogenous sector codes first.")

    build_time = datetime.now(timezone.utc).isoformat()
    commit_hash = _git_commit()
    lambda_method = cfg.get("lambda", {}).get("method", "uniform")
    lambda_normalize = bool(cfg.get("lambda", {}).get("normalize", True))

    summary_rows: list[dict[str, Any]] = []
    prev_v_by_label: pd.Series | None = None

    for year in range(year_start, year_end + 1):
        npz_path = a_dir / f"A_{year}.npz"
        meta_path = a_dir / f"A_{year}_meta.parquet"
        raw_file = raw_by_year.get(year)
        if not npz_path.exists() or not meta_path.exists():
            continue
        if raw_file is None:
            raise ValueError(f"Year {year}: missing raw CSV file in {raw_dir}.")

        a = np.load(npz_path)["A"]
        meta = pd.read_parquet(meta_path)
        labels = meta["sector"].astype(str).tolist()
        v_full, vc_full, zero_out_count = _compute_v_and_v_over_out(raw_file, labels)

        if a.shape[0] != len(labels) or a.shape[1] != len(labels):
            raise ValueError(
                f"Dimension mismatch for year {year}: A shape {a.shape} vs labels {len(labels)}"
            )

        if prev_v_by_label is None:
            pct_change_v_full = np.full(len(labels), np.nan, dtype=float)
        else:
            prev_v_full = prev_v_by_label.reindex(labels).to_numpy(dtype=float)
            pct_change_v_full = np.zeros(len(labels), dtype=float)
            valid = np.abs(prev_v_full) > ZERO_TOL
            pct_change_v_full[valid] = (v_full[valid] - prev_v_full[valid]) / prev_v_full[valid]

        prev_v_by_label = pd.Series(v_full, index=labels, dtype=float)

        e_idx = np.array(
            [i for i, s in enumerate(labels) if _sector_code(s) in exo_codes],
            dtype=int,
        )
        n_idx = np.array(
            [i for i, s in enumerate(labels) if _sector_code(s) not in exo_codes],
            dtype=int,
        )

        if e_idx.size == 0:
            raise ValueError(
                f"Year {year}: no sectors classified as exogenous from codes {sorted(exo_codes)}"
            )
        if n_idx.size == 0:
            raise ValueError(f"Year {year}: no sectors classified as non-exogenous.")

        a_nn = a[np.ix_(n_idx, n_idx)]
        a_en = a[np.ix_(e_idx, n_idx)]

        a_nn_sp = sparse.csr_matrix(a_nn)
        m = sparse.eye(a_nn_sp.shape[0], format="csr") - a_nn_sp.transpose().tocsr()

        lam = _build_lambda(e_idx.size, cfg)
        tau_dir = np.asarray(a_en.transpose() @ lam, dtype=float)

        solve_success = True
        solve_note = "ok"
        try:
            tau = spla.spsolve(m, tau_dir)
            tau = np.asarray(tau, dtype=float)
            tau_amp = tau - tau_dir
        except Exception as exc:
            solve_success = False
            solve_note = str(exc)
            tau = np.full(n_idx.size, np.nan, dtype=float)
            tau_amp = np.full(n_idx.size, np.nan, dtype=float)

        if solve_success:
            tau_sq = np.square(tau)
            lam_sq = np.square(lam)
            kappa = np.asarray(
                a_nn.transpose() @ tau_sq + a_en.transpose() @ lam_sq - tau_sq,
                dtype=float,
            )
        else:
            kappa = np.full(n_idx.size, np.nan, dtype=float)

        chi_solve_success = True
        chi_solve_note = "ok"
        try:
            chi = spla.spsolve(m, kappa)
            chi = np.asarray(chi, dtype=float)
        except Exception as exc:
            chi_solve_success = False
            chi_solve_note = str(exc)
            chi = np.full(n_idx.size, np.nan, dtype=float)

        spectral_radius, spectral_method = _spectral_radius_estimate(a_nn_sp)
        cond_est = _condition_number_estimate(m)

        blocks_npz = out_dir / f"blocks_{year}.npz"

        labels_n = [labels[i] for i in n_idx]
        labels_e = [labels[i] for i in e_idx]
        vc_n = vc_full[n_idx]
        pct_change_v_n = pct_change_v_full[n_idx]
        v_scaled_n = vc_n * pct_change_v_n
        c_n = np.full(n_idx.size, np.nan, dtype=float)
        c_rhs = v_scaled_n.copy()
        countries = sorted(
            {_country_from_label(label) for label in labels_n if _country_from_label(label)}
        )
        country_missing_pct = int(
            sum(
                np.isnan(
                    c_rhs[
                        [i for i, label in enumerate(labels_n) if label.startswith(f"{country}_")]
                    ]
                ).any()
                for country in countries
            )
        )

        c_solve_success = True
        c_solve_note = "ok"
        c_residual_inf = np.nan
        if np.isnan(c_rhs).any():
            c_solve_success = False
            c_solve_note = "missing_pct_change_v: global solve skipped because rhs contains NaN."
        else:
            try:
                c_n = spla.spsolve(m, c_rhs)
                c_n = np.asarray(c_n, dtype=float)
                c_residual_inf = float(np.linalg.norm(m.dot(c_n) - c_rhs, ord=np.inf))
            except Exception as exc:
                c_solve_success = False
                c_solve_note = str(exc)

        country_solve_ok = len(countries) if c_solve_success else 0
        country_solve_fail = 0 if c_solve_success else len(countries)
        country_residual_max = float(c_residual_inf) if c_solve_success else np.nan

        country_pos: dict[str, int] = {}
        c_country_rows: list[dict[str, Any]] = []
        for idx_local, label in enumerate(labels_n):
            country = _country_from_label(label)
            pos = country_pos.get(country, 0)
            country_pos[country] = pos + 1
            c_country_rows.append(
                {
                    "year": year,
                    "variant": variant,
                    "country": country,
                    "label": label,
                    "index_in_N": idx_local,
                    "index_in_country": pos,
                    "v_over_out": float(vc_n[idx_local]),
                    "pct_change_v": float(pct_change_v_n[idx_local]),
                    "v_over_out_times_pct_change_v": float(c_rhs[idx_local]),
                    "c": float(c_n[idx_local]),
                    "solve_status": "ok" if c_solve_success else "solve_failed",
                    "solve_note": c_solve_note,
                    "residual_inf": c_residual_inf,
                }
            )

        c_country_df = pd.DataFrame(c_country_rows)
        if not c_country_df.empty:
            c_country_df = c_country_df.sort_values(["country", "index_in_country"]).reset_index(
                drop=True
            )
        c_country_path = out_dir / f"blocks_{year}_country_c.parquet"
        c_country_df.to_parquet(c_country_path, index=False)

        gdp_growth_n = np.full(n_idx.size, np.nan, dtype=float)
        gdp_growth_missing = 0
        for idx_local, label in enumerate(labels_n):
            country = _country_from_label(label)
            value = gdp_growth_by_country_year.get((country, year), np.nan)
            if pd.isna(value):
                gdp_growth_missing += 1
            gdp_growth_n[idx_local] = float(value) if pd.notna(value) else np.nan

        gdp_growth_n_filled = np.where(np.isnan(gdp_growth_n), 0.0, gdp_growth_n)
        gdp_scaled_n = vc_n * gdp_growth_n_filled
        c_gdp_n = np.full(n_idx.size, np.nan, dtype=float)
        c_gdp_solve_success = True
        c_gdp_solve_note = "ok"
        c_gdp_residual_inf = np.nan
        try:
            c_gdp_n = spla.spsolve(m, gdp_scaled_n)
            c_gdp_n = np.asarray(c_gdp_n, dtype=float)
            c_gdp_residual_inf = float(np.linalg.norm(m.dot(c_gdp_n) - gdp_scaled_n, ord=np.inf))
        except Exception as exc:
            c_gdp_solve_success = False
            c_gdp_solve_note = str(exc)

        country_pos_gdp: dict[str, int] = {}
        c_gdp_country_rows: list[dict[str, Any]] = []
        for idx_local, label in enumerate(labels_n):
            country = _country_from_label(label)
            pos = country_pos_gdp.get(country, 0)
            country_pos_gdp[country] = pos + 1
            c_gdp_country_rows.append(
                {
                    "year": year,
                    "variant": variant,
                    "country": country,
                    "label": label,
                    "index_in_N": idx_local,
                    "index_in_country": pos,
                    "v_over_out": float(vc_n[idx_local]),
                    "real_gdp_growth": float(gdp_growth_n[idx_local]),
                    "real_gdp_growth_filled": float(gdp_growth_n_filled[idx_local]),
                    "v_over_out_times_real_gdp_growth": float(gdp_scaled_n[idx_local]),
                    "c_gdp": float(c_gdp_n[idx_local]),
                    "solve_status": "ok" if c_gdp_solve_success else "solve_failed",
                    "solve_note": c_gdp_solve_note,
                    "residual_inf": c_gdp_residual_inf,
                }
            )

        c_gdp_country_df = pd.DataFrame(c_gdp_country_rows)
        if not c_gdp_country_df.empty:
            c_gdp_country_df = c_gdp_country_df.sort_values(
                ["country", "index_in_country"]
            ).reset_index(drop=True)
        c_gdp_country_path = out_dir / f"blocks_{year}_country_c_gdp.parquet"
        c_gdp_country_df.to_parquet(c_gdp_country_path, index=False)

        np.savez_compressed(
            blocks_npz,
            A_NN=a_nn,
            A_EN=a_en,
            tau=tau,
            tau_dir=tau_dir,
            tau_amp=tau_amp,
            kappa=kappa,
            chi=chi,
            idx_N=n_idx,
            idx_E=e_idx,
            lambda_E=lam,
            v_over_out_N=vc_n,
            pct_change_v_N=pct_change_v_n,
            v_over_out_times_pct_change_v_N=v_scaled_n,
            c_N=c_n,
            gdp_growth_N=gdp_growth_n,
            gdp_growth_filled_N=gdp_growth_n_filled,
            v_over_out_times_gdp_growth_N=gdp_scaled_n,
            c_gdp_N=c_gdp_n,
        )

        labels_df = pd.DataFrame(
            {
                "year": year,
                "variant": variant,
                "group": ["N"] * len(labels_n) + ["E"] * len(labels_e),
                "label": labels_n + labels_e,
                "index_in_full": n_idx.tolist() + e_idx.tolist(),
                "index_in_group": list(range(len(labels_n))) + list(range(len(labels_e))),
            }
        )
        labels_out = out_dir / f"blocks_{year}_meta.parquet"
        labels_df.to_parquet(labels_out, index=False)

        summary_rows.append(
            {
                "year": year,
                "variant": variant,
                "n_total": int(len(labels)),
                "n_E": int(e_idx.size),
                "n_N": int(n_idx.size),
                "exo_codes": "|".join(sorted(exo_codes)),
                "lambda_method": lambda_method,
                "lambda_normalize": lambda_normalize,
                "solve_success": solve_success,
                "solve_note": solve_note,
                "tau_nan_count": int(np.isnan(tau).sum()),
                "tau_inf_count": int(np.isinf(tau).sum()),
                "tau_dir_nan_count": int(np.isnan(tau_dir).sum()),
                "tau_amp_nan_count": int(np.isnan(tau_amp).sum()),
                "kappa_nan_count": int(np.isnan(kappa).sum()),
                "kappa_inf_count": int(np.isinf(kappa).sum()),
                "chi_solve_success": chi_solve_success,
                "chi_solve_note": chi_solve_note,
                "chi_nan_count": int(np.isnan(chi).sum()),
                "chi_inf_count": int(np.isinf(chi).sum()),
                "tau_mean": float(np.nanmean(tau)),
                "tau_dir_mean": float(np.nanmean(tau_dir)),
                "tau_amp_mean": float(np.nanmean(tau_amp)),
                "kappa_mean": float(np.nanmean(kappa)),
                "chi_mean": float(np.nanmean(chi)),
                "raw_zero_out_count": zero_out_count,
                "country_solve_ok": country_solve_ok,
                "country_solve_fail": country_solve_fail,
                "country_missing_pct_change": country_missing_pct,
                "country_solve_max_inf_residual": country_residual_max,
                "gdp_growth_missing_count": gdp_growth_missing,
                "c_gdp_solve_success": c_gdp_solve_success,
                "c_gdp_solve_note": c_gdp_solve_note,
                "c_gdp_nan_count": int(np.isnan(c_gdp_n).sum()),
                "c_gdp_inf_count": int(np.isinf(c_gdp_n).sum()),
                "c_gdp_mean": float(np.nanmean(c_gdp_n)),
                "c_gdp_rhs_mean": float(np.nanmean(gdp_scaled_n)),
                "c_gdp_residual_inf": c_gdp_residual_inf,
                "spectral_radius_estimate": spectral_radius,
                "spectral_radius_method": spectral_method,
                "cond_estimate": cond_est,
                "npz_file": blocks_npz.name,
                "meta_file": labels_out.name,
                "country_c_file": c_country_path.name,
                "country_c_gdp_file": c_gdp_country_path.name,
                "build_timestamp_utc": build_time,
                "git_commit": commit_hash,
            }
        )

    if not summary_rows:
        raise ValueError(
            f"No yearly A artifacts found in {a_dir}. "
            "Expected files like A_<year>.npz and A_<year>_meta.parquet."
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("year").reset_index(drop=True)
    summary_path = out_dir / "blocks_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return summary_path
