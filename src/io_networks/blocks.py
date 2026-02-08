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


def _build_lambda(n_exo: int, cfg: dict[str, Any]) -> np.ndarray:
    method = cfg.get("lambda", {}).get("method", "uniform")
    normalize = bool(cfg.get("lambda", {}).get("normalize", True))

    if n_exo <= 0:
        raise ValueError("Exogenous sector set E is empty. Check sectors.exo_codes.")

    if method != "uniform":
        raise ValueError(f"Unsupported lambda.method: {method}. Currently only 'uniform' is implemented.")

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

    out_dir = paths["matrices"] / "blocks" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    exo_codes = set(cfg.get("sectors", {}).get("exo_codes", []))
    if not exo_codes:
        raise ValueError("config.sectors.exo_codes is empty. Define exogenous sector codes first.")

    build_time = datetime.now(timezone.utc).isoformat()
    commit_hash = _git_commit()
    lambda_method = cfg.get("lambda", {}).get("method", "uniform")
    lambda_normalize = bool(cfg.get("lambda", {}).get("normalize", True))

    summary_rows: list[dict[str, Any]] = []

    for year in range(year_start, year_end + 1):
        npz_path = a_dir / f"A_{year}.npz"
        meta_path = a_dir / f"A_{year}_meta.parquet"
        if not npz_path.exists() or not meta_path.exists():
            continue

        a = np.load(npz_path)["A"]
        meta = pd.read_parquet(meta_path)
        labels = meta["sector"].astype(str).tolist()

        if a.shape[0] != len(labels) or a.shape[1] != len(labels):
            raise ValueError(
                f"Dimension mismatch for year {year}: A shape {a.shape} vs labels {len(labels)}"
            )

        e_idx = np.array([i for i, s in enumerate(labels) if _sector_code(s) in exo_codes], dtype=int)
        n_idx = np.array([i for i, s in enumerate(labels) if _sector_code(s) not in exo_codes], dtype=int)

        if e_idx.size == 0:
            raise ValueError(f"Year {year}: no sectors classified as exogenous from codes {sorted(exo_codes)}")
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

        spectral_radius, spectral_method = _spectral_radius_estimate(a_nn_sp)
        cond_est = _condition_number_estimate(m)

        blocks_npz = out_dir / f"blocks_{year}.npz"
        np.savez_compressed(
            blocks_npz,
            A_NN=a_nn,
            A_EN=a_en,
            tau=tau,
            tau_dir=tau_dir,
            tau_amp=tau_amp,
            idx_N=n_idx,
            idx_E=e_idx,
            lambda_E=lam,
        )

        labels_n = [labels[i] for i in n_idx]
        labels_e = [labels[i] for i in e_idx]
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
                "tau_mean": float(np.nanmean(tau)),
                "tau_dir_mean": float(np.nanmean(tau_dir)),
                "tau_amp_mean": float(np.nanmean(tau_amp)),
                "spectral_radius_estimate": spectral_radius,
                "spectral_radius_method": spectral_method,
                "cond_estimate": cond_est,
                "npz_file": blocks_npz.name,
                "meta_file": labels_out.name,
                "build_timestamp_utc": build_time,
                "git_commit": commit_hash,
            }
        )

    if not summary_rows:
        raise ValueError(
            f"No yearly A artifacts found in {a_dir}. Expected files like A_<year>.npz and A_<year>_meta.parquet."
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("year").reset_index(drop=True)
    summary_path = out_dir / "blocks_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return summary_path