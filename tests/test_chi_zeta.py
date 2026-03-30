from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from io_networks.blocks import build_blocks
from io_networks.zeta import build_zeta


def _base_cfg(root: Path) -> dict:
    return {
        "project": {"name": "test"},
        "icio": {"extended": False, "year_range": {"start": 2000, "end": 2000}},
        "sectors": {"exo_codes": ["B06"]},
        "lambda": {"method": "uniform", "normalize": False},
        "paths": {
            "raw": str(root / "raw"),
            "interim": str(root / "interim"),
            "processed": str(root / "processed"),
            "matrices": str(root / "matrices"),
            "outputs": str(root / "outputs"),
        },
    }


def _write_world_bank_growth_csv(path: Path) -> None:
    lines = [
        "metadata line 1",
        "metadata line 2",
        "metadata line 3",
        "metadata line 4",
        "Country Name,Country Code,Indicator Name,Indicator Code,2000",
        "Country AAA,AAA,Real GDP growth,NY.GDP.MKTP.KD.ZG,5.0",
        "Country BBB,BBB,Real GDP growth,NY.GDP.MKTP.KD.ZG,3.0",
        "Country CCC,CCC,Real GDP growth,NY.GDP.MKTP.KD.ZG,1.0",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_build_blocks_computes_kappa_and_chi(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)

    raw_dir = tmp_path / "raw" / "regular"
    matrices_dir = tmp_path / "matrices" / "A" / "regular"
    gdp_dir = tmp_path / "raw" / "world_bank_real_gdp_growth"
    raw_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)
    gdp_dir.mkdir(parents=True, exist_ok=True)

    labels = ["AAA_N1", "AAA_N2", "AAA_B06", "BBB_N1", "BBB_B06"]
    a = np.zeros((5, 5), dtype=float)
    a_nn = np.array(
        [
            [0.10, 0.02, 0.03],
            [0.04, 0.15, 0.01],
            [0.02, 0.05, 0.12],
        ],
        dtype=float,
    )
    a_en = np.array(
        [
            [0.20, 0.10, 0.05],
            [0.10, 0.03, 0.07],
        ],
        dtype=float,
    )
    n_idx = [0, 1, 3]
    e_idx = [2, 4]
    a[np.ix_(n_idx, n_idx)] = a_nn
    a[np.ix_(e_idx, n_idx)] = a_en

    np.savez_compressed(matrices_dir / "A_2000.npz", A=a)
    pd.DataFrame(
        {
            "year": [2000] * len(labels),
            "sector": labels,
            "out": [20, 25, 15, 18, 14],
            "zero_out": [False] * len(labels),
        }
    ).to_parquet(matrices_dir / "A_2000_meta.parquet", index=False)

    raw_rows = pd.DataFrame(
        [
            {"V1": "AAA_N1", "AAA_HFCE": 4.0, "BBB_HFCE": 0.0},
            {"V1": "AAA_N2", "AAA_HFCE": 6.0, "BBB_HFCE": 0.0},
            {"V1": "AAA_B06", "AAA_HFCE": 0.0, "BBB_HFCE": 0.0},
            {"V1": "BBB_N1", "AAA_HFCE": 0.0, "BBB_HFCE": 8.0},
            {"V1": "BBB_B06", "AAA_HFCE": 0.0, "BBB_HFCE": 0.0},
            {"V1": "TLS", "AAA_HFCE": 0.0, "BBB_HFCE": 0.0},
            {"V1": "VA", "AAA_HFCE": 0.0, "BBB_HFCE": 0.0},
            {"V1": "OUT", "AAA_HFCE": 0.0, "BBB_HFCE": 0.0},
        ]
    )
    sector_rows = zip(
        labels,
        [2, 3, 1, 4, 2],
        [8, 7, 4, 5, 3],
        [20, 25, 15, 18, 14],
        strict=True,
    )
    for label, tls, va, out in sector_rows:
        raw_rows[label] = 0.0
        raw_rows.loc[raw_rows["V1"] == "TLS", label] = tls
        raw_rows.loc[raw_rows["V1"] == "VA", label] = va
        raw_rows.loc[raw_rows["V1"] == "OUT", label] = out
    raw_rows.to_csv(raw_dir / "icio_2000.csv", index=False)

    _write_world_bank_growth_csv(gdp_dir / "world_bank_real_gdp_growth.csv")

    build_blocks(cfg)

    arr = np.load(tmp_path / "matrices" / "blocks" / "regular" / "blocks_2000.npz")
    summary = pd.read_csv(tmp_path / "matrices" / "blocks" / "regular" / "blocks_summary.csv")

    assert "kappa" in arr.files
    assert "chi" in arr.files

    tau = arr["tau"]
    lam = arr["lambda_E"]
    kappa = arr["kappa"]
    chi = arr["chi"]

    expected_kappa = a_nn.transpose() @ (tau**2) + a_en.transpose() @ (lam**2) - (tau**2)
    np.testing.assert_allclose(kappa, expected_kappa)

    m = np.eye(a_nn.shape[0]) - a_nn.transpose()
    np.testing.assert_allclose(m @ chi, kappa)
    np.testing.assert_allclose(chi, kappa + a_nn.transpose() @ chi)

    assert bool(summary.loc[0, "chi_solve_success"])
    assert np.isfinite(summary.loc[0, "kappa_mean"])
    assert np.isfinite(summary.loc[0, "chi_mean"])


def test_build_zeta_aggregates_chi_with_hfce_weights(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)

    blocks_dir = tmp_path / "matrices" / "blocks" / "regular"
    raw_dir = tmp_path / "raw" / "regular"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    chi = np.array([1.0, 3.0, 5.0, 7.0], dtype=float)
    np.savez_compressed(blocks_dir / "blocks_2000.npz", chi=chi)
    pd.DataFrame(
        {
            "year": [2000] * 4,
            "variant": ["regular"] * 4,
            "group": ["N"] * 4,
            "label": ["AAA_N1", "AAA_N2", "BBB_N1", "CCC_N1"],
            "index_in_full": [0, 1, 2, 3],
            "index_in_group": [0, 1, 2, 3],
        }
    ).to_parquet(blocks_dir / "blocks_2000_meta.parquet", index=False)

    pd.DataFrame(
        [
            {"V1": "AAA_N1", "AAA_HFCE": 1.0, "BBB_HFCE": 0.0},
            {"V1": "AAA_N2", "AAA_HFCE": 3.0, "BBB_HFCE": 0.0},
            {"V1": "BBB_N1", "AAA_HFCE": 0.0, "BBB_HFCE": 0.0},
            {"V1": "CCC_N1", "AAA_HFCE": 0.0, "BBB_HFCE": 0.0},
        ]
    ).to_csv(raw_dir / "icio_2000.csv", index=False)

    zeta_path = build_zeta(cfg)

    zeta_df = pd.read_parquet(zeta_path).sort_values("country").reset_index(drop=True)
    weights_df = pd.read_parquet(
        tmp_path / "processed" / "zeta" / "regular" / "weights_by_country_sector.parquet"
    )
    diag_df = pd.read_parquet(
        tmp_path / "processed" / "zeta" / "regular" / "weights_diagnostics.parquet"
    )

    aaa_row = zeta_df[zeta_df["country"] == "AAA"].iloc[0]
    bbb_row = zeta_df[zeta_df["country"] == "BBB"].iloc[0]
    ccc_row = zeta_df[zeta_df["country"] == "CCC"].iloc[0]

    expected_aaa = 0.25 * 1.0 + 0.75 * 3.0
    assert aaa_row["status"] == "ok"
    assert np.isclose(aaa_row["zeta"], expected_aaa)
    assert bbb_row["status"] == "zero_hfce_mass"
    assert np.isnan(bbb_row["zeta"])
    assert ccc_row["status"] == "missing_hfce_column"
    assert np.isnan(ccc_row["zeta"])

    aaa_weights = weights_df[weights_df["country"] == "AAA"].sort_values("sector_label")
    aaa_weights = aaa_weights.reset_index(drop=True)
    np.testing.assert_allclose(aaa_weights["weight_norm"].to_numpy(), np.array([0.25, 0.75]))
    np.testing.assert_allclose(aaa_weights["chi"].to_numpy(), np.array([1.0, 3.0]))
    assert np.isclose(aaa_weights["contrib_zeta"].sum(), expected_aaa)

    assert set(diag_df["status"]) == {"ok", "zero_hfce_mass", "missing_hfce_column"}
