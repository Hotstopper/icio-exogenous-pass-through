from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from io_networks.c_diff_quarterly import build_c_diff_quarterly


def _base_cfg(root: Path) -> dict:
    return {
        "project": {"name": "test"},
        "icio": {"extended": False, "year_range": {"start": 2000, "end": 2000}},
        "paths": {
            "raw": str(root / "raw"),
            "interim": str(root / "interim"),
            "processed": str(root / "processed"),
            "matrices": str(root / "matrices"),
            "outputs": str(root / "outputs"),
        },
    }


def test_build_c_diff_quarterly_aggregates_quarterly_gdp_adjustment(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)

    blocks_dir = tmp_path / "matrices" / "blocks" / "regular"
    raw_dir = tmp_path / "raw" / "regular"
    real_gdp_dir = tmp_path / "processed" / "real_gdp"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    real_gdp_dir.mkdir(parents=True, exist_ok=True)

    a_nn = np.array([[0.2, 0.0], [0.1, 0.3]], dtype=float)
    v_over_out_n = np.array([2.0, 4.0], dtype=float)
    c_n = np.array([8.0, 12.0], dtype=float)

    np.savez_compressed(
        blocks_dir / "blocks_2000.npz",
        A_NN=a_nn,
        c_N=c_n,
        v_over_out_N=v_over_out_n,
    )
    pd.DataFrame(
        {
            "year": [2000, 2000],
            "variant": ["regular", "regular"],
            "group": ["N", "N"],
            "label": ["AAA_S1", "AAA_S2"],
            "index_in_full": [0, 1],
            "index_in_group": [0, 1],
        }
    ).to_parquet(blocks_dir / "blocks_2000_meta.parquet", index=False)

    pd.DataFrame(
        [
            {"V1": "AAA_S1", "AAA_HFCE": 1.0},
            {"V1": "AAA_S2", "AAA_HFCE": 3.0},
        ]
    ).to_csv(raw_dir / "icio_2000.csv", index=False)

    pd.DataFrame(
        {
            "country": ["AAA", "AAA", "AAA"],
            "country_name": ["Country AAA"] * 3,
            "period": ["1999-Q4", "2000-Q1", "2000-Q2"],
            "year": [1999, 2000, 2000],
            "quarter": [4, 1, 2],
            "real_gdp": [100.0, np.exp(0.1) * 100.0, np.exp(0.3) * 100.0],
        }
    ).to_csv(real_gdp_dir / "quarterly_real_gdp.csv", index=False)

    out_path = build_c_diff_quarterly(cfg)

    out = pd.read_parquet(out_path).sort_values(["year", "quarter", "country"]).reset_index(drop=True)
    weights = pd.read_parquet(
        tmp_path / "processed" / "c_diff" / "regular" / "weights_by_country_sector_quarterly.parquet"
    ).sort_values(["year", "quarter", "sector_label"]).reset_index(drop=True)

    m = np.eye(2) - a_nn.transpose()
    growth_q1 = 0.1
    growth_q2 = 0.2
    c_gdp_q1 = np.linalg.solve(m, v_over_out_n * growth_q1)
    c_gdp_q2 = np.linalg.solve(m, v_over_out_n * growth_q2)
    c_quarter = c_n / 4.0
    expected_q1 = np.dot(np.array([0.25, 0.75]), c_quarter - c_gdp_q1)
    expected_q2 = np.dot(np.array([0.25, 0.75]), c_quarter - c_gdp_q2)

    assert out["period"].tolist() == ["2000-Q1", "2000-Q2"]
    assert set(out["status"]) == {"ok"}
    np.testing.assert_allclose(out["c_diff"].to_numpy(), np.array([expected_q1, expected_q2]))

    q1_weights = weights[weights["period"] == "2000-Q1"].reset_index(drop=True)
    np.testing.assert_allclose(q1_weights["weight_norm"].to_numpy(), np.array([0.25, 0.75]))
    np.testing.assert_allclose(q1_weights["c_N_over_4"].to_numpy(), c_quarter)
    np.testing.assert_allclose(q1_weights["c_gdp_q_N"].to_numpy(), c_gdp_q1)
    np.testing.assert_allclose(q1_weights["contrib_c_diff"].sum(), expected_q1)


def test_build_c_diff_quarterly_marks_missing_growth_as_missing_vector(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)

    blocks_dir = tmp_path / "matrices" / "blocks" / "regular"
    raw_dir = tmp_path / "raw" / "regular"
    real_gdp_dir = tmp_path / "processed" / "real_gdp"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    real_gdp_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        blocks_dir / "blocks_2000.npz",
        A_NN=np.array([[0.1]], dtype=float),
        c_N=np.array([4.0], dtype=float),
        v_over_out_N=np.array([2.0], dtype=float),
    )
    pd.DataFrame(
        {
            "year": [2000],
            "variant": ["regular"],
            "group": ["N"],
            "label": ["AAA_S1"],
            "index_in_full": [0],
            "index_in_group": [0],
        }
    ).to_parquet(blocks_dir / "blocks_2000_meta.parquet", index=False)

    pd.DataFrame([{"V1": "AAA_S1", "AAA_HFCE": 1.0}]).to_csv(raw_dir / "icio_2000.csv", index=False)
    pd.DataFrame(
        {
            "country": ["AAA"],
            "country_name": ["Country AAA"],
            "period": ["2000-Q1"],
            "year": [2000],
            "quarter": [1],
            "real_gdp": [100.0],
        }
    ).to_csv(real_gdp_dir / "quarterly_real_gdp.csv", index=False)

    out_path = build_c_diff_quarterly(cfg)
    out = pd.read_parquet(out_path)

    assert out.loc[0, "period"] == "2000-Q1"
    assert out.loc[0, "status"] == "missing_quarterly_real_gdp_growth"
    assert np.isnan(out.loc[0, "c_diff"])
