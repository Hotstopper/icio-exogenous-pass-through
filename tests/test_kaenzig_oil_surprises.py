from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from io_networks.kaenzig_oil_surprises import build_quarterly_vintage_table


def test_build_quarterly_vintage_table_sums_three_months_per_quarter():
    monthly = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                    "2024-06-01",
                ]
            ),
            "period": ["2024M01", "2024M02", "2024M03", "2024M04", "2024M05", "2024M06"],
            "surprise": [1.0, 2.0, 3.0, 0.5, 0.5, 1.0],
            "news_shock": [10.0, 20.0, 30.0, 4.0, 5.0, 6.0],
        }
    )

    quarterly = build_quarterly_vintage_table(monthly)

    assert quarterly["period"].tolist() == ["2024Q1", "2024Q2"]
    assert quarterly["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-03-31", "2024-06-30"]
    assert quarterly["surprise"].tolist() == [6.0, 2.0]
    assert quarterly["news_shock"].tolist() == [60.0, 15.0]
