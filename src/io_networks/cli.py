from __future__ import annotations

import argparse
from pathlib import Path

from io_networks.blocks import build_blocks
from io_networks.c import build_c
from io_networks.c_diff import build_c_diff
from io_networks.config import load_config
from io_networks.eda import run_eda
from io_networks.kaenzig_oil_surprises import build_kaenzig_oil_surprise_tables
from io_networks.matrices import build_yearly_a
from io_networks.policy_rates import build_policy_rate_table
from io_networks.real_gdp import build_output_gap_table, build_real_gdp_table
from io_networks.xi import build_xi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="io-net", description="ICIO dissertation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eda = subparsers.add_parser("eda", help="Run exploratory data analysis")
    eda.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    build_a = subparsers.add_parser("build-a", help="Build technical coefficient matrices A")
    build_a.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    build_blocks_cmd = subparsers.add_parser(
        "build-blocks", help="Build E/N block matrices and tau from yearly A"
    )
    build_blocks_cmd.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    build_xi_cmd = subparsers.add_parser(
        "build-xi", help="Build country-year xi metrics from tau and HFCE weights"
    )
    build_xi_cmd.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    build_c_cmd = subparsers.add_parser(
        "build-c", help="Build country-year c metrics from c_N and HFCE weights"
    )
    build_c_cmd.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    build_c_diff_cmd = subparsers.add_parser(
        "build-c-diff",
        help="Build country-year c_diff metrics from (c_N - c_gdp_N) and HFCE weights",
    )
    build_c_diff_cmd.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    build_real_gdp_cmd = subparsers.add_parser(
        "build-real-gdp",
        help="Build a unified quarterly real GDP table from raw country sources",
    )
    build_real_gdp_cmd.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    build_output_gap_cmd = subparsers.add_parser(
        "build-output-gap",
        help="Build quarterly log real GDP and HP-filtered output gaps",
    )
    build_output_gap_cmd.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    build_policy_rates_cmd = subparsers.add_parser(
        "build-policy-rates",
        help="Build a quarterly policy-rate table from monthly BIS policy rates",
    )
    build_policy_rates_cmd.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    build_kaenzig_oil_cmd = subparsers.add_parser(
        "build-kaenzig-oil-surprises",
        help="Build monthly and quarterly Kaenzig oil supply surprise vintage tables",
    )
    build_kaenzig_oil_cmd.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "eda":
        output_file = run_eda(cfg)
        print(f"EDA summary written to: {output_file}")
        return

    if args.command == "build-a":
        summary_file = build_yearly_a(cfg)
        print(f"A matrices summary written to: {summary_file}")
        return

    if args.command == "build-blocks":
        summary_file = build_blocks(cfg)
        print(f"Block matrices summary written to: {summary_file}")
        return

    if args.command == "build-xi":
        output_file = build_xi(cfg)
        print(f"Xi output written to: {output_file}")
        return

    if args.command == "build-c":
        output_file = build_c(cfg)
        print(f"C output written to: {output_file}")
        return

    if args.command == "build-c-diff":
        output_file = build_c_diff(cfg)
        print(f"C diff output written to: {output_file}")
        return

    if args.command == "build-real-gdp":
        output_file = build_real_gdp_table(args.config)
        print(f"Real GDP output written to: {output_file}")
        return

    if args.command == "build-output-gap":
        output_file = build_output_gap_table(args.config)
        print(f"Output gap table written to: {output_file}")
        return

    if args.command == "build-policy-rates":
        output_file = build_policy_rate_table(args.config)
        print(f"Policy-rate table written to: {output_file}")
        return

    if args.command == "build-kaenzig-oil-surprises":
        monthly_path, quarterly_path = build_kaenzig_oil_surprise_tables(cfg)
        print(f"Monthly Kaenzig oil surprise table written to: {monthly_path}")
        print(f"Quarterly Kaenzig oil surprise table written to: {quarterly_path}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
