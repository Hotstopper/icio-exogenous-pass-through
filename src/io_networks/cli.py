from __future__ import annotations

import argparse
from pathlib import Path

from io_networks.config import load_config
from io_networks.eda import run_eda


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="io-net", description="ICIO dissertation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eda = subparsers.add_parser("eda", help="Run exploratory data analysis")
    eda.add_argument("--config", type=Path, default=Path("config/default.yaml"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "eda":
        output_file = run_eda(cfg)
        print(f"EDA summary written to: {output_file}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
