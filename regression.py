from __future__ import annotations

# Backward-compatible shim: regression logic now lives in src/io_networks/regression.py.
from io_networks.regression import *  # noqa: F403
from io_networks.regression import main


if __name__ == "__main__":
    main()
