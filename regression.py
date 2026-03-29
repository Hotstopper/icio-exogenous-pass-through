from __future__ import annotations

import sys
from pathlib import Path

# Backward-compatible shim: regression logic now lives in src/io_networks/regression.py.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from io_networks.regression import *  # noqa: F403
from io_networks.regression import main


if __name__ == "__main__":
    main()
