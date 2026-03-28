from __future__ import annotations

import sys
from pathlib import Path

# Backward-compatible shim: notebook-friendly import path.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from io_networks.local_projections import *  # noqa: F403
