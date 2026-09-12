"""Entry point for the Oswald Globe desktop application."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oswald_globe.app import (  # noqa: E402
    DEFAULT_HEIGHTFIELD,
    create_window,
    load_elevation,
    main,
)

if __name__ == "__main__":
    main()
