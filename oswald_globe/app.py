"""Compatibility entrypoint for the desktop application."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .globe_main_window import (
        GlobeMainWindow,
        IcosphereBuildWorker,
        IcosphereProgressDialog,
        create_window,
        load_elevation,
        main,
    )
except ImportError:  # pragma: no cover - supports running as a script
    from oswald_globe.globe_main_window import (
        GlobeMainWindow,
        IcosphereBuildWorker,
        IcosphereProgressDialog,
        create_window,
        load_elevation,
        main,
    )

__all__ = [
    "GlobeMainWindow",
    "IcosphereBuildWorker",
    "IcosphereProgressDialog",
    "create_window",
    "load_elevation",
    "main",
]


if __name__ == "__main__":
    main()
