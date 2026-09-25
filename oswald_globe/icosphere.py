"""Compatibility exports for adaptive icosphere mesh support."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .icosphere_grid import IcosphereBuildCancelled, IcosphereGrid, _xyz_to_latlon
    from .layer import Layer, LayerLegend
except ImportError:  # pragma: no cover - supports running as a script
    from oswald_globe.icosphere_grid import IcosphereBuildCancelled, IcosphereGrid, _xyz_to_latlon
    from oswald_globe.layer import Layer, LayerLegend

__all__ = [
    "IcosphereBuildCancelled",
    "IcosphereGrid",
    "Layer",
    "LayerLegend",
    "_xyz_to_latlon",
]
