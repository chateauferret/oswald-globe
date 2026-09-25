"""Shared helpers for the desktop application UI."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor

from oswald_globe.icosphere import IcosphereGrid

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEIGHTFIELD = PACKAGE_ROOT / "data" / "heightfield.tif"
APP_SETTINGS = QSettings("Oswald Globe", "Oswald Globe")


def empty_globe_elevation() -> np.ndarray:
    """Return a sea-level data array for a default globe without a heightfield."""
    return np.zeros((2, 2), dtype=np.float32)


def generic_icosphere_mesh(level: int = 7) -> IcosphereGrid:
    """Build a uniform icosphere mesh at the requested subdivision level."""
    mesh = IcosphereGrid()
    mesh.subdivide_uniform(max(0, int(level)))
    return mesh


def load_elevation(heightfield: Path) -> np.ndarray:
    """Load the normalized heightfield using terrain elevation units."""
    with Image.open(heightfield) as image:
        heightfield_data = np.asarray(image, dtype=np.float32)
    return heightfield_data * 4096.0 - 1024.0


def color_to_qcolor(color: Union[QColor, Tuple[float, float, float], str]) -> QColor:
    if isinstance(color, QColor):
        return QColor(color)
    if isinstance(color, str):
        qcolor = QColor(color)
        if not qcolor.isValid():
            raise ValueError(f"Invalid color value: {color!r}")
        return qcolor
    return QColor.fromRgbF(float(color[0]), float(color[1]), float(color[2]))


def qcolor_to_rgb(color: QColor) -> Tuple[float, float, float]:
    return (color.redF(), color.greenF(), color.blueF())
