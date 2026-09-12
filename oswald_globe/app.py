"""Desktop application for exploring the heightfield globe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication, QMainWindow

from oswald_globe.globe_viewer import GlobeViewer

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEIGHTFIELD = PACKAGE_ROOT / "data" / "heightfield.tif"
if not DEFAULT_HEIGHTFIELD.is_file():
    DEFAULT_HEIGHTFIELD = PACKAGE_ROOT / "working" / "92757802" / "heightfield.tif"


def load_elevation(heightfield: Path) -> np.ndarray:
    """Load the normalized heightfield using terrain elevation units."""
    with Image.open(heightfield) as image:
        heightfield_data = np.asarray(image, dtype=np.float32)
    return heightfield_data * 4096.0 - 1024.0


def create_window(
    heightfield: Path = DEFAULT_HEIGHTFIELD,
    mesh_min_level: int = 2,
    mesh_max_level: int = 7,
    mesh_threshold: float = 100.0,
    width: int = 850,
    height: int = 900,
) -> QMainWindow:
    """Create a native desktop window hosting the interactive 3D globe."""
    heightfield = Path(heightfield).expanduser().resolve()
    if not heightfield.is_file():
        raise FileNotFoundError(f"Heightfield not found: {heightfield}")

    elevation = load_elevation(heightfield)

    window = QMainWindow()
    window.setWindowTitle(heightfield.stem)
    window.resize(width, height)
    window.setMinimumSize(480, 540)

    viewer = GlobeViewer(
        elevation,
        cmap="topo",
        responsive=True,
        title=heightfield.stem,
        mesh=True,
        mesh_min_level=mesh_min_level,
        mesh_max_level=mesh_max_level,
        mesh_threshold=mesh_threshold,
    )
    window.setCentralWidget(viewer)
    return window


def _configure_opengl_surface_format() -> None:
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D heightfield desktop globe.")
    parser.add_argument("--heightfield", type=Path, default=DEFAULT_HEIGHTFIELD)
    parser.add_argument("--mesh-min-level", type=int, default=2, help="Minimum icosphere subdivision level.")
    parser.add_argument("--mesh-max-level", type=int, default=7, help="Maximum icosphere subdivision level.")
    parser.add_argument("--mesh-threshold", type=float, default=100.0, help="Gradient threshold driving adaptive refinement.")
    parser.add_argument("--width", type=int, default=850, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=900, help="Window height in pixels.")
    args = parser.parse_args()

    _configure_opengl_surface_format()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = create_window(
        args.heightfield,
        mesh_min_level=args.mesh_min_level,
        mesh_max_level=args.mesh_max_level,
        mesh_threshold=args.mesh_threshold,
        width=args.width,
        height=args.height,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
