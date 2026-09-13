"""Desktop application for exploring the heightfield globe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction, QCloseEvent, QColor, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from oswald_globe.globe_viewer import GlobeViewer

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEIGHTFIELD = PACKAGE_ROOT / "data" / "heightfield.tif"
APP_SETTINGS = QSettings("Oswald Globe", "Oswald Globe")


def load_elevation(heightfield: Path) -> np.ndarray:
    """Load the normalized heightfield using terrain elevation units."""
    with Image.open(heightfield) as image:
        heightfield_data = np.asarray(image, dtype=np.float32)
    return heightfield_data * 4096.0 - 1024.0


def _color_to_qcolor(color: Union[QColor, Tuple[float, float, float], str]) -> QColor:
    if isinstance(color, QColor):
        return QColor(color)
    if isinstance(color, str):
        qcolor = QColor(color)
        if not qcolor.isValid():
            raise ValueError(f"Invalid color value: {color!r}")
        return qcolor
    return QColor.fromRgbF(float(color[0]), float(color[1]), float(color[2]))


def _qcolor_to_rgb(color: QColor) -> Tuple[float, float, float]:
    return (color.redF(), color.greenF(), color.blueF())


class ColorButton(QPushButton):
    def __init__(self, color: Union[QColor, Tuple[float, float, float], str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._color = _color_to_qcolor(color)
        self.clicked.connect(self._choose_color)
        self._update_label()

    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: Union[QColor, Tuple[float, float, float], str]) -> None:
        self._color = _color_to_qcolor(color)
        self._update_label()

    def _choose_color(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Select color")
        if chosen.isValid():
            self._color = chosen
            self._update_label()

    def _update_label(self) -> None:
        self.setText(self._color.name().upper())
        self.setToolTip(self._color.name().upper())
        self.setStyleSheet(
            f"QPushButton {{ background-color: {self._color.name()}; color: white; padding: 4px 8px; }}"
        )


class GridSettingsTab(QWidget):
    def __init__(self, viewer: GlobeViewer, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._viewer = viewer

        layout = QGridLayout(self)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        layout.addWidget(QLabel("Grid"), 0, 0)
        layout.addWidget(QLabel("Visible"), 0, 1)
        layout.addWidget(QLabel("Color"), 0, 2)
        layout.addWidget(QLabel("Opacity"), 0, 3)

        self._rows: Dict[str, Tuple[QCheckBox, ColorButton, QDoubleSpinBox]] = {}
        self._add_row(
            layout,
            1,
            "Icosphere",
            viewer.gl_widget.show_wireframe,
            viewer.gl_widget.mesh_color,
            viewer.gl_widget.mesh_opacity,
        )
        self._add_row(
            layout,
            2,
            "Graticule",
            viewer.gl_widget.is_graticule,
            viewer.gl_widget.graticule_color,
            viewer.gl_widget.graticule_opacity,
        )
        layout.setRowStretch(3, 1)

    def _add_row(
        self,
        layout: QGridLayout,
        row: int,
        name: str,
        visible: bool,
        color: Tuple[float, float, float],
        opacity: float,
    ) -> None:
        label = QLabel(name, self)
        checkbox = QCheckBox(self)
        checkbox.setChecked(bool(visible))
        color_button = ColorButton(QColor.fromRgbF(*color), self)
        opacity_spin = QDoubleSpinBox(self)
        opacity_spin.setRange(0.0, 100.0)
        opacity_spin.setDecimals(0)
        opacity_spin.setSuffix("%")
        opacity_spin.setSingleStep(5.0)
        opacity_spin.setValue(max(0.0, min(100.0, float(opacity) * 100.0)))

        self._rows[name.lower()] = (checkbox, color_button, opacity_spin)
        layout.addWidget(label, row, 0)
        layout.addWidget(checkbox, row, 1)
        layout.addWidget(color_button, row, 2)
        layout.addWidget(opacity_spin, row, 3)

    def settings(self) -> Dict[str, Dict[str, Union[bool, float, QColor]]]:
        result: Dict[str, Dict[str, Union[bool, float, QColor]]] = {}
        for name, (checkbox, color_button, opacity_spin) in self._rows.items():
            result[name] = {
                "visible": checkbox.isChecked(),
                "color": color_button.color(),
                "opacity": opacity_spin.value() / 100.0,
            }
        return result


class SettingsDialog(QDialog):
    def __init__(
        self,
        viewer: GlobeViewer,
        apply_callback: Callable[[Dict[str, Dict[str, Union[bool, float, QColor]]]], None],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(520, 220)
        self._apply_callback = apply_callback

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        self.grid_tab = GridSettingsTab(viewer, self.tabs)
        self.tabs.addTab(self.grid_tab, "Grids")
        main_layout.addWidget(self.tabs)

        buttons = QDialogButtonBox(self)
        self.apply_button = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.ApplyRole)
        self.ok_button = buttons.addButton("OK", QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_button = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self.apply_button.clicked.connect(self.apply_changes)
        self.ok_button.clicked.connect(self.accept_changes)
        self.cancel_button.clicked.connect(self.reject)
        main_layout.addWidget(buttons)

    def apply_changes(self) -> None:
        self._apply_callback(self.grid_tab.settings())

    def accept_changes(self) -> None:
        self.apply_changes()
        self.accept()


class GlobeMainWindow(QMainWindow):
    def __init__(
        self,
        heightfield: Path = DEFAULT_HEIGHTFIELD,
        mesh_min_level: int = 2,
        mesh_max_level: int = 7,
        mesh_threshold: float = 100.0,
        width: int = 850,
        height: int = 900,
        settings: Optional[QSettings] = None,
    ):
        super().__init__()
        self._mesh_min_level = mesh_min_level
        self._mesh_max_level = mesh_max_level
        self._mesh_threshold = mesh_threshold
        self._save_path: Optional[Path] = None
        self._settings = settings or APP_SETTINGS

        self._set_heightfield(Path(heightfield))
        self.resize(width, height)
        self.setMinimumSize(480, 540)
        self._build_menu_bar()

    @property
    def viewer(self) -> GlobeViewer:
        return self.centralWidget()  # type: ignore[return-value]

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_heightfield)
        file_menu.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_heightfield)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save As...", self)
        save_as_action.triggered.connect(self.save_heightfield_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        file_menu.addAction(settings_action)

    def _set_heightfield(self, heightfield: Path) -> None:
        heightfield = Path(heightfield).expanduser().resolve()
        if not heightfield.is_file():
            raise FileNotFoundError(f"Heightfield not found: {heightfield}")

        elevation = load_elevation(heightfield)
        existing_settings = None
        if isinstance(self.centralWidget(), GlobeViewer):
            existing_settings = self._current_grid_settings()

        viewer = GlobeViewer(
            elevation,
            cmap="topo",
            responsive=True,
            title=heightfield.stem,
            mesh=True,
            mesh_min_level=self._mesh_min_level,
            mesh_max_level=self._mesh_max_level,
            mesh_threshold=self._mesh_threshold,
        )

        if existing_settings is not None:
            self._apply_grid_settings_to_viewer(viewer, existing_settings)
        else:
            self._load_grid_settings(viewer)

        old_viewer = self.centralWidget()
        self.setCentralWidget(viewer)
        if old_viewer is not None:
            old_viewer.deleteLater()

        self.setWindowTitle(heightfield.stem)
        self._heightfield = heightfield
        self._save_path = None

    def _current_grid_settings(self) -> Dict[str, Dict[str, Union[bool, float, QColor]]]:
        viewer = self.viewer
        return {
            "icosphere": {
                "visible": viewer.gl_widget.show_wireframe,
                "color": self._grid_color("icosphere", QColor.fromRgbF(*viewer.gl_widget.mesh_color)),
                "opacity": self._grid_opacity("icosphere", viewer.gl_widget.mesh_opacity),
            },
            "graticule": {
                "visible": viewer.gl_widget.is_graticule,
                "color": self._grid_color("graticule", QColor.fromRgbF(*viewer.gl_widget.graticule_color)),
                "opacity": self._grid_opacity("graticule", viewer.gl_widget.graticule_opacity),
            },
        }

    def _grid_color(self, grid: str, default: QColor) -> QColor:
        color = QColor(str(self._settings.value(f"grids/{grid}/color", default.name())))
        return color if color.isValid() else default

    def _grid_opacity(self, grid: str, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(self._settings.value(f"grids/{grid}/opacity", default))))
        except (TypeError, ValueError):
            return default

    def _grid_visible(self, grid: str, default: bool) -> bool:
        value = self._settings.value(f"grids/{grid}/visible", default)
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes")
        return bool(value)

    def _load_grid_settings(self, viewer: GlobeViewer) -> None:
        self._apply_grid_settings_to_viewer(viewer, {
            "icosphere": {
                "visible": self._grid_visible("icosphere", viewer.gl_widget.show_wireframe),
                "color": self._grid_color("icosphere", QColor.fromRgbF(*viewer.gl_widget.mesh_color)),
                "opacity": self._grid_opacity("icosphere", viewer.gl_widget.mesh_opacity),
            },
            "graticule": {
                "visible": self._grid_visible("graticule", viewer.gl_widget.is_graticule),
                "color": self._grid_color("graticule", QColor.fromRgbF(*viewer.gl_widget.graticule_color)),
                "opacity": self._grid_opacity("graticule", viewer.gl_widget.graticule_opacity),
            },
        })

    def _save_grid_settings(self) -> None:
        settings = self._current_grid_settings()
        for grid in ("icosphere", "graticule"):
            self._settings.setValue(f"grids/{grid}/visible", settings[grid]["visible"])
            self._settings.setValue(f"grids/{grid}/color", settings[grid]["color"].name())
            self._settings.setValue(f"grids/{grid}/opacity", settings[grid]["opacity"])

    def _apply_grid_settings_to_viewer(
        self,
        viewer: GlobeViewer,
        settings: Dict[str, Dict[str, Union[bool, float, QColor]]],
    ) -> None:
        icosphere = settings["icosphere"]
        graticule = settings["graticule"]
        viewer.set_mesh_wireframe(
            bool(icosphere["visible"]),
            icosphere["color"],
            float(icosphere["opacity"]),
        )
        viewer.set_graticule(
            bool(graticule["visible"]),
            graticule["color"],
            float(graticule["opacity"]),
        )

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.viewer, self.apply_grid_settings, self)
        dialog.exec()

    def apply_grid_settings(self, settings: Dict[str, Dict[str, Union[bool, float, QColor]]]) -> None:
        self._apply_grid_settings_to_viewer(self.viewer, settings)
        self._save_grid_settings()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_grid_settings()
        self._settings.sync()
        super().closeEvent(event)

    def open_heightfield(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Heightfield",
            str(self._heightfield.parent),
            "Supported Files (*.tif *.tiff *.npy *.npz);;All Files (*)",
        )
        if file_name:
            self._set_heightfield(Path(file_name))

    def save_heightfield(self) -> None:
        if self._save_path is None:
            self.save_heightfield_as()
            return
        self._save_heightfield(self._save_path)

    def save_heightfield_as(self) -> None:
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Heightfield",
            str(self._heightfield.with_suffix(".npy")),
            "NumPy Array (*.npy);;Compressed NumPy Array (*.npz);;TIFF (*.tif *.tiff)",
        )
        if file_name:
            self._save_path = Path(file_name)
            self._save_heightfield(self._save_path)

    def _save_heightfield(self, path: Path) -> None:
        path = Path(path)
        data = self.viewer.data
        suffix = path.suffix.lower()
        if suffix == ".npy":
            np.save(path, data)
        elif suffix == ".npz":
            np.savez_compressed(path, elevation=data)
        elif suffix in (".tif", ".tiff"):
            Image.fromarray(np.asarray(data, dtype=np.float32)).save(path)
        else:
            raise ValueError(f"Unsupported save format: {path.suffix}")


def create_window(
    heightfield: Path = DEFAULT_HEIGHTFIELD,
    mesh_min_level: int = 2,
    mesh_max_level: int = 7,
    mesh_threshold: float = 100.0,
    width: int = 850,
    height: int = 900,
    settings: Optional[QSettings] = None,
) -> QMainWindow:
    """Create a native desktop window hosting the interactive 3D globe."""
    return GlobeMainWindow(
        heightfield=heightfield,
        mesh_min_level=mesh_min_level,
        mesh_max_level=mesh_max_level,
        mesh_threshold=mesh_threshold,
        width=width,
        height=height,
        settings=settings,
    )


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
    parser.add_argument("--mesh-max-level", type=int, default=8, help="Maximum icosphere subdivision level.")
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
