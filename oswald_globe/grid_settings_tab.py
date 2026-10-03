"""Grid settings tab for the settings dialog."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QGridLayout, QLabel, QSpinBox, QWidget

from oswald_globe.app_support import APP_SETTINGS
from oswald_globe.color_button import ColorButton
from oswald_globe.globe_viewer import GlobeViewer


class ApplicationSettingsTab(QWidget):
    DEFAULT_INITIAL_GRID_DEPTH = 6
    DEFAULT_MAXIMUM_VERTEX_COUNT = 1_000_000

    def __init__(self, settings: Optional[QSettings] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._settings = settings or APP_SETTINGS

        layout = QGridLayout(self)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        layout.addWidget(QLabel("Initial grid depth"), 0, 0)
        self._initial_grid_depth = QSpinBox(self)
        self._initial_grid_depth.setRange(4, 8)
        self._initial_grid_depth.setSingleStep(1)
        self._initial_grid_depth.setValue(
            int(self._settings.value("application/initial_grid_depth", self.DEFAULT_INITIAL_GRID_DEPTH))
        )
        layout.addWidget(self._initial_grid_depth, 0, 1)

        layout.addWidget(QLabel("Maximum vertex count"), 1, 0)
        self._maximum_vertex_count = QSpinBox(self)
        self._maximum_vertex_count.setRange(100_000, 2_000_000)
        self._maximum_vertex_count.setSingleStep(100_000)
        self._maximum_vertex_count.setValue(
            int(self._settings.value("application/maximum_vertex_count", self.DEFAULT_MAXIMUM_VERTEX_COUNT))
        )
        layout.addWidget(self._maximum_vertex_count, 1, 1)

        layout.setRowStretch(2, 1)

    def settings(self) -> Dict[str, int]:
        return {
            "initial_grid_depth": int(self._initial_grid_depth.value()),
            "maximum_vertex_count": int(self._maximum_vertex_count.value()),
        }


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
