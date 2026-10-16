"""Simplex-noise paint tool and shared noise parameter options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QAbstractSpinBox, QCheckBox, QDoubleSpinBox, QLabel, QSpinBox, QWidget

from oswald_globe.algorithms.apply_value import ApplyValue
from oswald_globe.algorithms.geo_simplex_noise import GeoSimplexNoise
from oswald_globe.tools.brush_command import BrushCommand

from .tool import BrushTool
from .tool_options_dialog import ToolOptionsDialog

__all__ = ["NoiseBrushCommand", "NoiseTool", "NoiseToolOptionsDialog"]


class NoiseToolOptionsDialog(ToolOptionsDialog):
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        title: str = "Noise Tool Options",
        minimum_value: float = -32767.0,
        maximum_value: float = 32767.0,
        frequency: float = 1.0,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        octaves: int = 6,
        seed: int = 0,
        billow: bool = False,
        radius_km: int = 100,
        falloff_percent: int = 50,
        include_radius: bool = True,
        include_falloff: bool = True,
    ):
        super().__init__(
            parent,
            title=title,
            radius_km=radius_km,
            falloff_percent=falloff_percent,
            include_value=False,
            include_mode=False,
            include_radius=include_radius,
            include_falloff=include_falloff,
        )

        self.minimum_value_spin = self._create_double_spin(-32767.0, 32767.0, minimum_value, 1.0)
        self.maximum_value_spin = self._create_double_spin(-32767.0, 32767.0, maximum_value, 1.0)
        self.frequency_spin = self._create_double_spin(0.0, 100.0, frequency, 0.1)
        self.persistence_spin = self._create_double_spin(0.0, 5.0, persistence, 0.1)
        self.lacunarity_spin = self._create_double_spin(0.0, 5.0, lacunarity, 0.1)
        self.octaves_spin = self._create_int_spin(0, 20, octaves, 1)
        self.seed_spin = self._create_int_spin(-(2**31), 2**31 - 1, seed, 1)
        self.billow_check = QCheckBox("Billow")
        self.billow_check.setChecked(bool(billow))

        inherited_rows = list(self._rows)
        self._rows = [
            (QLabel("Minimum value"), self.minimum_value_spin, None),
            (QLabel("Maximum value"), self.maximum_value_spin, None),
            (QLabel("Frequency"), self.frequency_spin, None),
            (QLabel("Persistence"), self.persistence_spin, None),
            (QLabel("Lacunarity"), self.lacunarity_spin, None),
            (QLabel("Octaves"), self.octaves_spin, None),
            (QLabel("Seed"), self.seed_spin, None),
            (QLabel(""), self.billow_check, None),
            *inherited_rows,
        ]
        self._relayout_controls()

    @staticmethod
    def _create_double_spin(minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        spin.setKeyboardTracking(False)
        spin.setValue(float(value))
        spin.setFixedSize(spin.sizeHint())
        return spin

    @staticmethod
    def _create_int_spin(minimum: int, maximum: int, value: int, step: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        spin.setKeyboardTracking(False)
        spin.setValue(int(value))
        spin.setFixedSize(spin.sizeHint())
        return spin


@dataclass(slots=True)
class NoiseBrushCommand(BrushCommand):
    """Paint simplex-noise values onto the mesh-backed elevation layer."""

    minimum_value: float
    maximum_value: float
    frequency: float
    persistence: float
    lacunarity: float
    octaves: int
    seed: int
    apply_vertex_values: Callable[[np.ndarray, np.ndarray], None]
    billow: bool = False

    def _apply(self, indices: np.ndarray, values: np.ndarray) -> None:
        self.apply_vertex_values(indices, values)

    def _capture_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inner_radius, outer_radius, angular_distances, vertex_indices = self._capture_vertex_distances(
            self.mesh_grid,
            self.target_lat_deg,
            self.target_lon_deg,
            self.radius_km,
            self.falloff_percent,
        )
        layer = self.mesh_grid.get_layer("elevation")
        before_values = layer.values[vertex_indices].astype(np.float64, copy=True)
        after_values = before_values.copy()
        if vertex_indices.size == 0:
            return vertex_indices, before_values, after_values

        brush_weights = np.zeros(before_values.shape, dtype=np.float64)
        if inner_radius >= outer_radius - 1e-12:
            brush_weights.fill(1.0)
        else:
            affected_distances = angular_distances[vertex_indices]
            inner_mask = affected_distances <= inner_radius + 1e-12
            brush_weights[inner_mask] = 1.0
            transition_mask = ~inner_mask
            if np.any(transition_mask):
                brush_weights[transition_mask] = 1.0 - (
                    (affected_distances[transition_mask] - inner_radius) / (outer_radius - inner_radius)
                )

        if self.mesh_grid.has_selection:
            brush_weights *= self.mesh_grid.selection_values[vertex_indices]

        lat_deg, lon_deg = self.mesh_grid.vertex_lat_lon()
        noise_values = GeoSimplexNoise.sample_many(
            lat_deg[vertex_indices],
            lon_deg[vertex_indices],
            minimum_value=self.minimum_value,
            maximum_value=self.maximum_value,
            frequency=self.frequency,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
            octaves=self.octaves,
            seed=self.seed,
            billow=self.billow,
        )
        after_values = ApplyValue.blend(before_values, noise_values, weights=brush_weights)
        return vertex_indices, before_values, after_values


class NoiseTool(BrushTool):
    _menu_label = "Noise"

    def __init__(self, parent: QWidget, on_brush_changed: Callable[[], None]):
        super().__init__(parent, on_brush_changed)
        self._minimum_value = -32767.0
        self._maximum_value = 32767.0
        self._frequency = 1.0
        self._persistence = 0.5
        self._lacunarity = 2.0
        self._octaves = 6
        self._seed = 0
        self._billow = False

    def create_menu_action(
        self,
        tools_menu,
        action_group: Any,
        on_selected: Callable[[], None],
    ) -> Any:
        noise_menu = tools_menu.addMenu(self._menu_label)
        noise_menu.menuAction().setCheckable(True)
        noise_menu.menuAction().triggered.connect(
            lambda checked=False: (noise_menu.menuAction().setChecked(True), on_selected())[1]
        )
        action = QAction("Simplex", self._parent)
        action.setCheckable(True)
        action.triggered.connect(
            lambda checked=False: (noise_menu.menuAction().setChecked(True), on_selected())[1]
        )
        noise_menu.addAction(action)
        action_group.addAction(noise_menu.menuAction())
        action_group.addAction(action)
        self._menu_action = noise_menu.menuAction()
        return noise_menu.menuAction()

    def create_options_dialog(self) -> NoiseToolOptionsDialog:
        dialog = NoiseToolOptionsDialog(
            self._parent,
            minimum_value=self._minimum_value,
            maximum_value=self._maximum_value,
            frequency=self._frequency,
            persistence=self._persistence,
            lacunarity=self._lacunarity,
            octaves=self._octaves,
            seed=self._seed,
            billow=self._billow,
            radius_km=self._radius_km,
            falloff_percent=self._falloff_percent,
        )
        dialog.radius_slider.valueChanged.connect(self._notify_brush_changed)
        dialog.falloff_slider.valueChanged.connect(self._notify_brush_changed)
        return dialog

    def dispose_options_dialog(self) -> None:
        if isinstance(self._options_dialog, NoiseToolOptionsDialog):
            self._minimum_value = float(self._options_dialog.minimum_value_spin.value())
            self._maximum_value = float(self._options_dialog.maximum_value_spin.value())
            self._frequency = float(self._options_dialog.frequency_spin.value())
            self._persistence = float(self._options_dialog.persistence_spin.value())
            self._lacunarity = float(self._options_dialog.lacunarity_spin.value())
            self._octaves = int(self._options_dialog.octaves_spin.value())
            self._seed = int(self._options_dialog.seed_spin.value())
            self._billow = bool(self._options_dialog.billow_check.isChecked())
        super().dispose_options_dialog()

    def minimum_value(self) -> float:
        if isinstance(self._options_dialog, NoiseToolOptionsDialog):
            return float(self._options_dialog.minimum_value_spin.value())
        return self._minimum_value

    def maximum_value(self) -> float:
        if isinstance(self._options_dialog, NoiseToolOptionsDialog):
            return float(self._options_dialog.maximum_value_spin.value())
        return self._maximum_value

    def frequency(self) -> float:
        if isinstance(self._options_dialog, NoiseToolOptionsDialog):
            return float(self._options_dialog.frequency_spin.value())
        return self._frequency

    def persistence(self) -> float:
        if isinstance(self._options_dialog, NoiseToolOptionsDialog):
            return float(self._options_dialog.persistence_spin.value())
        return self._persistence

    def lacunarity(self) -> float:
        if isinstance(self._options_dialog, NoiseToolOptionsDialog):
            return float(self._options_dialog.lacunarity_spin.value())
        return self._lacunarity

    def octaves(self) -> int:
        if isinstance(self._options_dialog, NoiseToolOptionsDialog):
            return int(self._options_dialog.octaves_spin.value())
        return self._octaves

    def seed(self) -> int:
        if isinstance(self._options_dialog, NoiseToolOptionsDialog):
            return int(self._options_dialog.seed_spin.value())
        return self._seed

    def billow(self) -> bool:
        if isinstance(self._options_dialog, NoiseToolOptionsDialog):
            return bool(self._options_dialog.billow_check.isChecked())
        return self._billow

    def create_brush_command(self, payload: Dict[str, Any], gl_widget: Any) -> NoiseBrushCommand:
        mesh_grid = gl_widget.mesh_grid
        if mesh_grid is None:
            raise RuntimeError("Noise tool requires a mesh-backed globe.")
        return NoiseBrushCommand(
            target_lat_deg=float(payload["target_lat_deg"]),
            target_lon_deg=float(payload["target_lon_deg"]),
            radius_km=float(payload["radius_km"]),
            falloff_percent=float(payload["falloff_percent"]),
            minimum_value=self.minimum_value(),
            maximum_value=self.maximum_value(),
            frequency=self.frequency(),
            persistence=self.persistence(),
            lacunarity=self.lacunarity(),
            octaves=self.octaves(),
            seed=self.seed(),
            billow=self.billow(),
            mesh_grid=mesh_grid,
            apply_vertex_values=gl_widget.apply_mesh_vertex_values,
        )
